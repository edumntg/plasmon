"""Funded jobs: escrow at submission, holds per round, release and fee at the end, refund on
cancel, the download gate, and sealed shards that only the trainer and the coordinator open."""

from __future__ import annotations

import socket
import threading
import time
from email import message_from_bytes, policy

import httpx
import pytest
import uvicorn
from plasmon import jobs
from plasmon.client import ApiError, Client
from plasmon.coordinator.app import create_app
from plasmon.coordinator.config import (
    AuthConfig,
    CreditsConfig,
    EmailConfig,
    PolicyConfig,
    ServerConfig,
)
from plasmon.core import jobspec, sealed
from plasmon.core.identity import Identity
from plasmon.trainer.agent import Agent
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.timeout(420)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def funded(tmp_path_factory):
    root = tmp_path_factory.mktemp("funded")
    port = _port()
    cfg = ServerConfig(
        mode="public", host="127.0.0.1", port=port, data_dir=str(root), public_url=f"http://127.0.0.1:{port}",
        auth=AuthConfig(open_registration=True), policy=PolicyConfig(heartbeat_interval_s=2, idle_poll_interval_s=1), tick_interval_s=0.3,
        credits=CreditsConfig(enabled=True, fee_pct=10, grant_on_register=1000), email=EmailConfig(outbox_dir=str(root / "outbox")),
    )
    srv = uvicorn.Server(uvicorn.Config(create_app(cfg), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{base}/v1/healthz", timeout=1).raise_for_status()
            break
        except Exception:
            time.sleep(0.1)
    yield base, root / "outbox"
    srv.should_exit = True


def _spec(dataset_dir, name: str, rounds: int, extra: str) -> jobspec.JobSpec:
    return jobspec.loads(
        f"""
name: {name}
model: {{arch: mlp}}
dataset: {{source: "{dataset_dir.as_posix()}", shard_size: 1000}}
recipe: {{inner_steps: 10, batch_size: 64}}
requirements: {{min_trainers: 1, max_trainers: 1, round_timeout_s: 60}}
{extra}
"""
    )


def _user(base: str, email: str) -> Client:
    c = Client(base)
    try:
        c.register(email, "password-123")
    except ApiError as e:  # the same account from an earlier test of this module
        if e.status != 409:
            raise
    c.login(email, "password-123")
    return c


def _agent(base: str, token: str, name: str, path) -> tuple[Agent, threading.Thread]:
    agent = Agent(base, token, Identity.generate(), name=name, device="cpu")
    agent.machine_creds_path = path
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    return agent, thread


def _stop(agent: Agent, thread: threading.Thread) -> None:
    agent.stop()
    thread.join(timeout=30)


def test_funded_job_holds_then_releases(funded, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PLASMON_CACHE_DIR", str(tmp_path / "cache"))
    base, outbox = funded
    admin = _user(base, "admin@funded.test")  # the first account owns the server; the people below are members
    owner = _user(base, "owner@funded.test")
    lender = _user(base, "lender@funded.test")
    job = jobs.submit(owner, _spec(dataset_dir, "funded", 2, "budget: {rounds: 2, funding: 200}"), progress=lambda s: None)
    assert job["funding"] == 200 and job["per_round"] == 100 and job["held"] == 0 and job["downloadable"] is False and job["theta"] is None
    me = owner.get("/v1/credits/me")
    assert me["balance"] == 800 and me["locked"] == 200
    with httpx.Client(base_url=base, follow_redirects=False, timeout=30) as web:
        r = web.post("/login", data={"email": "owner@funded.test", "password": "password-123", "next": "/"})
        web.cookies.update(r.cookies)
        assert web.get(f"/jobs/{job['id']}/download").status_code == 403
        page = web.get(f"/jobs/{job['id']}")
        assert "Weights unlock when the job ends" in page.text and "Download latest weights" not in page.text

    agent, thread = _agent(base, lender.token, "lender-pc", tmp_path / "lender.toml")
    try:
        final = owner.wait_job(job["id"], timeout_s=180)
    finally:
        _stop(agent, thread)
    assert final["status"] == "completed", final
    assert final["held"] == 200 and final["downloadable"] is True and final["theta"]
    settlement = final["settlement"]
    assert settlement["outcome"] == "completed" and settlement["funding"] == 200 and settlement["paid"] == 180 and settlement["fee"] == 20 and settlement["refund"] == 0
    assert len(settlement["payouts"]) == 1 and settlement["payouts"][0]["machine"] == "lender-pc" and settlement["payouts"][0]["rounds"] == 2
    assert final["credits_spent"] == 0  # funding is not pay-per-round spend

    lender_me = lender.get("/v1/credits/me")
    assert lender_me["balance"] == 1000 + 180 and lender_me["on_hold"] == 0
    assert [h["status"] for h in lender_me["holds"]] == ["released", "released"] and sum(h["amount"] for h in lender_me["holds"]) == 180
    assert {e["kind"] for e in lender_me["entries"]} == {"grant", "earn"}
    owner_me = owner.get("/v1/credits/me")
    assert owner_me["balance"] == 800 and owner_me["locked"] == 0
    assert [e["kind"] for e in owner_me["entries"]][:1] == ["escrow"]
    balances = {b["email"]: b["balance"] for b in admin.get("/v1/credits/users")}
    assert balances["(fee account)"] == 20 and sum(balances.values()) == 3000
    updates = owner.job_updates(job["id"])
    assert all(u["status"] == "accepted" and u["credits"] == 90 for u in updates)
    kinds = [e["kind"] for e in owner.ledger(job=job["id"])]
    assert "job_settled" in kinds and "job_finished" in kinds
    settled = next(e for e in owner.ledger(job=job["id"]) if e["kind"] == "job_settled")
    assert settled["body"]["paid"] == 180 and settled["body"]["payouts"][0]["amount"] == 180
    rounds = [e for e in owner.ledger(job=job["id"]) if e["kind"] == "round"]
    assert all(e["body"]["credits_held"] == 100 and e["body"]["updates"][0]["held"] == 90 for e in rounds)

    with httpx.Client(base_url=base, follow_redirects=False, timeout=30) as web:
        r = web.post("/login", data={"email": "owner@funded.test", "password": "password-123", "next": "/"})
        web.cookies.update(r.cookies)
        assert web.get(f"/jobs/{job['id']}/download").status_code == 200
        page = web.get(f"/jobs/{job['id']}")
        # a member sees machines by node id prefix, as everywhere else on the dashboard
        assert "Payouts" in page.text and agent.identity.node_id[:8] in page.text and "lender-pc" not in page.text and "Data exposure" in page.text
        credits_page = web.get("/credits")
        assert credits_page.status_code == 200 and "locked in your running jobs" in credits_page.text
    for _ in range(50):
        if len(list(outbox.glob("*.eml"))) >= 2:
            break
        time.sleep(0.1)
    mails = [message_from_bytes(p.read_bytes(), policy=policy.default) for p in outbox.glob("*.eml")]
    subjects = {(m["To"], m["Subject"]) for m in mails}
    assert ("owner@funded.test", "plasmon: funded completed") in subjects
    assert ("lender@funded.test", "plasmon: 180 credits released from funded") in subjects


def test_cancelled_funded_job_pays_closed_rounds_and_refunds_the_rest(funded, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PLASMON_CACHE_DIR", str(tmp_path / "cache"))
    base, _ = funded
    owner = _user(base, "canceller@funded.test")
    lender = _user(base, "patient@funded.test")
    job = jobs.submit(owner, _spec(dataset_dir, "cancel-me", 6, "budget: {rounds: 6, funding: 600}"), progress=lambda s: None)
    agent, thread = _agent(base, lender.token, "patient-pc", tmp_path / "patient.toml")
    try:
        for _ in range(600):
            current = owner.job(job["id"])
            if current["round"] >= 1:
                break
            time.sleep(0.2)
        final = owner.cancel_job(job["id"])
    finally:
        _stop(agent, thread)
    closed = final["round"]
    assert final["status"] == "cancelled" and 1 <= closed < 6
    s = final["settlement"]
    assert s["outcome"] == "cancelled" and s["paid"] == 90 * closed and s["fee"] == 10 * closed and s["refund"] == 600 - 100 * closed
    assert owner.get("/v1/credits/me")["balance"] == 1000 - 100 * closed
    assert lender.get("/v1/credits/me")["balance"] == 1000 + 90 * closed
    assert final["downloadable"] is True and final["theta"]


def test_sealed_sticky_shards_train_and_stay_private(funded, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PLASMON_CACHE_DIR", str(tmp_path / "cache"))
    base, _ = funded
    admin = _user(base, "admin@funded.test")
    owner = _user(base, "private@funded.test")
    spec = _spec(dataset_dir, "sealed", 3, "privacy: {encrypt_shards: true, sticky_shards: true, max_shards_per_machine: 1}\nbudget: {rounds: 3}")
    job = jobs.submit(owner, spec, progress=lambda s: None)
    assert job["sealed"] is True
    url = admin.server_status()["db"]["url"]
    engine = create_engine(url)
    with engine.connect() as conn:
        shards = conn.execute(text("SELECT shards, eval_blob FROM jobs WHERE id = :id"), {"id": job["id"]}).one()
    engine.dispose()
    import json

    manifest = json.loads(shards[0]) if isinstance(shards[0], str) else shards[0]
    first = owner.get_blob(manifest[0]["blob"])
    assert sealed.is_sealed(first) and sealed.is_sealed(owner.get_blob(shards[1]))
    assert not first.startswith(b"PK")  # not an .npz: the blob store never sees the samples

    agent, thread = _agent(base, owner.token, "private-pc", tmp_path / "private.toml")
    try:
        final = owner.wait_job(job["id"], timeout_s=180)
    finally:
        _stop(agent, thread)
    assert final["status"] == "completed" and final["eval_acc"] > 0.3, final
    updates = owner.job_updates(job["id"])
    assert len(updates) == 3 and len({u["shard"] for u in updates}) == 1  # one machine, one shard, every round
    assert final["exposure"] == [{"machine": "private-pc", "node": agent.identity.node_id, "shards": [updates[0]["shard"]], "count": 1, "fraction": 1 / len(manifest)}]
    cached = tmp_path / "cache" / "blobs" / manifest[updates[0]["shard"]]["blob"]
    assert cached.exists() and sealed.is_sealed(cached.read_bytes())  # the trainer caches ciphertext only
