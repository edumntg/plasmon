"""Open jobs, join mode, owner approval, mail to the owner and to the machine's owner.

Machines here are fake: they register and heartbeat without training, so the tests take
seconds. Training with funding and sealed shards is in test_funding.py.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from email import message_from_bytes, policy
from pathlib import Path
from urllib.parse import unquote

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
from plasmon.core import jobspec
from plasmon.core.identity import Identity

pytestmark = pytest.mark.timeout(300)

IDLE = {"status": "idle", "status_detail": "", "job_id": None, "round": None, "metrics": {}, "logs": []}


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
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


def _spec(dataset_dir, name: str, extra: str = "") -> jobspec.JobSpec:
    return jobspec.loads(
        f"""
name: {name}
model: {{arch: mlp}}
dataset: {{source: "{dataset_dir.as_posix()}", shard_size: 1000}}
recipe: {{inner_steps: 5}}
requirements: {{min_trainers: 1, max_trainers: 4, round_timeout_s: 60}}
budget: {{rounds: 2}}
{extra}
"""
    )


def _user(base: str, email: str) -> Client:
    c = Client(base)
    c.register(email, "password-123")
    c.login(email, "password-123")
    return c


def _machine(base: str, user: Client, name: str, tflops: float | None = None) -> tuple[Client, str]:
    ident = Identity.generate()
    uid = user.me()["user"]["id"]
    hardware = {"os": "Linux 6.8", "cpu_count": 16, "ram_gb": 64, "gpu": {"kind": "cuda", "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24}}
    if tflops is not None:
        hardware["tflops"] = tflops
    out = user.register_machine({"node_id": ident.node_id, "name": name, "hardware": hardware, "versions": {}, "signature": ident.sign({"node_id": ident.node_id, "user": uid})})
    return Client(base, out["machine_token"]), ident.node_id


def _mails(outbox: Path) -> list:
    return sorted((message_from_bytes(p.read_bytes(), policy=policy.default) for p in outbox.glob("*.eml")), key=lambda m: m["Date"] or "")


def test_join_mode_open_list_and_leave(market, dataset_dir):
    base, _ = market
    owner = _user(base, "owner@market.test")
    lender = _user(base, "lender@market.test")
    pc_a, node_a = _machine(base, lender, "pc-a", tflops=40)
    pc_b, node_b = _machine(base, lender, "pc-b", tflops=40)

    joinable = jobs.submit(owner, _spec(dataset_dir, "join-me", "enrolment: {mode: join}"), progress=lambda s: None)
    assert joinable["enrolment"] == {"mode": "join", "approval": "none"}
    # nobody is assigned until a machine joins
    assert pc_a.heartbeat(IDLE)["assignment"] is None
    reason = owner.job(joinable["id"])["waiting_reason"]
    assert reason.startswith("2 free machines fit; none has joined") and joinable["id"] in reason

    listed = lender.open_jobs()
    assert [j["id"] for j in listed] == [joinable["id"]]
    view = listed[0]
    assert view["enrolment"]["mode"] == "join" and view["trainers_now"] == 0 and view["funding"] == 0
    assert {m["name"]: m["standing"] for m in view["mine"]} == {"pc-a": "can join", "pc-b": "can join"}
    assert "join-me" in [j["name"] for j in pc_a.open_jobs()]  # a machine token sees the list too

    # join by name, then the next heartbeat carries the round
    row = lender.join_job(joinable["id"], "pc-a")
    assert row["status"] == "approved" and row["machine"] == "pc-a"
    assert lender.join_job(joinable["id"], "pc-a")["status"] == "approved"  # idempotent
    assigned = pc_a.heartbeat(IDLE)["assignment"]
    assert assigned is not None and assigned["job_id"] == joinable["id"] and "data_key" not in assigned
    assert [m["standing"] for m in lender.open_jobs()[0]["mine"] if m["name"] == "pc-a"] == ["approved"]

    # an automatic job reaches pc-b unless it opts out
    automatic = jobs.submit(owner, _spec(dataset_dir, "auto-job"), progress=lambda s: None)
    assert pc_b.leave_job(automatic["id"])["status"] == "left"
    assert pc_b.heartbeat(IDLE)["assignment"] is None
    assert pc_b.join_job(joinable["id"])["status"] == "approved"  # a machine joins itself
    assert pc_b.heartbeat(IDLE)["assignment"]["job_id"] == joinable["id"]
    with pytest.raises(ApiError) as e:
        lender.join_job(joinable["id"], "no-such-machine")
    assert e.value.status == 404
    owner.cancel_job(joinable["id"])
    owner.cancel_job(automatic["id"])


def test_owner_approval_by_mail_and_page(market, dataset_dir):
    base, outbox = market
    owner = _user(base, "approver@market.test")
    lender = _user(base, "gpu-farm@market.test")
    slow, node_slow = _machine(base, lender, "slow-box", tflops=0.5)
    before = len(list(outbox.glob("*.eml")))

    job = jobs.submit(owner, _spec(dataset_dir, "approve-me", "enrolment: {approval: owner}\nrequirements: {min_trainers: 1, max_trainers: 4, round_timeout_s: 60, min_tflops: 50, min_honesty: 0.5}"), progress=lambda s: None)
    # a machine below the requirements is never asked; other tests leave slower machines online too
    assert slow.heartbeat(IDLE)["assignment"] is None
    assert owner.job_enrolments(job["id"]) == []
    reason = owner.job(job["id"])["waiting_reason"]
    assert re.fullmatch(r"\d+ free machines? do(es)? not meet the job requirements: 50 TFLOPS and honesty 0.5 or more", reason), reason
    with pytest.raises(ApiError) as e:
        slow.join_job(job["id"])
    assert e.value.status == 409 and "TFLOPS" in e.value.message

    # a fitting machine makes the scheduler ask the owner, once
    fast, node_fast = _machine(base, lender, "fast-box", tflops=82.6)
    reply = fast.heartbeat(IDLE)
    assert reply["assignment"] is None
    assert reply["enrolments"] == [{"job": job["id"], "name": "approve-me", "status": "pending"}]
    fast.heartbeat(IDLE)
    rows = owner.job_enrolments(job["id"])
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["source"] == "auto"
    assert rows[0]["tflops"] == 82.6 and "RTX 4090 24 GB" in rows[0]["hardware_text"] and rows[0]["owner"] == "gpu-farm@market.test"
    assert owner.job(job["id"])["waiting_reason"] == "1 machine waiting for the owner's approval"
    for _ in range(50):
        if len(list(outbox.glob("*.eml"))) >= before + 1:
            break
        time.sleep(0.1)
    mails = _mails(outbox)[before:]
    assert len(mails) == 1 and mails[0]["To"] == "approver@market.test"
    assert mails[0]["Subject"] == "plasmon: fast-box asks to train approve-me"
    body = mails[0].get_content()
    assert "82.6 TFLOPS" in body and "RTX 4090" in body and node_fast in body and f"plasmon job approve {job['id']} fast-box" in body

    # the owner approves by machine name; the next heartbeat carries the round
    decided = owner.approve(job["id"], "fast-box", note="welcome")
    assert decided["status"] == "approved" and decided["note"] == "welcome"
    assigned = fast.heartbeat(IDLE)["assignment"]
    assert assigned is not None and assigned["job_id"] == job["id"]
    assert fast.heartbeat({**IDLE, "status": "training", "job_id": job["id"], "round": 0})["enrolments"] == []
    approved_mail = [m for m in _mails(outbox)[before:] if m["To"] == "gpu-farm@market.test"]
    assert approved_mail and approved_mail[-1]["Subject"] == "plasmon: fast-box approved for approve-me" and "welcome" in approved_mail[-1].get_content()

    # a rejected machine stays off the job and cannot ask again
    other, node_other = _machine(base, lender, "other-box", tflops=60)
    other.join_job(job["id"])
    assert owner.reject(job["id"], node_other, note="not now")["status"] == "rejected"
    reply = other.heartbeat(IDLE)
    assert reply["assignment"] is None and reply["enrolments"][0]["status"] == "rejected"
    with pytest.raises(ApiError) as e:
        other.join_job(job["id"])
    assert e.value.status == 409
    assert {m["name"]: m["standing"] for m in lender.open_jobs()[0]["mine"]} == {"fast-box": "approved", "other-box": "rejected", "slow-box": "unfit"}
    with pytest.raises(ApiError):
        lender.approve(job["id"], node_other)  # not the owner

    # the pages: the owner sees the request table, the lender sees the open job and joins from the browser
    with httpx.Client(base_url=base, follow_redirects=False, timeout=30) as web:
        r = web.post("/login", data={"email": "approver@market.test", "password": "password-123", "next": "/"})
        web.cookies.update(r.cookies)
        page = web.get(f"/jobs/{job['id']}")
        assert page.status_code == 200 and 'id="trainers"' in page.text and "fast-box" in page.text and "other-box" in page.text
        assert "Weights unlock" not in page.text  # not a funded job
        r = web.post(f"/jobs/{job['id']}/enrolments/{node_other}/approve", data={"note": "changed my mind"})
        assert r.status_code == 303 and r.headers["location"] == f"/jobs/{job['id']}?result=approve:other-box#trainers"
        assert [e["status"] for e in owner.job_enrolments(job["id"]) if e["machine"] == "other-box"] == ["approved"]
        r = web.post(f"/jobs/{job['id']}/enrolments/{node_other}/reject", data={"note": ""})
        assert r.status_code == 303
    with httpx.Client(base_url=base, follow_redirects=False, timeout=30) as web:
        r = web.post("/login", data={"email": "gpu-farm@market.test", "password": "password-123", "next": "/"})
        web.cookies.update(r.cookies)
        page = web.get("/jobs/open")
        assert page.status_code == 200 and "approve-me" in page.text and "does not fit" in page.text and "Open jobs" in page.text
        assert web.get("/partials/jobs-open").status_code == 200
        machine_page = web.get(f"/machine/{node_fast}")
        assert machine_page.status_code == 200 and "Jobs for this machine" in machine_page.text and "joined" in machine_page.text and "82.6" in machine_page.text
        r = web.post(f"/jobs/{job['id']}/leave", data={"node_id": node_fast, "next": f"/machine/{node_fast}"})
        assert r.status_code == 303 and r.headers["location"] == f"/machine/{node_fast}?result=left:approve-me"
        r = web.post(f"/jobs/{job['id']}/join", data={"node_id": node_fast, "next": "/jobs/open"})
        assert r.headers["location"] == "/jobs/open?result=pending:approve-me"
        r = web.post(f"/jobs/{job['id']}/join", data={"node_id": node_slow, "next": "/jobs/open"})
        assert unquote(r.headers["location"]).startswith("/jobs/open?result=error:slow-box does not meet the job requirements: needs 50 TFLOPS")
    owner.cancel_job(job["id"])


def test_funding_needs_balance_and_one_payment_model(market, dataset_dir):
    base, _ = market
    owner = _user(base, "poor-owner@market.test")
    with pytest.raises(ApiError) as e:
        jobs.submit(owner, _spec(dataset_dir, "too-rich", "budget: {rounds: 2, funding: 5000}"), progress=lambda s: None)
    assert e.value.status == 402 and "locks 5000" in e.value.message
    with pytest.raises(ValueError):
        _spec(dataset_dir, "two-models", "budget: {rounds: 2, funding: 100, credits_per_1k_samples: 5}")
    with pytest.raises(ApiError) as e:
        owner.create_job({"spec": _spec(dataset_dir, "x", "privacy: {encrypt_shards: true}").model_dump(mode="json"), "init_blob": "0" * 64, "eval_blob": "0" * 64, "shards": [{"index": 0, "blob": "0" * 64, "n": 1}]})
    assert e.value.status in (400, 409)


def test_cli_open_join_approvals(market, dataset_dir, tmp_path, monkeypatch, capsys):
    from plasmon import credentials
    from plasmon.cli import main

    base, _ = market
    owner = _user(base, "cli-owner@market.test")
    lender = _user(base, "cli-lender@market.test")
    _, node = _machine(base, lender, "cli-box", tflops=12)
    job = jobs.submit(owner, _spec(dataset_dir, "cli-job", "enrolment: {mode: join, approval: owner}"), progress=lambda s: None)
    monkeypatch.setenv("PLASMON_CONFIG_DIR", str(tmp_path / "cfg"))

    credentials.save_user(credentials.UserCredentials(base, "cli-lender@market.test", lender.token))
    assert main(["job", "open"]) == 0
    out = capsys.readouterr().out
    assert "cli-job" in out and "join + approval" in out and "cli-box: can join" in out and "plasmon trainer join" in out
    assert main(["trainer", "join", job["id"], "--machine", "cli-box"]) == 0
    assert "asked to join" in capsys.readouterr().out
    assert main(["--json", "job", "open"]) == 0
    assert '"standing": "pending"' in capsys.readouterr().out

    credentials.save_user(credentials.UserCredentials(base, "cli-owner@market.test", owner.token))
    assert main(["job", "approvals", job["id"]]) == 0
    out = capsys.readouterr().out
    assert "cli-box" in out and "pending" in out and "RTX 4090" in out and f"plasmon job approve {job['id']}" in out
    assert main(["job", "approve", job["id"], "cli-box", "--note", "ok"]) == 0
    assert "approved for job" in capsys.readouterr().out
    assert [e["status"] for e in owner.job_enrolments(job["id"])] == ["approved"]

    credentials.save_user(credentials.UserCredentials(base, "cli-lender@market.test", lender.token))
    assert main(["trainer", "leave", job["id"], "--machine", node[:12]]) == 0
    assert "left job" in capsys.readouterr().out
    assert main(["trainer", "join", "job_missing", "--machine", "cli-box"]) == 1
    assert "no such job" in capsys.readouterr().err
    owner.cancel_job(job["id"])
