"""`python -m plasmon`: plain-text commands for the engine. The Rust `plasmon` binary
calls into these for the steps that need PyTorch and adds the TUI."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__, credentials
from .client import ApiError, Client
from .paths import machine_key_path

Handler = Callable[[argparse.Namespace], int]


def _print(args: argparse.Namespace, data, text: str | None = None) -> None:
    if getattr(args, "json", False):
        print(json.dumps(data, default=str, indent=2))
    elif text is not None:
        print(text)


def _client(args: argparse.Namespace, need_login: bool = True) -> Client:
    server = getattr(args, "server", None)
    creds = credentials.load_user()
    if server is None and creds is not None:
        server = creds.server
    if server is None:
        raise SystemExit("not logged in. Run: plasmon login --server http://<host>:7117")
    token = creds.token if creds and creds.server == server.rstrip("/") else None
    if need_login and token is None:
        raise SystemExit(f"not logged in to {server}. Run: plasmon login --server {server}")
    return Client(server, token)


def _ago(value) -> str:
    """`12 s`, `3 min`, `2 h` since an ISO timestamp in UTC, as the dashboard shows it."""
    if not value:
        return "never"
    import datetime as dt

    try:
        then = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if then.tzinfo is not None:
        then = then.astimezone(dt.UTC).replace(tzinfo=None)
    s = int((dt.datetime.now(dt.UTC).replace(tzinfo=None) - then).total_seconds())
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        return f"{s // 3600} h"
    return f"{s // 86400} d"


def _fmt_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


# ----- identity and login --------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    from .core.identity import Identity

    path = machine_key_path()
    if path.exists() and not args.force:
        ident = Identity.load(path)
        _print(args, {"node_id": ident.node_id, "path": str(path), "created": False}, f"machine key exists: {path}\nnode id: {ident.node_id}")
        return 0
    ident = Identity.generate()
    ident.save(path)
    _print(args, {"node_id": ident.node_id, "path": str(path), "created": True}, f"created machine key: {path}\nnode id: {ident.node_id}")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    server = args.server.rstrip("/")
    client = Client(server)
    try:
        client.healthz()
    except Exception as e:  # unreachable, wrong port, not a plasmon server
        print(f"cannot reach {server}: {e}", file=sys.stderr)
        return 1
    if args.email:
        password = args.password or getpass.getpass("password: ")
        out = client.login(args.email, password, label=f"cli on {_hostname()}")
    else:
        start = client.device_start(label=f"cli on {_hostname()}")
        url = start["verification_uri_complete"]
        print(f"Open this address in a browser and confirm the code.\n\n  {url}\n\n  code: {start['user_code']}\n")
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        deadline = time.time() + start["expires_in"]
        out = None
        while time.time() < deadline:
            time.sleep(start["interval"])
            reply = client.device_poll(start["device_code"])
            if reply.get("status") != "pending":
                out = reply
                break
            print(".", end="", flush=True)
        print()
        if out is None:
            print("code expired; run login again", file=sys.stderr)
            return 1
    credentials.save_user(credentials.UserCredentials(server, out["user"]["email"], out["token"]))
    _print(args, {"server": server, "user": out["user"]}, f"logged in to {server} as {out['user']['email']} ({out['user']['role']})")
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    creds = credentials.load_user()
    if creds:
        try:
            Client(creds.server, creds.token).post("/v1/auth/logout")
        except Exception:
            pass
        credentials.clear_user()
    print("logged out")
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    from .core.identity import Identity

    creds = credentials.load_user()
    key = machine_key_path()
    node_id = Identity.load(key).node_id if key.exists() else None
    data = {"node_id": node_id, "server": creds.server if creds else None, "user": creds.user if creds else None, "role": None}
    if creds:
        try:
            data["role"] = Client(creds.server, creds.token).me()["user"]["role"]
        except Exception as e:
            data["error"] = str(e)
    lines = [f"node id: {node_id or '(none; run plasmon init)'}"]
    lines.append(f"server:  {creds.server}\nuser:    {creds.user} ({data.get('role') or 'unknown role'})" if creds else "not logged in")
    _print(args, data, "\n".join(lines))
    return 0


def _hostname() -> str:
    import platform

    return platform.node() or "machine"


# ----- server ----------------------------------------------------------------------------

def cmd_server_init(args: argparse.Namespace) -> int:
    from .coordinator import config

    cfg = config.ServerConfig(mode=args.mode, port=args.port, org_name=args.org, public_url=args.public_url)
    cfg.auth.open_registration = args.mode in ("home", "public")
    if args.sso_issuer:
        cfg.oidc.issuer = args.sso_issuer
        cfg.oidc.client_id = args.sso_client_id
        cfg.oidc.client_secret = args.sso_client_secret
        cfg.oidc.admin_group = args.admin_group
    if args.bundle == "compose":
        from .coordinator import bundle

        if not args.domain:
            print("--domain is required for the compose bundle", file=sys.stderr)
            return 2
        out_dir = Path(args.output or "plasmon-deploy")
        files = bundle.write_bundle(out_dir, cfg, args.domain, args.storage or "minio", args.db, tls_internal=args.tls_internal)
        _print(args, {"files": [str(f) for f in files]}, "wrote:\n" + "\n".join(f"  {f}" for f in files) + f"\nnext:\n  cd {out_dir}\n  docker compose up -d\n  plasmon server bootstrap --owner you@company.com --config plasmon-server.yaml   (inside the api container, or with the same database URL)")
        return 0
    path = config.write(cfg, Path(args.config) if args.config else None)
    _print(args, {"config": str(path), "data_dir": str(cfg.resolved_data_dir())}, f"wrote {path}\ndata dir: {cfg.resolved_data_dir()}\nnext: plasmon server start")
    return 0


def cmd_server_start(args: argparse.Namespace) -> int:
    from .coordinator import run

    return run.main([args.config] if args.config else [])


def cmd_server_bootstrap(args: argparse.Namespace) -> int:
    """Create the owner account directly in the database, for a server with no users yet."""
    import secrets

    from sqlalchemy import func, select

    from .coordinator import auth, config, db

    cfg = config.load(Path(args.config) if args.config else None)
    engine = db.make_engine(cfg.resolved_db_url())
    with db.make_session_factory(engine)() as session:
        if session.scalar(select(func.count()).select_from(db.User)):
            print("the server already has users; use the dashboard to manage them", file=sys.stderr)
            return 1
        password = args.password or secrets.token_urlsafe(12)
        user = db.User(email=args.owner.lower(), name=args.name or "", password_hash=auth.hash_password(password), role="owner")
        session.add(user)
        session.commit()
    _print(args, {"email": args.owner, "password": password}, f"owner created: {args.owner}\npassword: {password}\nlog in at the dashboard and change it in Account.")
    return 0


def cmd_server_status(args: argparse.Namespace) -> int:
    client = _client(args)
    status = client.server_status()
    text = (
        f"version {status['version']}  mode {status['mode']}  uptime {status['uptime_s']} s\n"
        f"db {status['db']['url']} ({status['db']['ping_ms']} ms)\nblobs {status['blobs']['bytes']:,} B at {status['blobs']['path']}\n"
        f"scheduler running: {status['scheduler']['running']}  sse clients: {status['sse_clients']}\n"
        f"ledger entries {status['ledger']['entries']}  head {status['ledger']['head'][:16]}…\n"
        f"users {status['counts']['users']}  machines {status['counts']['machines']}  jobs {status['counts']['jobs']} ({status['counts']['jobs_running']} running)"
    )
    _print(args, status, text)
    return 0


# ----- jobs ------------------------------------------------------------------------------

def cmd_job_submit(args: argparse.Namespace) -> int:
    from . import jobs
    from .core import jobspec

    spec = jobspec.load(args.file)
    client = _client(args)
    try:
        job = jobs.submit(client, spec)
    except (FileNotFoundError, ValueError, RuntimeError) as e:  # dataset problems: a message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    url = f"{client.server}/jobs/{job['id']}"
    _print(args, job, f"submitted {job['name']} as {job['id']}\n  rounds: {job['total_rounds']}  shards: {job['shards']}  params: {job['param_count']:,}\n  watch: plasmon job watch {job['id']}\n  page:  {url}")
    return 0


def _job_rows(jobs: list[dict]) -> list[list[str]]:
    return [[j["id"], j["name"], j["status"], f"{j['round']}/{j['total_rounds']}", f"{j['eval_loss']:.3f}" if j["eval_loss"] is not None else "", f"{100 * j['eval_acc']:.1f} %" if j["eval_acc"] is not None else ""] for j in jobs]


def cmd_job_list(args: argparse.Namespace) -> int:
    jobs = _client(args).jobs(all=args.all)
    _print(args, jobs, _fmt_table(_job_rows(jobs), ["id", "name", "status", "round", "eval loss", "eval acc"]) if jobs else "no jobs")
    return 0


def cmd_job_status(args: argparse.Namespace) -> int:
    job = _client(args).job(args.id)
    rows = [[r["index"], r["status"], r["accepted"], f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "", f"{100 * r['eval_acc']:.1f} %" if r["eval_acc"] is not None else "", f"{r['bytes_in']:,}"] for r in job["rounds"]]
    head = f"{job['name']} ({job['id']})  {job['status']}  round {job['round']}/{job['total_rounds']}  params {job['param_count']:,}"
    if job.get("waiting_reason"):
        head += f"\n  waiting: {job['waiting_reason']}"
    text = head + "\n" + _fmt_table(rows, ["round", "status", "trainers", "eval loss", "eval acc", "bytes in"])
    if args.updates:
        ups = _client(args).job_updates(args.id)
        urows = [[u["round"], u["machine"], u["shard"], u["status"], f"{u['gain_assigned']:.3f}" if u["gain_assigned"] is not None else "", f"{u['gain_random']:.3f}" if u["gain_random"] is not None else "", f"{u['score']:.3f}" if u["score"] is not None else "", u["reject_reason"]] for u in ups]
        text += "\n\n" + _fmt_table(urows, ["round", "machine", "shard", "status", "gain own", "gain other", "score", "reason"])
        job["updates"] = ups
    _print(args, job, text)
    return 0


def cmd_job_watch(args: argparse.Namespace) -> int:
    client = _client(args)
    seen = -1
    while True:
        job = client.job(args.id)
        for r in job["rounds"]:
            if r["status"] == "closed" and r["index"] > seen:
                seen = r["index"]
                print(f"round {r['index']:>4}  trainers {r['accepted']:>3}  eval loss {r['eval_loss']:.4f}  acc {100 * r['eval_acc']:.1f} %  {r['bytes_in']:,} B in")
        if job["status"] != "running":
            print(f"job {job['status']}" + (f": {job['status_detail']}" if job["status_detail"] else ""))
            return 0 if job["status"] == "completed" else 1
        time.sleep(args.interval)


def cmd_job_download(args: argparse.Namespace) -> int:
    client = _client(args)
    job = client.job(args.id)
    out = Path(args.output or f"{job['name']}.safetensors")
    out.write_bytes(client.get_blob(job["theta"]))
    _print(args, {"path": str(out), "blob": job["theta"], "round": job["round"]}, f"wrote {out} (weights after round {job['round']}, blob {job['theta'][:12]}…)")
    return 0


def cmd_job_cancel(args: argparse.Namespace) -> int:
    job = _client(args).cancel_job(args.id)
    _print(args, job, f"job {job['id']} {job['status']}")
    return 0


def _pays(j: dict) -> str:
    if j.get("funding"):
        return f"{j['per_round']}/round of {j['funding']}"
    if j.get("credits_per_1k_samples"):
        return f"{j['credits_per_1k_samples']:g}/1k samples"
    return "reputation"


def _needs(req: dict) -> str:
    parts = []
    if req.get("device", "any") != "any":
        parts.append(req["device"])
    if req.get("min_vram_gb"):
        parts.append(f"{req['min_vram_gb']:g} GB VRAM")
    if req.get("min_tflops"):
        parts.append(f"{req['min_tflops']:g} TFLOPS")
    if req.get("min_honesty"):
        parts.append(f"honesty {req['min_honesty']:g}")
    return ", ".join(parts) or "any machine"


def cmd_job_open(args: argparse.Namespace) -> int:
    jobs = _client(args).open_jobs()
    rows = []
    for j in jobs:
        enrol = j["enrolment"]["mode"] + (" + approval" if j["enrolment"]["approval"] == "owner" else "")
        mine = ", ".join(f"{m['name']}: {m['standing']}" for m in j["mine"]) or "-"
        rows.append([j["id"], j["name"], j["owner"], _pays(j), _needs(j["requirements"]), enrol, f"{j['round']}/{j['total_rounds']}", j["trainers_now"], mine])
    text = _fmt_table(rows, ["id", "name", "owner", "pays", "needs", "enrolment", "round", "trainers", "your machines"]) if rows else "no job is running"
    if rows:
        text += "\n\njoin one with: plasmon trainer join <id>"
    _print(args, jobs, text)
    return 0


def _this_machine() -> str | None:
    from .core.identity import Identity

    key = machine_key_path()
    return Identity.load(key).node_id if key.exists() else None


def cmd_trainer_enrol(args: argparse.Namespace) -> int:
    client = _client(args)
    node = args.machine or _this_machine()
    joining = args.trainer_command == "join"
    out = client.join_job(args.job, node) if joining else client.leave_job(args.job, node)
    status = out["status"]
    if status == "approved":
        text = f"{out['machine']} joined job {args.job}; it takes a round at its next heartbeat"
    elif status == "pending":
        text = f"{out['machine']} asked to join job {args.job}; the owner decides and you get a mail either way"
    elif status == "left":
        text = f"{out['machine']} left job {args.job}; it finishes its current round and takes no more"
    else:
        text = f"{out['machine']}: {status}"
    _print(args, out, text)
    return 0


def _enrolment_rows(rows: list[dict]) -> list[list[str]]:
    return [[e["machine"], e["status"], e.get("owner") or "", e["hardware_text"], f"{e['tflops']:g}" if e.get("tflops") else "-", f"{e['honesty']:.2f}", e["rounds_served"], _ago(e.get("requested_at")), e.get("note") or ""] for e in rows]


def cmd_job_approvals(args: argparse.Namespace) -> int:
    rows = _client(args).job_enrolments(args.id)
    text = _fmt_table(_enrolment_rows(rows), ["machine", "status", "owner", "hardware", "tflops", "honesty", "rounds", "asked", "note"]) if rows else "no machine has asked to join this job"
    pending = [e for e in rows if e["status"] == "pending"]
    if pending:
        text += f"\n\n{len(pending)} waiting: plasmon job approve {args.id} <machine>   or   plasmon job reject {args.id} <machine>"
    _print(args, rows, text)
    return 0


def cmd_job_decide(args: argparse.Namespace) -> int:
    client = _client(args)
    approving = args.job_command == "approve"
    out = client.approve(args.id, args.machine, args.note or "") if approving else client.reject(args.id, args.machine, args.note or "")
    _print(args, out, f"{out['machine']} {out['status']} for job {args.id}" + ("; it takes a round at its next heartbeat" if approving else ""))
    return 0


# ----- trainer ---------------------------------------------------------------------------

def cmd_trainer_start(args: argparse.Namespace) -> int:
    from .trainer import agent

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise next to the round lines
    creds = credentials.load_user()
    server = args.server or (creds.server if creds else None)
    if server is None:
        print("not logged in. Run: plasmon login --server http://<host>:7117", file=sys.stderr)
        return 1
    ident = agent.machine_key_or_create(machine_key_path())
    from .coordinator import policy as policy_mod

    windows = [policy_mod.parse_hours(h) for h in (args.hours or [])]
    a = agent.Agent(server, creds.token if creds and creds.server == server else None, ident, args.name or agent.default_name(), device=args.device, max_hours=args.max_hours, local_windows=windows, never_on_battery=args.never_on_battery)
    try:
        a.run()
    except KeyboardInterrupt:
        a.stop()
        print("\nstopped")
    return 0


# ----- fleet and ledger ------------------------------------------------------------------

def cmd_fleet(args: argparse.Namespace) -> int:
    client = _client(args)
    machines = client.fleet(status=args.status)
    rows = []
    for m in machines:
        met = m.get("metrics") or {}
        gpu = (m.get("hardware") or {}).get("gpu") or {}
        rows.append([
            m["name"], m.get("owner") or "", m["status"], gpu.get("name", "none" if gpu.get("kind") in (None, "none") else gpu.get("kind")),
            f"{met.get('gpu_pct', '')}", f"{met.get('cpu_pct', '')}", f"{met.get('ram_pct', '')}",
            f"{m['current_job_id'] or ''} {('r' + str(m['current_round'])) if m['current_round'] is not None else ''}".strip(),
            f"{m['honesty']:.2f}", _ago(m.get("last_seen_at")),
        ])
    _print(args, machines, _fmt_table(rows, ["machine", "owner", "status", "gpu", "gpu%", "cpu%", "ram%", "job / round", "honesty", "seen"]) if rows else "no machines")
    return 0


def cmd_ledger_verify(args: argparse.Namespace) -> int:
    out = _client(args).ledger_verify()
    _print(args, out, f"ledger ok: {out['ok']}  entries: {out['entries']}" + (f"  problem: {out['problem']}" if out["problem"] else ""))
    return 0 if out["ok"] else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from .core import jobspec
    from .train import data, simulate

    spec = jobspec.load(args.job) if args.job else jobspec.loads(
        "name: bench\nmodel: {arch: mnist_cnn}\ndataset: {source: builtin://mnist}\n"
        "recipe: {inner_steps: 50, inner_optimizer: {lr: 2.0e-3}}\n"
    )
    train, test = data.load_mnist()
    train = data.Shard(train.x[: args.train_samples], train.y[: args.train_samples])
    test = data.Shard(test.x[:2000], test.y[:2000])
    shards = data.split_shards(train, spec.dataset.shard_size, seed=1)
    sim = simulate.run(spec, shards, test, trainers=args.trainers, rounds=args.rounds)
    steps = args.rounds * spec.recipe.inner_steps
    _, base_acc = simulate.baseline(spec, train, test, steps=steps)
    print(f"{'round':>5} {'eval loss':>9} {'eval acc':>8} {'frame bytes':>11} {'dense bytes':>11}")
    for r in sim.rounds:
        print(f"{r.round:>5} {r.eval_loss:>9.4f} {r.eval_acc:>8.4f} {r.frame_bytes:>11,} {r.dense_bytes:>11,}")
    pct = 100 * sim.total_frame_bytes / sim.total_dense_bytes
    print(f"diloco x{args.trainers}: {sim.final_acc:.4f}   single worker, {steps} steps: {base_acc:.4f}")
    print(f"traffic: {sim.total_frame_bytes:,} B sent vs {sim.total_dense_bytes:,} B dense ({pct:.1f} %)")
    return 0


# ----- fleet control, users, audit, policy -------------------------------------------------

def cmd_fleet_control(args: argparse.Namespace) -> int:
    client = _client(args)
    action = args.fleet_action
    if action == "logs":
        since = 0
        grep = args.grep
        while True:
            lines = client.get(f"/v1/fleet/{args.node}/logs", since_id=since or None, limit=200, grep=grep)
            for line in lines:
                since = max(since, line["id"])
                print(f"{str(line['at'])[11:19]} {line['level'].upper():<7} {line['message']}")
            if not args.follow:
                return 0
            time.sleep(2)
    if action == "show":
        m = client.machine(args.node)
        met = m.get("metrics") or {}
        text = (
            f"{m['name']}  {m['status']}  {m.get('status_detail') or ''}\n"
            f"owner {m.get('owner')}  node {m['node_id'][:16]}…  seen {m.get('last_seen_at')}\n"
            f"cpu {met.get('cpu_pct', '-')} %  ram {met.get('ram_pct', '-')} %  gpu {met.get('gpu_pct', '-')} %  "
            f"job {m.get('current_job_id') or '-'} round {m.get('current_round') if m.get('current_round') is not None else '-'}\n"
            f"rounds served {m['rounds_served']}  samples verified {m['samples_verified']:,}  honesty {m['honesty']:.2f}\n"
            f"paused by admin {m['paused_by_admin']}  draining {m['draining']}  tags {', '.join(m.get('tags') or []) or '-'}"
        )
        _print(args, m, text)
        return 0
    body = {"reason": getattr(args, "reason", "") or ""}
    out = client.post(f"/v1/fleet/{args.node}/{action}", body if action != "resume" else None)
    _print(args, out, f"{out['name']}: {action} requested (paused_by_admin={out['paused_by_admin']}, draining={out['draining']})")
    return 0


def cmd_users(args: argparse.Namespace) -> int:
    client = _client(args)
    action = args.users_action
    if action == "list":
        users = client.get("/v1/users")
        rows = [[u["email"], u["name"], u["role"], u["machines"], "disabled" if u["disabled"] else ""] for u in users]
        _print(args, users, _fmt_table(rows, ["email", "name", "role", "machines", ""]))
        return 0
    if action == "invite":
        out = client.post("/v1/users/invite", {"email": args.email, "role": args.role, "expires_days": args.days})
        _print(args, out, f"invite link ({out['role']}, expires {str(out['expires_at'])[:10]}):\n  {out['url']}")
        return 0
    users = client.get("/v1/users")
    target = next((u for u in users if u["email"] == args.email.lower()), None)
    if target is None:
        print(f"no user {args.email}", file=sys.stderr)
        return 1
    if action == "set-role":
        out = client.post(f"/v1/users/{target['id']}/role", {"role": args.role})
        _print(args, out, f"{out['email']} is now {out['role']}")
    elif action in ("disable", "enable"):
        out = client.post(f"/v1/users/{target['id']}/disable", {"disabled": action == "disable"})
        _print(args, out, f"{out['email']} {'disabled' if out['disabled'] else 'enabled'}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    rows = _client(args).get("/v1/audit", since_hours=args.since_hours, limit=args.limit)
    table = [[str(r["at"])[:19], r["actor"] or "", r["action"], r["target"], json.dumps(r["detail"]) if r["detail"] else ""] for r in rows]
    _print(args, rows, _fmt_table(table, ["when (UTC)", "who", "action", "target", "detail"]) if rows else "no audit events")
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.policy_action == "show":
        pol = client.get("/v1/policy")
        windows = "; ".join(f"{','.join(w['days'])} {w['start']}-{w['end']}" for w in pol["windows"]) or "always"
        _print(args, pol, f"windows: {windows}\npause on battery: {pol['pause_on_battery']}\ndrain at window end: {pol['drain_at_window_end']}")
        return 0
    from .coordinator import policy as policy_mod

    current = client.get("/v1/policy")
    if args.windows is not None:
        current["windows"] = [policy_mod.parse_hours(w).model_dump(mode="json") for w in args.windows if w]
    if args.battery is not None:
        current["pause_on_battery"] = args.battery == "pause"
    out = client.request("PUT", "/v1/policy", json=current)
    _print(args, out, "policy saved; trainers apply it at their next heartbeat")
    return 0


def cmd_trainer_service(args: argparse.Namespace) -> int:
    from .trainer import services

    if args.trainer_command == "enable":
        extra = []
        if args.name:
            extra += ["--name", args.name]
        if args.device != "any":
            extra += ["--device", args.device]
        for h in args.hours or []:
            extra += ["--hours", h]
        print(services.enable(extra, dry_run=args.dry_run))
    else:
        print(services.disable(dry_run=args.dry_run))
    return 0


def cmd_credits(args: argparse.Namespace) -> int:
    client = _client(args)
    action = getattr(args, "credits_action", None)
    if action == "grant":
        out = client.post("/v1/credits/grant", {"email": args.email, "amount": args.amount, "memo": args.memo or ""})
        _print(args, out, f"granted {out['granted']} to {out['email']}; balance now {out['balance']}")
        return 0
    if action == "users":
        rows = client.get("/v1/credits/users")
        _print(args, rows, _fmt_table([[r["email"], r["role"], r["balance"]] for r in rows], ["user", "role", "balance"]))
        return 0
    if action == "export":
        text = client._http.get("/v1/credits/export.csv", params={"since_days": args.since_days}, headers=client._headers()).text
        out_path = Path(args.output or f"plasmon-credits-{args.since_days}d.csv")
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path} ({text.count(chr(10)) - 1} entries)")
        return 0
    me = client.get("/v1/credits/me", limit=30)
    if not me["enabled"]:
        _print(args, me, "credits are off on this server")
        return 0
    rows = [[str(e["at"])[:19], e["kind"], e["amount"], e["job_id"] or "", e["memo"]] for e in me["entries"]]
    _print(args, me, f"balance: {me['balance']} {me['unit']}s\n" + (_fmt_table(rows, ["when (UTC)", "kind", "amount", "job", "memo"]) if rows else "no movements yet"))
    return 0


# ----- parser ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plasmon", description="plasmon engine")
    parser.add_argument("--version", action="version", version=f"plasmon {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--server", help="coordinator URL (default: the one you logged in to)")
    # The same two options are accepted after a subcommand. SUPPRESS keeps a leaf parser
    # from overwriting a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--server", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="create this machine's Ed25519 key", parents=[common])
    p.add_argument("--force", action="store_true")
    p.set_defaults(handler=cmd_init)

    p = sub.add_parser("login", help="log in to a coordinator (device code, or --email)", parents=[common])
    p.add_argument("--email")
    p.add_argument("--password")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(handler=cmd_login)
    sub.add_parser("logout", parents=[common]).set_defaults(handler=cmd_logout)
    sub.add_parser("whoami", parents=[common]).set_defaults(handler=cmd_whoami)

    server = sub.add_parser("server", help="run and manage a coordinator").add_subparsers(dest="server_command")
    p = server.add_parser("init", help="write plasmon-server.yaml", parents=[common])
    p.add_argument("--mode", choices=["home", "private", "public"], default="home")
    p.add_argument("--port", type=int, default=7117)
    p.add_argument("--org", default="home")
    p.add_argument("--public-url")
    p.add_argument("--config")
    p.add_argument("--bundle", choices=["compose"], help="also write a deployment bundle")
    p.add_argument("--domain", help="public host name for the bundle, e.g. plasmon.acme.com")
    p.add_argument("--storage", help="minio (bundled) or s3://bucket[@https://endpoint]")
    p.add_argument("--db", help="postgresql+psycopg://... (default: bundled PostgreSQL)")
    p.add_argument("--output", help="bundle directory (default: ./plasmon-deploy)")
    p.add_argument("--tls-internal", action="store_true", help="self-signed certificate instead of Let's Encrypt")
    p.add_argument("--sso-issuer", help="OIDC issuer URL")
    p.add_argument("--sso-client-id")
    p.add_argument("--sso-client-secret")
    p.add_argument("--admin-group", help="OIDC group whose members become admins")
    p.set_defaults(handler=cmd_server_init)
    p = server.add_parser("start", help="start the coordinator", parents=[common])
    p.add_argument("--config")
    p.set_defaults(handler=cmd_server_start)
    p = server.add_parser("bootstrap", help="create the owner account on an empty server", parents=[common])
    p.add_argument("--owner", required=True, help="owner email")
    p.add_argument("--password")
    p.add_argument("--name")
    p.add_argument("--config")
    p.set_defaults(handler=cmd_server_bootstrap)
    server.add_parser("status", help="coordinator health (operator role)", parents=[common]).set_defaults(handler=cmd_server_status)

    job = sub.add_parser("job", help="submit and follow jobs").add_subparsers(dest="job_command")
    p = job.add_parser("submit", parents=[common])
    p.add_argument("file")
    p.set_defaults(handler=cmd_job_submit)
    p = job.add_parser("list", parents=[common])
    p.add_argument("--all", action="store_true", help="every job in the org (operator role)")
    p.set_defaults(handler=cmd_job_list)
    p = job.add_parser("status", parents=[common])
    p.add_argument("id")
    p.add_argument("--updates", action="store_true", help="also list every trainer update with its score")
    p.set_defaults(handler=cmd_job_status)
    p = job.add_parser("watch", parents=[common])
    p.add_argument("id")
    p.add_argument("--interval", type=float, default=2.0)
    p.set_defaults(handler=cmd_job_watch)
    p = job.add_parser("download", parents=[common])
    p.add_argument("id")
    p.add_argument("-o", "--output")
    p.set_defaults(handler=cmd_job_download)
    p = job.add_parser("cancel", parents=[common])
    p.add_argument("id")
    p.set_defaults(handler=cmd_job_cancel)
    job.add_parser("open", help="running jobs a trainer can join, with pay and requirements", parents=[common]).set_defaults(handler=cmd_job_open)
    p = job.add_parser("approvals", help="machines that asked to train a job of yours", parents=[common])
    p.add_argument("id")
    p.set_defaults(handler=cmd_job_approvals)
    for name, help_ in (("approve", "let a machine train your job"), ("reject", "keep a machine off your job")):
        p = job.add_parser(name, help=help_, parents=[common])
        p.add_argument("id")
        p.add_argument("machine", help="machine name, node id or its prefix")
        p.add_argument("--note", default="", help="a sentence the machine's owner receives")
        p.set_defaults(handler=cmd_job_decide)

    trainer = sub.add_parser("trainer", help="offer this machine").add_subparsers(dest="trainer_command")
    p = trainer.add_parser("start", parents=[common])
    p.add_argument("--name", help="machine name shown in the fleet (default: hostname)")
    p.add_argument("--device", choices=["any", "cuda", "mps", "cpu"], default="any")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--hours", action="append", help="only train inside this window, e.g. 'weekdays 19:00-08:00' (repeatable)")
    p.add_argument("--never-on-battery", action="store_true")
    p.set_defaults(handler=cmd_trainer_start)
    for name, help_ in (("join", "offer this machine to one open job"), ("leave", "take this machine off a job")):
        p = trainer.add_parser(name, help=help_, parents=[common])
        p.add_argument("job", help="job id, see `plasmon job open`")
        p.add_argument("--machine", help="another machine of yours: name, node id or its prefix (default: this machine)")
        p.set_defaults(handler=cmd_trainer_enrol)
    for name, help_ in (("enable", "run the trainer at login as a user service"), ("disable", "remove the user service")):
        p = trainer.add_parser(name, help=help_, parents=[common])
        p.add_argument("--name")
        p.add_argument("--device", choices=["any", "cuda", "mps", "cpu"], default="any")
        p.add_argument("--hours", action="append")
        p.add_argument("--dry-run", action="store_true", help="print what would be written")
        p.set_defaults(handler=cmd_trainer_service)

    p = sub.add_parser("fleet", help="machines you can see", parents=[common])
    p.add_argument("--status")
    p.set_defaults(handler=cmd_fleet)
    fleet_sub = p.add_subparsers(dest="fleet_action")
    for action in ("pause", "resume", "drain"):
        q = fleet_sub.add_parser(action, parents=[common])
        q.add_argument("node", help="node id (see `plasmon fleet --json`)")
        q.add_argument("--reason", default="")
        q.set_defaults(handler=cmd_fleet_control)
    q = fleet_sub.add_parser("show", parents=[common])
    q.add_argument("node")
    q.set_defaults(handler=cmd_fleet_control)
    q = fleet_sub.add_parser("logs", parents=[common])
    q.add_argument("node")
    q.add_argument("--follow", "-f", action="store_true")
    q.add_argument("--grep")
    q.set_defaults(handler=cmd_fleet_control)

    users = sub.add_parser("users", help="people and roles (admin)").add_subparsers(dest="users_action")
    users.add_parser("list", parents=[common]).set_defaults(handler=cmd_users)
    q = users.add_parser("invite", parents=[common])
    q.add_argument("--email")
    q.add_argument("--role", default="member", choices=["member", "operator", "admin", "viewer"])
    q.add_argument("--days", type=int, default=7)
    q.set_defaults(handler=cmd_users)
    q = users.add_parser("set-role", parents=[common])
    q.add_argument("email")
    q.add_argument("role", choices=["owner", "admin", "operator", "member", "viewer"])
    q.set_defaults(handler=cmd_users)
    for name in ("disable", "enable"):
        q = users.add_parser(name, parents=[common])
        q.add_argument("email")
        q.set_defaults(handler=cmd_users)

    p = sub.add_parser("credits", help="your balance and movements", parents=[common])
    p.set_defaults(handler=cmd_credits)
    cred = p.add_subparsers(dest="credits_action")
    q = cred.add_parser("grant", parents=[common])
    q.add_argument("email")
    q.add_argument("amount", type=int)
    q.add_argument("--memo", default="")
    q.set_defaults(handler=cmd_credits)
    cred.add_parser("users", parents=[common]).set_defaults(handler=cmd_credits)
    q = cred.add_parser("export", parents=[common])
    q.add_argument("--since-days", type=int, default=30)
    q.add_argument("-o", "--output")
    q.set_defaults(handler=cmd_credits)

    p = sub.add_parser("audit", help="who did what (operator)", parents=[common])
    p.add_argument("--since-hours", type=int, default=24)
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(handler=cmd_audit)

    pol = sub.add_parser("policy", help="org trainer policy").add_subparsers(dest="policy_action")
    pol.add_parser("show", parents=[common]).set_defaults(handler=cmd_policy)
    q = pol.add_parser("set", parents=[common])
    q.add_argument("--windows", nargs="*", help="e.g. 'weekdays 19:00-08:00' 'weekends 00:00-23:59'; pass nothing to clear")
    q.add_argument("--battery", choices=["pause", "allow"])
    q.set_defaults(handler=cmd_policy)

    ledger = sub.add_parser("ledger").add_subparsers(dest="ledger_command")
    ledger.add_parser("verify", parents=[common]).set_defaults(handler=cmd_ledger_verify)

    p = sub.add_parser("bench-diloco", help="simulate N trainers in one process and compare with one worker", parents=[common])
    p.add_argument("--job")
    p.add_argument("--trainers", type=int, default=2)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-samples", type=int, default=12000)
    p.set_defaults(handler=cmd_bench)
    return parser


def main(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except ApiError as e:
        print(f"error: {e.message}", file=sys.stderr)
        return 1
