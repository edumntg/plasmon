"""HTML pages. Server-rendered Jinja2 with HTMX partials polled every few seconds."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..core.jobspec import JobSpec
from . import api_auth, api_credits, api_fleet, api_jobs, auth, credits, db, ledger, policy
from .config import OrgPolicy
from .deps import Principal, get_session, get_state, principal_optional
from .engine import EngineError, exposure, waiting_reason

router = APIRouter(include_in_schema=False)


def install_filters(templates: Jinja2Templates) -> None:
    templates.env.filters["ago"] = ago
    templates.env.filters["bytes"] = fmt_bytes
    templates.env.filters["pct"] = lambda v: "" if v is None else f"{100 * v:.1f} %"
    templates.env.filters["num"] = lambda v: "" if v is None else f"{v:,}"
    templates.env.filters["f3"] = lambda v: "" if v is None else f"{v:.3f}"
    templates.env.globals["role_at_least"] = auth.role_at_least
    templates.env.globals["version"] = __version__
    templates.env.globals["describe_hardware"] = mail_describe_hardware


def ago(value: dt.datetime | None) -> str:
    if value is None:
        return "never"
    if value.tzinfo is not None:
        value = value.astimezone(dt.UTC).replace(tzinfo=None)
    s = int((db.now() - value).total_seconds())
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        return f"{s // 3600} h"
    return f"{s // 86400} d"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def mail_describe_hardware(hw) -> str:
    from .mail import describe_hardware

    return describe_hardware(hw)


def render(request: Request, name: str, p: Principal, **ctx):
    state = request.app.state.plasmon
    return state.templates.TemplateResponse(request, name, {"user": p.user, "org": state.cfg.org_name, "mode": state.cfg.mode, **ctx})


def need_user(p: Principal) -> db.User:
    if p.user is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return p.user


# ----- auth pages ----------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", p: Principal = Depends(principal_optional), state=Depends(get_state)):
    if p.user:
        return RedirectResponse(next, status_code=303)
    return render(request, "login.html", p, next=next, error=None, open_registration=state.cfg.auth.open_registration, sso=state.cfg.oidc.enabled)


@router.post("/login")
def login_submit(request: Request, email: str = Form(), password: str = Form(), next: str = Form("/"), session: Session = Depends(get_session), state=Depends(get_state)):
    user = session.scalar(select(db.User).where(db.User.email == email.lower().strip()))
    if user is None or user.disabled or not auth.verify_password(user.password_hash, password):
        return render(request, "login.html", Principal(), next=next, error="Wrong email or password.", open_registration=state.cfg.auth.open_registration, sso=state.cfg.oidc.enabled)
    resp = RedirectResponse(next if next.startswith("/") else "/", status_code=303)
    state.sessions.write(resp, user.id, secure=request.url.scheme == "https")
    return resp


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request, invite: str = "", p: Principal = Depends(principal_optional), state=Depends(get_state), session: Session = Depends(get_session)):
    first = session.scalar(select(func.count()).select_from(db.User)) == 0
    closed = not first and not state.cfg.auth.open_registration
    if closed and not invite:
        return render(request, "register.html", p, error="Registration is closed. Ask an admin for an invite link.", closed=True, first=False, invite="")
    return render(request, "register.html", p, error=None, closed=False, first=first, invite=invite)


@router.post("/register")
def register_submit(request: Request, email: str = Form(), password: str = Form(), name: str = Form(""), invite: str = Form(""), session: Session = Depends(get_session), state=Depends(get_state)):
    try:
        user = api_auth.register_user(session, state, email, password, name, invite or None)
    except HTTPException as e:
        return render(request, "register.html", Principal(), error=e.detail, closed=False, first=False, invite=invite)
    resp = RedirectResponse("/", status_code=303)
    state.sessions.write(resp, user.id, secure=request.url.scheme == "https")
    return resp


@router.post("/logout")
def logout(state=Depends(get_state)):
    resp = RedirectResponse("/login", status_code=303)
    state.sessions.clear(resp)
    return resp


@router.get("/device", response_class=HTMLResponse)
def device_page(request: Request, code: str = "", p: Principal = Depends(principal_optional)):
    if p.user is None:
        return RedirectResponse(f"/login?next=/device%3Fcode%3D{code}", status_code=303)
    return render(request, "device.html", p, code=code, result=None)


@router.post("/device", response_class=HTMLResponse)
def device_submit(request: Request, code: str = Form(), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    try:
        out = api_auth.device_confirm(api_auth.DeviceConfirm(user_code=code), user, session, state)
        result = {"ok": True, "label": out["label"]}
    except HTTPException as e:
        result = {"ok": False, "error": e.detail}
    return render(request, "device.html", p, code=code, result=result)


# ----- dashboard pages -----------------------------------------------------------

def _summary(session: Session) -> dict:
    counts = dict(session.execute(select(db.Machine.status, func.count()).group_by(db.Machine.status)).all())
    total = sum(counts.values())
    since = db.now() - dt.timedelta(hours=1)
    return {
        "machines": total,
        "online": total - counts.get("offline", 0),
        "training": counts.get("training", 0),
        "idle": counts.get("idle", 0),
        "paused": counts.get("paused", 0),
        "unavailable": counts.get("unavailable", 0),
        "offline": counts.get("offline", 0),
        "error": counts.get("error", 0),
        "jobs_running": session.scalar(select(func.count()).select_from(db.Job).where(db.Job.status == "running")),
        "rounds_last_hour": session.scalar(select(func.count()).select_from(db.Round).where(db.Round.status == "closed", db.Round.closed_at >= since)),
        "samples_verified": int(session.scalar(select(func.coalesce(func.sum(db.Machine.samples_verified), 0))) or 0),
    }


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    active = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at.desc()).limit(6)).all()
    recent = session.scalars(select(db.Job).where(db.Job.status != "running").order_by(db.Job.finished_at.desc()).limit(5)).all()
    mine = session.scalars(select(db.Machine).where(db.Machine.user_id == user.id)).all()
    last = session.scalar(select(db.LedgerEntry).order_by(db.LedgerEntry.seq.desc()).limit(1))
    top = session.scalars(select(db.Machine).where(db.Machine.samples_verified > 0).order_by(db.Machine.samples_verified.desc()).limit(10)).all()
    return render(request, "overview.html", p, summary=_summary(session), active_jobs=active, losses=_losses(session, active), waiting=_waiting(session, active), recent=recent, mine=mine, last=last, top=top)


def _waiting(session: Session, jobs: list[db.Job]) -> dict[str, str]:
    return {j.id: r for j in jobs if (r := waiting_reason(session, j))}


def _losses(session: Session, jobs: list[db.Job]) -> dict[str, list[float]]:
    out = {}
    for j in jobs:
        rows = session.scalars(select(db.Round.eval_loss).where(db.Round.job_id == j.id, db.Round.status == "closed").order_by(db.Round.index)).all()
        out[j.id] = [float(v) for v in rows if v is not None]
    return out


@router.get("/partials/overview", response_class=HTMLResponse)
def overview_partial(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    need_user(p)
    active = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at.desc()).limit(6)).all()
    return render(request, "partials/overview_stats.html", p, summary=_summary(session), active_jobs=active, losses=_losses(session, active), waiting=_waiting(session, active))


@router.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    return render(request, "jobs.html", p, jobs=_jobs_for(session, user), can_see_all=auth.role_at_least(user.role, "operator"))


def _jobs_for(session: Session, user: db.User) -> list[db.Job]:
    q = select(db.Job).order_by(db.Job.created_at.desc())
    if not auth.role_at_least(user.role, "operator"):
        q = q.where(db.Job.owner_id == user.id)
    return session.scalars(q).all()


@router.get("/partials/jobs", response_class=HTMLResponse)
def jobs_partial(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    return render(request, "partials/jobs_table.html", p, jobs=_jobs_for(session, user))


@router.get("/jobs/new", response_class=HTMLResponse)
def job_new_page(request: Request, p: Principal = Depends(principal_optional)):
    user = need_user(p)
    if "jobs:write" not in auth.ROLE_SCOPES[user.role]:
        raise HTTPException(403, "your role cannot submit jobs")
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / "examples" / "mnist" / "job.yaml"
    text = example.read_text() if example.exists() else "name: my-job\nmodel: {arch: mnist_cnn}\ndataset: {source: builtin://mnist}\nbudget: {rounds: 10}\n"
    return render(request, "job_new.html", p, yaml_text=text, error=None)


@router.post("/jobs/new")
def job_new_submit(request: Request, yaml_text: str = Form(), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    """Prepare the job on the server: the dataset must be builtin or readable by the server."""
    user = need_user(p)
    if "jobs:write" not in auth.ROLE_SCOPES[user.role]:
        raise HTTPException(403, "your role cannot submit jobs")
    from .. import jobs as jobs_mod
    from ..core import jobspec

    try:
        spec = jobspec.loads(yaml_text)
        prepared = jobs_mod.prepare(spec)
        for blob_id, data in prepared["blobs"].items():
            if not state.blobs.exists(blob_id):
                state.blobs.put(data, expected_id=blob_id)
        job = state.engine.create_job(session, user, spec, prepared["init_blob"], prepared["shards"], prepared["eval_blob"], prepared["param_count"])
    except (ValueError, FileNotFoundError) as e:
        return render(request, "job_new.html", p, yaml_text=yaml_text, error=str(e))
    except Exception as e:  # engine errors carry a message for the user
        return render(request, "job_new.html", p, yaml_text=yaml_text, error=str(e))
    session.add(db.AuditEvent(actor_id=user.id, action="job.create", target=job.id, detail={"via": "dashboard"}))
    session.commit()
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


def _open_jobs_ctx(session: Session, user: db.User) -> dict:
    machines = session.scalars(select(db.Machine).where(db.Machine.user_id == user.id).order_by(db.Machine.name)).all()
    jobs = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at.desc())).all()
    return {"jobs": [api_jobs.open_job_out(session, j, machines) for j in jobs], "machines": machines}


@router.get("/jobs/open", response_class=HTMLResponse)
def open_jobs_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    return render(request, "jobs_open.html", p, result=request.query_params.get("result", ""), credits_enabled=state.cfg.credits.enabled, unit=state.cfg.credits.unit, **_open_jobs_ctx(session, user))


@router.get("/partials/jobs-open", response_class=HTMLResponse)
def open_jobs_partial(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    return render(request, "partials/jobs_open_table.html", p, credits_enabled=state.cfg.credits.enabled, unit=state.cfg.credits.unit, **_open_jobs_ctx(session, user))



def _job_ctx(session: Session, user: db.User, job: db.Job) -> dict:
    rounds = session.scalars(select(db.Round).where(db.Round.job_id == job.id).order_by(db.Round.index)).all()
    updates = session.scalars(select(db.Update).where(db.Update.job_id == job.id).order_by(db.Update.round_index.desc(), db.Update.id).limit(200)).all()
    closed = [r for r in rounds if r.status == "closed"]
    series = {"rounds": [r.index for r in closed], "loss": [r.eval_loss for r in closed], "acc": [r.eval_acc for r in closed]}
    operator = auth.role_at_least(user.role, "operator")
    enrolments = session.execute(
        select(db.Enrolment, db.Machine).join(db.Machine, db.Machine.id == db.Enrolment.machine_id).where(db.Enrolment.job_id == job.id).order_by(db.Enrolment.requested_at)
    ).all()
    spec = JobSpec.model_validate(job.spec)
    return {
        "job": job,
        "spec": spec,
        "rounds": rounds,
        "updates": updates,
        "series": series,
        "after": _weights_after(job, rounds),
        "waiting": waiting_reason(session, job),
        "show_names": operator,
        "can_download": api_jobs.downloadable(job) or operator,
        "enrolments": [(e, m) for e, m in enrolments],
        "pending": [(e, m) for e, m in enrolments if e.status == "pending"],
        "exposure": exposure(session, job),
        "per_round": spec.round_slices[0] if job.funding else 0,
    }


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    job = session.get(db.Job, job_id)
    if job is None or (job.owner_id != user.id and not auth.role_at_least(user.role, "operator")):
        raise HTTPException(404, "no such job")
    return render(request, "job.html", p, result=request.query_params.get("result", ""), **_job_ctx(session, user, job))


@router.post("/jobs/{job_id}/enrolments/{node}/{decision}")
def job_decide(job_id: str, node: str, decision: str, note: str = Form(""), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if decision not in ("approve", "reject"):
        raise HTTPException(404, "unknown decision")
    try:
        out = api_jobs._decide(job_id, node, decision == "approve", api_jobs.DecisionIn(note=note), user, session, state)
    except HTTPException as e:
        return RedirectResponse(f"/jobs/{job_id}?result=error:{e.detail}#trainers", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}?result={decision}:{out['machine']}#trainers", status_code=303)


def _weights_after(job: db.Job, rounds: list[db.Round]) -> dict[int, str]:
    """Blob id of the weights produced by each closed round: the next round's start, or the job's latest."""
    out = {}
    for i, r in enumerate(rounds):
        if r.status != "closed":
            continue
        out[r.index] = rounds[i + 1].theta_blob if i + 1 < len(rounds) else job.theta_blob
    return out


@router.get("/partials/jobs/{job_id}", response_class=HTMLResponse)
def job_partial(request: Request, job_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    job = session.get(db.Job, job_id)
    if job is None or (job.owner_id != user.id and not auth.role_at_least(user.role, "operator")):
        raise HTTPException(404, "no such job")
    return render(request, "partials/job_live.html", p, result="", **_job_ctx(session, user, job))


@router.get("/jobs/{job_id}/download")
def job_download(job_id: str, blob: str | None = None, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    """The latest weights, or the weights after a given round (`blob` must belong to the job)."""
    from fastapi.responses import Response

    user = need_user(p)
    job = session.get(db.Job, job_id)
    if job is None or (job.owner_id != user.id and not auth.role_at_least(user.role, "operator")):
        raise HTTPException(404, "no such job")
    if not api_jobs.downloadable(job) and not auth.role_at_least(user.role, "operator"):
        raise HTTPException(403, "the weights of a funded job are released when it ends and the trainers are paid")
    allowed = {job.theta_blob, job.init_blob, *session.scalars(select(db.Round.theta_blob).where(db.Round.job_id == job.id)).all()}
    chosen = blob or job.theta_blob
    if chosen not in allowed or not state.blobs.exists(chosen):
        raise HTTPException(404, "no such checkpoint for this job")
    suffix = "latest" if chosen == job.theta_blob else chosen[:8]
    return Response(state.blobs.get(chosen), media_type="application/octet-stream", headers={"content-disposition": f'attachment; filename="{job.name}-{suffix}.safetensors"'})


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    job = session.get(db.Job, job_id)
    if job is None or (job.owner_id != user.id and not auth.role_at_least(user.role, "operator")):
        raise HTTPException(404, "no such job")
    if job.status == "running":
        state.engine.cancel_job(session, job)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/{verb}")
def job_join_or_leave(job_id: str, verb: str, node_id: str = Form(""), next: str = Form("/jobs/open"), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if verb not in ("join", "leave"):
        raise HTTPException(404, "unknown action")
    back = next if next.startswith("/") else "/jobs/open"
    job = session.get(db.Job, job_id)
    if job is None:
        return RedirectResponse(f"{back}?result=error:no such job", status_code=303)
    try:
        machine = api_jobs._machine_for_principal(session, p, node_id or None)
        row = state.engine.join_job(session, job, machine) if verb == "join" else state.engine.leave_job(session, job, machine)
    except (HTTPException, EngineError) as e:
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        return RedirectResponse(f"{back}?result=error:{detail}", status_code=303)
    session.add(db.AuditEvent(actor_id=user.id, action=f"job.{verb}", target=job.id, detail={"node": machine.node_id, "status": row.status}))
    session.commit()
    return RedirectResponse(f"{back}?result={row.status}:{job.name}", status_code=303)


@router.get("/machine", response_class=HTMLResponse)
def my_machines(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    machines = session.scalars(select(db.Machine).where(db.Machine.user_id == user.id).order_by(db.Machine.name)).all()
    if len(machines) == 1:
        return RedirectResponse(f"/machine/{machines[0].node_id}", status_code=303)
    return render(request, "machines.html", p, machines=machines)


def _machine_for(session: Session, user: db.User, node_id: str) -> db.Machine:
    m = session.scalar(select(db.Machine).where(db.Machine.node_id == node_id))
    if m is None or (m.user_id != user.id and not auth.role_at_least(user.role, "operator")):
        raise HTTPException(404, "no such machine")
    return m


def _machine_ctx(session: Session, m: db.Machine) -> dict:
    since = db.now() - dt.timedelta(hours=1)
    history = session.scalars(select(db.Heartbeat).where(db.Heartbeat.machine_id == m.id, db.Heartbeat.at >= since).order_by(db.Heartbeat.at)).all()
    job = session.get(db.Job, m.current_job_id) if m.current_job_id else None
    recent = session.scalars(select(db.Update).where(db.Update.machine_id == m.id).order_by(db.Update.id.desc()).limit(20)).all()
    logs = session.scalars(select(db.LogLine).where(db.LogLine.machine_id == m.id).order_by(db.LogLine.id.desc()).limit(100)).all()
    spark = {k: [float((h.metrics or {}).get(k) or 0) for h in history] for k in ("cpu_pct", "ram_pct", "gpu_pct")}
    standing = session.execute(
        select(db.Enrolment, db.Job).join(db.Job, db.Job.id == db.Enrolment.job_id).where(db.Enrolment.machine_id == m.id, db.Job.status == "running").order_by(db.Enrolment.requested_at.desc())
    ).all()
    held = credits.held_by_machine(session).get(m.id, 0)
    return {"m": m, "job": job, "history": history, "recent": recent, "logs": list(reversed(logs)), "spark": spark, "standing": [(e, j) for e, j in standing], "held": held}


@router.get("/machine/{node_id}", response_class=HTMLResponse)
def machine_page(request: Request, node_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    m = _machine_for(session, user, node_id)
    joinable = _joinable(session, m)
    return render(request, "machine.html", p, can_control=auth.role_at_least(user.role, "operator"), joinable=joinable, result=request.query_params.get("result", ""), **_machine_ctx(session, m))


def _joinable(session: Session, m: db.Machine) -> list[dict]:
    """Open jobs this machine could join right now: fits the requirements, not yet enrolled."""
    jobs = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at.desc())).all()
    out = []
    for j in jobs:
        view = api_jobs.open_job_out(session, j, [m])
        if view["mine"] and view["mine"][0]["standing"] == "can join":
            out.append(view)
    return out


@router.post("/machine/{node_id}/{action}")
def machine_action(node_id: str, action: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "operator"):
        raise HTTPException(403, "operator role needed")
    body = api_fleet.ControlIn(reason="from the dashboard")
    if action == "pause":
        api_fleet.pause(node_id, body, user, session, state)
    elif action == "resume":
        api_fleet.resume(node_id, user, session, state)
    elif action == "drain":
        api_fleet.drain(node_id, body, user, session, state)
    else:
        raise HTTPException(404, "unknown action")
    return RedirectResponse(f"/machine/{node_id}", status_code=303)


@router.get("/partials/machine/{node_id}", response_class=HTMLResponse)
def machine_partial(request: Request, node_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    return render(request, "partials/machine_live.html", p, **_machine_ctx(session, _machine_for(session, user, node_id)))


@router.get("/fleet", response_class=HTMLResponse)
def fleet_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "operator"):
        raise HTTPException(403, "the fleet page needs the operator role")
    machines = session.scalars(select(db.Machine).order_by(db.Machine.status, db.Machine.name)).all()
    return render(request, "fleet.html", p, machines=machines, summary=_summary(session))


@router.get("/partials/fleet", response_class=HTMLResponse)
def fleet_partial(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "operator"):
        raise HTTPException(403, "operator role needed")
    machines = session.scalars(select(db.Machine).order_by(db.Machine.status, db.Machine.name)).all()
    return render(request, "partials/fleet_table.html", p, machines=machines, summary=_summary(session))


@router.get("/server", response_class=HTMLResponse)
def server_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "operator"):
        raise HTTPException(403, "the server page needs the operator role")
    from .api_misc import server_status

    return render(request, "server.html", p, status=server_status(user, session, state))


@router.get("/ledger", response_class=HTMLResponse)
def ledger_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    need_user(p)
    entries = session.scalars(select(db.LedgerEntry).order_by(db.LedgerEntry.seq.desc()).limit(200)).all()
    ok, count, problem = ledger.verify(session, state.server.node_id)
    return render(request, "ledger.html", p, entries=entries, ok=ok, count=count, problem=problem, server_node_id=state.server.node_id)


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "admin"):
        raise HTTPException(403, "the users page needs the admin role")
    users = api_fleet.list_users(user, session)
    invites = api_fleet.list_invites(user, session)
    audit = api_fleet.audit(24, 50, user, session)
    return render(request, "users.html", p, users=users, invites=invites, audit=audit, base=state.public_url(request), result=request.query_params.get("result", ""))


@router.post("/users/invite")
def users_invite(request: Request, email: str = Form(""), role: str = Form("member"), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    out = api_fleet.invite(api_fleet.InviteIn(email=email or None, role=role), user, session, state)
    return RedirectResponse(f"/users?result=invite:{out['code']}", status_code=303)


@router.post("/users/{user_id}/role")
def users_role(user_id: str, role: str = Form(), p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    try:
        api_fleet.set_role(user_id, api_fleet.RoleIn(role=role), user, session)
    except HTTPException as e:
        return RedirectResponse(f"/users?result=error:{e.detail}", status_code=303)
    return RedirectResponse("/users?result=role", status_code=303)


@router.post("/users/{user_id}/disable")
def users_disable(user_id: str, disabled: str = Form("1"), p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    try:
        api_fleet.disable_user(user_id, api_fleet.DisableIn(disabled=disabled == "1"), user, session)
    except HTTPException as e:
        return RedirectResponse(f"/users?result=error:{e.detail}", status_code=303)
    return RedirectResponse("/users?result=disabled", status_code=303)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "admin"):
        raise HTTPException(403, "the settings page needs the admin role")
    current = policy.load(session, state.cfg.policy.defaults)
    return render(request, "settings.html", p, policy=current, days=policy.DAYS, result=request.query_params.get("result", ""), cfg=state.cfg)


@router.post("/settings/policy")
def settings_policy(request: Request, windows_text: str = Form(""), pause_on_battery: str = Form(""), drain_at_window_end: str = Form(""), p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    if not auth.role_at_least(user.role, "admin"):
        raise HTTPException(403, "admin role needed")
    try:
        windows = [policy.parse_hours(line.strip()) for line in windows_text.splitlines() if line.strip()]
        new = OrgPolicy(windows=windows, pause_on_battery=pause_on_battery == "on", drain_at_window_end=drain_at_window_end == "on")
    except (ValueError, IndexError) as e:
        return RedirectResponse(f"/settings?result=error:{e}", status_code=303)
    api_fleet.put_policy(new, user, session, state)
    return RedirectResponse("/settings?result=saved", status_code=303)


@router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    need_user(p)
    rows = session.scalars(select(db.Machine).where(db.Machine.samples_verified > 0).order_by(db.Machine.samples_verified.desc()).limit(100)).all()
    earned = credits.earned_by_machine(session) if request.app.state.plasmon.cfg.credits.enabled else None
    return render(request, "leaderboard.html", p, rows=rows, earned=earned)


@router.get("/account", response_class=HTMLResponse)
def account_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    tokens = session.scalars(select(db.Token).where(db.Token.user_id == user.id, db.Token.revoked.is_(False)).order_by(db.Token.created_at.desc())).all()
    machines = session.scalars(select(db.Machine).where(db.Machine.user_id == user.id)).all()
    return render(request, "account.html", p, tokens=tokens, machines=machines, result=request.query_params.get("result", ""), has_password=bool(user.password_hash))


@router.post("/account/password")
def account_password(current: str = Form(""), new: str = Form(), p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    try:
        api_auth.change_password(api_auth.PasswordIn(current=current, new=new), user, session)
    except HTTPException as e:
        return RedirectResponse(f"/account?result=error:{e.detail}", status_code=303)
    return RedirectResponse("/account?result=password", status_code=303)


@router.post("/account/tokens/{token_id}/revoke")
def account_revoke(token_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    tok = session.get(db.Token, token_id)
    if tok is not None and tok.user_id == user.id:
        tok.revoked = True
        session.add(db.AuditEvent(actor_id=user.id, action="token.revoke", target=token_id))
        session.commit()
    return RedirectResponse("/account?result=revoked", status_code=303)


@router.get("/credits", response_class=HTMLResponse)
def credits_page(request: Request, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    user = need_user(p)
    is_admin = auth.role_at_least(user.role, "admin")
    holds = credits.holds_for(session, user.id, 50)
    job_names = {j.id: j.name for j in session.scalars(select(db.Job).where(db.Job.id.in_({h.job_id for h in holds}))).all()} if holds else {}
    return render(
        request, "credits.html", p,
        enabled=state.cfg.credits.enabled, unit=state.cfg.credits.unit, fee_pct=state.cfg.credits.fee_pct,
        balance=credits.balance(session, user.id), entries=credits.entries_for(session, user.id, 100),
        on_hold=credits.on_hold(session, user.id), locked=credits.locked(session, user.id), holds=holds, job_names=job_names,
        all_balances=credits.balances(session) if is_admin else None, is_admin=is_admin,
        result=request.query_params.get("result", ""),
    )


@router.post("/credits/grant")
def credits_grant(email: str = Form(), amount: int = Form(), memo: str = Form(""), p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    user = need_user(p)
    try:
        api_credits.grant(api_credits.GrantIn(email=email, amount=amount, memo=memo), user, session)
    except HTTPException as e:
        return RedirectResponse(f"/credits?result=error:{e.detail}", status_code=303)
    except ValueError as e:
        return RedirectResponse(f"/credits?result=error:{e}", status_code=303)
    return RedirectResponse("/credits?result=granted", status_code=303)
