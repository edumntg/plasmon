"""Jobs: create, list, inspect, cancel, download the latest weights, open jobs, enrolment."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.jobspec import JobSpec
from . import auth, db
from .deps import Principal, current_user, get_session, get_state, principal_optional
from .engine import EngineError, exposure, recent_trainers, unfit_reason, waiting_reason
from .mail import describe_hardware

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class ShardIn(BaseModel):
    index: int
    blob: str = Field(min_length=64, max_length=64)
    n: int


class JobCreate(BaseModel):
    spec: dict[str, Any]
    init_blob: str = Field(min_length=64, max_length=64)
    eval_blob: str = Field(min_length=64, max_length=64)
    shards: list[ShardIn]
    param_count: int = 0
    data_key: str | None = Field(default=None, min_length=64, max_length=64, description="hex key that sealed the shards; only with privacy.encrypt_shards")


def downloadable(job: db.Job) -> bool:
    """A funded job hands out its weights once it ended and the trainers were paid."""
    return not job.funding or job.status != "running"


def job_out(job: db.Job, rounds: list[db.Round] | None = None, session: Session | None = None, show_names: bool = True, operator: bool = True) -> dict[str, Any]:
    spec = job.spec or {}
    can_download = downloadable(job) or operator
    out = {
        "id": job.id,
        "name": job.name,
        "owner": job.owner.email if job.owner else job.owner_id,
        "status": job.status,
        "status_detail": job.status_detail,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "round": job.round_index,
        "total_rounds": job.total_rounds,
        "theta": job.theta_blob if can_download else None,
        "downloadable": can_download,
        "param_count": job.param_count,
        "eval_loss": job.last_eval_loss,
        "eval_acc": job.last_eval_acc,
        "credits_spent": job.credits_spent,
        "funding": job.funding,
        "held": job.held,
        "per_round": JobSpec.model_validate(spec).round_slices[0] if job.funding else 0,
        "settlement": job.settlement,
        "sealed": bool(job.data_key),
        "enrolment": spec.get("enrolment") or {"mode": "auto", "approval": "none"},
        "shards": len(job.shards),
        "spec": spec,
    }
    if rounds is not None:
        out["rounds"] = [round_out(r) for r in rounds]
    if session is not None:
        out["waiting_reason"] = waiting_reason(session, job)
        out["trainers"] = recent_trainers(session, job, show_names)
    return out


def round_out(r: db.Round) -> dict[str, Any]:
    return {
        "index": r.index,
        "status": r.status,
        "theta": r.theta_blob,
        "opened_at": r.opened_at,
        "deadline_at": r.deadline_at,
        "closed_at": r.closed_at,
        "accepted": r.accepted,
        "eval_loss": r.eval_loss,
        "eval_acc": r.eval_acc,
        "mean_loss_end": r.mean_loss_end,
        "bytes_in": r.bytes_in,
        "timings": r.timings,
    }


def enrolment_out(e: db.Enrolment, m: db.Machine, show_owner: bool = True) -> dict[str, Any]:
    hw = m.hardware or {}
    return {
        "node": m.node_id,
        "machine": m.name,
        "owner": m.user.email if show_owner and m.user else None,
        "status": e.status,
        "source": e.source,
        "requested_at": e.requested_at,
        "decided_at": e.decided_at,
        "note": e.note,
        "hardware": hw,
        "hardware_text": describe_hardware(hw),
        "tflops": hw.get("tflops"),
        "honesty": m.honesty,
        "rounds_served": m.rounds_served,
        "samples_verified": m.samples_verified,
        "machine_status": m.status,
        "last_seen_at": m.last_seen_at,
    }


def _visible(job: db.Job, user: db.User) -> bool:
    return job.owner_id == user.id or auth.role_at_least(user.role, "operator")


def _job_or_404(session: Session, job_id: str, user: db.User) -> db.Job:
    job = session.get(db.Job, job_id)
    if job is None or not _visible(job, user):
        raise HTTPException(404, "no such job")
    return job


@router.post("")
def create(body: JobCreate, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    if "jobs:write" not in auth.ROLE_SCOPES[user.role]:
        raise HTTPException(403, "your role cannot submit jobs")
    try:
        spec = JobSpec.model_validate(body.spec)
    except ValueError as e:
        raise HTTPException(422, f"invalid job spec: {e}") from e
    try:
        key = bytes.fromhex(body.data_key) if body.data_key else None
    except ValueError as e:
        raise HTTPException(422, "data_key must be 64 hex characters") from e
    try:
        job = state.engine.create_job(session, user, spec, body.init_blob, [s.model_dump() for s in body.shards], body.eval_blob, body.param_count, data_key=key)
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    return job_out(job, session=session, operator=auth.role_at_least(user.role, "operator"))


@router.get("")
def list_jobs(all: bool = False, user: db.User = Depends(current_user), session: Session = Depends(get_session)):
    q = select(db.Job).order_by(db.Job.created_at.desc())
    operator = auth.role_at_least(user.role, "operator")
    if not (all and operator):
        q = q.where(db.Job.owner_id == user.id)
    return [job_out(j, session=session, show_names=operator, operator=operator) for j in session.scalars(q).all()]


# ----- open jobs: what a trainer can join -----------------------------------------------

def open_job_out(session: Session, job: db.Job, machines: list[db.Machine]) -> dict[str, Any]:
    """A running job as the marketplace shows it, with the standing of the caller's machines."""
    spec = JobSpec.model_validate(job.spec)
    rnd_trainers = session.scalar(
        select(func.count()).select_from(db.Update).where(db.Update.job_id == job.id, db.Update.round_index == job.round_index, db.Update.status.in_(("assigned", "committed", "revealed")))
    )
    approved = session.scalar(select(func.count()).select_from(db.Enrolment).where(db.Enrolment.job_id == job.id, db.Enrolment.status == "approved"))
    pending = session.scalar(select(func.count()).select_from(db.Enrolment).where(db.Enrolment.job_id == job.id, db.Enrolment.status == "pending"))
    rows = {e.machine_id: e for e in session.scalars(select(db.Enrolment).where(db.Enrolment.job_id == job.id, db.Enrolment.machine_id.in_([m.id for m in machines]))).all()} if machines else {}
    mine = []
    for m in machines:
        row = rows.get(m.id)
        why = unfit_reason(m, spec)
        if row is not None and row.status != "left":
            standing = row.status
        elif why:
            standing = "unfit"
        elif spec.enrolment.mode == "auto" and spec.enrolment.approval == "none":
            standing = "automatic"
        else:
            standing = "can join"
        mine.append({"node": m.node_id, "name": m.name, "status": m.status, "standing": standing, "reason": why})
    return {
        "id": job.id,
        "name": job.name,
        "owner": job.owner.email if job.owner else job.owner_id,
        "created_at": job.created_at,
        "round": job.round_index,
        "total_rounds": job.total_rounds,
        "arch": spec.model.arch,
        "param_count": job.param_count,
        "shards": len(job.shards),
        "shard_size": spec.dataset.shard_size,
        "inner_steps": spec.recipe.inner_steps,
        "batch_size": spec.recipe.batch_size,
        "requirements": spec.requirements.model_dump(mode="json"),
        "enrolment": spec.enrolment.model_dump(mode="json"),
        "privacy": spec.privacy.model_dump(mode="json"),
        "sealed": bool(job.data_key),
        "funding": job.funding,
        "per_round": spec.round_slices[0] if job.funding else 0,
        "credits_per_1k_samples": spec.budget.credits_per_1k_samples,
        "trainers_now": rnd_trainers,
        "approved": approved,
        "pending": pending,
        "eval_loss": job.last_eval_loss,
        "waiting_reason": waiting_reason(session, job),
        "mine": mine,
    }


@router.get("/open")
def open_jobs(p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    """Running jobs any trainer may look at, with the standing of the caller's machines."""
    if p.user is None and p.machine is None:
        raise HTTPException(401, "login required")
    if p.user is not None:
        machines = session.scalars(select(db.Machine).where(db.Machine.user_id == p.user.id).order_by(db.Machine.name)).all()
    else:
        machines = [session.get(db.Machine, p.machine.id)]
    jobs = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at.desc())).all()
    return [open_job_out(session, j, machines) for j in jobs]


class JoinIn(BaseModel):
    node_id: str | None = Field(default=None, min_length=1, max_length=120, description="node id, its prefix, or the machine name; omitted when the account has one machine")


def _machine_for_principal(session: Session, p: Principal, node_id: str | None) -> db.Machine:
    if p.machine is not None:
        return session.get(db.Machine, p.machine.id)
    if p.user is None:
        raise HTTPException(401, "login required")
    mine = session.scalars(select(db.Machine).where(db.Machine.user_id == p.user.id)).all()
    if node_id:
        m = next((m for m in mine if m.node_id == node_id or m.node_id.startswith(node_id) or m.name == node_id), None)
        if m is None and auth.role_at_least(p.user.role, "admin"):
            m = session.scalar(select(db.Machine).where(db.Machine.node_id == node_id))
        if m is None:
            raise HTTPException(404, "no machine of yours with that node id")
        return m
    if len(mine) == 1:
        return mine[0]
    if not mine:
        raise HTTPException(404, "no machine is linked to your account; run plasmon trainer start once")
    raise HTTPException(409, "you have several machines; pass node_id")


@router.post("/{job_id}/join")
def join(job_id: str, body: JoinIn | None = None, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    job = session.get(db.Job, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    machine = _machine_for_principal(session, p, body.node_id if body else None)
    try:
        row = state.engine.join_job(session, job, machine)
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    session.add(db.AuditEvent(actor_id=p.user.id if p.user else None, action="job.join", target=job.id, detail={"node": machine.node_id, "status": row.status}))
    session.commit()
    return enrolment_out(row, machine)


@router.post("/{job_id}/leave")
def leave(job_id: str, body: JoinIn | None = None, p: Principal = Depends(principal_optional), session: Session = Depends(get_session), state=Depends(get_state)):
    job = session.get(db.Job, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    machine = _machine_for_principal(session, p, body.node_id if body else None)
    row = state.engine.leave_job(session, job, machine)
    session.add(db.AuditEvent(actor_id=p.user.id if p.user else None, action="job.leave", target=job.id, detail={"node": machine.node_id}))
    session.commit()
    return enrolment_out(row, machine)


@router.get("/{job_id}/enrolments")
def enrolments(job_id: str, user: db.User = Depends(current_user), session: Session = Depends(get_session)):
    job = _job_or_404(session, job_id, user)
    rows = session.execute(
        select(db.Enrolment, db.Machine).join(db.Machine, db.Machine.id == db.Enrolment.machine_id).where(db.Enrolment.job_id == job.id).order_by(db.Enrolment.requested_at)
    ).all()
    return [enrolment_out(e, m) for e, m in rows]


class DecisionIn(BaseModel):
    note: str = Field(default="", max_length=500)


def _decide(job_id: str, node: str, approve: bool, body: DecisionIn | None, user: db.User, session: Session, state) -> dict[str, Any]:
    job = _job_or_404(session, job_id, user)
    machine = session.scalar(select(db.Machine).where(db.Machine.node_id == node))
    if machine is None:  # the owner typed a name from the mail or the page
        candidates = session.execute(
            select(db.Machine).join(db.Enrolment, db.Enrolment.machine_id == db.Machine.id).where(db.Enrolment.job_id == job.id, (db.Machine.name == node) | db.Machine.node_id.startswith(node))
        ).scalars().all()
        if len(candidates) != 1:
            raise HTTPException(404, "no enrolled machine matches" if not candidates else "several machines match; use the node id")
        machine = candidates[0]
    row = state.engine.enrolment_of(session, job, machine)
    if row is None:
        raise HTTPException(404, "this machine has not asked to join")
    try:
        row = state.engine.decide_enrolment(session, job, row, approve, user, body.note if body else "")
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    session.add(db.AuditEvent(actor_id=user.id, action="job.approve" if approve else "job.reject", target=job.id, detail={"node": machine.node_id}))
    session.commit()
    return enrolment_out(row, machine)


@router.post("/{job_id}/enrolments/{node}/approve")
def approve(job_id: str, node: str, body: DecisionIn | None = None, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    return _decide(job_id, node, True, body, user, session, state)


@router.post("/{job_id}/enrolments/{node}/reject")
def reject(job_id: str, node: str, body: DecisionIn | None = None, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    return _decide(job_id, node, False, body, user, session, state)


# ----- one job -------------------------------------------------------------------------

@router.get("/{job_id}")
def get_job(job_id: str, p: Principal = Depends(principal_optional), session: Session = Depends(get_session)):
    job = session.get(db.Job, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if p.user is None and p.machine is None:
        raise HTTPException(401, "login required")
    if p.user is not None and not _visible(job, p.user):
        raise HTTPException(403, "not your job")
    rounds = session.scalars(select(db.Round).where(db.Round.job_id == job.id).order_by(db.Round.index)).all()
    operator = p.user is not None and auth.role_at_least(p.user.role, "operator")
    out = job_out(job, rounds, session=session, show_names=operator, operator=operator or p.machine is not None)
    out["exposure"] = exposure(session, job) if operator or (p.user is not None and job.owner_id == p.user.id) else []
    return out


@router.get("/{job_id}/updates")
def list_updates(job_id: str, round: int | None = None, user: db.User = Depends(current_user), session: Session = Depends(get_session)):
    job = _job_or_404(session, job_id, user)
    q = select(db.Update).where(db.Update.job_id == job.id).order_by(db.Update.round_index.desc(), db.Update.id)
    if round is not None:
        q = q.where(db.Update.round_index == round)
    show_names = auth.role_at_least(user.role, "operator")
    return [
        {
            "round": u.round_index,
            "machine": u.machine.name if show_names else u.machine.node_id[:8],
            "node": u.machine.node_id if show_names else u.machine.node_id[:8],
            "shard": u.shard_index,
            "status": u.status,
            "samples": u.samples,
            "loss_start": u.loss_start,
            "loss_end": u.loss_end,
            "frame_bytes": u.frame_bytes,
            "score": u.score,
            "gain_assigned": u.gain_assigned,
            "gain_random": u.gain_random,
            "reject_reason": u.reject_reason,
            "credits": u.credits,
        }
        for u in session.scalars(q).all()
    ]


@router.post("/{job_id}/cancel")
def cancel(job_id: str, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    job = _job_or_404(session, job_id, user)
    try:
        state.engine.cancel_job(session, job)
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    session.add(db.AuditEvent(actor_id=user.id, action="job.cancel", target=job.id))
    session.commit()
    return job_out(job, session=session, operator=auth.role_at_least(user.role, "operator"))
