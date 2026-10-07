"""Machines: registration, heartbeats with assignments and commands, commit and reveal."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core import identity
from . import auth, db, policy
from .deps import current_machine, current_user, get_session, get_state
from .engine import EngineError

router = APIRouter(prefix="/v1/machines", tags=["machines"])


class RegisterIn(BaseModel):
    node_id: str = Field(min_length=64, max_length=64)
    name: str = Field(default="", max_length=120)
    hardware: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, Any] = Field(default_factory=dict)
    signature: str  # over {"node_id", "user": user id}


def machine_out(m: db.Machine) -> dict[str, Any]:
    return {
        "id": m.id,
        "node_id": m.node_id,
        "name": m.name,
        "owner": m.user.email if m.user else None,
        "status": m.status,
        "status_detail": m.status_detail,
        "hardware": m.hardware,
        "tags": m.tags,
        "metrics": m.metrics,
        "current_job_id": m.current_job_id,
        "current_round": m.current_round,
        "last_seen_at": m.last_seen_at,
        "paused_by_admin": m.paused_by_admin,
        "draining": m.draining,
        "honesty": m.honesty,
        "rounds_served": m.rounds_served,
        "samples_verified": m.samples_verified,
        "versions": m.versions,
    }


@router.post("/register")
def register(body: RegisterIn, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    if not identity.verify(body.node_id, {"node_id": body.node_id, "user": user.id}, body.signature):
        raise HTTPException(401, "bad machine signature")
    machine = session.scalar(select(db.Machine).where(db.Machine.node_id == body.node_id))
    if machine is None:
        machine = db.Machine(node_id=body.node_id, user_id=user.id)
        session.add(machine)
    elif machine.user_id != user.id and not auth.role_at_least(user.role, "admin"):
        raise HTTPException(403, "this machine belongs to another account")
    machine.name = body.name or machine.name or body.node_id[:8]
    machine.hardware = body.hardware
    machine.versions = body.versions
    machine.last_seen_at = db.now()
    machine.status = "idle"
    session.flush()
    for old in session.scalars(select(db.Token).where(db.Token.machine_id == machine.id, db.Token.revoked.is_(False))):
        old.revoked = True
    secret = auth.issue_machine_token(session, machine)
    session.add(db.AuditEvent(actor_id=user.id, action="machine.register", target=machine.node_id, detail={"name": machine.name}))
    session.commit()
    state.bus.publish("fleet", {"event": "registered", "node": machine.node_id})
    return {"machine_token": secret, "machine": machine_out(machine)}


class HeartbeatIn(BaseModel):
    status: str = Field(pattern=r"^(idle|training|paused|unavailable|error)$")
    status_detail: str = ""
    job_id: str | None = None
    round: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    logs: list[dict[str, Any]] = Field(default_factory=list)  # [{at, level, message}]
    ready: bool = False  # asks for a round; idle heartbeats ask implicitly
    hardware: dict[str, Any] | None = None  # sent once per trainer start; replaces what enrolment recorded


@router.post("/heartbeat")
def heartbeat(body: HeartbeatIn, machine: db.Machine = Depends(current_machine), session: Session = Depends(get_session), state=Depends(get_state)):
    machine = session.get(db.Machine, machine.id)
    was = machine.status
    machine.last_seen_at = db.now()
    machine.status = body.status
    machine.status_detail = body.status_detail
    machine.current_job_id = body.job_id
    machine.current_round = body.round
    machine.metrics = body.metrics
    if body.hardware:
        machine.hardware = body.hardware
    session.add(db.Heartbeat(machine_id=machine.id, status=body.status, metrics=body.metrics))
    for line in body.logs[:200]:
        session.add(db.LogLine(machine_id=machine.id, at=_parse_at(line.get("at")), level=str(line.get("level", "info"))[:8], message=str(line.get("message", ""))[:4000]))
    commands: list[dict[str, Any]] = []
    if machine.paused_by_admin:
        commands.append({"type": "pause", "reason": "paused by admin"})
    if machine.draining:
        commands.append({"type": "drain"})
    assignment = None
    if body.status == "idle" or body.ready:
        a = state.engine.try_assign(session, machine)
        if a is not None:
            assignment = a.as_dict()
    job_info = None
    if body.job_id:
        j = session.get(db.Job, body.job_id)
        job_info = {"status": j.status, "round": j.round_index} if j is not None else {"status": "unknown", "round": None}
    # jobs where this machine is not (yet) allowed to train, so the trainer log can say why it waits
    standing = [
        {"job": e.job_id, "name": j.name, "status": e.status}
        for e, j in session.execute(
            select(db.Enrolment, db.Job).join(db.Job, db.Job.id == db.Enrolment.job_id).where(db.Enrolment.machine_id == machine.id, db.Job.status == "running", db.Enrolment.status.in_(("pending", "rejected")))
        ).all()
    ]
    session.commit()
    state.bus.publish("fleet", {"event": "heartbeat", "node": machine.node_id, "status": body.status, "job": body.job_id, "round": body.round})
    if body.status == "error" and was != "error":
        state.engine.notify("machine.error", f"machine {machine.name} reports an error: {body.status_detail}", node=machine.node_id, name=machine.name)
    for line in body.logs[:200]:
        state.bus.publish(f"logs:{machine.node_id}", {"node": machine.node_id, **line})
    return {
        "interval": state.cfg.policy.heartbeat_interval_s,
        "idle_interval": state.cfg.policy.idle_poll_interval_s,
        "commands": commands,
        "assignment": assignment,
        "job": job_info,
        "enrolments": standing,
        "policy": policy.load(session, state.cfg.policy.defaults).model_dump(mode="json"),
    }


def _parse_at(value):
    import datetime as dt

    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(dt.UTC).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
    return db.now()


@router.get("/me")
def me(machine: db.Machine = Depends(current_machine), session: Session = Depends(get_session)):
    return machine_out(session.get(db.Machine, machine.id))


class CommitIn(BaseModel):
    blob: str = Field(min_length=64, max_length=64)
    signature: str


@router.post("/rounds/{job_id}/{round_index}/commit")
def commit(job_id: str, round_index: int, body: CommitIn, machine: db.Machine = Depends(current_machine), session: Session = Depends(get_session), state=Depends(get_state)):
    try:
        upd = state.engine.commit(session, machine, job_id, round_index, body.blob, body.signature)
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"status": upd.status}


class RevealIn(BaseModel):
    blob: str = Field(min_length=64, max_length=64)


@router.post("/rounds/{job_id}/{round_index}/reveal")
def reveal(job_id: str, round_index: int, body: RevealIn, machine: db.Machine = Depends(current_machine), session: Session = Depends(get_session), state=Depends(get_state)):
    try:
        upd = state.engine.reveal(session, machine, job_id, round_index, body.blob)
    except EngineError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"status": upd.status, "samples": upd.samples}
