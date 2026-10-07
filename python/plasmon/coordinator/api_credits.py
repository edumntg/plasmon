"""Credits: my balance and entries, admin grants, balances and CSV export."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import credits, db
from .api_auth import Email
from .deps import current_user, get_session, get_state, require_role

router = APIRouter(prefix="/v1/credits", tags=["credits"])


def entry_out(e: db.CreditEntry) -> dict:
    return {"id": e.id, "at": e.at, "amount": e.amount, "kind": e.kind, "job_id": e.job_id, "round": e.round_index, "machine_id": e.machine_id, "memo": e.memo}


def hold_out(h: db.Hold, job_names: dict[str, str]) -> dict:
    return {"id": h.id, "at": h.created_at, "job_id": h.job_id, "job": job_names.get(h.job_id, h.job_id), "round": h.round_index, "machine_id": h.machine_id, "amount": h.amount, "status": h.status, "released_at": h.released_at}


@router.get("/me")
def me(limit: int = 100, user: db.User = Depends(current_user), session: Session = Depends(get_session), state=Depends(get_state)):
    holds = credits.holds_for(session, user.id, limit)
    names = {j.id: j.name for j in session.scalars(select(db.Job).where(db.Job.id.in_({h.job_id for h in holds}))).all()} if holds else {}
    return {
        "enabled": state.cfg.credits.enabled,
        "unit": state.cfg.credits.unit,
        "fee_pct": state.cfg.credits.fee_pct,
        "balance": credits.balance(session, user.id),
        "on_hold": credits.on_hold(session, user.id),
        "locked": credits.locked(session, user.id),
        "entries": [entry_out(e) for e in credits.entries_for(session, user.id, limit)],
        "holds": [hold_out(h, names) for h in holds],
    }


@router.get("/users")
def users(actor: db.User = Depends(require_role("admin")), session: Session = Depends(get_session)):
    return credits.balances(session)


class GrantIn(BaseModel):
    email: Email
    amount: int = Field(gt=0, le=10_000_000)
    memo: str = Field(default="", max_length=200)


@router.post("/grant")
def grant(body: GrantIn, actor: db.User = Depends(require_role("admin")), session: Session = Depends(get_session)):
    target = session.scalar(select(db.User).where(db.User.email == body.email))
    if target is None:
        raise HTTPException(404, "no such user")
    entry = credits.grant(session, target, body.amount, body.memo, actor)
    session.commit()
    return {"email": target.email, "granted": body.amount, "balance": credits.balance(session, target.id), "entry": entry_out(entry)}


@router.get("/export.csv")
def export(since_days: int = 30, actor: db.User = Depends(require_role("admin")), session: Session = Depends(get_session)):
    text = credits.export_csv(session, db.now() - dt.timedelta(days=since_days))
    return Response(text, media_type="text/csv", headers={"content-disposition": f"attachment; filename=plasmon-credits-{since_days}d.csv"})
