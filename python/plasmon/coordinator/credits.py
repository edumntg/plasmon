"""Credit accounting: balances, grants, the per-round settlement."""

from __future__ import annotations

import csv
import datetime as dt
import io

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import db
from .config import CreditsConfig


def balance(session: Session, user_id: str) -> int:
    """Credits the user can spend or withdraw. Holds are not in it until a job releases them."""
    return int(session.scalar(select(func.coalesce(func.sum(db.CreditEntry.amount), 0)).where(db.CreditEntry.user_id == user_id)) or 0)


def on_hold(session: Session, user_id: str) -> int:
    """Credits set aside for this user's machines by rounds of jobs that have not ended."""
    return int(session.scalar(select(func.coalesce(func.sum(db.Hold.amount), 0)).where(db.Hold.user_id == user_id, db.Hold.status == "held")) or 0)


def locked(session: Session, user_id: str) -> int:
    """Funding of this user's running jobs. It returns as payouts to trainers, the fee, and a refund."""
    return int(session.scalar(select(func.coalesce(func.sum(db.Job.funding), 0)).where(db.Job.owner_id == user_id, db.Job.status == "running")) or 0)


def holds_for(session: Session, user_id: str, limit: int = 100) -> list[db.Hold]:
    return session.scalars(select(db.Hold).where(db.Hold.user_id == user_id).order_by(db.Hold.id.desc()).limit(limit)).all()


def balances(session: Session) -> list[dict]:
    rows = session.execute(
        select(db.User.id, db.User.email, db.User.role, func.coalesce(func.sum(db.CreditEntry.amount), 0))
        .outerjoin(db.CreditEntry, db.CreditEntry.user_id == db.User.id)
        .group_by(db.User.id, db.User.email, db.User.role)
        .order_by(db.User.email)
    ).all()
    fee = int(session.scalar(select(func.coalesce(func.sum(db.CreditEntry.amount), 0)).where(db.CreditEntry.user_id.is_(None))) or 0)
    return [{"user_id": r[0], "email": r[1], "role": r[2], "balance": int(r[3])} for r in rows] + [{"user_id": None, "email": "(fee account)", "role": "", "balance": fee}]


def grant(session: Session, user: db.User, amount: int, memo: str, actor: db.User | None) -> db.CreditEntry:
    entry = db.CreditEntry(user_id=user.id, amount=amount, kind="grant", memo=memo)
    session.add(entry)
    session.add(db.AuditEvent(actor_id=actor.id if actor else None, action="credits.grant", target=user.id, detail={"amount": amount, "memo": memo}))
    return entry


def settle_round(session: Session, cfg: CreditsConfig, job: db.Job, round_index: int, accepted: list[db.Update], price_per_1k: float) -> tuple[int, bool]:
    """Charge the owner for the accepted samples and pay the machines' owners.

    Returns (credits spent, owner can continue). Weights: score × samples, or samples when
    every score is zero. The fee goes to the fee account (user None).
    """
    if not cfg.enabled or price_per_1k <= 0 or not accepted:
        return 0, True
    total_samples = sum(u.samples for u in accepted)
    cost = round(price_per_1k * total_samples / 1000)
    if cost <= 0:
        return 0, True
    available = balance(session, job.owner_id)
    if available <= 0:
        return 0, False
    cost = min(cost, available)
    session.add(db.CreditEntry(user_id=job.owner_id, job_id=job.id, round_index=round_index, amount=-cost, kind="spend", memo=f"{job.name} round {round_index}: {total_samples} samples"))
    fee = round(cost * cfg.fee_pct / 100)
    pool = cost - fee
    shares = split_integer(pool, contribution_weights(accepted))
    paid = 0
    for u, share in zip(accepted, shares):
        if share <= 0:
            continue
        u.credits = share
        paid += share
        session.add(db.CreditEntry(user_id=u.machine.user_id, machine_id=u.machine_id, job_id=job.id, round_index=round_index, amount=share, kind="earn", memo=f"{job.name} round {round_index}: {u.samples} samples, score {u.score or 0:.3f}"))
    remainder = cost - paid  # exactly the fee: the shares sum to the pool
    if remainder:
        session.add(db.CreditEntry(user_id=None, job_id=job.id, round_index=round_index, amount=remainder, kind="fee", memo=f"{job.name} round {round_index}"))
    job.credits_spent += cost
    stop = (job.spec.get("budget", {}).get("max_credits") is not None and job.credits_spent >= job.spec["budget"]["max_credits"]) or available - cost <= 0
    return cost, not stop


def escrow(session: Session, job: db.Job, owner_id: str, funding: int) -> None:
    """Lock the funding: it leaves the owner's balance now and comes back only as a refund."""
    session.add(db.CreditEntry(user_id=owner_id, job_id=job.id, amount=-funding, kind="escrow", memo=f"{job.name}: funding locked for {job.total_rounds} rounds"))
    job.funding = funding


def contribution_weights(accepted: list[db.Update]) -> list[float]:
    """Verified contribution: score × samples, or samples alone when every score is zero."""
    weights = [max(u.score or 0.0, 0.0) * max(u.samples, 1) for u in accepted]
    if sum(weights) <= 0:
        weights = [float(max(u.samples, 1)) for u in accepted]
    return weights


def hold_round(session: Session, cfg: CreditsConfig, job: db.Job, round_index: int, accepted: list[db.Update], slice_credits: int) -> int:
    """Set this round's share of the funding aside: the fee for the fee account, the rest for
    the machines whose updates were accepted, by contribution. Nothing is paid yet."""
    if slice_credits <= 0 or not accepted:
        return 0
    fee = round(slice_credits * cfg.fee_pct / 100)
    pool = slice_credits - fee
    shares = split_integer(pool, contribution_weights(accepted))
    for u, share in zip(accepted, shares):
        u.credits = share
        if share > 0:
            session.add(db.Hold(job_id=job.id, round_index=round_index, update_id=u.id, machine_id=u.machine_id, user_id=u.machine.user_id, amount=share))
    if fee > 0:
        session.add(db.Hold(job_id=job.id, round_index=round_index, user_id=None, amount=fee))
    job.held += slice_credits
    return slice_credits


def settle_job(session: Session, job: db.Job, outcome: str) -> dict:
    """Release every hold of a job that ended. Trainers are paid for the rounds that closed,
    whatever the outcome. The fee is charged when the job completed or the owner cancelled
    it, and waived when the server failed it. The rest of the funding goes back to the owner."""
    holds = session.scalars(select(db.Hold).where(db.Hold.job_id == job.id, db.Hold.status == "held").order_by(db.Hold.id)).all()
    now = db.now()
    per_machine: dict[tuple[str | None, str | None], dict] = {}
    fee_total = 0
    for h in holds:
        if h.user_id is None:
            fee_total += h.amount
            h.status = "released" if outcome != "failed" else "voided"
            h.released_at = now
            continue
        entry = per_machine.setdefault((h.user_id, h.machine_id), {"user_id": h.user_id, "machine_id": h.machine_id, "amount": 0, "rounds": 0})
        entry["amount"] += h.amount
        entry["rounds"] += 1
        h.status = "released"
        h.released_at = now
    paid = 0
    machines = {m.id: m for m in session.scalars(select(db.Machine).where(db.Machine.id.in_([k[1] for k in per_machine if k[1]]))).all()} if per_machine else {}
    payouts = []
    for (user_id, machine_id), entry in per_machine.items():
        machine = machines.get(machine_id)
        session.add(db.CreditEntry(user_id=user_id, machine_id=machine_id, job_id=job.id, amount=entry["amount"], kind="earn", memo=f"{job.name}: {entry['rounds']} rounds released"))
        paid += entry["amount"]
        payouts.append({"user_id": user_id, "machine_id": machine_id, "machine": machine.name if machine else "", "node": machine.node_id if machine else "", "amount": entry["amount"], "rounds": entry["rounds"]})
    fee_charged = fee_total if outcome != "failed" else 0
    if fee_charged > 0:
        session.add(db.CreditEntry(user_id=None, job_id=job.id, amount=fee_charged, kind="fee", memo=f"{job.name}: fee on {paid + fee_charged} released"))
    refund = job.funding - paid - fee_charged
    if refund > 0:
        session.add(db.CreditEntry(user_id=job.owner_id, job_id=job.id, amount=refund, kind="refund", memo=f"{job.name}: unspent funding" + (", fee waived" if outcome == "failed" and fee_total else "")))
    summary = {"outcome": outcome, "funding": job.funding, "paid": paid, "fee": fee_charged, "refund": refund, "payouts": sorted(payouts, key=lambda p: -p["amount"]), "at": now.isoformat()}
    job.settlement = summary
    return summary


def split_integer(total: int, weights: list[float]) -> list[int]:
    """Split `total` into integers proportional to `weights`, summing exactly to `total`
    (largest remainder method). Avoids the float rounding that turns 900 into 899."""
    if total <= 0 or not weights:
        return [0] * len(weights)
    scale = sum(weights)
    if scale <= 0:
        weights = [1.0] * len(weights)
        scale = float(len(weights))
    exact = [total * w / scale for w in weights]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def entries_for(session: Session, user_id: str, limit: int = 100) -> list[db.CreditEntry]:
    return session.scalars(select(db.CreditEntry).where(db.CreditEntry.user_id == user_id).order_by(db.CreditEntry.id.desc()).limit(limit)).all()


def earned_by_machine(session: Session) -> dict[str, int]:
    rows = session.execute(select(db.CreditEntry.machine_id, func.sum(db.CreditEntry.amount)).where(db.CreditEntry.kind == "earn").group_by(db.CreditEntry.machine_id)).all()
    return {r[0]: int(r[1]) for r in rows if r[0]}


def held_by_machine(session: Session) -> dict[str, int]:
    rows = session.execute(select(db.Hold.machine_id, func.sum(db.Hold.amount)).where(db.Hold.status == "held").group_by(db.Hold.machine_id)).all()
    return {r[0]: int(r[1]) for r in rows if r[0]}


def export_csv(session: Session, since: dt.datetime) -> str:
    users = {u.id: u.email for u in session.scalars(select(db.User)).all()}
    machines = {m.id: m.name for m in session.scalars(select(db.Machine)).all()}
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["at_utc", "user", "machine", "job", "round", "kind", "amount", "memo"])
    for e in session.scalars(select(db.CreditEntry).where(db.CreditEntry.at >= since).order_by(db.CreditEntry.id)).all():
        w.writerow([e.at.isoformat(timespec="seconds"), users.get(e.user_id, "" if e.user_id else "fee account"), machines.get(e.machine_id, ""), e.job_id or "", e.round_index if e.round_index is not None else "", e.kind, e.amount, e.memo])
    return out.getvalue()
