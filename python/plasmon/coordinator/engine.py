"""Jobs, rounds, assignment, commit-reveal and aggregation.

The engine holds no state of its own except caches. Everything persistent is in the
database and the blob store, so a restarted coordinator continues where it stopped.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..core import assignment, identity, sealed
from ..core import frame as fr
from ..core.jobspec import JobSpec
from ..train import data, diloco, weights
from ..validator import scoring
from . import credits, db, ledger, mail
from .events import Bus
from .keys import KeyWrapper

log = logging.getLogger("plasmon.engine")


class EngineError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Assignment:
    job_id: str
    round_index: int
    theta_blob: str
    shard: dict[str, Any]
    spec: dict[str, Any]
    deadline_at: str
    data_key: str | None = None  # hex, only for sealed shards; never written to disk by the trainer

    def as_dict(self) -> dict[str, Any]:
        out = {
            "job_id": self.job_id,
            "round": self.round_index,
            "theta": self.theta_blob,
            "shard": self.shard,
            "spec": self.spec,
            "deadline_at": self.deadline_at,
        }
        if self.data_key:
            out["data_key"] = self.data_key
        return out


class Engine:
    def __init__(self, blobs, bus: Bus, server: identity.Identity, heartbeat_interval_s: int, retention=None, scoring_cfg=None, notifier=None, credits_cfg=None, idle_poll_interval_s: int = 3, public_url: str = ""):
        # A round does not close with fewer trainers than there are idle machines that fit until
        # the slower pollers had two idle polls to join. Keeps a fast machine from taking every round.
        self.gather_window_s = 2 * idle_poll_interval_s + 2
        from .config import CreditsConfig

        self.credits_cfg = credits_cfg or CreditsConfig()
        self.notifier = notifier
        self.public_url = public_url.rstrip("/")
        self.blobs = blobs
        self.bus = bus
        self.server = server
        self.keys = KeyWrapper(server)
        self.heartbeat_interval_s = heartbeat_interval_s
        self.retention = retention
        self.scoring = scoring.ScoringConfig(**scoring_cfg.model_dump()) if scoring_cfg is not None else scoring.ScoringConfig()
        self._last_cleanup = 0.0
        self._eval_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.RLock()

    # ----- jobs -------------------------------------------------------------------

    def create_job(
        self,
        session: Session,
        owner: db.User,
        spec: JobSpec,
        init_blob: str,
        shards: list[dict[str, Any]],
        eval_blob: str,
        param_count: int,
        data_key: bytes | None = None,
    ) -> db.Job:
        for blob_id in [init_blob, eval_blob, *[s["blob"] for s in shards]]:
            if not self.blobs.exists(blob_id):
                raise EngineError(f"blob {blob_id[:12]} was not uploaded", 409)
        if not shards:
            raise EngineError("a job needs at least one shard")
        if spec.privacy.encrypt_shards:
            if data_key is None:
                raise EngineError("privacy.encrypt_shards needs the data key that sealed the shards")
            try:
                sealed.unseal(data_key, self.blobs.get(eval_blob))
            except sealed.SealError as e:
                raise EngineError(f"the data key does not open the eval shard: {e}") from e
        elif data_key is not None:
            raise EngineError("a data key was sent but privacy.encrypt_shards is off")
        funding = spec.budget.funding or 0
        if funding:
            if not self.credits_cfg.enabled:
                raise EngineError("budget.funding needs credits enabled on this server", 400)
            have = credits.balance(session, owner.id)
            if have < funding:
                raise EngineError(f"not enough credits: balance {have}, this job locks {funding}", 402)
        elif self.credits_cfg.enabled and spec.budget.credits_per_1k_samples > 0:
            have = credits.balance(session, owner.id)
            need = spec.budget.max_credits if spec.budget.max_credits is not None else 1
            if have < max(need, 1):
                raise EngineError(f"not enough credits: balance {have}, this job needs {need}", 402)
        job = db.Job(
            owner_id=owner.id,
            name=spec.name,
            spec=spec.model_dump(mode="json"),
            seed=f"{spec.name}:{db.new_id('seed')}",
            total_rounds=spec.budget.rounds,
            theta_blob=init_blob,
            init_blob=init_blob,
            shards=shards,
            eval_blob=eval_blob,
            param_count=param_count,
            data_key=self.keys.wrap(data_key) if data_key else None,
        )
        session.add(job)
        session.flush()
        if funding:
            credits.escrow(session, job, owner.id, funding)
        self._open_round(session, job, spec)
        ledger.append(
            session,
            self.server,
            "job_created",
            {
                "job": job.id,
                "owner": owner.id,
                "name": job.name,
                "init": init_blob,
                "shards": len(shards),
                "rounds": job.total_rounds,
                "funding": funding,
                "enrolment": f"{spec.enrolment.mode}/{spec.enrolment.approval}",
                "sealed": spec.privacy.encrypt_shards,
            },
        )
        session.commit()
        self.bus.publish("jobs", {"event": "created", "job": job.id})
        return job

    def cancel_job(self, session: Session, job: db.Job, reason: str = "cancelled by owner") -> None:
        if job.status != "running":
            raise EngineError(f"job is {job.status}", 409)
        for u in session.scalars(select(db.Update).where(db.Update.job_id == job.id, db.Update.status.in_(("assigned", "committed")))):
            u.status = "expired"
        summary = self._end_job(session, job, "cancelled", reason)
        session.commit()
        self.bus.publish(f"job:{job.id}", {"event": "cancelled", "job": job.id})
        self.bus.publish("jobs", {"event": "cancelled", "job": job.id})
        self.notify("job.cancelled", f"job {job.name} ({job.id}) cancelled after {job.round_index} rounds: {reason}", job=job.id, name=job.name)
        self._mail_job_ended(session, job, summary)

    def _end_job(self, session: Session, job: db.Job, status: str, detail: str = "", theta: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Every way a job stops ends here: the status, the ledger entry, and for a funded job the
        settlement that releases the holds. Returns the settlement summary, or None."""
        job.status = status
        job.status_detail = detail
        job.finished_at = db.now()
        body = {"job": job.id, "status": status, "rounds": job.round_index, "credits_spent": job.credits_spent, **(extra or {})}
        if theta:
            body["theta"] = theta
        if detail:
            body["reason"] = detail
        ledger.append(session, self.server, "job_finished", body)
        summary = None
        if job.funding and job.settlement is None:
            summary = credits.settle_job(session, job, status)
            ledger.append(
                session,
                self.server,
                "job_settled",
                {
                    "job": job.id,
                    "outcome": status,
                    "funding": summary["funding"],
                    "paid": summary["paid"],
                    "fee": summary["fee"],
                    "refund": summary["refund"],
                    "payouts": [{"node": p["node"], "amount": p["amount"], "rounds": p["rounds"]} for p in summary["payouts"]],
                },
            )
        return summary

    def _open_round(self, session: Session, job: db.Job, spec: JobSpec) -> db.Round:
        rnd = db.Round(
            job_id=job.id,
            index=job.round_index,
            theta_blob=job.theta_blob,
            deadline_at=db.now() + dt.timedelta(seconds=spec.requirements.round_timeout_s),
        )
        session.add(rnd)
        session.flush()
        return rnd

    @staticmethod
    def current_round(session: Session, job: db.Job) -> db.Round | None:
        return session.scalar(select(db.Round).where(db.Round.job_id == job.id, db.Round.index == job.round_index))

    def shard_bytes(self, job: db.Job, blob_id: str) -> bytes:
        """A shard as the trainer sees it: unsealed when the job's shards are encrypted."""
        raw = self.blobs.get(blob_id)
        if job.data_key:
            return sealed.unseal(self.keys.unwrap(job.data_key), raw)
        return raw

    # ----- enrolment ----------------------------------------------------------------

    @staticmethod
    def enrolment_of(session: Session, job: db.Job, machine: db.Machine) -> db.Enrolment | None:
        return session.scalar(select(db.Enrolment).where(db.Enrolment.job_id == job.id, db.Enrolment.machine_id == machine.id))

    def _may_train(self, session: Session, job: db.Job, spec: JobSpec, machine: db.Machine) -> bool:
        """Enrolment rules at assignment time. A job in `auto` mode with owner approval asks the
        owner the first time a fitting machine shows up."""
        row = self.enrolment_of(session, job, machine)
        if row is not None:
            return row.status == "approved"
        if spec.enrolment.mode == "join":
            return False
        if spec.enrolment.approval == "owner":
            self.request_enrolment(session, job, spec, machine, source="auto")
            return False
        return True

    def request_enrolment(self, session: Session, job: db.Job, spec: JobSpec, machine: db.Machine, source: str) -> db.Enrolment:
        row = self.enrolment_of(session, job, machine)
        status = "pending" if spec.enrolment.approval == "owner" else "approved"
        if row is None:
            row = db.Enrolment(job_id=job.id, machine_id=machine.id, status=status, source=source)
            session.add(row)
        else:
            row.status, row.source, row.requested_at, row.decided_at, row.decided_by, row.note = status, source, db.now(), None, None, ""
        session.flush()
        self.bus.publish(f"job:{job.id}", {"event": "enrolment", "job": job.id, "node": machine.node_id, "status": status})
        if status == "pending":
            owner = session.get(db.User, job.owner_id)
            machine_owner = session.get(db.User, machine.user_id)
            self.notify("enrolment.requested", f"{machine.name} asks to train {job.name} ({job.id}); approval needed", job=job.id, name=job.name, node=machine.node_id, machine=machine.name)
            if owner is not None:
                subject, body = mail.enrolment_requested(job, machine, machine_owner.email if machine_owner else "", self.public_url)
                self.mail(owner.email, subject, body)
        return row

    def join_job(self, session: Session, job: db.Job, machine: db.Machine) -> db.Enrolment:
        if job.status != "running":
            raise EngineError(f"job is {job.status}", 409)
        spec = JobSpec.model_validate(job.spec)
        why = unfit_reason(machine, spec)
        if why:
            raise EngineError(f"{machine.name} does not meet the job requirements: {why}", 409)
        row = self.enrolment_of(session, job, machine)
        if row is not None and row.status in ("pending", "approved"):
            return row
        if row is not None and row.status == "rejected":
            raise EngineError("the owner rejected this machine for this job", 409)
        return self.request_enrolment(session, job, spec, machine, source="join")

    def leave_job(self, session: Session, job: db.Job, machine: db.Machine) -> db.Enrolment:
        row = self.enrolment_of(session, job, machine)
        if row is None:
            row = db.Enrolment(job_id=job.id, machine_id=machine.id, status="left", source="join")
            session.add(row)
        else:
            row.status = "left"
            row.decided_at = db.now()
        session.flush()
        self.bus.publish(f"job:{job.id}", {"event": "enrolment", "job": job.id, "node": machine.node_id, "status": "left"})
        return row

    def decide_enrolment(self, session: Session, job: db.Job, row: db.Enrolment, approve: bool, actor: db.User, note: str = "") -> db.Enrolment:
        if row.status == "left":
            raise EngineError("the machine left this job", 409)
        row.status = "approved" if approve else "rejected"
        row.decided_at = db.now()
        row.decided_by = actor.id
        row.note = note[:500]
        session.flush()
        machine = session.get(db.Machine, row.machine_id)
        self.bus.publish(f"job:{job.id}", {"event": "enrolment", "job": job.id, "node": machine.node_id, "status": row.status})
        self.notify(f"enrolment.{row.status}", f"{machine.name} {row.status} for {job.name} ({job.id})", job=job.id, name=job.name, node=machine.node_id, machine=machine.name)
        machine_owner = session.get(db.User, machine.user_id)
        if machine_owner is not None:
            subject, body = mail.enrolment_decided(job, machine, approve, note, self.public_url)
            self.mail(machine_owner.email, subject, body)
        return row

    # ----- assignment -------------------------------------------------------------

    def try_assign(self, session: Session, machine: db.Machine) -> Assignment | None:
        """Give an idle machine the current round of the job that needs it most."""
        if machine.paused_by_admin or machine.draining:
            return None
        if _holds_update(session, machine):
            return None
        jobs = session.scalars(select(db.Job).where(db.Job.status == "running").order_by(db.Job.created_at)).all()
        candidates: list[tuple[int, db.Job, db.Round, JobSpec, db.Update | None, int]] = []
        for job in jobs:
            spec = JobSpec.model_validate(job.spec)
            if not _machine_fits(machine, spec):
                continue
            rnd = self.current_round(session, job)
            if rnd is None or rnd.status != "open":
                continue
            if not self._may_train(session, job, spec, machine):
                continue
            taken = session.scalar(
                select(func.count()).select_from(db.Update).where(
                    db.Update.job_id == job.id,
                    db.Update.round_index == rnd.index,
                    db.Update.status.in_(("assigned", "committed", "revealed")),
                )
            )
            if taken >= spec.requirements.max_trainers:
                continue
            already = session.scalar(
                select(db.Update).where(
                    db.Update.job_id == job.id, db.Update.round_index == rnd.index, db.Update.machine_id == machine.id
                )
            )
            if already is not None and already.status not in ("expired", "rejected"):
                continue
            shard_index = pick_shard(session, job, spec, rnd, machine)
            if shard_index is None:
                continue
            candidates.append((taken, job, rnd, spec, already, shard_index))
        if not candidates:
            return None
        taken, job, rnd, spec, previous, shard_index = min(candidates, key=lambda c: c[0])
        shard = job.shards[shard_index]
        if previous is not None:  # the row is unique per (job, round, machine): reuse it
            previous.status = "assigned"
            previous.assigned_at = db.now()
            previous.shard_index = shard_index
            previous.commit_blob = previous.signature = None
            previous.committed_at = previous.revealed_at = None
            previous.reject_reason = ""
        else:
            session.add(db.Update(job_id=job.id, round_index=rnd.index, machine_id=machine.id, shard_index=shard_index))
        session.flush()
        self.bus.publish(f"job:{job.id}", {"event": "assigned", "job": job.id, "round": rnd.index, "node": machine.node_id})
        key = self.keys.unwrap(job.data_key).hex() if job.data_key else None
        return Assignment(job.id, rnd.index, rnd.theta_blob, {"index": shard_index, **shard}, job.spec, rnd.deadline_at.isoformat(), key)

    # ----- commit and reveal ------------------------------------------------------

    def _update_for(self, session: Session, machine: db.Machine, job_id: str, round_index: int) -> tuple[db.Job, db.Round, db.Update]:
        job = session.get(db.Job, job_id)
        if job is None:
            raise EngineError("unknown job", 404)
        rnd = session.scalar(select(db.Round).where(db.Round.job_id == job_id, db.Round.index == round_index))
        if rnd is None:
            raise EngineError("unknown round", 404)
        upd = session.scalar(
            select(db.Update).where(db.Update.job_id == job_id, db.Update.round_index == round_index, db.Update.machine_id == machine.id)
        )
        if upd is None:
            raise EngineError("this machine is not assigned to that round", 409)
        return job, rnd, upd

    def commit(self, session: Session, machine: db.Machine, job_id: str, round_index: int, blob: str, signature: str) -> db.Update:
        job, rnd, upd = self._update_for(session, machine, job_id, round_index)
        if upd.status not in ("assigned", "expired"):
            raise EngineError(f"update is already {upd.status}", 409)
        if rnd.status != "open":
            raise EngineError("round is closed", 409)
        payload = {"job": job_id, "round": round_index, "node": machine.node_id, "blob": blob}
        if not identity.verify(machine.node_id, payload, signature):
            raise EngineError("bad signature", 401)
        upd.status = "committed"
        upd.commit_blob = blob
        upd.signature = signature
        upd.committed_at = db.now()
        session.commit()
        return upd

    def reveal(self, session: Session, machine: db.Machine, job_id: str, round_index: int, blob: str) -> db.Update:
        job, rnd, upd = self._update_for(session, machine, job_id, round_index)
        if upd.status != "committed":
            raise EngineError(f"update is {upd.status}, expected committed", 409)
        if upd.commit_blob != blob:
            upd.status = "rejected"
            upd.reject_reason = "revealed blob differs from commit"
            session.commit()
            raise EngineError("revealed blob differs from the commit", 409)
        if not self.blobs.exists(blob):
            raise EngineError("frame not uploaded yet", 409)
        try:
            frame = fr.decode(self.blobs.get(blob))
        except fr.FrameError as e:
            upd.status = "rejected"
            upd.reject_reason = f"bad frame: {e}"
            session.commit()
            raise EngineError(f"bad frame: {e}") from e
        problems = []
        if frame.job != job_id:
            problems.append("job")
        if frame.round != round_index:
            problems.append("round")
        if frame.node != machine.node_id:
            problems.append("node")
        if frame.theta != rnd.theta_blob:
            problems.append("theta")
        if problems:
            upd.status = "rejected"
            upd.reject_reason = "frame header mismatch: " + ", ".join(problems)
            session.commit()
            raise EngineError(upd.reject_reason)
        upd.status = "revealed"
        upd.revealed_at = db.now()
        upd.samples = frame.samples
        upd.loss_start = _f(frame.meta.get("loss_start"))
        upd.loss_end = _f(frame.meta.get("loss_end"))
        upd.frame_bytes = self.blobs.size(blob)
        session.commit()
        self.bus.publish(f"job:{job_id}", {"event": "revealed", "job": job_id, "round": round_index, "node": machine.node_id})
        return upd

    # ----- scheduler tick ---------------------------------------------------------

    def tick(self, session: Session) -> None:
        with self._lock:
            self._expire_offline_machines(session)
            self._cleanup(session)
            for job in session.scalars(select(db.Job).where(db.Job.status == "running")).all():
                try:
                    self._tick_job(session, job)
                except Exception:  # keep other jobs alive; the failure is logged and stored
                    log.exception("job %s failed", job.id)
                    session.rollback()
                    job = session.get(db.Job, job.id)
                    if job is not None:
                        for u in session.scalars(select(db.Update).where(db.Update.job_id == job.id, db.Update.status.in_(("assigned", "committed")))):
                            u.status = "expired"
                            u.reject_reason = "job failed"
                        summary = self._end_job(session, job, "failed", "aggregation error, see server log")
                        session.commit()
                        self.bus.publish(f"job:{job.id}", {"event": "failed", "job": job.id})
                        self.notify("job.failed", f"job {job.name} ({job.id}) failed: aggregation error", job=job.id, name=job.name)
                        self._mail_job_ended(session, job, summary)

    def _cleanup(self, session: Session) -> None:
        """Delete old heartbeats, logs and device codes on the retention schedule."""
        if self.retention is None or time.time() - self._last_cleanup < self.retention.cleanup_interval_s:
            return
        self._last_cleanup = time.time()
        now = db.now()
        session.execute(delete(db.Heartbeat).where(db.Heartbeat.at < now - dt.timedelta(hours=self.retention.heartbeats_hours)))
        session.execute(delete(db.LogLine).where(db.LogLine.at < now - dt.timedelta(days=self.retention.logs_days)))
        session.execute(delete(db.DeviceCode).where(db.DeviceCode.expires_at < now))
        session.commit()

    def _expire_offline_machines(self, session: Session) -> None:
        cutoff = db.now() - dt.timedelta(seconds=3 * self.heartbeat_interval_s + 10)
        for m in session.scalars(select(db.Machine).where(db.Machine.status != "offline", db.Machine.last_seen_at < cutoff)):
            m.status = "offline"
            m.current_job_id = None
            m.current_round = None
            for u in session.scalars(select(db.Update).where(db.Update.machine_id == m.id, db.Update.status.in_(("assigned", "committed")))):
                u.status = "expired"
                u.reject_reason = "machine went offline"
            self.bus.publish("fleet", {"event": "offline", "node": m.node_id})
            self.notify("machine.offline", f"machine {m.name} went offline", node=m.node_id, name=m.name)
        session.commit()

    def _tick_job(self, session: Session, job: db.Job) -> None:
        spec = JobSpec.model_validate(job.spec)
        rnd = self.current_round(session, job)
        if rnd is None or rnd.status != "open":
            return
        counts = dict(
            session.execute(
                select(db.Update.status, func.count()).where(db.Update.job_id == job.id, db.Update.round_index == rnd.index).group_by(db.Update.status)
            ).all()
        )
        revealed = counts.get("revealed", 0)
        in_flight = counts.get("assigned", 0) + counts.get("committed", 0)
        now = db.now()
        if revealed >= spec.requirements.min_trainers and in_flight == 0:
            expected = self._expected_trainers(session, job, spec)
            age = (now - rnd.opened_at).total_seconds()
            if revealed >= expected or age >= self.gather_window_s:
                self._close_round(session, job, rnd, spec)
        elif now >= rnd.deadline_at:
            for u in session.scalars(
                select(db.Update).where(db.Update.job_id == job.id, db.Update.round_index == rnd.index, db.Update.status.in_(("assigned", "committed")))
            ):
                u.status = "expired"
                u.reject_reason = "round deadline passed"
            if revealed >= 1:
                self._close_round(session, job, rnd, spec)
            else:
                rnd.deadline_at = now + dt.timedelta(seconds=spec.requirements.round_timeout_s)
                session.commit()
                self.bus.publish(f"job:{job.id}", {"event": "round_extended", "job": job.id, "round": rnd.index})

    def _expected_trainers(self, session: Session, job: db.Job, spec: JobSpec) -> int:
        """How many machines could take this round now: online, idle or training, not held back,
        fitting, and allowed by the job's enrolment rules."""
        machines = session.scalars(
            select(db.Machine).where(db.Machine.status.in_(("idle", "training")), db.Machine.paused_by_admin.is_(False), db.Machine.draining.is_(False))
        ).all()
        n = sum(1 for m in machines if _machine_fits(m, spec) and self._allowed_now(session, job, spec, m))
        return min(n, spec.requirements.max_trainers)

    def _allowed_now(self, session: Session, job: db.Job, spec: JobSpec, machine: db.Machine) -> bool:
        """Like `_may_train`, without asking the owner: a read-only check for counts and reasons."""
        row = self.enrolment_of(session, job, machine)
        if row is not None:
            return row.status == "approved"
        return spec.enrolment.mode == "auto" and spec.enrolment.approval == "none"

    def _eval_tensors(self, job: db.Job) -> tuple[torch.Tensor, torch.Tensor]:
        if job.eval_blob not in self._eval_cache:
            shard = data.Shard.from_bytes(self.shard_bytes(job, job.eval_blob))
            self._eval_cache[job.eval_blob] = data.to_tensors(shard)
        return self._eval_cache[job.eval_blob]

    def _close_round(self, session: Session, job: db.Job, rnd: db.Round, spec: JobSpec) -> None:
        t0 = time.perf_counter()
        rnd.status = "aggregating"
        session.commit()
        updates = session.scalars(
            select(db.Update).where(db.Update.job_id == job.id, db.Update.round_index == rnd.index, db.Update.status == "revealed")
        ).all()
        theta = weights.from_bytes(self.blobs.get(job.theta_blob))
        decoded: list[tuple[db.Update, dict]] = []
        for u in updates:
            try:
                _, delta = scoring.decode_delta(self.blobs.get(u.commit_blob))
                decoded.append((u, delta))
            except (fr.FrameError, FileNotFoundError) as e:
                u.status = "rejected"
                u.reject_reason = f"unreadable frame: {e}"
        deltas, weights_, accepted = self._score(session, job, rnd, spec, theta, decoded)
        if not accepted:
            rnd.status = "open"
            rnd.deadline_at = db.now() + dt.timedelta(seconds=spec.requirements.round_timeout_s)
            session.commit()
            self.bus.publish(f"job:{job.id}", {"event": "round_reopened", "job": job.id, "round": rnd.index, "reason": "no update passed verification"})
            return
        avg = diloco.average(deltas, weights_)
        outer = diloco.Outer(spec.recipe.outer_optimizer)
        if job.outer_state_blob:
            outer.load_state_dict(weights.from_bytes(self.blobs.get(job.outer_state_blob)))
        new_theta = outer.step(theta, avg)
        t_agg = time.perf_counter()
        ex, ey = self._eval_tensors(job)
        eval_loss, eval_acc = diloco.evaluate(spec.model.arch, spec.model.config, new_theta, ex, ey, device=torch.device("cpu"))
        t_eval = time.perf_counter()
        new_blob = self.blobs.put(weights.to_bytes(new_theta))
        state_blob = self.blobs.put(weights.to_bytes(outer.state_dict()))
        session.add(db.Blob(id=new_blob, size=self.blobs.size(new_blob), kind="theta"))
        for u in accepted:
            u.status = "accepted"
            m = session.get(db.Machine, u.machine_id)
            m.rounds_served += 1
            m.samples_verified += u.samples
        rnd.status = "closed"
        rnd.closed_at = db.now()
        rnd.accepted = len(accepted)
        rnd.eval_loss, rnd.eval_acc = eval_loss, eval_acc
        losses = [u.loss_end for u in accepted if u.loss_end is not None]
        rnd.mean_loss_end = sum(losses) / len(losses) if losses else None
        rnd.bytes_in = sum(u.frame_bytes for u in accepted)
        rnd.timings = {"aggregate_s": round(t_agg - t0, 3), "eval_s": round(t_eval - t_agg, 3)}
        spent, can_continue = credits.settle_round(session, self.credits_cfg, job, rnd.index, accepted, spec.budget.credits_per_1k_samples)
        held = 0
        if job.funding:
            slices = spec.round_slices
            held = credits.hold_round(session, self.credits_cfg, job, rnd.index, accepted, slices[rnd.index] if rnd.index < len(slices) else 0)
        ledger.append(
            session,
            self.server,
            "round",
            {
                "job": job.id,
                "round": rnd.index,
                "theta_in": job.theta_blob,
                "theta_out": new_blob,
                "updates": [{"node": u.machine.node_id, "blob": u.commit_blob, "shard": u.shard_index, "samples": u.samples, "score_milli": int((u.score or 0) * 1000), "held": u.credits if job.funding else 0} for u in accepted],
                "rejected": [{"node": u.machine.node_id, "reason": u.reject_reason} for u in updates if u.status == "rejected"],
                "eval_loss_milli": int(eval_loss * 1000),
                "eval_acc_milli": int(eval_acc * 1000),
                "credits_spent": spent,
                "credits_held": held,
            },
        )
        job.theta_blob = new_blob
        job.outer_state_blob = state_blob
        job.last_eval_loss, job.last_eval_acc = eval_loss, eval_acc
        job.round_index += 1
        finished = job.round_index >= job.total_rounds
        out_of_credits = not can_continue and not finished
        summary = None
        if finished:
            summary = self._end_job(session, job, "completed", theta=new_blob)
        elif out_of_credits:
            detail = "out of credits" if job.spec.get("budget", {}).get("max_credits") is None or job.credits_spent < job.spec["budget"]["max_credits"] else "credit budget reached"
            summary = self._end_job(session, job, "cancelled", detail, theta=new_blob)
        else:
            self._open_round(session, job, spec)
        session.commit()
        if out_of_credits:
            self.notify("job.cancelled", f"job {job.name} ({job.id}) stopped after {job.round_index} rounds: {job.status_detail}", job=job.id, name=job.name)
        if finished:
            self.notify("job.completed", f"job {job.name} ({job.id}) completed: eval loss {eval_loss:.4f}, accuracy {100 * eval_acc:.1f} %", job=job.id, name=job.name, eval_loss=eval_loss, eval_acc=eval_acc)
        if finished or out_of_credits:
            self._mail_job_ended(session, job, summary)
        self.bus.publish(
            f"job:{job.id}",
            {"event": "round_closed", "job": job.id, "round": rnd.index, "eval_loss": eval_loss, "eval_acc": eval_acc, "accepted": len(accepted), "status": job.status},
        )
        self.bus.publish("jobs", {"event": "round_closed", "job": job.id, "round": rnd.index, "status": job.status})

    def _score(self, session: Session, job: db.Job, rnd: db.Round, spec: JobSpec, theta, decoded):
        """Run verification; mark rejected updates; return (deltas, weights, accepted updates)."""
        import random

        if not decoded:
            return [], [], []
        if not self.scoring.enabled:
            for u, _ in decoded:
                u.score = 1.0
            return [d for _, d in decoded], [float(max(u.samples, 1)) for u, _ in decoded], [u for u, _ in decoded]
        rng = random.Random(f"{job.seed}:{rnd.index}")
        needed = {u.shard_index for u, _ in decoded}
        random_shard = rng.randrange(len(job.shards))
        needed.add(random_shard)
        shard_bytes = {i: self.shard_bytes(job, job.shards[i]["blob"]) for i in needed}
        scorer = scoring.Scorer(self.scoring, spec, theta, shard_bytes, random_shard)
        norms = sorted(n for n in (scoring.delta_norm(d) for _, d in decoded) if math.isfinite(n))
        median_norm = norms[len(norms) // 2] if len(norms) >= 3 else None
        # commit order decides who copied whom: the later commit is the duplicate
        ordered = sorted(range(len(decoded)), key=lambda i: decoded[i][0].committed_at or db.now())
        dupes = scoring.duplicates([(i, decoded[i][1]) for i in ordered])
        deltas, weights_, accepted = [], [], []
        for idx, (u, delta) in enumerate(decoded):
            m = session.get(db.Machine, u.machine_id)
            sampled = rng.random() < self.scoring.sample
            if idx in dupes:
                earlier, cos = dupes[idx]
                verdict = scoring.Verdict(False, f"duplicate of an update committed earlier by {decoded[earlier][0].machine.name} (cosine {cos:.3f})", score=0.0, signal=0)
            else:
                verdict = scorer.judge(delta, u.shard_index, median_norm, m.honesty, sampled)
            u.score = verdict.score
            u.gain_assigned = verdict.gain_assigned
            u.gain_random = verdict.gain_random
            m.honesty = scoring.update_honesty(m.honesty, verdict.signal, self.scoring.honesty_alpha)
            if verdict.accepted:
                deltas.append(delta)
                weights_.append(float(max(u.samples, 1)))
                accepted.append(u)
            else:
                u.status = "rejected"
                u.reject_reason = verdict.reason
                self.bus.publish(f"job:{job.id}", {"event": "rejected", "job": job.id, "round": rnd.index, "node": m.node_id, "reason": verdict.reason})
        return deltas, weights_, accepted

    # ----- notifications ----------------------------------------------------------

    def notify(self, event: str, text: str, **data) -> None:
        if self.notifier is not None:
            self.notifier.send(event, text, data)

    def mail(self, to: str, subject: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier.email(to, subject, body)

    def _mail_job_ended(self, session: Session, job: db.Job, summary: dict[str, Any] | None) -> None:
        owner = session.get(db.User, job.owner_id)
        if owner is not None:
            subject, body = mail.job_ended(job, summary, self.public_url)
            self.mail(owner.email, subject, body)
        for payout in (summary or {}).get("payouts", []):
            user = session.get(db.User, payout["user_id"]) if payout.get("user_id") else None
            if user is not None:
                subject, body = mail.payout_released(job, payout, self.public_url)
                self.mail(user.email, subject, body)


def pick_shard(session: Session, job: db.Job, spec: JobSpec, rnd: db.Round, machine: db.Machine) -> int | None:
    """The shard this machine trains in this round, or None when the job's exposure cap
    leaves it nothing new to take.

    Two trainers must not train the same data in one round, so the hashed index is a start
    and the first shard nobody holds wins. With `sticky_shards` a machine prefers a shard it
    already trained; with `max_shards_per_machine` it never learns more shards than that."""
    in_use = set(
        session.scalars(
            select(db.Update.shard_index).where(
                db.Update.job_id == job.id, db.Update.round_index == rnd.index, db.Update.status.in_(("assigned", "committed", "revealed"))
            )
        ).all()
    )
    seen: set[int] = set()
    if spec.privacy.sticky_shards or spec.privacy.max_shards_per_machine:
        seen = set(session.scalars(select(db.Update.shard_index).where(db.Update.job_id == job.id, db.Update.machine_id == machine.id, db.Update.round_index < rnd.index)).all())
    if spec.privacy.sticky_shards:
        own = sorted(seen - in_use)
        if own:
            return own[0]
    if spec.privacy.max_shards_per_machine and len(seen) >= spec.privacy.max_shards_per_machine:
        return None
    start = assignment.shard_index(job.seed, rnd.index, machine.node_id, len(job.shards))
    for step in range(len(job.shards)):
        candidate = (start + step) % len(job.shards)
        if candidate not in in_use:
            return candidate
    return start  # more trainers than shards: shards repeat


def exposure(session: Session, job: db.Job) -> list[dict[str, Any]]:
    """Which machine saw which shards of a job: the dataset exposure report."""
    rows = session.execute(
        select(db.Machine.id, db.Machine.name, db.Machine.node_id, db.Update.shard_index)
        .join(db.Update, db.Update.machine_id == db.Machine.id)
        .where(db.Update.job_id == job.id)
        .distinct()
    ).all()
    per: dict[str, dict[str, Any]] = {}
    for mid, name, node, shard in rows:
        entry = per.setdefault(mid, {"machine": name, "node": node, "shards": []})
        entry["shards"].append(shard)
    total = max(len(job.shards), 1)
    out = []
    for entry in per.values():
        entry["shards"].sort()
        entry["count"] = len(entry["shards"])
        entry["fraction"] = entry["count"] / total
        out.append(entry)
    return sorted(out, key=lambda e: (-e["count"], e["machine"]))


def _holds_update(session: Session, machine: db.Machine) -> bool:
    """True while the machine has an update in flight for a round of a running job."""
    n = session.scalar(
        select(func.count())
        .select_from(db.Update)
        .join(db.Job, db.Job.id == db.Update.job_id)
        .where(db.Update.machine_id == machine.id, db.Update.status.in_(("assigned", "committed")), db.Job.status == "running")
    )
    return bool(n)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def waiting_reason(session: Session, job: db.Job) -> str | None:
    """One sentence that says why a running job has no trainer right now. None when it has one."""
    if job.status != "running":
        return None
    rnd = Engine.current_round(session, job)
    if rnd is None or rnd.status != "open":
        return None
    # Between rounds of a job that progresses, the round is young and empty: that is not waiting.
    if rnd.index > 0 and rnd.opened_at is not None and (db.now() - rnd.opened_at).total_seconds() < 20:
        return None
    in_flight = session.scalar(
        select(func.count())
        .select_from(db.Update)
        .where(db.Update.job_id == job.id, db.Update.round_index == rnd.index, db.Update.status.in_(("assigned", "committed", "revealed")))
    )
    if in_flight:
        return None
    spec = JobSpec.model_validate(job.spec)
    machines = session.scalars(select(db.Machine).where(db.Machine.status != "offline")).all()
    if not machines:
        return "No machine is online. Start one with: plasmon trainer start"

    def held_back(m: db.Machine) -> str | None:
        if m.paused_by_admin:
            return "paused"
        if m.draining:
            return "draining"
        if m.status not in ("idle", "training"):
            return m.status
        if _holds_update(session, m):
            return "busy with another round"
        return None

    kinds: dict[str, int] = {}
    free: list[db.Machine] = []
    for m in machines:
        kind = held_back(m)
        if kind is None:
            free.append(m)
        else:
            kinds[kind] = kinds.get(kind, 0) + 1
    if not free:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        return f"{_plural(len(machines), 'machine')} online but none is free: {detail}"
    fit = [m for m in free if _machine_fits(m, spec)]
    if not fit:
        req = spec.requirements
        need = []
        if req.device != "any":
            need.append(f"device {req.device}")
        if req.min_vram_gb:
            need.append(f"{req.min_vram_gb:g} GB of GPU memory")
        if req.min_tflops:
            need.append(f"{req.min_tflops:g} TFLOPS")
        if req.min_honesty:
            need.append(f"honesty {req.min_honesty:g} or more")
        verb = "does" if len(free) == 1 else "do"
        return f"{_plural(len(free), 'free machine')} {verb} not meet the job requirements: {' and '.join(need)}"
    enrolments = {e.machine_id: e for e in session.scalars(select(db.Enrolment).where(db.Enrolment.job_id == job.id)).all()}
    allowed = []
    pending = 0
    for m in fit:
        row = enrolments.get(m.id)
        if row is not None and row.status == "approved":
            allowed.append(m)
        elif row is not None and row.status == "pending":
            pending += 1
        elif row is None and spec.enrolment.mode == "auto" and spec.enrolment.approval == "none":
            allowed.append(m)
    if not allowed:
        if pending:
            return f"{_plural(pending, 'machine')} waiting for the owner's approval"
        if spec.enrolment.mode == "join":
            return f"{_plural(len(fit), 'free machine')} fit; none has joined. A trainer joins with: plasmon trainer join {job.id}"
        return f"{_plural(len(fit), 'free machine')} fit; the owner must approve a machine before it trains"
    capped = [m for m in allowed if pick_shard(session, job, spec, rnd, m) is None]
    if len(capped) == len(allowed):
        return f"{_plural(len(allowed), 'free machine')} reached this job's limit of {spec.privacy.max_shards_per_machine} shards per machine"
    verb = "fits" if len(allowed) == 1 else "fit"
    return f"{_plural(len(allowed), 'free machine')} {verb}; the next heartbeat assigns the round"


def recent_trainers(session: Session, job: db.Job, show_names: bool = True) -> list[dict[str, Any]]:
    """Machines that took part in the current or the previous round, newest round first, one entry each.
    The dashboard keeps them in the job's group between rounds instead of showing an empty job."""
    if job.status != "running":
        return []
    rows = session.execute(
        select(db.Update, db.Machine)
        .join(db.Machine, db.Machine.id == db.Update.machine_id)
        .where(db.Update.job_id == job.id, db.Update.round_index >= job.round_index - 1)
        .order_by(db.Update.round_index.desc(), db.Update.id)
    ).all()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for u, m in rows:
        if m.node_id in seen:
            continue
        seen.add(m.node_id)
        out.append(
            {
                "node": m.node_id if show_names else m.node_id[:8],
                "name": m.name if show_names else m.node_id[:8],
                "round": u.round_index,
                "status": u.status,
                "current": u.round_index == job.round_index,
            }
        )
    return out


def unfit_reason(machine: db.Machine, spec: JobSpec) -> str:
    """Why a machine does not meet a job's requirements, or an empty string when it does."""
    hw = machine.hardware or {}
    req = spec.requirements
    gpu = hw.get("gpu") or {}
    if req.device == "cuda" and gpu.get("kind") != "cuda":
        return "needs an NVIDIA GPU"
    if req.device == "mps" and gpu.get("kind") != "mps":
        return "needs an Apple GPU"
    if req.min_vram_gb and float(gpu.get("vram_gb") or 0) < req.min_vram_gb:
        return f"needs {req.min_vram_gb:g} GB of GPU memory, this machine reports {float(gpu.get('vram_gb') or 0):g}"
    if req.min_tflops and float(hw.get("tflops") or 0) < req.min_tflops:
        return f"needs {req.min_tflops:g} TFLOPS, this machine measured {float(hw.get('tflops') or 0):g}"
    if req.min_honesty and machine.honesty < req.min_honesty:
        return f"needs honesty {req.min_honesty:g}, this machine has {machine.honesty:.2f}"
    return ""


def _machine_fits(machine: db.Machine, spec: JobSpec) -> bool:
    return unfit_reason(machine, spec) == ""


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class Scheduler(threading.Thread):
    """Runs `engine.tick` on an interval in its own thread with its own sessions."""

    def __init__(self, engine: Engine, session_factory, interval_s: float):
        super().__init__(name="plasmon-scheduler", daemon=True)
        self.engine = engine
        self.session_factory = session_factory
        self.interval_s = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                with self.session_factory() as session:
                    self.engine.tick(session)
            except Exception:
                log.exception("scheduler tick failed")
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
