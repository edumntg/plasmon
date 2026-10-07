"""Database schema. SQLite for home mode, PostgreSQL for a company server."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

ROLES = ("owner", "admin", "operator", "member", "viewer")
ROLE_RANK = {r: i for i, r in enumerate(reversed(ROLES))}  # owner highest


def now() -> dt.datetime:
    """Naive UTC. SQLite stores no offset, so every stored time is naive UTC by rule."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(16), default="member")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)


class Token(Base):
    """API tokens for users and machines. Only the SHA-256 of the secret is stored."""

    __tablename__ = "tokens"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("tok"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(ForeignKey("machines.id"), nullable=True)
    scopes: Mapped[str] = mapped_column(Text, default="")  # comma separated
    label: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class DeviceCode(Base):
    __tablename__ = "device_codes"
    device_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    token_secret: Mapped[str | None] = mapped_column(Text, nullable=True)  # handed out once, then cleared
    label: Mapped[str] = mapped_column(String(120), default="")


class Machine(Base):
    __tablename__ = "machines"
    # status: training|idle|paused|unavailable|offline|error
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("mch"))
    node_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="offline")
    status_detail: Mapped[str] = mapped_column(Text, default="")
    current_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # latest heartbeat
    paused_by_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    draining: Mapped[bool] = mapped_column(Boolean, default=False)
    honesty: Mapped[float] = mapped_column(Float, default=1.0)
    rounds_served: Mapped[int] = mapped_column(Integer, default=0)
    samples_verified: Mapped[int] = mapped_column(Integer, default=0)
    versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    user: Mapped[User] = relationship()


class Heartbeat(Base):
    __tablename__ = "heartbeats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(ForeignKey("machines.id"), index=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)
    status: Mapped[str] = mapped_column(String(16))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("job"))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    seed: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)  # running|completed|failed|cancelled
    status_detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    round_index: Mapped[int] = mapped_column(Integer, default=0)
    total_rounds: Mapped[int] = mapped_column(Integer)
    theta_blob: Mapped[str] = mapped_column(String(64))
    init_blob: Mapped[str] = mapped_column(String(64))
    outer_state_blob: Mapped[str | None] = mapped_column(String(64), nullable=True)
    shards: Mapped[list[dict[str, Any]]] = mapped_column(JSON)  # [{index, blob, n}]
    eval_blob: Mapped[str] = mapped_column(String(64))
    param_count: Mapped[int] = mapped_column(Integer, default=0)
    last_eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_eval_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    credits_spent: Mapped[int] = mapped_column(Integer, default=0)
    funding: Mapped[int] = mapped_column(Integer, default=0)  # credits locked at submission
    held: Mapped[int] = mapped_column(Integer, default=0)  # of the funding, set aside by closed rounds for trainers and the fee
    settlement: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # written once, when the job ends
    data_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # wrapped shard key, hex; None when the shards are in clear
    owner: Mapped[User] = relationship()


class Round(Base):
    __tablename__ = "rounds"
    __table_args__ = (UniqueConstraint("job_id", "index", name="uq_round"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    theta_blob: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|aggregating|closed
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    deadline_at: Mapped[dt.datetime] = mapped_column(DateTime)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    accepted: Mapped[int] = mapped_column(Integer, default=0)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_loss_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    bytes_in: Mapped[int] = mapped_column(Integer, default=0)
    timings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Update(Base):
    """One trainer's work in one round: assignment, commit, reveal, score."""

    __tablename__ = "updates"
    __table_args__ = (
        UniqueConstraint("job_id", "round_index", "machine_id", name="uq_update"),
        Index("ix_updates_round", "job_id", "round_index"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    round_index: Mapped[int] = mapped_column(Integer)
    machine_id: Mapped[str] = mapped_column(ForeignKey("machines.id"), index=True)
    shard_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="assigned")
    # assigned|committed|revealed|accepted|rejected|expired
    assigned_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    committed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    revealed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    commit_blob: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    loss_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_bytes: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    gain_assigned: Mapped[float | None] = mapped_column(Float, nullable=True)
    gain_random: Mapped[float | None] = mapped_column(Float, nullable=True)
    reject_reason: Mapped[str] = mapped_column(Text, default="")
    credits: Mapped[int] = mapped_column(Integer, default=0)
    machine: Mapped[Machine] = relationship()


class Enrolment(Base):
    """One machine's standing on one job: asked to join, approved, rejected, or left."""

    __tablename__ = "enrolments"
    __table_args__ = (UniqueConstraint("job_id", "machine_id", name="uq_enrolment"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    machine_id: Mapped[str] = mapped_column(ForeignKey("machines.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected|left
    source: Mapped[str] = mapped_column(String(8), default="join")  # join: the machine asked; auto: the scheduler asked for it
    requested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    machine: Mapped[Machine] = relationship()


class Hold(Base):
    """Credits a closed round set aside from a job's funding. Released when the job ends."""

    __tablename__ = "holds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    round_index: Mapped[int] = mapped_column(Integer)
    update_id: Mapped[int | None] = mapped_column(ForeignKey("updates.id"), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(ForeignKey("machines.id"), nullable=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # None: the fee account
    amount: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="held")  # held|released|voided
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    released_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class LedgerEntry(Base):
    """Hash-chained, server-signed record of each closed round."""

    __tablename__ = "ledger"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    kind: Mapped[str] = mapped_column(String(24))  # round|job_created|job_finished|credit
    body: Mapped[dict[str, Any]] = mapped_column(JSON)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    signature: Mapped[str] = mapped_column(Text)


class Blob(Base):
    __tablename__ = "blobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    size: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(24), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class LogLine(Base):
    __tablename__ = "logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(ForeignKey("machines.id"), index=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    level: Mapped[str] = mapped_column(String(8))
    message: Mapped[str] = mapped_column(Text)


class AuditEvent(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Invite(Base):
    __tablename__ = "invites"
    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), default="member")
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime)
    used_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class CreditEntry(Base):
    """One movement of credits. Balance of a user = sum of amount over their entries."""

    __tablename__ = "credits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # None: the protocol fee account
    machine_id: Mapped[str | None] = mapped_column(ForeignKey("machines.id"), nullable=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True, index=True)
    round_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    amount: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # earn|spend|grant|fee|adjust|escrow|refund
    memo: Mapped[str] = mapped_column(Text, default="")


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary)


def make_engine(url: str):
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        path = url.split("sqlite:///", 1)[-1]
        if path and path != ":memory:":
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(conn, _):  # WAL lets the scheduler thread write while the API reads
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    Base.metadata.create_all(engine)
    add_missing_columns(engine)
    return engine


def add_missing_columns(engine) -> list[str]:
    """Columns added to a table after a server was first installed. `create_all` only creates
    tables, so a release that adds a column would otherwise break an existing database.
    Returns the columns added, as `table.column`."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"
                default = column.default.arg if column.default is not None and column.default.is_scalar else None
                if default is not None:
                    ddl += f" DEFAULT {default!r}" if isinstance(default, str) else f" DEFAULT {int(default) if isinstance(default, bool) else default}"
                    if not column.nullable:
                        ddl += " NOT NULL"
                conn.execute(text(ddl))
                added.append(f"{table.name}.{column.name}")
    return added


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False, future=True)
