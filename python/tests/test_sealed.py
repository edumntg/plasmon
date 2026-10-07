"""Sealed shards: round trip, tamper detection, key wrapping, schema migration of a column."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag
from plasmon.coordinator import db
from plasmon.coordinator.keys import KeyWrapper
from plasmon.core import sealed
from plasmon.core.identity import Identity
from sqlalchemy import inspect, text


def test_seal_roundtrip_and_tamper():
    key = sealed.new_key()
    blob = sealed.seal(key, b"x" * 5000)
    assert sealed.is_sealed(blob) and not sealed.is_sealed(b"PK\x03\x04")
    assert blob[:4] == b"PLSE" and blob[4] == 1
    assert sealed.unseal(key, blob) == b"x" * 5000
    assert sealed.seal(key, b"same") != sealed.seal(key, b"same")  # a fresh nonce every time
    with pytest.raises(sealed.SealError):
        sealed.unseal(sealed.new_key(), blob)
    flipped = bytearray(blob)
    flipped[-1] ^= 1
    with pytest.raises(sealed.SealError):
        sealed.unseal(key, bytes(flipped))
    with pytest.raises(sealed.SealError):
        sealed.unseal(key, b"not sealed at all")
    with pytest.raises(sealed.SealError):
        sealed.unseal(key, blob[:10])
    with pytest.raises(sealed.SealError):
        sealed.seal(b"short", b"data")


def test_key_wrapper_is_bound_to_the_server_key():
    server = Identity.generate()
    key = sealed.new_key()
    wrapped = KeyWrapper(server).wrap(key)
    assert KeyWrapper(server).unwrap(wrapped) == key
    assert KeyWrapper(server).wrap(key) != wrapped
    with pytest.raises(InvalidTag):
        KeyWrapper(Identity.generate()).unwrap(wrapped)


def test_missing_columns_are_added_on_start(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.sqlite3'}"
    engine = db.make_engine(url)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE jobs DROP COLUMN funding"))
        conn.execute(text("ALTER TABLE jobs DROP COLUMN settlement"))
    assert "funding" not in {c["name"] for c in inspect(engine).get_columns("jobs")}
    engine.dispose()
    engine = db.make_engine(url)
    columns = {c["name"] for c in inspect(engine).get_columns("jobs")}
    assert {"funding", "settlement", "held", "data_key"} <= columns
    with db.make_session_factory(engine)() as session:
        user = db.User(email="a@b.c")
        session.add(user)
        session.flush()
        job = db.Job(owner_id=user.id, name="j", spec={}, seed="s", total_rounds=1, theta_blob="0" * 64, init_blob="0" * 64, shards=[], eval_blob="0" * 64)
        session.add(job)
        session.commit()
        assert session.get(db.Job, job.id).funding == 0
    engine.dispose()
