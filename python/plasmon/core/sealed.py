"""Sealed shards: AES-256-GCM with one random key per job.

A submitter seals the shards before upload, so the blob store only ever holds
ciphertext. The coordinator keeps the key and hands it to a machine together with an
assignment. A machine that was never assigned the job cannot read its data.

Layout: magic `PLSE`, version `1`, a 12-byte nonce, then ciphertext and tag.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"PLSE"
VERSION = 1
NONCE_LEN = 12
KEY_LEN = 32
AAD = b"plasmon shard v1"


class SealError(ValueError):
    pass


def new_key() -> bytes:
    return os.urandom(KEY_LEN)


def is_sealed(blob: bytes) -> bool:
    return blob[:4] == MAGIC


def seal(key: bytes, plaintext: bytes) -> bytes:
    _check_key(key)
    nonce = os.urandom(NONCE_LEN)
    return MAGIC + bytes([VERSION]) + nonce + AESGCM(key).encrypt(nonce, plaintext, AAD)


def unseal(key: bytes, blob: bytes) -> bytes:
    _check_key(key)
    if blob[:4] != MAGIC:
        raise SealError("not a sealed blob")
    if blob[4] != VERSION:
        raise SealError(f"unsupported sealed blob version {blob[4]}")
    head = 5 + NONCE_LEN
    if len(blob) < head + 16:
        raise SealError("sealed blob is truncated")
    try:
        return AESGCM(key).decrypt(blob[5:head], blob[head:], AAD)
    except InvalidTag as e:
        raise SealError("wrong key, or the blob was altered") from e


def _check_key(key: bytes) -> None:
    if len(key) != KEY_LEN:
        raise SealError(f"key must be {KEY_LEN} bytes")
