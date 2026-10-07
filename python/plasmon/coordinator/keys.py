"""Per-job shard keys at rest. Each key is wrapped with AES-GCM under a key derived from the
server's own Ed25519 seed, so a copy of the database alone does not open any dataset."""

from __future__ import annotations

import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..core.identity import Identity

NONCE_LEN = 12


class KeyWrapper:
    def __init__(self, server: Identity):
        seed = server.private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        self.kek = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"plasmon shard key wrapping").derive(seed)

    def wrap(self, key: bytes) -> str:
        nonce = os.urandom(NONCE_LEN)
        return (nonce + AESGCM(self.kek).encrypt(nonce, key, b"")).hex()

    def unwrap(self, wrapped: str) -> bytes:
        raw = bytes.fromhex(wrapped)
        return AESGCM(self.kek).decrypt(raw[:NONCE_LEN], raw[NONCE_LEN:], b"")
