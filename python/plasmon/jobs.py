"""Submit a job: validate, prepare blobs, upload what the server lacks, create."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .client import Client
from .core import hashing, jobspec, sealed
from .train import data, models, weights


def prepare(spec: jobspec.JobSpec, progress: Callable[[str], None] = lambda s: None, data_key: bytes | None = None) -> dict[str, Any]:
    """Build the initial weights, the shards and the eval split. Returns blobs by id.

    With `privacy.encrypt_shards` every shard and the eval split are sealed with one random
    key (or `data_key`), so only ciphertext leaves this machine. The key is returned as
    `data_key` for the coordinator; the weights stay in clear because every trainer needs them."""
    blobs: dict[str, bytes] = {}
    key = (data_key or sealed.new_key()) if spec.privacy.encrypt_shards else None

    def pack(raw: bytes) -> bytes:
        return sealed.seal(key, raw) if key else raw
    progress(f"initial weights: {spec.model.arch} seed {spec.recipe.seed}")
    theta = models.init_weights(spec.model.arch, spec.model.config, spec.recipe.seed)
    if spec.model.init:
        path = Path(spec.model.init)
        if path.exists():
            theta = weights.from_bytes(path.read_bytes())
        else:
            raise FileNotFoundError(f"model.init {spec.model.init} not found")
    init_bytes = weights.to_bytes(theta)
    init_id = hashing.digest(init_bytes)
    blobs[init_id] = init_bytes
    progress(f"dataset: {spec.dataset.source}" + (" (downloading on first use)" if spec.dataset.source.startswith(("builtin://", "http")) else ""))
    train, eval_shard = data.load_source(spec.dataset.source, spec.dataset.eval_source, spec.dataset.eval_fraction, spec.dataset.label_column, tuple(spec.dataset.image_shape), spec.dataset.block_size)
    if len(eval_shard) > 2000:
        eval_shard = data.Shard(eval_shard.x[:2000], eval_shard.y[:2000])
    shards = data.split_shards(train, spec.dataset.shard_size, seed=spec.recipe.seed)
    progress(f"shards: {len(shards)} × {spec.dataset.shard_size} samples, eval {len(eval_shard)} samples")
    if key:
        progress("sealing the shards: only ciphertext is uploaded; the key goes to the coordinator")
    manifest = []
    for i, shard in enumerate(shards):
        b = pack(shard.to_bytes())
        sid = hashing.digest(b)
        blobs[sid] = b
        manifest.append({"index": i, "blob": sid, "n": len(shard)})
    eval_bytes = pack(eval_shard.to_bytes())
    eval_id = hashing.digest(eval_bytes)
    blobs[eval_id] = eval_bytes
    return {"init_blob": init_id, "eval_blob": eval_id, "shards": manifest, "blobs": blobs, "param_count": models.parameter_count(theta), "data_key": key.hex() if key else None}


def submit(client: Client, spec: jobspec.JobSpec, progress: Callable[[str], None] = lambda s: print(s, file=sys.stderr)) -> dict[str, Any]:
    prepared = prepare(spec, progress)
    blobs: dict[str, bytes] = prepared["blobs"]
    missing = client.missing_blobs(list(blobs))
    total = sum(len(blobs[m]) for m in missing)
    progress(f"uploading {len(missing)} of {len(blobs)} blobs ({total / 1e6:.1f} MB); the rest is already on the server")
    for i, blob_id in enumerate(missing, 1):
        client.put_blob(blobs[blob_id], kind="shard" if blob_id != prepared["init_blob"] else "theta")
        if i % 10 == 0 or i == len(missing):
            progress(f"  {i}/{len(missing)}")
    body = {
        "spec": spec.model_dump(mode="json"),
        "init_blob": prepared["init_blob"],
        "eval_blob": prepared["eval_blob"],
        "shards": prepared["shards"],
        "param_count": prepared["param_count"],
    }
    if prepared["data_key"]:
        body["data_key"] = prepared["data_key"]
    return client.create_job(body)
