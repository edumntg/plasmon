# Protocol, version 1

This document specifies the data formats that the coordinator, the trainers and the
CLI exchange. Python (`plasmon.core`) is the reference implementation. Rust
(`plasmon-core`) must produce the same bytes. The file
`python/tests/vectors/v1.json` holds test vectors for both.

## 1. Identity

- Each machine has one Ed25519 key pair.
- The node id is the public key as 64 lowercase hex characters.
- The private seed is 32 bytes in `machine.key`, file mode 0600.
- A signature is 64 bytes as 128 lowercase hex characters.
- A signature covers the canonical JSON of a payload (section 2).

## 2. Canonical JSON

Signed payloads use canonical JSON:

- Keys are sorted by Unicode code point.
- There is no whitespace between tokens.
- Strings are UTF-8. Only `"`, `\` and control characters are escaped.
- Allowed values: string, integer in the signed 64-bit range, boolean, null, array, object.
- Floats are not allowed. Put decimal values in strings or scale them to integers.

Example: the object `{"b": 1, "a": [true, null, "ñ"]}` serializes to
`{"a":[true,null,"ñ"],"b":1}`.

## 3. Content addressing

- A blob id is the BLAKE3 digest of the blob bytes, as 64 lowercase hex characters.
- The blob store keys blobs by blob id. A blob is immutable.
- A party verifies the digest after download. A mismatch is an error.

## 4. Shard assignment

The shard for one trainer in one round is:

```
index = u64_le(blake3(job_seed || u64_le(round) || node_id)[0:8]) mod num_shards
```

- `job_seed` and `node_id` are UTF-8 strings.
- `round` is a 64-bit little-endian integer.
- The result is deterministic. Any party can recompute it.
- The coordinator uses `index` as the start: if that shard is already assigned to another
  trainer in the same round, it takes the next free one, `(index + k) mod num_shards`
  for the smallest `k`. Two trainers in one round therefore train different shards; the
  round's ledger entry records which shard each update used. Only when a round has more
  trainers than the job has shards does a shard repeat.
- A job with `privacy.sticky_shards` prefers, for each trainer, the lowest-numbered shard
  that trainer already trained in an earlier round and that is free in this round; the
  hashed start applies only when none is free. A job with `privacy.max_shards_per_machine`
  gives a trainer no new shard once it has seen that many; the trainer sits the round out
  when all of its shards are taken.

## 5. Δ frame

A Δ frame carries one trainer's compressed pseudo-gradient for one round.

| Field | Size | Value |
|---|---|---|
| magic | 4 bytes | `PLSM` |
| version | 1 byte | `1` |
| kind | 1 byte | `1` sparse top-k, `2` dense |
| reserved | 2 bytes | `0` |
| header length | 8 bytes, little endian | byte length of the header |
| header | header length | JSON, UTF-8 |
| payload | rest | tensor data |

Header fields: `job`, `round`, `node`, `theta` (blob id of the weights the trainer
started from), `samples`, `meta` (free JSON, for example losses), `tensors`.

Each entry in `tensors` has `name`, `shape`, `numel`, `val_offset`, `val_len`, and for
sparse tensors `idx_offset` and `idx_len`. Offsets are relative to the start of the
payload.

- Sparse values: `uint32` little-endian indices, then `float16` values.
- Dense values: `float16` values in row-major order.
- The frame id is the BLAKE3 digest of the complete frame.

A receiver rejects a frame when: the magic or the version is wrong, the header does not
fit, an offset points outside the payload, or an index is not smaller than `numel`.

## 6. Commit and reveal

1. The trainer computes the frame and its frame id.
2. The trainer sends a commit: `{"job", "round", "node", "blob": frame id}` with a
   signature. The coordinator stores the commit and the time.
3. The trainer uploads the frame to the blob store.
4. The coordinator verifies the digest, matches it to the commit and marks the update as
   revealed.

A trainer cannot change an update after the commit. A trainer that reveals a frame with
a different id gets no credit for the round.

## 7. Verification at round close

The coordinator, or a validator with the same code, checks every revealed update before
aggregation.

Cheap checks, in this order. A failure rejects the update:

1. Tensor names must be the names of the model. Shapes must match.
2. All values must be finite.
3. The update must not be empty.
4. The L2 norm of the update must not exceed `norm_clip` times the median norm of the
   round. The check needs at least three updates.
5. The machine's honesty must not be below `honesty_floor`.

Loss delta, on a sample of `sample` × updates (default: all):

```
gain_assigned = L(θ_r; assigned shard) − L(θ_r − Δ; assigned shard)
gain_random   = L(θ_r; random shard)   − L(θ_r − Δ; random shard)
```

The random shard is one shard per round, chosen from the job seed and the round index.
The loss uses at most `max_eval_samples` samples of each shard.

- An update with `gain_assigned < min_gain` is rejected: the training made the model
  worse on the data the trainer had.
- The score of an accepted update is `max(gain_assigned, 0)`.
- Duplicates: updates of one round are compared pairwise by cosine similarity, in
  commit order. An update with similarity above 0.98 to an update committed earlier is
  rejected as a duplicate. Commit-reveal makes this sound: a copier can only copy an
  update after it was revealed, so the copier's commit is the later one.
- The honesty signal of the round is 1 when `gain_assigned > 0` and the update is not a
  duplicate, else 0.
- `honesty ← (1 − α) · honesty + α · signal`, with `α = honesty_alpha`. A new machine
  starts at 1.0.
- `gain_random` is stored and shown for review. With shards from one distribution an
  honest update helps random data almost as much as its own, so the difference does not
  drive honesty on its own.

Accepted updates are averaged with weights equal to their sample counts. The ledger
entry of the round lists the accepted updates with their scores and the rejected updates
with their reasons.

Known limits: one bad update can enter the aggregate before its machine's honesty falls
below the floor. A trainer that trains honestly on wrong data is indistinguishable from a
weak trainer. These limits are the same in every live network today.

## 8. Sealed shards

A job with `privacy.encrypt_shards` uploads every data shard and the eval shard sealed:

| Field | Size | Value |
|---|---|---|
| magic | 4 bytes | `PLSE` |
| version | 1 byte | `1` |
| nonce | 12 bytes | random per blob |
| body | rest | AES-256-GCM ciphertext and tag of the shard bytes |

- The key is 32 random bytes chosen by the submitter, one per job, sent to the
  coordinator in the job creation request as 64 hex characters.
- The associated data is the UTF-8 string `plasmon shard v1`.
- The blob id is the BLAKE3 digest of the sealed bytes. Two seals of the same shard have
  different nonces and therefore different blob ids.
- The coordinator stores the key wrapped (AES-GCM under a key derived from its own
  Ed25519 seed with HKDF-SHA256, info `plasmon shard key wrapping`) and sends it to a
  trainer only inside an assignment, as `data_key`.
- A trainer caches the sealed bytes and unseals them in memory. The initial weights and the
  Δ frames are not sealed: every trainer needs the weights, and frames are the trainer's own.

## 9. Ledger entries

The ledger holds one signed entry per event. Kinds: `job_created` (with `funding`, the
enrolment mode and whether the shards are sealed), `round` (accepted updates with their
shard, samples, score and credits held, rejected updates with reasons, credits spent and
held), `job_finished` (status, rounds, the final weights blob, a reason when it did not
complete) and `job_settled` (funding, paid, fee, refund, and one payout per node). A funded
job writes `job_finished` and `job_settled` together.
