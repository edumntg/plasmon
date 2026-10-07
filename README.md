# plasmon

**A permissionless network for training machine-learning models on other people's GPUs.**
Anyone submits a model and a dataset. Volunteer *trainers* each train on a slice of the
data, their updates are merged, and they are paid in proportion to the training progress
they verifiably contributed.

**Why "plasmon".** A plasmon is a quantum of collective oscillation: billions of
electrons in a metal moving together as a single wave. plasmon is that for compute:
thousands of GPUs, each on its own desk, training as one model.

> **Status: working engine, early.** The coordinator, the trainer, the Python commands,
> the dashboard and the native CLI run end to end on one machine or on a home network:
> see [docs/QUICKSTART.md](docs/QUICKSTART.md) (one machine),
> [docs/LOCAL-NETWORK.md](docs/LOCAL-NETWORK.md) (one server and many participants on the
> same Wi-Fi) and [docs/HOME-LAB.md](docs/HOME-LAB.md) (a Mac and a Windows PC).
> The roadmap (§12) marks what is done. Everything else in this document is the
> specification the implementation follows.

---

## Table of contents

1. [The idea](#1-the-idea)
2. [Design principles](#2-design-principles)
3. [How it works](#3-how-it-works)
4. [Architecture](#4-architecture)
5. [Interfaces: CLI, dashboard, roles and watch mode](#5-interfaces-cli-dashboard-roles-and-watch-mode)
6. [Running your own network (self-hosting)](#6-running-your-own-network-self-hosting)
7. [Tech stack](#7-tech-stack)
8. [Is it a server or a blockchain?](#8-is-it-a-server-or-a-blockchain)
9. [Going live: from laptop to first users](#9-going-live-from-laptop-to-first-users)
10. [Prior art and competitors](#10-prior-art-and-competitors)
11. [Repository layout](#11-repository-layout)
12. [Roadmap](#12-roadmap)
13. [Contributing and license](#13-contributing-and-license)

---

## 1. The idea

There are three kinds of participants:

| Role | What they bring | What they get |
|---|---|---|
| **Requester** | A model definition, a dataset, a training recipe and a budget | A trained model, cheaper than renting a cluster |
| **Trainer** | An idle GPU (RTX 3060 and up), bandwidth, uptime | Credits proportional to *verified* training progress |
| **Validator** | A GPU plus stake or reputation | A fee for scoring trainers' work honestly |

A requester publishes a *job*. The network splits the dataset into shards, assigns shards
to trainers, and runs the job in **rounds**. In each round every trainer trains locally
for a few hundred steps on its own shard, then publishes a compressed *update*. Validators
score the updates, the scored updates are merged into a new global model, and the next
round starts. When the budget or the token count is exhausted the requester downloads the
final weights. Trainers are paid per round from the requester's deposit, weighted by their
verified contribution.

Everything is operated from two surfaces: a **CLI** (`plasmon`) that trainers and
requesters live in, and a **web dashboard** where accounts, credits and history live and
where anyone can watch the network run (§5). The same software runs the public network
and a private one: a company can deploy the coordinator on its own server and let its
employees' machines train each other's models (§6).

The goal is **not** to out-train hyperscalers. It is to make the long tail of training
work (fine-tunes, domain models, 100 M to 10 B parameter pre-training, RL post-training)
cheap and open by using hardware that is already switched on and idle.

## 2. Design principles

Three decisions shape everything else. Each one follows from what has and has not worked
in internet-scale training over the last three years (§10).

**Merging is DiLoCo, not step-wise all-reduce and not weight averaging.** Multi-GPU
training in a datacentre exchanges gradients *every step* over 400 Gbit/s links; over the
internet a per-step all-reduce of even a 150 M parameter model is impossible. Averaging
independently trained weights after many steps does not work either: the replicas drift
apart and the average is worse than any of them. The method every working decentralized
run uses is **DiLoCo** (Distributed Low-Communication training): each trainer runs an
inner optimizer (AdamW) for H ≈ 100–500 local steps, computes a *pseudo-gradient*
Δ = θ_start − θ_end, compresses it, and a single **outer optimizer** (Nesterov momentum)
applies the average of all pseudo-gradients to the global model. Communication drops by
100–500× and convergence matches data-parallel training within a few percent. With
pseudo-gradient compression (DeMo / SparseLoCo: top-k 1–3 % plus 2-bit quantization with
error feedback) the per-round traffic for a 1 B model is tens of megabytes. This is the
base algorithm of plasmon.

**A coordinator, not a bespoke blockchain.** Every live decentralized training network
(Templar, Psyche, IOTA, Prime Intellect, Pluralis) has a *logically central* coordinator,
whether a smart contract, a validator set or an orchestrator service, that owns
membership, data assignment and round transitions, with *physically decentralized*
workers doing the compute and a P2P or object-storage layer moving blobs. Payments settle
on an existing chain. Nobody runs a bespoke blockchain for coordination and nobody does a
fully peer-to-peer all-reduce in production. plasmon has the same shape (§4, §8), and
spends its engineering on the training, verification and incentive layers rather than on
a transport protocol.

**Verification is built in and public.** If trainers are paid per update, someone will
submit random tensors, copy a neighbour's update, or train on an easier dataset.
Cryptographic proof-of-learning has been broken; bitwise-deterministic re-execution needs
special kernels and doubles the cost. The approach that works in production is
**economic/statistical verification**: validators measure how much each update actually
lowers the loss on held-out data, compare the trainer's *assigned* shard against *random*
data to catch copiers, and re-execute a random sample of rounds with a tolerance. Scores
and scoring code are public, so anyone can recompute them (§3.5).

## 3. How it works

### 3.1 Job specification

A requester submits a `job.yaml` (or the equivalent through the CLI / API):

```yaml
name: tinyllama-es-150m
model:
  source: hf://plasmon/tinyllama-150m-init   # or a safetensors upload; content-addressed
  framework: pytorch
  arch: llama                                   # from an allow-list in v0 (see sandboxing)
  params: 150M
dataset:
  source: hf://HuggingFaceFW/fineweb-edu         # today: builtin://mnist, fashion-mnist, cifar10, tinyshakespeare, an https:// URL or a path
  tokenizer: hf://plasmon/tinyllama-150m-init    #        of a .csv/.npz file, or a folder of IDX files; hf:// and s3:// are later
  total_tokens: 3_000_000_000
  shard_size_tokens: 50_000_000
recipe:
  algorithm: diloco
  inner_steps: 300
  inner_optimizer: {name: adamw, lr: 4e-4, betas: [0.9, 0.95], weight_decay: 0.1}
  outer_optimizer: {name: nesterov, lr: 0.7, momentum: 0.9}
  compression: {name: sparseloco, topk: 0.02, bits: 2, error_feedback: true}
  per_trainer_batch_tokens: 262_144
  mixed_precision: bf16
requirements:
  min_vram_gb: 12
  min_tflops: 20                               # today: matmul throughput measured at trainer start
  min_honesty: 0.8                             # today: reputation floor, see §3.5
  min_upload_mbps: 20
  min_trainers: 8
  max_trainers: 64
enrolment:
  mode: join                                   # today: auto (any idle machine that fits) or join (trainers pick it from the open jobs)
  approval: owner                              # today: none, or owner (the requester approves each machine, by mail or on the job page)
privacy:
  encrypt_shards: true                         # today: shards sealed on the requester's machine; the key travels only with an assignment
  sticky_shards: true                          # today: a machine keeps the shards it already saw
  max_shards_per_machine: 4
budget:
  rounds: 200
  funding: 500000                              # today: credits locked at submission, funding/rounds held per round, released when the job ends
  deadline: 2026-11-15T00:00:00Z
```

The coordinator validates the spec, computes a content hash of model and dataset, converts
the dataset into fixed-size shards stored in object storage, and opens the job for
enrolment when the deposit is locked. [docs/MARKETPLACE.md](docs/MARKETPLACE.md) walks
through funding, enrolment, approval and data privacy as they work today.

### 3.2 Node identity and enrolment

Every node (trainer, validator, requester, coordinator) has an **Ed25519 keypair**. Its
node ID is the public key. Every message on the network is signed; every blob is
content-addressed by BLAKE3 hash. Trainers announce their hardware (GPU model, VRAM,
measured upload bandwidth, a short benchmark score) and are admitted to a job if they
meet its requirements. In the paid phases a trainer bonds a small stake that can be
slashed for provably bad behaviour (invalid tensor shapes, missed commits after accepting
a slot).

### 3.3 The training round

A job runs in numbered rounds (windows). Each round is roughly `inner_steps × step_time`
long: a few minutes on a consumer GPU.

```
          ┌──────────────── round r ────────────────┐
trainer   pull θ_r ─▶ train H steps on shard(seed,r,id) ─▶ Δ_i = θ_r − θ_i ─▶ compress ─▶ commit hash ─▶ reveal blob
validator                                   ─▶ fetch sampled Δ_i ─▶ score ─▶ publish scores (signed)
coordinator                                                          ─▶ select top-G ─▶ θ_{r+1} = θ_r − η·OuterOpt(mean Δ) ─▶ publish θ_{r+1}
```

1. **Pull.** Trainer fetches the global weights θ_r (only the *changed* parameters since the
   last round it saw; full checkpoint on first join).
2. **Data assignment.** The shard for trainer *i* in round *r* is
   `shard = H(job_seed ‖ r ‖ node_id) mod num_shards`. It is deterministic and public, so
   validators can re-derive it, but a trainer cannot choose an easy shard.
3. **Local training.** H inner steps with AdamW under bf16 autocast. Trainers log loss per
   step; the log is part of the submission.
4. **Pseudo-gradient and compression.** Δ_i is sparsified (top-k with error-feedback
   residual kept locally) and quantized. For a 150 M model at 2 % top-k and 2 bits this is
   ≈ 1–2 MB; for 1 B ≈ 10–20 MB; for 7 B ≈ 70–150 MB per round.
5. **Commit–reveal.** Trainer first publishes `H(Δ_i)` to the coordinator, then uploads the
   blob. This stops a trainer from waiting to see others' updates and copying them.
6. **Scoring (§3.5).** Validators score a sample of updates.
7. **Aggregation.** The coordinator (or, from Phase 3, each validator redundantly) takes the
   top-G scored updates, de-quantizes, averages, applies the outer optimizer, and publishes
   θ_{r+1} plus the round's score table, all signed.
8. **Settlement.** Credits for round *r* are split among the top-G trainers proportional to
   `score_i × tokens_i`. Late or malformed submissions earn nothing for that round.

Trainers may join or leave at round boundaries. A round closes when `min_trainers`
updates are in or a timeout fires; stragglers' updates for round *r* arriving during
round *r+1* are either discarded or applied with staleness correction (HeLoCo-style),
configurable per job.

### 3.4 Communication budget

Why this is feasible on home connections (upload is the constraint):

| Model | Dense Δ (fp32) | Compressed Δ (2 % top-k, 2-bit + indices) | Round length on RTX 4090 (300 steps) | Upload needed |
|---|---|---|---|---|
| 150 M | 600 MB | ≈ 2 MB | ≈ 3 min | < 1 Mbit/s |
| 1 B | 4 GB | ≈ 15 MB | ≈ 12 min | < 1 Mbit/s |
| 7 B (LoRA / partial) | 28 GB (full) / 0.3 GB (LoRA r=64) | ≈ 100 MB / 1 MB | ≈ 25 min | 1–2 Mbit/s |

Downloads of θ_{r+1} are larger (the coordinator can send the dense delta or the
compressed aggregate; the latter is the same order as one update). Pipeline parallelism
for models that do not fit on one GPU is deliberately out of scope until Phase 4 (§12).

### 3.5 Verification and anti-cheating

Validators are nodes with stake or earned reputation. In each round each validator:

- runs a **cheap check on every update**: signature, tensor shapes, quantization ranges,
  timeliness, and a *sync score* (the trainer's declared θ_r hash must match the real one);
- runs an **expensive check on a random sample** (≈ 5 updates per validator per round):
  `gain_assigned = L(θ_r) − L(θ_r − βΔ_i)` on the trainer's *assigned* shard and
  `gain_random = L(θ_r) − L(θ_r − βΔ_i)` on a random held-out shard. An honest update has
  `gain_assigned > gain_random > 0`. A copied update has `gain_assigned ≈ gain_random`. A
  random update has both ≈ 0 or negative. The running mean of the sign of
  `(gain_assigned − gain_random)` is the trainer's **honesty score**; its magnitude feeds
  an OpenSkill rating that determines selection into the top-G and payout weight;
- optionally **re-executes** a sampled trainer's round with the same seed and compares the
  compressed update by Jaccard/cosine similarity within a calibrated tolerance (bitwise
  equality is not possible across GPUs).

Validator scores are published and signed; the coordinator uses the stake-weighted median.
A validator whose scores are consistently outliers loses reputation. All scoring code is
open source and all score tables are public, so anyone can recompute them. This is a
direct answer to the "black-box validator" critique of Bittensor.

What this does **not** solve, stated plainly: one bad update can land in the aggregate
before its author is down-weighted (mitigated by top-G selection and by clipping each Δ to
a norm bound); collusion between a majority of validators; and a trainer who honestly trains
on *wrong* data cannot be distinguished from a slightly weak GPU. These are the open
problems of the whole field (§10).

### 3.6 Rewards and economics

- Unit of account: **credits**, 1 credit = 1 USD-cent equivalent. Requesters buy credits
  (card / USDC); trainers withdraw credits (USDC, or fiat payout in later phases).
- A requester funds a job up front: `budget.funding` is locked when the job is submitted.
  Every round sets aside `funding / rounds`: the protocol fee for the fee account, the rest
  for the trainers whose updates were accepted, split by `score_i × samples_i`. The amounts
  stay **on hold** until the job ends; then they are released in one settlement, the fee is
  charged, unspent funding returns to the requester, and the requester can download the
  weights. A job cancelled early pays the rounds that closed and refunds the rest; a job
  the server failed pays the trainers and waives the fee. A job can instead pay per round
  (`budget.credits_per_1k_samples`), which charges the requester as rounds close.
- A machine earns in proportion to verified contribution: one share per accepted update,
  so a farm that runs one trainer per GPU takes that many shares per round, a machine that
  misses the deadline earns nothing for the round, and an update that fails verification
  earns nothing and costs honesty.
- Phase 1 and 2 ledger: an **append-only, hash-chained, signed log** published by the
  coordinator (each entry references the previous entry's hash; anyone can audit; the
  coordinator cannot rewrite history without detection). This is "blockchain-shaped"
  without a consensus network.
- Phase 3: settlement contract on an existing L2 (Base) or Solana: deposits, per-job escrow,
  payouts by Merkle proof of the round score table, slashing of trainer and validator bonds.
  **No new chain, no new token in v1.** A governance/utility token is a Phase 4 question,
  only if there is something a token does that credits cannot.

### 3.7 Sandboxing and safety

Trainers execute code chosen by requesters. In **v0 there is no arbitrary code**: the model
must be one of an allow-listed set of architectures instantiated from a config (Llama,
GPT-2, Mistral, ResNet, ViT, plus LoRA adapters on allow-listed HF checkpoints), and data
loaders are built-in (WebDataset shards of tokenized text or images). This removes remote
code execution from the threat model entirely. From Phase 3, custom `nn.Module` code runs
inside a container with no network access, a read-only filesystem, and GPU-only
capabilities (Docker with `--gpus`, seccomp profile, later gVisor), and is signed by the
requester.

Dataset privacy: a trainer computes on the samples it receives, so **a machine that trains
a shard sees that shard in clear**; no scheme that runs on consumer GPUs changes that.
What plasmon bounds is who sees data, how much, and where it sits in clear:
`privacy.encrypt_shards` seals the shards on the requester's machine so the blob store
holds ciphertext and only an assigned machine gets the key; `enrolment.approval: owner`
lets the requester approve each machine, with its hardware, reputation and owner in front
of them; `privacy.sticky_shards` and `privacy.max_shards_per_machine` keep each machine on
a few shards instead of a new one every round; and the job page reports which machine saw
which shards. Data that no trainer may see needs attested enclaves (datacenter GPUs, not
supported yet) or features from a frozen encoder computed on the requester's side.
[docs/MARKETPLACE.md](docs/MARKETPLACE.md) states the limits plainly.

## 4. Architecture

```
   ┌─────────────────────┐          ┌─────────────────────┐
   │  CLI / TUI          │          │  Web dashboard      │
   │  plasmon login      │          │  sign-up · plans    │
   │  job · trainer ·    │          │  credits · history  │
   │  validator · net    │          │  live network view  │
   └──────────┬──────────┘          └──────────┬──────────┘
              │  signed API calls (HTTPS / gRPC) │  HTTPS + SSE
              ▼                                  ▼
                         ┌──────────────────────────────────────────────┐
                         │               Coordinator (API)              │
                         │  accounts · jobs · rounds · assignment       │◀──── validator agents
                         │  aggregation · score tables · ledger         │      (score, re-execute)
                         │  FastAPI · Postgres · Redis · Python workers │
                         └───────┬───────────────────────────┬──────────┘
                                 │ signed metadata (gRPC/HTTP)│
                                 ▼                           ▼
                   ┌─────────────────────┐        ┌─────────────────────┐
                   │ Blob store          │        │ Settlement          │
                   │ S3-compatible (R2 / │        │ Phase 1–2: ledger   │
                   │ MinIO) + P2P blobs  │        │ Phase 3: contracts  │
                   │ (Iroh / Hivemind)   │        │ on Base or Solana   │
                   └──────────┬──────────┘        └─────────────────────┘
                              │ θ_r, shards, Δ_i (content-addressed, signed)
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ trainer  │         │ trainer  │   ...   │ trainer  │     `plasmon trainer start`
   │ RTX 3090 │         │ RTX 4070 │         │ A100     │     Python · PyTorch · CUDA
   └──────────┘         └──────────┘         └──────────┘
```

**Components**

| Component | Responsibility | Trust assumption |
|---|---|---|
| `coordinator` | Job registry, round state machine, deterministic data assignment, aggregation, publishing θ_{r+1}, ledger | Phase 1–2: run by the project, fully auditable. Phase 3: stateless replicas behind a multi-validator attestation; aggregation recomputed redundantly by validators |
| `trainer` | Pull weights, train, compress, commit-reveal, upload | Untrusted; verified by validators |
| `daemon` | One long-lived connection per machine: warm path for CLI commands, heartbeats and metrics, log shipping, control messages (pause, drain, stream logs) | Runs as the user with a machine-scoped token; can only report about its own machine |
| `validator` | Score updates, re-execute samples, publish signed scores | Semi-trusted via stake/reputation; scores are public and recomputable |
| `blobstore` | Move checkpoints, shards and updates | Dumb storage; everything is hashed and signed, so a malicious store can only deny service |
| `settlement` | Hold deposits, pay out, slash | Phase 1–2: coordinator's ledger. Phase 3: smart contracts |
| `cli` | `plasmon` TUI and subcommands: login, jobs, trainers, validators, credits, network | Client; holds the node keypair and an API token |
| `web` | Dashboard: accounts, plans and credits, job and trainer history, live network view, public leaderboard and explorer | Client of the same API; no privileged access |
| `sdk` | Python package used by the CLI and by scripts (`plasmon.Client`) | Client |

**Why a coordinator and not a DHT for everything.** Membership, round transitions and
assignment need a single source of truth with sub-second latency; a DHT gives neither.
The coordinator holds no secrets that matter (every artefact is signed by its author and
content-addressed) and keeps a public hash-chained log, so it can be replaced or replicated
without trusting its history. Blob *transfer* is where P2P pays off (trainers seeding
θ_{r+1} to each other rather than everyone hitting one bucket), so that is where the P2P
layer goes, in Phase 2.

## 5. Interfaces: CLI, dashboard, roles and watch mode

Two front ends, one API. Everything the web app can do, the CLI can do, and vice versa,
except payments, which only happen in the browser. Both talk to the coordinator with the
same signed requests; neither has privileges the other lacks.

### 5.1 The CLI (`plasmon`)

The CLI is where trainers and requesters spend their time, so it has to feel good *and*
be instant. Three reference points shaped it:

- **boxd**: one static binary per platform, installed by a script that checks a SHA-256
  manifest, with shell completions, up in milliseconds. Its machines boot in under 10 ms
  because they *resume a snapshot instead of booting*. The CLI borrows the same idea for
  its network path: keep warm state around, never pay start-up twice.
- **pi**: a custom terminal renderer that repaints only the lines that changed and uses
  synchronized output so nothing flickers. That is the rendering bar.
- **Codex CLI**: rewritten from TypeScript/Node to Rust for millisecond start-up and a
  dependency-free install. That is the stack decision, confirmed by measurement below.

**Stack: Rust.** `clap` for commands, `ratatui` + `crossterm` for the TUI, `tokio` +
`reqwest` (rustls, HTTP/2 and HTTP/3) for the API, `ed25519-dalek` and `blake3` for
identity and content addressing, `keyring` for the OS keychain. One static binary per
platform, 5–10 MB, no runtime, no interpreter, no `npm`, `pip` or `node` required.

**Why not the alternatives.** Cold start of `--version`, median of 25 runs on a Linux
x86_64 development box:

| Stack | Start-up | Binary | Used by |
|---|---|---|---|
| Rust, release build | **3 ms** | 0.4 MB | Codex CLI, gitui, atuin, yazi, bottom |
| Go | ~9 ms | ~2 MB | boxd-style tools, lazygit, gh (public benchmark, not re-measured) |
| Bun-compiled TypeScript | 30 ms | 94 MB | Claude Code (embeds the whole runtime) |
| Node 24 script | 74 ms | needs Node | pi, Gemini CLI |
| Python 3.12, no imports | 38 ms | needs Python | |
| Python + click | 100 ms | | |
| Python + Rich | 177 ms | | prime (Typer + Rich) starts here, before its own imports |

Rust starts 10× faster than the Bun route and 50× faster than the Python route, and it is
the language of the pieces the network needs later anyway: Iroh for P2P blobs, the blake3
reference implementation, and the compression kernels. Go with Bubble Tea would be a fine
second choice. Python is ruled out for the CLI by the numbers above and stays the language
of the trainer, where PyTorch is.

**Start-up budget**, enforced in CI with `hyperfine`:

| Path | Budget |
|---|---|
| `plasmon --version`, `plasmon --help` | < 5 ms; no config read, no network |
| `plasmon` home screen, first paint | < 30 ms, from local cache, before any network reply |
| `plasmon job submit`, local work (validate, hash, sign) | < 50 ms plus upload time for blobs the network has not seen |
| `plasmon job submit`, round trips to the coordinator | 1 to create the job; 0 extra when all blobs are already known |

**How the submit path stays light**

1. **Nothing to boot.** Single static binary; the async runtime and TLS are initialised
   lazily by the commands that use the network. `--help` and `--version` never touch them.
2. **Warm connection.** `plasmon daemon` (the same binary in daemon mode, started on first use,
   optional) keeps an HTTP/2 connection with an established TLS session to the coordinator
   and listens on a local Unix socket. Commands connect to the socket in well under a
   millisecond and reuse the warm connection, skipping TCP and TLS handshakes (one to
   three round trips, 50–150 ms on a typical link). Without the daemon the CLI connects
   directly; the coordinator also speaks HTTP/3, so returning clients get QUIC 0-RTT.
   This is the CLI analogue of what boxd does for machines: do not boot, resume.
3. **Content addressing and dedupe.** Model and dataset are hashed locally with blake3
   (multi-threaded, gigabytes per second). The submit request carries hashes; the
   coordinator returns pre-signed upload URLs only for blobs it does not already have.
   Resubmitting a job or reusing a dataset uploads nothing.
4. **Direct-to-storage uploads.** Blobs go straight to object storage in parallel chunks.
   The coordinator never proxies bytes.
5. **One small signed request.** The job spec is validated against a schema compiled into
   the binary, signed with the machine key, and sent as a single request. The reply is the
   job id and the first round's estimated start.
6. **Stale-while-revalidate.** Every read command paints from `~/.cache/plasmon/` immediately
   and refreshes in place when the API answers.
7. **The intro never costs time.** The start-up animation plays only on the bare `plasmon`
   command, runs concurrently with the status fetch, is skipped by any key, and is off when
   stdout is not a TTY or `NO_COLOR` / `PLASMON_NO_ANIM` is set.

**Start-up screen.** Running `plasmon` with no arguments in a TTY plays a short sequence
(about 0.9 s, any key skips it): a line of scattered dots locks into a travelling wave while
the wordmark resolves out of noise from left to right. Then the dashboard opens: five tabs
(overview, jobs, fleet, my machines, server), cards for machines online with a status bar,
jobs running, the scheduler and anything that needs attention, tables with status chips and
load bars, a loss chart per job, and a key-hint footer. `plasmon trainer start` prints the
same wordmark once and then hands the terminal to the trainer log. The sequence is off when
stdout is not a TTY, with `--plain`, or when `NO_COLOR`, `PLASMON_NO_ANIM` or `TERM=dumb`
is set. Colours come from the terminal's own palette, so every view reads on a light or a
dark theme.

```
    ●●●●●●●●                        ●●●●●●●●                        ●●●●●●●●
  ●●        ●●                    ●●        ●●                    ●●        ●●
●●            ●●●●            ●●●●            ●●●●            ●●●●            ●●
                  ●●●●●●●●●●●●                    ●●●●●●●●●●●●
            ██████╗ ██╗      █████╗ ███████╗███╗   ███╗ ██████╗ ███╗   ██╗
            ██╔══██╗██║     ██╔══██╗██╔════╝████╗ ████║██╔═══██╗████╗  ██║
            ██████╔╝██║     ███████║███████╗██╔████╔██║██║   ██║██╔██╗ ██║
            ██╔═══╝ ██║     ██╔══██║╚════██║██║╚██╔╝██║██║   ██║██║╚██╗██║
            ██║     ███████╗██║  ██║███████║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║
            ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                              thousands of GPUs, one wave
                              v0.1.2 · http://192.168.1.20:7117

 ∿ plasmon   1 overview   2 jobs   3 fleet   4 my machines   5 server          eduardo@home (owner)
──────────────────────────────────────────────────────────────────────────────────────────────────
╭ machines online ───────────╮╭ jobs running ──────────────╮╭ scheduler ─────────╮╭ attention ─────╮
│ 7 / 8  3 training · 1 idle ││ 1   0 completed            ││ running            ││ 2              │
│ ██████████▓▓▓▒▒▒░░░        ││ 4 rounds in the last hour  ││ up 21 min          ││ error 1 · offline 1
╰────────────────────────────╯╰────────────────────────────╯╰────────────────────╯╰────────────────╯
 jobs ────────────────────────────────────────────────────────────────────────────────────────────
  job          state       round             eval loss  acc     loss, last rounds
  mnist-cnn    ● running   ▰▰▰▱▱▱▱▱▱▱ 6/20   0.412      88.3 %  █▇▆▅▄▃▂▂▁▁
 your machines ───────────────────────────────────────────────────────────────────────────────────
  machine       status       detail         cpu       mem       gpu %     job / round        seen
  eduardo-mbp   ● training   step 19/50     ▰▰▱▱▱ 38  ▰▰▱▱▱ 46  ▰▰▰▰▱ 71  job_faf2081e53 r6  3 s
  win-tower     ● training   step 50/50     ▰▱▱▱▱ 22  ▰▰▱▱▱ 31  ▰▰▰▰▰ 88  job_faf2081e53 r6  3 s
  mac-mini      ● idle       waiting        ▱▱▱▱▱ 4   ▰▰▱▱▱ 35  –                             3 s

 1-5 tabs   tab next   r refresh   ? help   q quit                        ↻ 2 s · updated 1 s ago
```

**Commands.** Flat verbs grouped by noun; every command accepts `--json` for scripting
and `--plain` for logs.

```
plasmon                                                      intro, then the live dashboard (five tabs)
plasmon login | logout | whoami                              device-code login (prints a code and URL; confirm in the browser)
plasmon init                                                 create this machine's Ed25519 keypair and link it to your account

plasmon job submit job.yaml                                  validate, hash, estimate cost, confirm, submit
plasmon job list | status <id> | logs <id> --follow | cancel <id> | download <id> [--round N]
plasmon job open                                             running jobs a trainer can join: pay, requirements, who approves
plasmon job approvals <id> | approve <id> <machine> | reject <id> <machine>   the machines that asked to train your job

plasmon trainer start [--gpus 0,1] [--job <id> | --any] [--max-hours 8]
plasmon trainer join <job> | leave <job> [--machine <name>]  offer a machine to one open job, or take it off
plasmon trainer status | stop | earnings

plasmon validator start | status

plasmon net status | peers | rounds <job>
plasmon credits                                              balance and recent ledger entries; `credits buy` opens the browser
plasmon ledger verify                                        re-verify the hash chain and signatures of the public ledger
plasmon dashboard                                            the same dashboard, as an explicit command

plasmon daemon start | stop | status                         warm-connection daemon (started automatically on first use)
plasmon update                                               self-update from the signed release manifest
plasmon completions <shell>                                  install shell completions
```

**Live views.** `trainer start` renders a live panel: GPU utilisation and temperature,
current round, inner-step progress, loss sparkline, bytes uploaded this round, verified
tokens and credits earned this session. `job logs --follow` renders loss per round,
trainers per round and spend. `net status` is a table of jobs and a histogram of GPUs by
model. ratatui's `Sparkline`, `Gauge`, `Chart` and `BarChart` widgets cover all of these;
effects such as the colour sweep use `tachyonfx`.

**Login flow.** `plasmon login` requests a device code from the coordinator, prints
`https://plasmon.dev/device` plus an 8-character code, and polls. The user confirms in the
browser (creating an account if needed). The CLI stores a scoped API token in the OS
keychain (fallback: `~/.config/plasmon/credentials`, mode 600). The machine keypair created by
`plasmon init` is registered to the account so earnings from that machine accrue to the right
wallet. Tokens are revocable from the dashboard.

**Install.** Today the binaries come from GitHub Releases:

```bash
curl -fsSL https://raw.githubusercontent.com/edumntg/plasmon/main/install/install.sh | sh      # Linux, macOS
irm https://raw.githubusercontent.com/edumntg/plasmon/main/install/install.ps1 | iex                       # Windows, in PowerShell
python -m pip install "plasmon[engine] @ git+https://github.com/edumntg/plasmon.git"           # the engine (server, trainer)
```

The script downloads the binary for the platform, checks its SHA-256 against the
published checksum, installs to `~/.local/bin` (Windows: `%LOCALAPPDATA%\plasmon\bin`) and
adds the directory to `PATH` if needed. Targets: linux-x86_64, linux-arm64, darwin-arm64,
darwin-x86_64, windows-x86_64. Later: `plasmon.dev/install.sh`, a Homebrew tap,
`cargo binstall plasmon`, `winget`, and `plasmon update` from the signed manifest.
Training on Windows uses the CPU, or WSL2 for CUDA.

**The trainer is a separate process.** The trainer is Python (PyTorch) and is launched
and supervised by the CLI: `plasmon trainer start` finds the `plasmon` Python package
(`python -m plasmon`), starts it, and talks to it over a local socket. The protocol has a
reference implementation in Python (`plasmon.core`) and a Rust implementation in
`plasmon-core`; shared test vectors keep the two byte-identical. Every engine command also
works without the binary through `python -m plasmon`, with plain text output.

### 5.2 Roles and permissions

Everyone in an organisation (or on the public network) has one role. API tokens carry
scopes that can only narrow it.

| Role | Can | Typical holder |
|---|---|---|
| **Owner** | Everything Admin can, plus billing and plan, SSO configuration, delete the org | Whoever deployed the server |
| **Admin** | Fleet view of every machine (metrics, logs, pause, drain), server health, every job, users and roles, audit log, org policies | IT or ML-platform team |
| **Operator** | Fleet and Server pages read-only plus pause and drain; no user management; cannot read other people's job artefacts | On-call engineer |
| **Member** | Submit jobs, offer their machine, see their own jobs and machine, see aggregate network stats and the leaderboard | Every employee, every public user |
| **Viewer** | Aggregate stats and leaderboard only | Guests, management screens |

Scopes: `jobs:read` `jobs:write` `trainer:read` `trainer:write` `fleet:read` `fleet:write`
`server:read` `users:write` `billing:write`. Machine tokens created by `plasmon init` are
limited to `trainer:*` and can only report about their own machine. Privacy default:
Members see per-machine metrics for their own machines only; an org can switch on "open
fleet" so everyone sees everyone, which small teams like. `plasmon whoami` prints the role; a
command outside the role fails with a clear message ("fleet requires Operator or Admin;
ask r.vega").

### 5.3 Dashboard design

One web app, one sidebar, pages gated by role. Everything that is live on a page is
live in the CLI too (§5.4).

```
Overview      everyone      the network at a glance: machines online, jobs running, rounds/h, loss curves of active jobs
Jobs          everyone      my jobs (Member) · all jobs (Admin): status, loss, rounds, spend, ETA, trainers per round, requests to approve, payouts, data exposure
Open jobs     everyone      running jobs a trainer can join: pay per round, requirements, enrolment, the standing of my machines, join
My machine    everyone      the PC I am offering: live status, what it is training, usage, schedule, earnings, logs
Fleet         Admin/Op      all machines: table, detail per machine, metrics, logs, actions
Server        Admin/Op      coordinator health, round timings, queues, storage, DB, SSE clients, version
Users         Admin         people, roles, machines per person, invites, SSO group mapping
Ledger        everyone      the hash-chained log with a verify button; credits or chargeback per team (Admin)
Settings      Owner/Admin   org policies (availability defaults, caps, retention), notifications, billing
```

Design rules: one page answers one question ("is my machine training?", "is the fleet
healthy?", "where is my job?"); the most important number is first and large, on a card;
every live number carries a sparkline of the last hour; every table row opens a detail page
whose URL the CLI also prints; status colours are the same everywhere: **training** green,
**idle** blue, **paused** yellow, **unavailable** (outside window, on battery) grey,
**offline** red, **error** magenta. The Overview and Fleet pages open with a live SVG
diagram of the coordinator, the running jobs and the machines; a machine's path moves
while it sends an update and its status dot pulses with each heartbeat. The job page shows
the round in progress the same way: shards, trainers, aggregation, weights. Motion only
carries information (data moving, a live heartbeat, first-paint order) and stops under
`prefers-reduced-motion`.

**Fleet (Admin, Operator).** The scenario is a company with 100 employee machines.

```
FLEET  acme                                   online 87 · training 31 · idle 42 · paused 9 · unavailable 5 · offline 13
[ all teams ▾ ] [ GPU ≥ 8 GB ▾ ] [ status ▾ ]   search ____________                                  sort: GPU % ▾

MACHINE       OWNER      STATUS        GPU              VRAM        GPU%  CPU%  RAM%  TEMP  JOB / ROUND                 ↑Mbps  HONESTY  SEEN
ws-eng-037    r.vega     ● training    RTX 4090 24G     18.2/24 G   96    41    62    71°   tinyllama-es-150m  r412     12.4   0.98     2 s
ws-eng-012    a.lopez    ● training    RTX 3080 10G      8.9/10 G   91    35    58    76°   tinyllama-es-150m  r412      6.1   0.97     3 s
ws-des-004    m.chen     ● idle        RTX 3060 12G      0.4/12 G    2    12    30    38°   —                              —    0.95     5 s
mbp-mkt-021   j.ortiz    ○ unavailable Apple M3 (MPS)    —           —    18    55    —     outside window 08–19          —    —        9 s
ws-fin-009    d.kim      ● paused      RTX 4070 12G      0.2/12 G    0     8    41    35°   paused by user, 1 h           —    0.99    11 s
ws-eng-048    s.ruiz     ✖ error       RTX 3090 24G      —           —    —     —     —     driver 535 < required 550     —    0.91     4 m
ws-ops-002    —          offline       RTX 2080 8G       —           —    —     —     —     last seen 3 d ago             —    0.88     3 d
… 93 more

row → MACHINE DETAIL   hourly sparklines (GPU %, VRAM, CPU %, RAM %, temp, Mbps) · rounds served · tokens verified ·
                       honesty history · versions · live log tail (last 1 000 lines, follow) · actions: pause · drain · tag · revoke
```

**Server (Admin, Operator).** Coordinator replicas and version; API p50 and p99 latency;
per-job round timings (collect, score, aggregate, publish); scheduler queue depth;
Postgres, Redis and storage health and usage; SSE clients connected; ledger head hash and
last verification; recent errors. Every number on this page is also a Prometheus metric,
and the Grafana board that ships in the deployment bundle shows the same thing.

**My machine (everyone).** The page for the person who is lending their PC.

```
MY MACHINE  ws-eng-037                                                        ● training   since 19:02 (2 h 14 m)
┌─ now ─────────────────────────────────────────┐  ┌─ schedule and limits ───────────────────────────────┐
│ tinyllama-es-150m        round 412 of ~1 100  │  │ ■ only when idle (no input for 10 min)              │
│ submitted by r.vega (you)                     │  │ ■ weekdays 19:00–08:00 · weekends all day            │
│ inner step 231 / 300   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░   │  │ □ never on battery                                   │
│ loss 2.981  ▂▃▃▄▅▅▆▆▇▇       3.9 steps/s      │  │ GPU cap 100 %  ·  VRAM leave 2 GB  ·  upload 20 Mbps │
│ this round  ↑ 12.4 MB   ↓ 9.8 MB               │  │ [ pause 1 h ]  [ pause until tomorrow ]  [ resume ]  │
└───────────────────────────────────────────────┘  └─────────────────────────────────────────────────────┘
GPU 96 % ▁▂▇▇▇▇▇▇▇▇   VRAM 18.2 / 24 GB   CPU 41 %   RAM 62 %   temp 71 °C   fan 58 %   net ↑ 12.4  ↓ 3.1 Mbps
this session   41 rounds · 1.02 B verified tokens · +184 credits · honesty 0.98
last 7 days    212 rounds · 5.3 B tokens · +961 credits                           [ view log ]  [ machine settings ]
```

**Jobs → job detail (the requester; Admins see all).** A large loss curve with round
markers; status and ETA; spend so far and projected; a table of rounds with the trainers
in each (machine, tokens, score, accepted into the top-G or not); checkpoints to download;
logs; cancel. Which colleagues' machines trained a job is visible only when the org has
open fleet enabled; otherwise machine names are shown hashed.

**Overview (everyone).** Machines online, training and idle; jobs running; rounds per
hour; aggregate GPU utilisation; loss curves of active jobs; top 10 of the leaderboard;
the last ledger entry.

**Notifications.** Email, Slack or webhook for: job finished or failed, machine offline
for more than an hour, machine in error, honesty score dropped, round stalled.

### 5.4 Watch mode in the CLI

Every read command takes `--watch` (`-w`) and becomes a live, full-width table or panel
fed by the coordinator's SSE stream (fallback: polling, `--interval 5s`). Columns, colours
and role gating are the same as the web pages, so the terminal and the browser are
interchangeable.

```
plasmon fleet [--team eng] [--status training] --watch       Admin/Operator: the Fleet table, live
plasmon fleet show ws-eng-037 --watch                        one machine: gauges, sparklines, log tail
plasmon fleet logs ws-eng-037 --follow [--since 1h] [--grep error]
plasmon fleet pause | drain | resume ws-eng-037 [--for 1h]
plasmon server status --watch                                Admin/Operator: health, round timings, queues
plasmon job watch <id>                                       loss, rounds, trainers, spend, ETA
plasmon jobs --all --watch                                   Admin: every job in the org
plasmon trainer watch                                        My machine, live
plasmon users list | invite <email> --role member | set-role <user> admin
plasmon audit --since 24h                                    who did what, when
```

`plasmon fleet --watch` on an Admin's terminal:

```
FLEET acme   online 87  training 31  idle 42  paused 9  unavail 5  offline 13        rounds/h 14.2    ↻ 2 s    q quit
MACHINE       OWNER     STATUS       GPU            VRAM       GPU% CPU% RAM% TEMP  JOB / ROUND               ↑Mbps  HON   SEEN
ws-eng-037    r.vega    ● training   RTX 4090 24G   18.2/24    96   41   62   71°   tinyllama-es-150m r412    12.4   .98   2 s
ws-eng-012    a.lopez   ● training   RTX 3080 10G    8.9/10    91   35   58   76°   tinyllama-es-150m r412     6.1   .97   3 s
ws-des-004    m.chen    ● idle       RTX 3060 12G    0.4/12     2   12   30   38°   —                            —   .95   5 s
mbp-mkt-021   j.ortiz   ○ unavail    M3 (MPS)        —          —   18   55    —    outside window 08–19         —    —    9 s
ws-eng-048    s.ruiz    ✖ error      RTX 3090 24G    —          —    —    —    —    driver 535 < 550             —   .91   4 m
```

### 5.5 What machines report

The `plasmon daemon` on each machine keeps one long-lived connection to the coordinator, the
same warm connection that makes submits fast. Over it the machine sends a **heartbeat
every 10 s** (status, current job and round, inner step, GPU %, VRAM, CPU %, RAM %,
temperature, fan, network throughput, battery, seconds idle, versions) and **batched
structured logs** (level-filtered; 7-day retention and a per-machine size cap by default).
Over the same connection the coordinator sends **control messages**: pause, resume, drain
(finish the current round, then stop), start or stop a live log stream, apply a policy,
update the version. Machines never need an inbound port, so this works behind NAT, VPNs
and corporate firewalls. Heartbeats are kept at 10 s resolution for 24 h and downsampled
to 1 min for 90 days. The Fleet and My machine pages and every `--watch` view read from
this stream. Telemetry is hardware and trainer state only: no screen, no files, no process
list beyond the trainer's own.

## 6. Running your own network (self-hosting)

plasmon is open source and runs in two ways.

| | Hosted (plasmon.dev) | Self-hosted (your servers) |
|---|---|---|
| Who runs the coordinator | plasmon | You, from the Compose bundle or the Helm chart |
| Identity | plasmon.dev accounts | Your SSO over OIDC (Google Workspace, Okta, Entra ID, Keycloak), or local accounts with invites |
| Who can join | Anyone | Only your people |
| Payments | Credits bought with Stripe or USDC; trainers are paid | Off by default; optional internal credits for chargeback per team |
| Trust model | Staked validators, Gauntlet scoring | "Trusted fleet": validators optional; scoring stays on to catch broken machines |
| Data | Public or licensable datasets only | Private data allowed; it never leaves your perimeter |
| Updates | Continuous | You pull releases; employee CLIs follow the version your server serves |

This section is written for one scenario: a company with 100 employees and their own
workstations and laptops. Employee 37 wants to train a model. The other 99 machines are
online; the ones with a suitable GPU and an open availability window should train it, and
everyone should be able to see what is happening.

### 6.1 What you need

- **One Linux host** for the coordinator: 4 vCPU, 16 GB RAM, 200 GB SSD to start, as a VM
  in your cloud or a box on premises. A GPU on this host is optional; it speeds up
  aggregation for models above roughly 1 B parameters.
- **Docker 24+ with Compose**, or Kubernetes if you prefer the Helm chart.
- **A DNS name** (`plasmon.acme.com`) pointing at the host. TLS is automatic through the
  bundled Caddy (Let's Encrypt), or mount your corporate certificate.
- **Object storage.** The bundle ships MinIO; or point it at S3, R2, GCS or Azure Blob you
  already run. Budget about 20 GB per 1 B-parameter job for checkpoints at default
  retention.
- **PostgreSQL.** Bundled, or a managed instance.
- **Optional: an OIDC application** in your identity provider (client id and secret,
  redirect `https://plasmon.acme.com/auth/callback`), with a group for admins.
- **Employee machines.** Anything runs the CLI. Machines with an NVIDIA GPU (8 GB VRAM or
  more, driver 550 or newer) train. Apple Silicon machines train small jobs through MPS.
  CPU-only machines can act as validators and blob seeders or be excluded; they do not
  contribute meaningfully to training, and the dashboard says so instead of hiding it.

### 6.2 Deploy the coordinator

```bash
# on the server
curl -fsSL https://raw.githubusercontent.com/edumntg/plasmon/main/install/install.sh | sh   # the same CLI, used here for administration
plasmon server init \
    --domain plasmon.acme.com \
    --storage minio \                                           # or s3://acme-plasmon-blobs
    --db bundled \                                              # or postgres://user:pass@host/plasmon
    --sso oidc --oidc-issuer https://accounts.google.com \
    --oidc-client-id ... --oidc-client-secret ... \
    --admin-group plasmon-admins \
    --mode private                                              # trusted fleet, payments off
# writes ./plasmon/{docker-compose.yml, plasmon-server.yaml, Caddyfile, .env}  (.env holds generated secrets)
cd plasmon && docker compose up -d
plasmon server bootstrap --owner you@acme.com                   # creates the org and its Owner, prints an invite link
plasmon server status                                           # api ✓  worker ✓  db ✓  redis ✓  storage ✓  web ✓  v0.x.y
```

The bundle starts `coordinator-api` (stateless, scale with `--scale`), `coordinator-worker`
(scheduler, round state machine, aggregation), `postgres`, `redis`, `minio`, `web` (the
dashboard), `caddy` (TLS and reverse proxy) and, under an optional profile, `prometheus`
and `grafana`. Configuration lives in `plasmon-server.yaml`, secrets in `.env`. On Kubernetes,
`helm install plasmon oci://ghcr.io/edumntg/charts/plasmon -f values.yaml` deploys the same
components. The only open port on the server is 443 (HTTPS and HTTP/3).

### 6.3 Connect the machines

Each employee, or your device-management tool on their behalf:

```bash
curl -fsSL https://plasmon.acme.com/install.sh | sh             # your server serves the installer and pins the CLI version
plasmon login --server https://plasmon.acme.com                 # device code → browser → company SSO
plasmon trainer enable                                          # installs a user service: daemon at login, trains within policy
plasmon trainer watch                                           # "My machine" in the terminal
```

`trainer enable` installs a systemd user unit (Linux), a launchd agent (macOS) or a
scheduled task (Windows; training itself needs WSL2 for CUDA). It runs under the user's
account with no admin rights and starts paused outside the availability window. For a
silent rollout through Intune, Jamf or Ansible:
`PLASMON_SERVER=https://plasmon.acme.com plasmon trainer enable --policy org --non-interactive`, with
login completed by the user on first use or pre-provisioned through SSO device trust.

Org policy defaults are set by an Admin in Settings; each user can tighten them, not
loosen them:

- **Availability:** only after 10 min idle; weekdays 19:00–08:00, weekends all day; never on
  battery.
- **Caps:** GPU 100 % when training is allowed; leave 2 GB VRAM free; upload 20 Mbit/s;
  CPU 50 %.
- **Behaviour:** pause within a second when the user touches keyboard or mouse; resume after
  idle; drain at the end of the window (finish the round, then stop).
- **Updates:** the CLI follows the version the server advertises.

Machines appear in Fleet within one heartbeat of enabling. An Admin tags them once (team,
office, GPU class); tags drive filters, scheduling and quotas.

### 6.4 Walkthrough: employee 37 trains a model

1. **Submit.** On their workstation, employee 37 writes `job.yaml` (model from the internal
   registry or an upload; dataset from the internal bucket) and runs `plasmon job submit
   job.yaml`. Validation and blake3 hashing of a multi-gigabyte dataset take a few hundred
   milliseconds; blobs the server already has are skipped. The CLI prints the job id, the
   estimated trainer count and the dashboard URL.
2. **Schedule.** The coordinator selects machines that are online, inside their
   availability window, idle, not already assigned, with a GPU that meets the job's
   `min_vram_gb`. Say 23 of the 99 qualify at 15:00 (engineering GPUs whose owners are in
   meetings): the job starts round 1 with 23 trainers rather than waiting for the evening.
   If several jobs are queued, fair-share gives each user and team a slice, Admin quotas
   adjust it, and preemption happens only at round boundaries.
3. **Grow and shrink.** At 19:00 the window opens on the rest of the fleet; eligible
   machines join at the next round boundary and the job grows from 23 to 60 trainers. A
   laptop whose owner comes back pauses within a second; its partial round is simply not
   counted and the round closes on the remaining updates.
4. **Watch.** Employee 37 runs `plasmon job watch <id>` or opens the job page: loss per round,
   trainers per round, ETA, spend in internal credits. Each colleague's `plasmon trainer watch`
   or My machine page reads "training tinyllama-es-150m for r.vega, round 412". The Admin's
   Fleet view shows 60 green rows, GPU utilisation and temperatures, and the one machine in
   error with its driver-mismatch message.
5. **Verify.** In private mode, scoring still runs on a sample of updates every round (on
   the coordinator's GPU or on machines tagged `validator`), so a machine with a flaky GPU
   that produces garbage is down-weighted and flagged in Fleet instead of silently damaging
   the model.
6. **Finish.** When the token budget is reached, the final checkpoint is published to the
   internal bucket, employee 37 gets an email or Slack notification, and `plasmon job download
   <id>` fetches `model.safetensors` plus the training report (loss curve, rounds, machines,
   cost). The ledger holds one signed entry per round that anyone in the org can verify.

### 6.5 Network, security and data

- **Outbound only.** Machines open one HTTPS connection to the coordinator and move blobs
  to and from storage over HTTPS. No inbound ports; works behind NAT, VPNs and corporate
  proxies (`HTTPS_PROXY` is honoured). LAN blob seeding between machines (QUIC on UDP 7117)
  is optional and off by default.
- **Identity.** SSO over OIDC with group-to-role mapping (`plasmon-admins` → Admin); local
  accounts with invites when there is no IdP. Every machine has its own Ed25519 key, every
  message is signed, tokens are scoped and revocable, and the audit log records every admin
  action.
- **Data stays inside.** Datasets, checkpoints and updates live in your storage. The
  coordinator makes no outbound calls except, optionally, a release check
  (`updates.check: false` disables it). Air-gapped deployments mirror the release manifest
  and container images to an internal registry; everything else is already internal.
- **Trainer isolation.** The trainer runs under the employee's account without privileges.
  Only allow-listed architectures run in v0, so no arbitrary code reaches employee
  machines; custom model code (Phase 3) runs in a container with no network and a
  read-only filesystem.
- **Privacy inside the company.** Members see their own machine and aggregate stats;
  Admins see per-machine metrics. Telemetry is hardware and trainer state only (§5.5).

### 6.6 Operating it

- **Backups.** Nightly `pg_dump` from a bundled cron container, plus storage versioning.
  Restore: `docker compose down`, restore the database, `docker compose up -d`.
- **Upgrades.** `docker compose pull && docker compose up -d`; migrations run on start.
  Employee CLIs update to the version the server advertises.
- **Monitoring.** `/metrics` on every component; the Grafana board mirrors the Server
  page. Alerts ship for: coordinator down, round stalled beyond twice its expected time,
  storage above 80 %, more than 5 % of machines in error.
- **Scaling.** 100 machines: heartbeats are about 10 requests per second, and a round of a
  1 B model moves about 1.5 GB of updates in and 5 GB of checkpoints out, which one host and
  a LAN handle easily. 1 000 machines: scale `coordinator-api` to three replicas, move
  Postgres to a managed instance, put aggregation on a GPU host. Beyond that, the Phase 3
  multi-coordinator design (§8).
- **Retention.** Heartbeats 90 days downsampled, logs 7 days, checkpoints the last three per
  job plus the final one; all adjustable in `plasmon-server.yaml`.
- **Chargeback.** Internal credits are optional. Switch them on to attribute GPU-hours and
  tokens to teams and export a monthly CSV. Nobody is paid, but the leaderboard and the
  "verified tokens trained" badges still work as recognition.

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Trainer runtime | **Python 3.11+, PyTorch 2.x, CUDA 12.x** (bf16 autocast, `torch.compile` optional); launched and supervised by the CLI | Where every model and every volunteer already is. CPU and Apple MPS backends supported for small jobs and for developer testing; CUDA is the first-class target |
| Training algorithm | **DiLoCo** inner/outer loop; reference from Prime Intellect's `OpenDiLoCo` / `prime` (Apache-2.0) | Proven at 1–100 B scale over WAN |
| Compression | **SparseLoCo / DeMo** style top-k + low-bit + error feedback; reference code from Templar (MIT) and Nous Psyche (Apache-2.0/MIT) | 100–500× bandwidth reduction, convergence proven |
| Tensor wire format | **safetensors** for checkpoints; custom flat binary frame (BLAKE3 hash, dtype, shape, packed indices + values) for compressed Δ | Zero-copy binary; tensors never travel as text |
| Dataset format | **WebDataset** `.tar` shards of pre-tokenized `uint16`/`uint32` arrays, or images; content-addressed | Streamable, sliceable, cacheable, standard |
| Identity / signing | **Ed25519** (PyNaCl / `cryptography`), **BLAKE3** hashing | Fast, small, standard; same as Iroh node IDs |
| Coordinator API | **FastAPI** + **Pydantic v2** + **SQLAlchemy 2.0** on **PostgreSQL**; **Redis** for round timers and queues; **gRPC** streaming for trainer heartbeats and round events | Python keeps coordinator and trainer in one language; Postgres gives real transactions and row locks for batch assignment |
| Blob storage | **S3-compatible** (Cloudflare R2 in production, MinIO locally); Phase 2: **Iroh** (Rust, QUIC, hole-punching, blobs + gossip) or **Hivemind** DHT for peer seeding | Start boring, add P2P where it reduces cost |
| Validator agent | Same Python package as trainer, `plasmon validator start` | One binary, two modes |
| Scoring | Templar **Gauntlet**-style loss-delta scoring, **OpenSkill** ratings (`openskill` PyPI) | Deployed in production for 72 B; MIT |
| Settlement (Phase 3) | **Solidity on Base** (OpenZeppelin, Foundry) or **Anchor on Solana**; Merkle-root payouts | Use an existing chain; both have public reference implementations in this space |
| Sandbox (Phase 3) | **Docker** with `--gpus`, no network, read-only rootfs, seccomp; **gVisor** when available | Standard GPU isolation story |
| CLI | **Rust**: `clap`, `ratatui` + `crossterm`, `tokio`, `reqwest` (rustls, HTTP/2, HTTP/3), `ed25519-dalek`, `blake3`, `keyring`, `tachyonfx`; one static binary per platform; `plasmon daemon` mode keeps a warm connection | 3 ms start-up measured; dependency-free install; same language as Iroh and blake3 |
| Protocol core | Python `plasmon.core` (reference) and Rust crate **`plasmon-core`**: identity, signatures, canonical JSON, content addressing, Δ frames, shard assignment; cross-checked by shared test vectors | Two implementations, one specification (`docs/PROTOCOL.md`) |
| SDK | Python package `plasmon` (`plasmon.Client`); TypeScript SDK for the web later | Shared by CLI-launched trainer and user scripts |
| Web dashboard | **Next.js** (React, TypeScript) + **Tailwind** + **shadcn/ui**; charts with **Recharts**; live updates over **SSE**; deployed on Vercel or beside the coordinator | Standard, fast to build, good charting; the API stays in Python |
| Accounts and auth | Coordinator owns accounts: email + password / magic link, GitHub and Google OAuth, **OIDC SSO with group→role mapping** for self-hosted orgs (**authlib**), device-code flow for the CLI, scoped API tokens, RBAC (Owner / Admin / Operator / Member / Viewer), audit log, passkeys later | One identity for CLI and web; companies bring their own IdP |
| Payments | **Stripe** (cards, subscriptions for plans, Connect for fiat payouts later); **USDC** via Coinbase Commerce in Phase 2, direct on-chain in Phase 3 | Credits are the unit; fiat and crypto are just on-ramps |
| Packaging / ops | Cargo workspace + `pyproject.toml` (uv/hatch), signed release manifests + `install.sh`, Docker images for coordinator and trainer, `docker compose` dev stack, GitHub Actions with a `hyperfine` start-up budget check, **pytest** two-trainer integration test | Testable from day one |
| Observability | Structured logs (structlog), Prometheus metrics, Grafana dashboard: loss per round, trainers online, bytes per round, score distribution | Trainers need a public leaderboard and requesters need a loss curve |
| Fleet telemetry | 10 s heartbeats and batched logs over the daemon's long-lived connection; GPU via NVML (`nvml-wrapper`), CPU/RAM/battery/idle via `sysinfo` in the Rust daemon; stored in Postgres (10 s for 24 h, 1 min for 90 d, TimescaleDB optional); fan-out to web and CLI over SSE | Powers Fleet, My machine and every `--watch` view |
| Self-hosting bundle | `plasmon server init` → Docker Compose (api, worker, Postgres, Redis, MinIO, web, Caddy auto-TLS, Prometheus/Grafana profile) or Helm chart; org policies (availability windows, caps, idle detection); `plasmon trainer enable` installs systemd / launchd / scheduled-task services | One-command private deployment (§6) |

Deliberately **not** in the stack: a new blockchain, a new P2P protocol, JSON tensors, an
interpreted or runtime-bundled CLI, raw CUDA kernels in v1.

## 8. Is it a server or a blockchain?

Both questions people ask, answered directly.

**Is there a server?** Yes. There is a coordinator service and it is the only way the
system can start simply and ship. In Phase 1–2 it is one deployment run by the project. It
is, however, *designed to be untrusted*: every artefact it relays is signed by its author
and content-addressed; its ledger is hash-chained and public; its aggregation can be
recomputed by any validator from public inputs. If it misbehaves, that is detectable; if it
dies, a replica can resume from the public log and the blob store. From Phase 3, several
validators each run a coordinator replica and sign the round result; the client accepts a
round only when a quorum agrees. That is a small permissioned BFT set, not a public chain.

**Is it a blockchain?** Not in the sense of a new consensus network with its own token.
The *chain* in plasmon refers to (a) the hash-chained ledger of rounds, scores and
payouts, and (b) settlement of value on an existing public chain once there is value to
settle. This is the same shape as Psyche (coordinator = Solana program), Templar
(coordinator = validator set + Bittensor for emissions) and Prime Intellect (Rust
orchestrator + contracts on Base). Running a bespoke L1 has consumed several well-funded
teams and delivered no training; this project will not repeat that.

**Is it decentralized?** Compute, yes, from day one: the GPUs are other people's. Trust,
progressively: Phase 1 trusts the project's coordinator (while making its behaviour
auditable); Phase 3 trusts a quorum of staked validators; the design never requires
trusting trainers.

## 9. Going live: from laptop to first users

Two ways to run plasmon: hosted at plasmon.dev, or self-hosted inside a company (§6). The phases
below are the public network's path. A company can self-host from milestone M3 (§12)
onward; the Compose bundle the alpha runs on is the same one companies deploy.

### Phase 0: it works on one machine (target: 6–8 weeks of work)

- `docker compose up` starts Postgres, Redis, MinIO, one coordinator and two trainers
  (CPU, or CUDA if present).
- Reference job: a 10–30 M parameter GPT-2 on TinyStories or Shakespeare, DiLoCo with
  H=50, 2 % top-k. **Acceptance test:** final loss within 5 % of the same model trained
  on one GPU with the same token budget, with network traffic measured and reported.
- Everything in §3.1–3.4 implemented; scoring (§3.5) stubbed to "all honest".
- Integration test suite: 2 trainers join mid-run, 1 leaves, 1 submits garbage.

Trainer hardware for the dev loop: any machine; GPU optional.

### Phase 1: closed alpha, one shared run (target: first 10–50 trainers)

Launch the way Pluralis (Node0), Nous (Psyche) and Macrocosmos (Train-at-Home) all did:
**not with a marketplace but with one public reference model everyone trains together.**
That gives early trainers a shared goal, gives the team one run to debug, and produces
an artefact (open weights) that proves the network works.

- **Infrastructure:** coordinator on one VPS (e.g. Hetzner AX or a 2-vCPU cloud box; the
  coordinator does aggregation of ≤ 1 B params, which fits a 16 GB CPU box or a small GPU
  instance), Postgres managed or on the box, Cloudflare R2 for blobs (free egress matters:
  every trainer pulls θ_{r+1} every round). Estimated cost < 100 USD/month at this scale.
- **Reference run:** a 150 M Llama-style model on 3–5 B tokens of FineWeb-Edu, DiLoCo
  H=300, bf16, 2 % top-k 2-bit. On 20 consumer GPUs this takes about one to two weeks.
  Publish loss curve and leaderboard live.
- **Trainer onboarding:** sign up on the dashboard, install the CLI and the engine (§5.1), then `plasmon login --server <url> && plasmon trainer start`
  or `docker run plasmon/trainer`. Minimum: NVIDIA GPU with ≥ 8 GB VRAM (RTX 3060 /
  3070 / 4060 Ti and up), Linux or WSL2, 20 Mbit/s upload, driver ≥ 535. Invite codes via
  Discord/GitHub; 10–50 people.
- **Verification:** validators run by the project only (2–3 GPUs), full Gauntlet scoring
  live, scores public. Credits accrue in the hash-chained ledger; not yet withdrawable.
  The public leaderboard, the live job page and a "verified tokens trained" badge are
  the reward. The dashboard ships in this phase with network status, job explorer,
  leaderboard and the account pages; payments come in Phase 2.
- **Exit criterion:** the 150 M model reaches the loss of a single-GPU baseline within
  5–10 %; at least one cheating attempt (seeded by the team) is caught and down-weighted;
  no round lost to coordinator failure.

### Phase 2: open alpha, paid jobs (target: first 3–5 external requesters)

- **Job submission opens** for allow-listed architectures and public datasets; requesters
  pay in credits bought by card (Stripe) or USDC. Fine-tunes and LoRA jobs on 1–7 B
  checkpoints are the first paid product because they are small, fast, and the market
  already exists (compare Gradients on Bittensor).
- Trainers can **withdraw** credits (USDC on Base; fiat later). Protocol fee 10 %.
- **P2P blob seeding** (Iroh or Hivemind) so trainers serve θ_{r+1} to each other and the
  bucket bill does not scale with trainer count.
- Validator set opens to **staked community validators** (bond in USDC); scores
  stake-weighted-median; validator slashing for outlier scoring.
- Public **status page and job explorer** (loss per round, trainers, bytes, scores).
- Target: 100–500 trainers, a handful of paying jobs, and the economics measured honestly
  (credits paid per verified token vs. the equivalent cloud GPU hour).

### Phase 3: decentralized trust

- Settlement contracts on Base (or Solana): escrow per job, Merkle payouts per round,
  trainer and validator bonds, slashing.
- Coordinator replicated across validators; round results accepted on quorum signature.
- Custom model code in sandboxed containers; RL post-training jobs with TOPLOC-style
  rollout verification (the cheapest verifiable workload, see Prime Intellect in §10).

### Phase 4: scale

- Pipeline parallelism for models that do not fit one GPU (SWARM / IOTA / Pluralis style),
  which also changes the accounting (per-stage credit).
- Asynchronous rounds with staleness correction (HeLoCo / Decoupled DiLoCo).
- Governance of parameters and fees; token only if it is needed for something credits
  cannot do.

## 10. Prior art and competitors

Full research notes with sources are in [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md). Summary
as of October 2026:

| Project | Model of coordination | Training method | Verification | Rewards | Status / scale | Open source |
|---|---|---|---|---|---|---|
| **Templar / Covenant AI** (was Bittensor SN3) | Validator set + cloud buckets; miners get deterministic shards | DiLoCo + DeMo → SparseLoCo | Gauntlet: loss-delta on assigned vs random data, OpenSkill | TAO emissions ∝ score | Covenant-72B (72.7 B, 1.1 T tokens, 70+ nodes, Mar 2026). Covenant left Bittensor Apr 2026 | Yes, MIT, Python |
| **Nous Research Psyche** | Coordinator is a **Solana program**; P2P over Iroh | DisTrO/DeMo | Witnesses + statistical similarity of recomputed updates | None yet (contribution-driven) | Consilience 40 B, Hermes 4.3 36 B trained end-to-end on Psyche | Yes, Rust, Apache/MIT |
| **Prime Intellect** | Rust orchestrator + contracts on Base; now mostly a compute marketplace + RL stack | OpenDiLoCo (on Hivemind); prime-rl async RL; SHARDCAST | TOPLOC for inference rollouts | Testnet payouts | INTELLECT-1 10 B, INTELLECT-2 32 B (decentralized RL). INTELLECT-3 trained centrally. $130 M at $1 B (Jul 2026); protocol repo archived | Yes, Python/Rust |
| **Macrocosmos IOTA** (Bittensor SN9) | Central orchestrator; pipeline-parallel stages | SWARM-style PP, Butterfly all-reduce, activation compression | CLASP Shapley-style credit; recompute samples | TAO | Orion-100B PoC, Orion-16B live on ~180 GPUs; "Train at Home" Mac app | Yes, Python |
| **Pluralis Research** | Hivemind-based; model-parallel so no node holds full weights | Protocol Models (compressed activations) | Research | Reputation only | Node0-7.5 B (1,642 GPUs, 198 cities); Pluralis-8B 500 B tokens at 63 % of H100-cluster efficiency | Yes, Python |
| **Gensyn** | Ethereum L2 rollup; RL Swarm | Verde refereed delegation, RepOps | Deterministic re-execution with bisection | $AI token (Apr 2026) | Testnet since Mar 2025, RL Swarm paused Jan 2026; training mainnet not live; Open-1B is *auditable* centralized training | Partly |
| **Hivemind / Petals / SWARM** | Library: libp2p DHT, decentralized averaging | Any | None | None | Hivemind 1.1.12 (Jan 2026) maintained; Petals swarm mostly idle | Yes, MIT |
| **Flower / FedML** | Federated learning frameworks (data stays with owner) | FedAvg family; Photon for LLMs | Honest-but-curious | None | Flower dominant FL framework; FedML pivoted to TensorOpera cloud | Yes |
| **Akash, io.net, Render, Golem, Vast.ai, Salad** | GPU *rental* marketplaces | None (you bring your own orchestration) | n/a | Token or fiat per GPU-hour | Large (io.net ~370 k GPUs claimed) | Varies |
| **Together AI, Exo** | Together: started from decentralized-training research, now a centralized neocloud ($8.3 B). Exo: LAN clusters for inference | | | | | |

**What this tells plasmon**

1. The algorithmic stack is settled and open: DiLoCo + sparse/low-bit pseudo-gradients. Use
   the reference implementations, do not reinvent.
2. The architecture is settled: logically central coordinator, decentralized workers,
   settlement on an existing chain. Nobody who tried "fully P2P" or "own L1" shipped
   training.
3. Every live network trains **its own** model. A **permissionless job market** where third
   parties submit model + dataset exists only as fine-tuning competitions (Gradients). That
   is the open space and the differentiator, together with transparent, recomputable
   scoring (the exact thing the Covenant/Bittensor split was about).
4. Economics are unproven at the frontier: the best-funded teams train flagships on
   InfiniBand clusters. plasmon targets the long tail (fine-tunes, ≤ 10 B pre-training,
   RL rollouts), where volunteer compute is already competitive.
5. Verification is the hard, unsolved problem. Ship statistical verification and publish
   detection rates rather than promising cryptographic proofs.

## 11. Repository layout

One repository: a Rust workspace for the CLI and protocol core, a Python package for the trainer, the coordinator and the web app.

```
plasmon/
├── Cargo.toml                    # rust workspace
├── pyproject.toml                # python package `plasmon` (engine: core, trainer, coordinator)
├── crates/
│   ├── plasmon-core/             # identity (ed25519), blake3 content addressing, canonical JSON, Δ frame header
│   └── plasmon-cli/              # `plasmon` binary: clap commands, ratatui TUI, self-update
├── python/
│   ├── plasmon/
│   │   ├── core/                 # reference protocol: identity, canonical JSON, frames, job spec, assignment
│   │   ├── train/                # diloco inner/outer loop, compression, allow-listed models, data shards
│   │   ├── trainer/              # agent: enrol, heartbeat, pull, train, commit-reveal, upload
│   │   ├── validator/            # scoring: cheap checks, loss-delta scoring, re-execution
│   │   └── coordinator/          # fastapi: accounts, jobs, rounds, aggregation, ledger, SSE, web dashboard
│   └── tests/                    # unit tests, two-trainer integration test, cross-language vectors
├── deploy/                       # docker compose bundle, Dockerfile, Caddyfile, helm chart
├── install/                      # install.sh, install.ps1, release manifest
├── examples/                     # mnist job and eval script; local-network scripts (server, join, submit)
├── scripts/                      # start-up budget, vector generation
└── docs/
    ├── PROTOCOL.md               # wire formats: identity, canonical JSON, frames, commit-reveal, sealed shards
    ├── MARKETPLACE.md            # funded jobs, open jobs, approval, what private data gets and does not get
    ├── QUICKSTART.md             # one machine: run the server, connect, train
    ├── LOCAL-NETWORK.md          # one server, many participants on the same Wi-Fi
    ├── HOME-LAB.md               # two machines at home (Mac + Windows), MNIST end to end
    ├── SELF-HOSTING.md           # company deployment
    └── LANDSCAPE.md              # competitor and research notes with sources
```

## 12. Roadmap

- [x] **M0 Scaffold.** Cargo workspace and `pyproject`; `plasmon-core` identity and signed
      messages; blob client; compose stack; CI with the `hyperfine` start-up budget. CLI
      skeleton: `install.sh`, `--version` under 5 ms, start-up animation, `login` (device
      code), `whoami`, `--json`.
- [x] **M1 DiLoCo locally.** Inner/outer loop, SparseLoCo compression, binary Δ frames,
      two in-process trainers reach single-GPU loss on a 10–30 M model. Traffic measured.
- [x] **M2 Coordinator.** Job spec, round state machine, deterministic assignment,
      commit-reveal, aggregation, hash-chained ledger, join/leave mid-round. Accounts,
      roles and API tokens, device-code login. Trainer agent with heartbeats and
      telemetry. Python commands (`server`, `login`, `job`, `trainer`, `fleet`,
      `ledger`), the native CLI (`plasmon`, `job watch`, `fleet --watch`, intro
      animation), content-addressed dedupe on submit, the first dashboard pages
      (overview, jobs, job, my machine, fleet, server, ledger). Not yet: the warm
      connection daemon, `net` commands.
- [x] **M3 Fleet and self-hosting.** `plasmon server init --bundle compose` (API, worker,
      PostgreSQL, MinIO, Caddy TLS), S3 blob store, OIDC with group→role mapping, org
      policy (availability windows, battery rule) editable in Settings, `plasmon trainer
      enable` services for Linux, macOS and Windows, log shipping and `fleet logs -f`,
      pause/resume/drain, users, invites, audit log, Prometheus metrics, retention,
      `fleet show --watch`, `server status --watch`. Idle detection and CPU caps are not
      enforced yet. Not yet run on a real office fleet.
- [x] **M4 Verification.** Loss-delta scoring on assigned versus random shards, norm and
      finiteness checks, per-machine honesty with a floor that excludes repeat offenders,
      scores in the ledger and the job page. Seeded cheaters (random Δ, copied Δ, wrong
      shard, NaN) are rejected or lose honesty in tests. Not yet: OpenSkill ratings,
      sampled re-execution, validators on separate machines.
- [x] **M5 Dashboard v1 and alpha runbook.** Overview with loss sparklines and the
      leaderboard, Leaderboard page, Account page (password, command-line logins),
      job submission form, webhook notifications (JSON or Slack), `public` mode with open
      registration, `plasmon dashboard` tabbed TUI, `docs/ALPHA-RUN.md`. The alpha run
      itself (a public server, a reference model, volunteers) has not been executed.
- [x] **M6 Credits.** Internal credits: welcome grants, per-round settlement (job owner
      pays per 1,000 accepted samples; machine owners earn by score × samples; fee
      account), balances, admin grants, CSV export for chargeback, Credits page,
      `plasmon credits`, jobs stop when the owner runs out. Not done: Stripe or USDC
      on-ramps and payouts, plans, invoices, LoRA jobs, staked community validators, P2P
      seeding.
- [x] **M6a Marketplace.** Funded jobs: funding locked at submission, one hold per round
      split by verified contribution, release, fee and refund in a settlement when the job
      ends, weights released with the holds. Open jobs page and `plasmon job open`; `join`
      enrolment mode and owner approval with mail to the requester (hardware, measured
      TFLOPS, reputation, owner) and to the machine's owner; `min_tflops` and
      `min_honesty` requirements. Private data: sealed shards with a per-job key that
      travels only with an assignment, sticky shards, an exposure cap per machine, and an
      exposure report on the job page. Mail over SMTP or into an outbox directory.
      Columns added to an existing database on start. Not done: an on-ramp for credits,
      attested enclaves, removing an approved machine mid-round.
- [ ] **M6b Payments.** Card and USDC on-ramps, payouts, plans and invoices on top of the
      credit ledger. (Phase 2.)
- [ ] **M7 Contracts and quorum coordinator.** (Phase 3.)
- [ ] **M8 Pipeline parallelism, async rounds, custom code sandbox.** (Phase 4.)

## 13. Contributing and license

Work happens on `main` under the layout in §11, starting with M0. Issues are welcome,
especially from people with a consumer GPU who want to be alpha trainers, and from anyone
who has run DiLoCo, Hivemind, Psyche or Templar nodes.

License: to be decided before M4; the intended choice is **Apache-2.0** for the client and
coordinator (matching the ecosystem it builds on) with scoring code required to stay open.
