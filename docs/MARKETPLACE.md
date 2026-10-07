# Marketplace: funded jobs, open jobs, approval, and private data

This guide is for a plasmon server that other people use: a hosted coordinator where
*requesters* pay to have a model trained and *trainers* lend machines for credits. It
covers how the money moves, how a trainer gets a job, how a requester decides who trains
it, and what the server can and cannot promise about a private dataset.

Everything here also works on a company server. The settings live in `job.yaml`, the
dashboard and the `plasmon` command.

## What a requester writes

```yaml
name: reviews-classifier
model: {arch: mlp}
dataset: {source: ./reviews.npz, shard_size: 2000}
requirements:
  device: cuda
  min_vram_gb: 12
  min_tflops: 20          # matmul throughput the machine measured when its trainer started
  min_honesty: 0.8        # reputation: share of recent rounds whose update helped and was not a copy
  min_trainers: 2
  max_trainers: 16
enrolment:
  mode: join              # auto: any idle machine that fits takes rounds; join: trainers pick the job from Open jobs
  approval: owner         # none: a fitting machine trains; owner: you approve each machine first
privacy:
  encrypt_shards: true    # shards are sealed on your machine; the key travels only with an assignment
  sticky_shards: true     # a machine keeps the shards it already saw
  max_shards_per_machine: 4
budget:
  rounds: 50
  funding: 5000           # credits locked when you submit; released to trainers when the job ends
```

`plasmon job submit job.yaml` prepares the shards on your machine, seals them when
`encrypt_shards` is on, uploads them, and locks the funding. The dashboard form prepares
them on the server, so use the command for a dataset that must not leave your computer in
clear.

## How the money moves

Credits are the unit of account on the server (see the Credits page). A funded job moves
them in four steps, all recorded in the hash-chained ledger:

1. **Lock.** At submission the funding leaves your available balance (`escrow` entry).
   The job is rejected with `402` when the balance is short. The Credits page shows the
   amount under *locked in your running jobs*.
2. **Hold, per round.** Every round sets aside `funding / rounds` credits. The fee
   (`credits.fee_pct`, 10 % by default) is set aside for the fee account; the rest is split
   among the machines whose updates were accepted, in proportion to their verified
   contribution, `score × samples`. Nothing is paid yet: the trainer sees the amount
   under *on hold* on the Credits page and on the machine page.
3. **Release, when the job ends.** Holds become `earn` entries for the trainers and one
   `fee` entry for the fee account, in one settlement that the ledger records as
   `job_settled`. The requester and every paid trainer get a mail.
4. **Refund.** Funding that no closed round set aside returns to the requester as a
   `refund` entry. A job cancelled after 7 of 50 rounds pays 7 slices and refunds 43.

The fee is charged only when the job completed or the requester cancelled it. When the
server fails a job, trainers are still paid for the rounds that closed and the fee is
waived.

**The weights of a funded job are released at the same time as the holds.** While the job
runs, the job page says so instead of showing a download button, and
`plasmon job download` reports `403`. Operators of the server are not gated.

Why a machine with more or faster hardware earns more: contribution is measured per
accepted update, and a machine takes one update per round per trainer process. A farm
runs one trainer per GPU and takes that many shares per round; a machine that misses the
round deadline earns nothing for that round; an update that fails verification earns
nothing and lowers the machine's honesty.

A job can still pay per round instead: `budget.credits_per_1k_samples` charges the owner
each round and pays trainers at once, as before. A job uses one model or the other.

## How a trainer gets a job

Open the **Open jobs** page, or run `plasmon job open`:

```
id              name        owner           pays                needs                 enrolment        round  trainers  your machines
job_3f2a...     reviews     ana@acme.test   100/round of 5000   cuda, 12 GB VRAM, …   join + approval  4/50   3         ws-01: can join
```

`enrolment` says what happens next:

- **auto**: the coordinator assigns the job to any idle machine that fits. Nothing to do.
  A machine can opt out with `plasmon trainer leave <job>`.
- **join**: run `plasmon trainer join <job>` (or press *Join* on the page). The machine takes
  a round at its next heartbeat.
- **+ approval**: the join becomes a request. The requester gets a mail with the machine's
  hardware, measured TFLOPS, reputation and owner, and approves or rejects it. The trainer
  log says `waiting for the owner to approve this machine`, and the trainer's owner gets a
  mail with the decision.

`plasmon trainer join <job> --machine <name>` enrols another machine of the same account.
Standing per machine is visible on the page, in `plasmon job open` and on the machine
page: *can join*, *waiting for approval*, *joined*, *not accepted*, *does not fit* (with the
reason), or *assigned automatically*.

## How a requester approves

With `approval: owner`, the job page has a **Trainers** section. Each request shows the
machine, its owner, hardware, TFLOPS, honesty, rounds served and samples verified, with
*Approve* (optionally with a note the machine's owner receives) and *Reject*. The same from
a terminal:

```
plasmon job approvals <job>
plasmon job approve <job> <machine> [--note "..."]
plasmon job reject <job> <machine>
```

Approval and rejection take effect at the machine's next heartbeat. An approved machine can
be removed later; a rejected one cannot ask again.

## What the server can promise about a private dataset

A trainer computes gradients on the samples it receives, so **a machine that trains a
shard sees that shard in clear**. No encryption scheme that runs on consumer GPUs changes
that: homomorphic encryption is orders of magnitude too slow for training, and secure
enclaves (NVIDIA confidential computing, AMD SEV-SNP, Intel TDX) exist only on datacenter
hardware and are not supported yet. What plasmon can bound is *who* sees data, *how much*
of it, and *where* it sits in clear.

| Setting | What it does | Protects against |
|---|---|---|
| `privacy.encrypt_shards` | Shards are sealed on the requester's machine (AES-256-GCM, one random key per job). The blob store holds ciphertext. The coordinator keeps the key wrapped under its own key and hands it to a machine with an assignment. The trainer caches ciphertext on disk and keeps the clear shard in memory only. | Anyone with a blob id or bucket access who is not assigned the job: a leaked bucket, a storage provider, a machine that was never approved. |
| `enrolment.approval: owner` | Only machines the requester approved receive assignments, and with them the key. | Machines the requester does not trust. The decision is yours, with hardware, reputation and owner in front of you. |
| `privacy.sticky_shards` | A machine keeps training the shards it already saw instead of a new one each round. | Exposure growing with the number of rounds: a 50-round job exposes one shard per machine, not 50. |
| `privacy.max_shards_per_machine` | A machine is never given more distinct shards than this. When its shards are all taken in a round, it sits that round out. | A single machine collecting a large part of the dataset. |
| Data exposure report | The job page lists which machine received which shards, as a count and a share of the dataset. | Not knowing. The report is also what you need for a data-processing record. |

What this does **not** cover: an approved trainer who copies the shards it received,
and the coordinator itself, which holds the key because scoring and evaluation need the
samples. For data that must not be seen by any trainer, the remaining options are to train
on features from a frozen encoder computed on your side (an attacker gets vectors, not
records), or to wait for attested enclaves. Terms of service and a data-processing
agreement with trainers belong in the product, not in the protocol.

## Reputation and requirements

Every machine carries `honesty` (0 to 1, the share of recent rounds whose update helped and
was not a copy), `rounds_served` and `samples_verified`. At every trainer start the machine
measures matmul throughput for a fraction of a second and reports it as `tflops` (fp16 on
CUDA, fp32 on a CPU). A job can require any of them: `requirements.min_tflops`,
`requirements.min_honesty`, plus the existing `device` and `min_vram_gb`. A machine below
the requirements is not asked, cannot join, and the job's waiting line says what is missing.

## Mail

Mail is sent to the requester when a machine asks to join and when the job ends, and to
a machine's owner when it is approved, rejected, or paid. Configure it in
`plasmon-server.yaml`:

```yaml
email:
  smtp_host: smtp.example.com
  smtp_port: 587
  username: plasmon@example.com
  password: ...            # or PLASMON_SMTP_PASSWORD in the environment
  from_addr: plasmon@example.com
  starttls: true
```

For development, `email: {outbox_dir: /tmp/plasmon-mail}` writes each message as an
`.eml` file instead of sending it. Without either setting no mail is sent; the dashboard
and the trainer log still show every state.

## API

| Call | Who | What |
|---|---|---|
| `GET /v1/jobs/open` | user or machine token | running jobs with pay, requirements, enrolment and the standing of the caller's machines |
| `POST /v1/jobs/{id}/join` `{node_id}` | user (own machine) or machine | join, or ask to join |
| `POST /v1/jobs/{id}/leave` `{node_id}` | user or machine | leave, or opt out of an automatic job |
| `GET /v1/jobs/{id}/enrolments` | owner, operator | every machine that asked, with hardware and reputation |
| `POST /v1/jobs/{id}/enrolments/{machine}/approve` `{note}` | owner, operator | `machine` is a node id, its prefix, or the machine name |
| `POST /v1/jobs/{id}/enrolments/{machine}/reject` `{note}` | owner, operator | |
| `GET /v1/credits/me` | user | balance, `on_hold`, `locked`, recent holds |
| `POST /v1/jobs` | user | accepts `data_key` (hex) with `privacy.encrypt_shards` |

A job's JSON carries `funding`, `held`, `per_round`, `settlement`, `sealed`, `enrolment`,
`downloadable` and, for the owner, `exposure`. The heartbeat reply carries `enrolments`:
the jobs where the machine waits for a decision or was rejected.

## Not done

- Buying credits with a card and withdrawing them (Stripe, USDC). Credits are granted by
  an admin today; the escrow and the settlement already run on the credit ledger that an
  on-ramp would fund.
- Attested enclaves as a requirement a job can set.
- Kicking an approved machine out of the round it is in: a rejection takes effect at the
  next round.
