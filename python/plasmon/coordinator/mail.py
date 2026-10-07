"""Subject and body of each mail the coordinator sends. Plain text, no markup."""

from __future__ import annotations

from typing import Any

from . import db


def describe_hardware(hw: dict[str, Any] | None) -> str:
    hw = hw or {}
    gpu = hw.get("gpu") or {}
    parts = [str(hw.get("os") or "unknown OS")]
    if gpu.get("kind") in (None, "none"):
        parts.append("no GPU")
    else:
        vram = f" {gpu['vram_gb']:g} GB" if gpu.get("vram_gb") else ""
        count = f" ×{gpu['count']}" if (gpu.get("count") or 1) > 1 else ""
        parts.append(f"{gpu.get('name') or gpu.get('kind')}{vram}{count}")
    if hw.get("cpu_count"):
        parts.append(f"{hw['cpu_count']} CPUs")
    if hw.get("ram_gb"):
        parts.append(f"{hw['ram_gb']:g} GB RAM")
    return " · ".join(parts)


def describe_throughput(hw: dict[str, Any] | None) -> str:
    tflops = (hw or {}).get("tflops")
    return f"{tflops:g} TFLOPS measured at start" if tflops else "not measured"


def describe_reputation(m: db.Machine) -> str:
    return f"honesty {m.honesty:.2f} · {m.rounds_served} rounds served · {m.samples_verified:,} samples verified"


def enrolment_requested(job: db.Job, m: db.Machine, machine_owner: str, base_url: str) -> tuple[str, str]:
    subject = f"plasmon: {m.name} asks to train {job.name}"
    body = f"""{m.name} asks to join your job {job.name} ({job.id}).

machine owner   {machine_owner}
hardware        {describe_hardware(m.hardware)}
throughput      {describe_throughput(m.hardware)}
reputation      {describe_reputation(m)}
node id         {m.node_id}

Approve or reject it on the job page:
{base_url}/jobs/{job.id}#trainers

From a terminal:
plasmon job approve {job.id} {m.name}
plasmon job reject {job.id} {m.name}
"""
    return subject, body


def enrolment_decided(job: db.Job, m: db.Machine, approved: bool, note: str, base_url: str) -> tuple[str, str]:
    verdict = "approved for" if approved else "not accepted on"
    subject = f"plasmon: {m.name} {verdict} {job.name}"
    next_step = "It receives a round at its next heartbeat." if approved else "It will not receive rounds of this job."
    body = f"""Your machine {m.name} was {verdict} the job {job.name} ({job.id}).
{next_step}
"""
    if note:
        body += f"\nNote from the owner: {note}\n"
    body += f"\n{base_url}/machine/{m.node_id}\n"
    return subject, body


def job_ended(job: db.Job, summary: dict[str, Any] | None, base_url: str) -> tuple[str, str]:
    subject = f"plasmon: {job.name} {job.status}"
    lines = [f"Your job {job.name} ({job.id}) {job.status} after {job.round_index} rounds."]
    if job.status_detail:
        lines.append(f"Reason: {job.status_detail}")
    if job.last_eval_loss is not None:
        lines.append(f"Eval loss {job.last_eval_loss:.4f}" + (f", accuracy {100 * job.last_eval_acc:.1f} %" if job.last_eval_acc is not None else ""))
    if summary:
        lines.append("")
        lines.append(f"funding       {summary['funding']:,} credits")
        lines.append(f"paid          {summary['paid']:,} to {len(summary['payouts'])} machines")
        lines.append(f"fee           {summary['fee']:,}")
        lines.append(f"refund        {summary['refund']:,} back to your balance")
    if job.status in ("completed", "cancelled"):
        lines.append("")
        lines.append(f"Download the weights: {base_url}/jobs/{job.id}")
        lines.append(f"From a terminal: plasmon job download {job.id}")
    return subject, "\n".join(lines) + "\n"


def payout_released(job: db.Job, payout: dict[str, Any], base_url: str) -> tuple[str, str]:
    subject = f"plasmon: {payout['amount']:,} credits released from {job.name}"
    body = f"""The job {job.name} ({job.id}) {job.status}. The credits your machine {payout['machine']} earned
in {payout['rounds']} rounds are now in your balance: {payout['amount']:,} credits.

{base_url}/credits
"""
    return subject, body
