"""The trainer loop.

    heartbeat → assignment → fetch θ and shard → inner round → compress →
    commit (signed) → upload → reveal → heartbeat ...

Control commands arrive in heartbeat replies: pause, drain. The agent never opens an
inbound port.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import torch

from ..client import ApiError, Client
from ..coordinator.config import OrgPolicy, Window
from ..coordinator.policy import available as policy_available
from ..core import frame as fr
from ..core import sealed
from ..core.identity import Identity
from ..core.jobspec import JobSpec
from ..credentials import MachineCredentials, load_machine, machine_path, save_machine
from ..paths import cache_dir
from ..train import compression, data, diloco, weights
from . import telemetry

log = logging.getLogger("plasmon.trainer")


class LogBuffer(logging.Handler):
    """Keeps recent log lines for the next heartbeat."""

    def __init__(self):
        super().__init__()
        self.lines: deque[dict[str, Any]] = deque(maxlen=500)

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append({"at": dt.datetime.now(dt.UTC).isoformat(), "level": record.levelname.lower(), "message": record.getMessage()[:2000]})

    def drain(self) -> list[dict[str, Any]]:
        out = list(self.lines)
        self.lines.clear()
        return out


class Agent:
    def __init__(
        self,
        server: str,
        user_token: str | None,
        identity: Identity,
        name: str,
        device: str = "any",
        max_hours: float | None = None,
        local_windows: list[Window] | None = None,
        never_on_battery: bool = False,
    ):
        self.server = server
        self.local_windows = local_windows or []
        self.never_on_battery = never_on_battery
        self.policy = OrgPolicy()
        self.identity = identity
        self.name = name
        self.device_pref = device
        self.device = diloco.pick_device(device)
        self.deadline = time.time() + max_hours * 3600 if max_hours else None
        self.user_client = Client(server, user_token) if user_token else None
        self.client: Client | None = None
        self.creds: MachineCredentials | None = None
        self.state = "idle"
        self.detail = ""
        # (job id, since, round timeout) while the machine waits for the next round of a running
        # job: it reports `training` and asks for work, instead of flipping to idle after each round
        self.in_job: tuple[str, float, int] | None = None
        self.in_round = False
        self.hardware_sent = False  # the first heartbeat of a run carries the current hardware
        self.job_id: str | None = None
        self.round: int | None = None
        self.step = 0
        self.steps_total = 0
        self.paused = False
        self.draining = False
        self.stop_event = threading.Event()
        self.compressors: dict[str, compression.Compressor] = {}
        self.buffer = LogBuffer()
        log.addHandler(self.buffer)
        if log.getEffectiveLevel() > logging.INFO:  # the buffer needs info lines even when the app did not configure logging
            log.setLevel(logging.INFO)
        self.blob_cache = cache_dir() / "blobs"
        self.blob_cache.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval = 3  # the server's value replaces this after the first reply
        self.idle_interval = 3
        self.session_rounds = 0
        self.session_samples = 0
        self.standing: dict[str, tuple[str, str]] = {}  # job id -> (pending|rejected, job name), as the last heartbeat reported it
        self.machine_creds_path: Path = machine_path()
        self._lock = threading.Lock()
        self._hb_lock = threading.Lock()  # one heartbeat on the wire at a time

    # ----- enrolment ---------------------------------------------------------------

    def enrol(self) -> None:
        existing = load_machine(self.machine_creds_path)
        if existing and existing.server == self.server and existing.node_id == self.identity.node_id:
            self.creds = existing
            self.client = Client(self.server, existing.token)
            try:
                self.client.me()
                log.info("machine %s already enrolled", self.identity.node_id[:8])
                return
            except ApiError:
                log.info("machine token rejected, registering again")
        if self.user_client is None:
            raise RuntimeError("not logged in on this machine; run `plasmon login` first")
        me = self.user_client.me()
        user_id = me["user"]["id"]
        body = {
            "node_id": self.identity.node_id,
            "name": self.name,
            "hardware": telemetry.hardware(),
            "versions": telemetry.versions(),
            "signature": self.identity.sign({"node_id": self.identity.node_id, "user": user_id}),
        }
        out = self.user_client.register_machine(body)
        self.creds = MachineCredentials(self.server, self.identity.node_id, out["machine_token"], out["machine"]["id"])
        save_machine(self.creds, self.machine_creds_path)
        self.client = Client(self.server, self.creds.token)
        log.info("enrolled machine %s as %s", self.identity.node_id[:8], self.name)

    # ----- heartbeat ---------------------------------------------------------------

    def heartbeat(self, ready: bool = False) -> dict[str, Any]:
        """`ready` asks the server for a round; only the main loop sets it, so a background
        heartbeat can never take an assignment the loop would not see."""
        with self._hb_lock:
            return self._heartbeat(ready)

    def _heartbeat(self, ready: bool = False) -> dict[str, Any]:
        assert self.client is not None
        metrics = telemetry.metrics()
        metrics.update({"step": self.step, "steps_total": self.steps_total, "session_rounds": self.session_rounds, "session_samples": self.session_samples})
        with self._lock:
            body = {"status": self.state, "status_detail": self.detail, "job_id": self.job_id, "round": self.round, "metrics": metrics, "logs": self.buffer.drain(), "ready": ready}
            if not self.hardware_sent:  # a GPU or a driver installed after enrolment shows up without re-enrolling
                body["hardware"] = telemetry.hardware()
        reply = self.client.heartbeat(body)
        self.hardware_sent = True
        job = reply.get("job")
        if isinstance(job, dict) and job.get("status") not in (None, "running") and self.in_job and self.in_job[0] == body["job_id"]:
            self.in_job = None  # the job ended: the loop goes idle at its next turn
        self.heartbeat_interval = reply.get("interval", 10)
        self.idle_interval = reply.get("idle_interval", 3)
        if reply.get("policy"):
            try:
                self.policy = OrgPolicy.model_validate(reply["policy"])
            except ValueError:
                pass
        self._log_standing(reply.get("enrolments") or [])
        for cmd in reply.get("commands", []):
            if cmd["type"] == "pause" and not self.paused:
                log.info("paused: %s", cmd.get("reason", ""))
                self.paused = True
            if cmd["type"] == "drain":
                self.draining = True
        if not any(c["type"] == "pause" for c in reply.get("commands", [])) and self.paused:
            log.info("resumed")
            self.paused = False
        return reply

    def _log_standing(self, rows: list[dict[str, Any]]) -> None:
        """One line when a job starts or stops waiting on the owner's decision about this machine."""
        current = {r["job"]: (r["status"], r.get("name") or r["job"]) for r in rows}
        for job_id, (status, name) in current.items():
            if self.standing.get(job_id, ("", ""))[0] == status:
                continue
            if status == "pending":
                log.info("job %s: waiting for the owner to approve this machine", name)
            elif status == "rejected":
                log.info("job %s: the owner did not accept this machine; it takes no rounds of this job", name)
        for job_id in set(self.standing) - set(current):
            if self.standing[job_id][0] == "pending":
                log.info("job %s: approved; this machine may take its rounds now", self.standing[job_id][1])
        self.standing = current

    def _heartbeat_thread(self) -> None:
        while not self.stop_event.is_set():
            if self.in_round:  # between rounds the main loop beats
                try:
                    self.heartbeat()
                except ApiError as e:
                    log.warning("heartbeat failed: %s", e)
                except Exception as e:  # network blips
                    log.warning("heartbeat error: %s", e)
            self.stop_event.wait(self.heartbeat_interval)

    # ----- blobs -------------------------------------------------------------------

    def fetch_blob(self, blob_id: str) -> bytes:
        assert self.client is not None
        path = self.blob_cache / blob_id
        if path.exists():
            return path.read_bytes()
        content = self.client.get_blob(blob_id)
        # Two agents on one computer share this cache: each writer needs its own temp file.
        tmp = path.with_name(f"{blob_id}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
        return content

    # ----- one round ---------------------------------------------------------------

    def run_assignment(self, a: dict[str, Any]) -> None:
        assert self.client is not None
        spec = JobSpec.model_validate(a["spec"])
        job_id, rnd = a["job_id"], a["round"]
        with self._lock:
            self.state, self.job_id, self.round, self.step, self.steps_total = "training", job_id, rnd, 0, spec.recipe.inner_steps
            self.detail = f"shard {a['shard']['index']}"
            self.in_round = True
            self.in_job = (job_id, time.time(), spec.requirements.round_timeout_s)
        log.info("round %s of %s: shard %s (%s samples)", rnd, spec.name, a["shard"]["index"], a["shard"]["n"])
        try:  # tell the server now; a short round can end before the next scheduled heartbeat
            self.heartbeat()
        except Exception as e:  # the round still runs; the next heartbeat retries
            log.debug("heartbeat at round start failed: %s", e)
        t0 = time.perf_counter()
        theta = weights.from_bytes(self.fetch_blob(a["theta"]))
        shard_bytes = self.fetch_blob(a["shard"]["blob"])
        if a.get("data_key"):  # the cache keeps the sealed bytes; the clear shard lives in memory only
            shard_bytes = sealed.unseal(bytes.fromhex(a["data_key"]), shard_bytes)
        shard = data.Shard.from_bytes(shard_bytes)
        x, y = data.to_tensors(shard)
        t_fetch = time.perf_counter()

        result = self.train_round(spec, theta, x, y, job_id, rnd, a)
        t_train = time.perf_counter()
        entries = self.make_entries(spec, job_id, result)
        frame = fr.Frame(job_id, rnd, self.identity.node_id, a["theta"], result.samples, entries, {"loss_start": result.loss_start, "loss_end": result.loss_end, "steps": result.steps, "device": str(self.device)})
        encoded = fr.encode(frame)
        blob_id = fr_digest(encoded)
        signature = self.identity.sign({"job": job_id, "round": rnd, "node": self.identity.node_id, "blob": blob_id})
        self.client.commit(job_id, rnd, blob_id, signature)
        self.client.put_blob(encoded, kind="delta")
        self.client.reveal(job_id, rnd, blob_id)
        self.after_reveal(job_id, rnd, result)
        t_done = time.perf_counter()
        self.session_rounds += 1
        with self._lock:
            self.in_round = False
            self.in_job = (job_id, time.time(), spec.requirements.round_timeout_s)
            self.detail = f"round {rnd} done"
        self.session_samples += result.samples
        log.info(
            "round %s done: loss %.3f→%.3f, %s bytes up, fetch %.1fs train %.1fs upload %.1fs",
            rnd, result.loss_start, result.loss_end, f"{len(encoded):,}", t_fetch - t0, t_train - t_fetch, t_done - t_train,
        )

    def availability(self) -> tuple[bool, str]:
        """Org policy first, then local tightening. Returns (available, reason)."""
        ok, reason = policy_available(self.policy)
        if not ok:
            return False, reason
        if self.local_windows:
            ok, reason = policy_available(OrgPolicy(windows=self.local_windows))
            if not ok:
                return False, reason.replace("outside window", "outside your window")
        if self.policy.pause_on_battery or self.never_on_battery:
            try:
                batt = telemetry.psutil.sensors_battery()
            except Exception:
                batt = None
            if batt is not None and not batt.power_plugged:
                return False, "on battery"
        return True, ""

    def train_round(self, spec: JobSpec, theta, x, y, job_id: str, rnd: int, assignment: dict[str, Any]) -> diloco.RoundResult:
        """The honest inner round. Tests subclass the agent and replace this step."""

        def on_step(i: int) -> None:
            self.step = i + 1

        return diloco.inner_round(spec.model.arch, spec.model.config, theta, x, y, spec.recipe, seed=(hash((job_id, rnd, self.identity.node_id)) & 0xFFFFFFFF), device=self.device, on_step=on_step)

    def after_reveal(self, job_id: str, rnd: int, result: diloco.RoundResult) -> None:
        """Called once the update is public. Tests use it to model a copier."""

    def make_entries(self, spec: JobSpec, job_id: str, result: diloco.RoundResult) -> list[fr.TensorEntry]:
        comp = self.compressors.setdefault(job_id, compression.Compressor(spec.recipe.compression.topk if spec.recipe.compression.name == "topk" else 1.0, spec.recipe.compression.error_feedback))
        return comp.compress(result.delta)

    # ----- main loop ---------------------------------------------------------------

    def run(self) -> None:
        while True:
            try:
                self.enrol()
                break
            except ApiError:
                raise  # a login or permission problem does not fix itself
            except Exception as e:  # the server is down or the network blocks this process
                hint = ""
                if "10013" in str(e):
                    hint = " (Windows refused the connection for this python.exe: allow it in Windows Defender Firewall or your antivirus)"
                log.warning("cannot reach %s: %s%s; retrying in 10 s", self.server, _short(e), hint)
                if self.stop_event.wait(10):
                    return
        hb = threading.Thread(target=self._heartbeat_thread, name="plasmon-heartbeat", daemon=True)
        hb.start()
        log.info("trainer %s on %s, device %s", self.name, self.server, self.device)
        _warn_if_cuda_missing()
        try:
            while not self.stop_event.is_set():
                if self.deadline and time.time() > self.deadline:
                    log.info("max hours reached, stopping")
                    break
                if self.draining:
                    log.info("drained, stopping")
                    break
                avail, why = self.availability()
                with self._lock:
                    if self.paused:
                        self.state, self.detail, self.job_id, self.round = "paused", "paused by admin", None, None
                    elif not avail:
                        self.state, self.detail, self.job_id, self.round = "unavailable", why, None, None
                    elif self.in_job and time.time() - self.in_job[1] < self.in_job[2] + 30:
                        self.state = "training"  # between rounds of a job that still runs
                    else:
                        self.in_job = None
                        self.state, self.detail, self.job_id, self.round = "idle", "", None, None
                try:
                    reply = self.heartbeat(ready=True)
                except ApiError as e:
                    if e.status == 401:
                        log.warning("machine token rejected; re-enrolling")
                        self.enrol()
                        continue
                    log.warning("heartbeat failed: %s", e)
                    self.stop_event.wait(self.idle_interval)
                    continue
                except Exception as e:
                    log.warning("server unreachable: %s", e)
                    self.stop_event.wait(5)
                    continue
                assignment = reply.get("assignment")
                if assignment and not self.paused and avail:
                    try:
                        self.run_assignment(assignment)
                    except ApiError as e:
                        log.warning("round rejected by server: %s", e)
                    except Exception:
                        log.exception("round failed")
                        with self._lock:
                            self.in_round = False
                            self.state, self.detail = "error", "round failed, see log"
                        self.stop_event.wait(self.idle_interval)
                else:
                    self.stop_event.wait(self.idle_interval)
        finally:
            self.stop_event.set()
            with self._lock:
                self.state = "idle"

    def stop(self) -> None:
        self.stop_event.set()


def _short(e: BaseException) -> str:
    """The message without the httpx chain: one line a person can act on."""
    text = str(e).strip() or e.__class__.__name__
    return text.splitlines()[0][:200]


def _warn_if_cuda_missing() -> None:
    """An NVIDIA driver is installed but this torch build has no CUDA: the usual case on Windows,
    where the default wheel is CPU only."""
    try:
        import torch

        if torch.cuda.is_available() or shutil.which("nvidia-smi") is None:
            return
    except Exception:
        return
    log.warning(
        "an NVIDIA GPU is present but this PyTorch has no CUDA, so the trainer uses the CPU. "
        "Install the CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu126"
    )


def fr_digest(data: bytes) -> str:
    from ..core import hashing

    return hashing.digest(data)


def default_name() -> str:
    import platform

    return platform.node() or "machine"


def machine_key_or_create(path: Path) -> Identity:
    if path.exists():
        return Identity.load(path)
    ident = Identity.generate()
    ident.save(path)
    return ident


__all__ = ["Agent", "default_name", "machine_key_or_create", "torch"]
