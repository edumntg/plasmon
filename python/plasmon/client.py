"""HTTP client for the coordinator API. Used by the CLI, the trainer and the tests."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from .core import hashing


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class Client:
    def __init__(self, server: str, token: str | None = None, timeout: float = 60.0):
        self.server = server.rstrip("/")
        self.token = token
        self._http = httpx.Client(base_url=self.server, timeout=timeout)

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"accept": "application/json"}
        if self.token:
            h["authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    def request(self, method: str, path: str, **kwargs) -> Any:
        r = self._http.request(method, path, headers=self._headers(kwargs.pop("headers", None)), **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, str(detail))
        if r.status_code == 428:
            return {"status": "pending"}
        return r.json() if r.content else None

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: Any = None) -> Any:
        return self.request("POST", path, json=body)

    # auth
    def register(self, email: str, password: str, name: str = "") -> dict:
        return self.post("/v1/auth/register", {"email": email, "password": password, "name": name})

    def login(self, email: str, password: str, label: str = "cli") -> dict:
        out = self.post("/v1/auth/login", {"email": email, "password": password, "label": label})
        self.token = out["token"]
        return out

    def device_start(self, label: str = "cli") -> dict:
        return self.post("/v1/auth/device", {"label": label})

    def device_poll(self, device_code: str) -> dict:
        r = self._http.post("/v1/auth/device/token", json={"device_code": device_code}, headers=self._headers())
        if r.status_code == 428:
            return {"status": "pending"}
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.json().get("detail", r.text))
        out = r.json()
        self.token = out["token"]
        return out

    def me(self) -> dict:
        return self.get("/v1/auth/me")

    # blobs
    def blob_exists(self, blob_id: str) -> bool:
        r = self._http.head(f"/v1/blobs/{blob_id}", headers=self._headers())
        return r.status_code == 200

    def put_blob(self, data: bytes, kind: str = "") -> str:
        blob_id = hashing.digest(data)
        if self.blob_exists(blob_id):
            return blob_id
        r = self._http.put(f"/v1/blobs/{blob_id}", content=data, headers=self._headers({"content-type": "application/octet-stream", "x-plasmon-kind": kind}))
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text)
        return blob_id

    def get_blob(self, blob_id: str) -> bytes:
        r = self._http.get(f"/v1/blobs/{blob_id}", headers=self._headers())
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text)
        if hashing.digest(r.content) != blob_id:
            raise ApiError(502, "blob digest mismatch")
        return r.content

    def missing_blobs(self, ids: list[str]) -> list[str]:
        return self.post("/v1/blobs/check", {"ids": ids})["missing"]

    # jobs
    def create_job(self, body: dict) -> dict:
        return self.post("/v1/jobs", body)

    def jobs(self, all: bool = False) -> list[dict]:
        return self.get("/v1/jobs", all="true" if all else None)

    def job(self, job_id: str) -> dict:
        return self.get(f"/v1/jobs/{job_id}")

    def job_updates(self, job_id: str, round: int | None = None) -> list[dict]:
        return self.get(f"/v1/jobs/{job_id}/updates", round=round)

    def cancel_job(self, job_id: str) -> dict:
        return self.post(f"/v1/jobs/{job_id}/cancel")

    # open jobs and enrolment
    def open_jobs(self) -> list[dict]:
        return self.get("/v1/jobs/open")

    def join_job(self, job_id: str, node_id: str | None = None) -> dict:
        return self.post(f"/v1/jobs/{job_id}/join", {"node_id": node_id})

    def leave_job(self, job_id: str, node_id: str | None = None) -> dict:
        return self.post(f"/v1/jobs/{job_id}/leave", {"node_id": node_id})

    def job_enrolments(self, job_id: str) -> list[dict]:
        return self.get(f"/v1/jobs/{job_id}/enrolments")

    def approve(self, job_id: str, machine: str, note: str = "") -> dict:
        return self.post(f"/v1/jobs/{job_id}/enrolments/{machine}/approve", {"note": note})

    def reject(self, job_id: str, machine: str, note: str = "") -> dict:
        return self.post(f"/v1/jobs/{job_id}/enrolments/{machine}/reject", {"note": note})

    # machines
    def register_machine(self, body: dict) -> dict:
        return self.post("/v1/machines/register", body)

    def heartbeat(self, body: dict) -> dict:
        return self.post("/v1/machines/heartbeat", body)

    def commit(self, job_id: str, round_index: int, blob: str, signature: str) -> dict:
        return self.post(f"/v1/machines/rounds/{job_id}/{round_index}/commit", {"blob": blob, "signature": signature})

    def reveal(self, job_id: str, round_index: int, blob: str) -> dict:
        return self.post(f"/v1/machines/rounds/{job_id}/{round_index}/reveal", {"blob": blob})

    # fleet, server, ledger
    def fleet(self, status: str | None = None) -> list[dict]:
        return self.get("/v1/fleet", status=status)

    def fleet_summary(self) -> dict:
        return self.get("/v1/fleet/summary")

    def machine(self, node_id: str) -> dict:
        return self.get(f"/v1/fleet/{node_id}")

    def server_status(self) -> dict:
        return self.get("/v1/server/status")

    def ledger(self, job: str | None = None, limit: int = 100) -> list[dict]:
        return self.get("/v1/ledger", job=job, limit=limit)

    def ledger_verify(self) -> dict:
        return self.get("/v1/ledger/verify")

    def healthz(self) -> dict:
        return self.get("/v1/healthz")

    def events(self, topics: list[str]) -> Iterator[tuple[str, dict]]:
        """Yield (topic, data) from the SSE stream until the connection drops."""
        with self._http.stream("GET", "/v1/events", params={"topics": ",".join(topics)}, headers=self._headers({"accept": "text/event-stream"}), timeout=None) as r:
            event, data = None, []
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "":
                    if event and data:
                        yield event, json.loads("\n".join(data))
                    event, data = None, []

    def wait_job(self, job_id: str, timeout_s: float, poll_s: float = 1.0) -> dict:
        deadline = time.time() + timeout_s
        while True:
            job = self.job(job_id)
            if job["status"] != "running" or time.time() > deadline:
                return job
            time.sleep(poll_s)

    def close(self) -> None:
        self._http.close()
