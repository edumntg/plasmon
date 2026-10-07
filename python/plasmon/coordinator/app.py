"""Application factory. One process serves the API, the dashboard and the scheduler."""

from __future__ import annotations

import datetime as dt
import logging
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..core import identity
from . import (
    api_auth,
    api_credits,
    api_fleet,
    api_jobs,
    api_machines,
    api_misc,
    db,
    notify,
    oidc,
    web,
)
from .blobs import make_store
from .config import ServerConfig
from .engine import Engine, Scheduler
from .events import Bus
from .sessions import CookieSessions

log = logging.getLogger("plasmon.server")
WEB = Path(__file__).parent / "web"


class State:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        data_dir = cfg.resolved_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        key_path = cfg.server_key_path()
        if key_path.exists():
            self.server = identity.Identity.load(key_path)
        else:
            self.server = identity.Identity.generate()
            self.server.save(key_path)
        self.engine_db = db.make_engine(cfg.resolved_db_url())
        self.session_factory = db.make_session_factory(self.engine_db)
        self.blobs = make_store(cfg)
        self.bus = Bus()
        self.notifier = notify.Notifier(cfg.webhooks, cfg.public_url or "", cfg.email)
        self.engine = Engine(self.blobs, self.bus, self.server, cfg.policy.heartbeat_interval_s, cfg.retention, cfg.scoring, self.notifier, cfg.credits, idle_poll_interval_s=cfg.policy.idle_poll_interval_s, public_url=self.public_url())
        self.oidc = oidc.Provider(cfg.oidc) if cfg.oidc.enabled else None
        self.sessions = CookieSessions(cfg.auth.session_secret)
        self.scheduler: Scheduler | None = None
        self.started_ts = time.time()
        self.started_at = dt.datetime.now(dt.UTC).isoformat()
        self.templates = Jinja2Templates(directory=str(WEB / "templates"))
        web.install_filters(self.templates)

    def public_url(self, request: Request | None = None) -> str:
        if self.cfg.public_url:
            return self.cfg.public_url.rstrip("/")
        if request is not None:
            return str(request.base_url).rstrip("/")
        return f"http://{lan_ip()}:{self.cfg.port}"


# Interfaces that are not the local network: loopback, VPN tunnels, VM and container
# adapters, Apple's peer-to-peer links. A corporate VPN often routes 10.0.0.0/8, which
# made the old "route to 10.255.255.255" trick return the tunnel address.
_SKIP_PREFIXES = ("lo", "utun", "tun", "tap", "awdl", "llw", "bridge", "vmnet", "vnic", "docker", "veth", "br-", "gif", "stf", "anpi", "ap", "ppp", "ipsec", "zt", "ts", "wg", "tailscale", "virbr", "vboxnet", "cni", "flannel")
_PREFER_PREFIXES = ("en0", "en1", "eth", "wlan", "wlp", "enp", "eno", "wi-fi", "ethernet", "en")


def lan_candidates() -> list[tuple[str, str]]:
    """(interface, IPv4) pairs that can be the Wi-Fi or wired address, best first."""
    try:
        import psutil

        addrs = psutil.net_if_addrs()
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for name, entries in addrs.items():
        lname = name.lower()
        if lname.startswith(_SKIP_PREFIXES):
            continue
        for e in entries:
            if e.family != socket.AF_INET:
                continue
            ip = e.address
            if ip.startswith(("127.", "169.254.", "0.")):
                continue
            out.append((name, ip))

    def rank(item: tuple[str, str]) -> tuple[int, int, str]:
        name, ip = item
        lname = name.lower()
        pref = next((i for i, p in enumerate(_PREFER_PREFIXES) if lname.startswith(p)), len(_PREFER_PREFIXES))
        private = 0 if ip.startswith("192.168.") else 1 if ip.startswith("10.") else 2 if _is_172_private(ip) else 3
        return (pref, private, name)

    return sorted(out, key=rank)


def _is_172_private(ip: str) -> bool:
    parts = ip.split(".")
    return parts[0] == "172" and 16 <= int(parts[1]) <= 31


def lan_ip() -> str:
    """Best guess of the address other computers on the network can reach."""
    candidates = lan_candidates()
    if candidates:
        return candidates[0][1]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def create_app(cfg: ServerConfig) -> FastAPI:
    state = State(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.run_worker:
            state.scheduler = Scheduler(state.engine, state.session_factory, cfg.tick_interval_s)
            state.scheduler.start()
        log.info("plasmon %s listening on %s:%s (%s mode)", __version__, cfg.host, cfg.port, cfg.mode)
        yield
        if state.scheduler:
            state.scheduler.stop()

    app = FastAPI(title="plasmon coordinator", version=__version__, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.plasmon = state
    app.include_router(api_auth.router)
    app.include_router(api_machines.router)
    app.include_router(api_jobs.router)
    app.include_router(api_misc.router)
    app.include_router(api_fleet.router)
    app.include_router(api_credits.router)
    app.include_router(oidc.router)
    app.include_router(web.router)
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "internal error, see server log"}, status_code=500)

    return app
