"""Webhooks and mail. Each webhook gets a JSON POST; Slack-style hooks get {"text"}.
Mail goes over SMTP, or into an outbox directory as .eml files for development."""

from __future__ import annotations

import datetime as dt
import logging
import smtplib
import threading
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx

from .config import EmailConfig, WebhookConfig

log = logging.getLogger("plasmon.notify")


class Notifier:
    def __init__(self, hooks: list[WebhookConfig], public_url: str = "", email: EmailConfig | None = None):
        self.hooks = hooks
        self.public_url = public_url
        self.email_cfg = email or EmailConfig()
        self._outbox_seq = 0
        self._lock = threading.Lock()

    def send(self, event: str, text: str, data: dict[str, Any]) -> None:
        for hook in self.hooks:
            if hook.events and event not in hook.events:
                continue
            payload = {"text": text} if hook.format == "slack" else {"event": event, "text": text, "data": data, "server": self.public_url}
            threading.Thread(target=self._post, args=(hook.url, payload), daemon=True).start()

    @staticmethod
    def _post(url: str, payload: dict[str, Any]) -> None:
        try:
            httpx.post(url, json=payload, timeout=10)
        except Exception as e:  # a dead webhook must not affect the round
            log.warning("webhook %s failed: %s", url, e)

    @property
    def mail_enabled(self) -> bool:
        return self.email_cfg.enabled

    def email(self, to: str, subject: str, body: str) -> None:
        if not to or not self.email_cfg.enabled:
            return
        msg = EmailMessage()
        msg["From"] = self.email_cfg.from_addr
        msg["To"] = to
        msg["Subject"] = subject
        msg["Date"] = dt.datetime.now(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S +0000")
        msg.set_content(body)
        if self.email_cfg.outbox_dir:
            self._write_outbox(msg)
            return
        threading.Thread(target=self._smtp, args=(msg,), daemon=True).start()

    def _write_outbox(self, msg: EmailMessage) -> None:
        directory = Path(self.email_cfg.outbox_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._outbox_seq += 1
            name = f"{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%S')}-{self._outbox_seq:04d}.eml"
        (directory / name).write_bytes(bytes(msg))

    def _smtp(self, msg: EmailMessage) -> None:
        cfg = self.email_cfg
        try:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as smtp:
                if cfg.starttls:
                    smtp.starttls()
                if cfg.username:
                    smtp.login(cfg.username, cfg.password or "")
                smtp.send_message(msg)
        except Exception as e:  # mail is a courtesy; the round and the job do not depend on it
            log.warning("mail to %s failed: %s", msg["To"], e)
