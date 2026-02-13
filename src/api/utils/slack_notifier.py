from __future__ import annotations

import os
from typing import Any

import requests


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class SlackNotifier:
    def __init__(
        self,
        enabled: bool,
        webhook_url: str,
        timeout_sec: float = 3.0,
        prefix: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.webhook_url = str(webhook_url or "").strip()
        self.timeout_sec = float(timeout_sec)
        self.prefix = str(prefix or "")

    def send(self, message: str) -> None:
        if not self.enabled or not self.webhook_url:
            return
        text = f"{self.prefix}{message}" if self.prefix else str(message)
        try:
            requests.post(
                self.webhook_url,
                json={"text": text},
                timeout=self.timeout_sec,
            )
        except Exception:
            return


def create_slack_notifier(
    slack_config: dict[str, Any] | None = None,
    prefix: str = "",
) -> SlackNotifier:
    conf = dict(slack_config or {})

    webhook_url = str(
        conf.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL", "")
    ).strip()

    enabled_default = bool(webhook_url)
    enabled = _to_bool(
        conf.get("enabled", os.getenv("SLACK_NOTIFICATIONS_ENABLED", None)),
        default=enabled_default,
    )

    timeout_sec = float(conf.get("timeout_sec", os.getenv("SLACK_TIMEOUT_SEC", 3.0)))

    return SlackNotifier(
        enabled=enabled,
        webhook_url=webhook_url,
        timeout_sec=timeout_sec,
        prefix=prefix,
    )
