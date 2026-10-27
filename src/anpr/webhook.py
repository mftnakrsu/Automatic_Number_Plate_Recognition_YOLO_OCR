"""Webhook calls for watchlist matches.

Each match is POSTed as JSON to `ANPR_WEBHOOK_URL` from a background task, so
a slow receiver never holds up the video. The body carries the plate's HMAC
and the watchlist label, never the plate text.

With `ANPR_WEBHOOK_SECRET` set, every request is signed the way Stripe and
GitHub sign theirs:

    X-ANPR-Timestamp: <unix seconds>
    X-ANPR-Signature: sha256=<hex HMAC-SHA256 of "<timestamp>.<body>">

The receiver recomputes the HMAC over the raw body and rejects old timestamps
to stop replays. Failed calls are retried with exponential backoff; 4xx
answers other than 429 are not retried. The same plate is announced at most
once per cooldown, since one car often comes back as a new track a few
seconds later.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from anpr import __version__
from anpr.config import Settings
from anpr.logging import get_logger
from anpr.observability import WEBHOOK_DELIVERIES
from anpr.storage.models import WatchlistEntry

_log = get_logger(__name__)

Sender = Callable[[str, bytes, dict[str, str], float], int]


def post_json(url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
    """POST `body` and return the HTTP status. Network errors raise `OSError`."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def sign(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def match_event(
    entry: WatchlistEntry,
    *,
    source: str,
    track_id: int,
    frame_idx: int,
    province_code: str | None,
    confidence: float,
) -> dict[str, Any]:
    return {
        "event": "watchlist.match",
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "plate_hmac": entry.plate_hmac,
        "label": entry.label,
        "province_code": province_code,
        "track_id": track_id,
        "frame_idx": frame_idx,
        "confidence": confidence,
    }


class WebhookNotifier:
    def __init__(
        self,
        url: str,
        secret: str = "",
        *,
        timeout: float = 5.0,
        cooldown_seconds: float = 60.0,
        attempts: int = 3,
        backoff_seconds: float = 1.0,
        sender: Sender | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self._url = url
        self._secret = secret
        self._timeout = timeout
        self._cooldown = cooldown_seconds
        self._attempts = attempts
        self._backoff = backoff_seconds
        self._sender = sender or post_json
        self._clock = clock
        self._last_sent: dict[str, float] = {}
        self._tasks: set[asyncio.Task[bool]] = set()

    def notify(self, event: dict[str, Any]) -> bool:
        """Queue a delivery. Returns False when the plate is still in its cooldown."""
        key = event["plate_hmac"]
        now = self._clock()
        last = self._last_sent.get(key)
        if last is not None and now - last < self._cooldown:
            return False
        self._last_sent[key] = now
        body = json.dumps(event, ensure_ascii=False).encode("utf-8")
        task = asyncio.create_task(self._deliver(body))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def _headers(self, body: bytes) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"anpr-pipeline/{__version__}",
        }
        if self._secret:
            timestamp = str(int(time.time()))
            headers["X-ANPR-Timestamp"] = timestamp
            headers["X-ANPR-Signature"] = sign(self._secret, timestamp, body)
        return headers

    async def _deliver(self, body: bytes) -> bool:
        loop = asyncio.get_running_loop()
        status: int | None = None
        tried = 0
        for attempt in range(1, self._attempts + 1):
            tried = attempt
            try:
                status = await loop.run_in_executor(
                    None, self._sender, self._url, body, self._headers(body), self._timeout
                )
            except (OSError, ValueError) as e:
                status = None
                _log.warning("webhook.error", attempt=attempt, error=type(e).__name__)
            if status is not None and 200 <= status < 300:
                WEBHOOK_DELIVERIES.labels(result="ok").inc()
                return True
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt < self._attempts:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
        WEBHOOK_DELIVERIES.labels(result="failed").inc()
        _log.warning("webhook.failed", status=status, attempts=tried)
        return False

    async def aclose(self, timeout: float = 10.0) -> None:
        """Wait up to `timeout` seconds for queued deliveries, then cancel the rest."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()


def notifier_from_settings(settings: Settings) -> WebhookNotifier | None:
    """The notifier for `ANPR_WEBHOOK_URL`, or None when it is unset."""
    url = settings.webhook_url.get_secret_value()
    if not url:
        return None
    return WebhookNotifier(
        url,
        settings.webhook_secret.get_secret_value(),
        timeout=settings.webhook_timeout_seconds,
        cooldown_seconds=settings.webhook_cooldown_seconds,
    )
