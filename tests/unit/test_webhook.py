from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest

from anpr.storage.models import WatchlistEntry
from anpr.webhook import WebhookNotifier, match_event, sign


class _Receiver:
    def __init__(self, *statuses: int | Exception) -> None:
        self.statuses = list(statuses)
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
        self.calls.append((url, body, headers))
        status = self.statuses.pop(0) if self.statuses else 200
        if isinstance(status, Exception):
            raise status
        return status


def _event(plate_hmac: str = "a" * 64) -> dict[str, Any]:
    return match_event(
        WatchlistEntry(plate_hmac=plate_hmac, label="van"),
        source="watch",
        track_id=3,
        frame_idx=42,
        province_code="34",
        confidence=0.9,
    )


def _notifier(receiver: _Receiver, **kwargs: Any) -> WebhookNotifier:
    return WebhookNotifier(
        "http://hooks.test/anpr",
        sender=receiver,
        backoff_seconds=0,
        **kwargs,
    )


async def test_delivers_json_without_the_plate_text() -> None:
    receiver = _Receiver()
    notifier = _notifier(receiver)
    assert notifier.notify(_event())
    await notifier.aclose()

    assert len(receiver.calls) == 1
    url, body, headers = receiver.calls[0]
    assert url == "http://hooks.test/anpr"
    payload = json.loads(body)
    assert payload["event"] == "watchlist.match"
    assert payload["label"] == "van"
    assert payload["plate_hmac"] == "a" * 64
    assert "plate" not in payload
    assert headers["Content-Type"] == "application/json"
    assert "X-ANPR-Signature" not in headers


async def test_signature_covers_timestamp_and_body() -> None:
    receiver = _Receiver()
    notifier = _notifier(receiver, secret="s3cret")
    notifier.notify(_event())
    await notifier.aclose()

    _, body, headers = receiver.calls[0]
    timestamp = headers["X-ANPR-Timestamp"]
    expected = hmac.new(b"s3cret", f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    assert headers["X-ANPR-Signature"] == f"sha256={expected}"
    assert sign("s3cret", timestamp, body) == headers["X-ANPR-Signature"]


async def test_cooldown_per_plate() -> None:
    now = [100.0]
    receiver = _Receiver()
    notifier = _notifier(receiver, cooldown_seconds=60, clock=lambda: now[0])
    assert notifier.notify(_event("a" * 64))
    assert not notifier.notify(_event("a" * 64))
    assert notifier.notify(_event("b" * 64))
    now[0] += 61
    assert notifier.notify(_event("a" * 64))
    await notifier.aclose()
    assert len(receiver.calls) == 3


@pytest.mark.parametrize("failure", [503, 429, OSError("connection refused")])
async def test_retries_until_success(failure: int | Exception) -> None:
    receiver = _Receiver(failure, failure, 204)
    notifier = _notifier(receiver, attempts=3)
    notifier.notify(_event())
    await notifier.aclose()
    assert len(receiver.calls) == 3


async def test_client_error_is_not_retried() -> None:
    receiver = _Receiver(400)
    notifier = _notifier(receiver, attempts=3)
    notifier.notify(_event())
    await notifier.aclose()
    assert len(receiver.calls) == 1


async def test_gives_up_after_the_last_attempt() -> None:
    receiver = _Receiver(500, 500, 500, 500)
    notifier = _notifier(receiver, attempts=2)
    notifier.notify(_event())
    await notifier.aclose()
    assert len(receiver.calls) == 2


def test_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attempts"):
        WebhookNotifier("http://hooks.test", attempts=0)
