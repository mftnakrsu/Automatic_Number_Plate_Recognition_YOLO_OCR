"""Watchlist: the plates to raise an alert for, held as HMAC hashes.

Plates go in as text and are hashed with the deployment pepper straight away
(`anpr.storage.hashing.hash_plate`), so the watchlist holds no more than the
detections table does. A confirmed read matches when its hash is listed.

`Watchlist` keeps the table in memory so a match costs a dict lookup per
confirmed plate. `refresh_worker` reloads it now and then, which picks up
entries added with `anpr watchlist add` while the server is running.
"""

from __future__ import annotations

import asyncio

from anpr.logging import get_logger
from anpr.observability import WATCHLIST_MATCHES
from anpr.storage.models import WatchlistEntry
from anpr.storage.repository import WatchlistRepository
from anpr.webhook import WebhookNotifier, match_event

_log = get_logger(__name__)


class Watchlist:
    """In-memory view of the watchlist table; without a repository it lives only in memory."""

    def __init__(self, repo: WatchlistRepository | None = None) -> None:
        self._repo = repo
        self._entries: dict[str, WatchlistEntry] = {}

    async def load(self) -> None:
        if self._repo is not None:
            self._entries = {e.plate_hmac: e for e in await self._repo.list_all()}

    async def add(self, plate_hmac: str, label: str | None = None) -> WatchlistEntry:
        if self._repo is not None:
            entry = await self._repo.upsert(plate_hmac, label)
        else:
            old = self._entries.get(plate_hmac)
            keep = label if label is not None or old is None else old.label
            entry = WatchlistEntry(plate_hmac=plate_hmac, label=keep)
        self._entries[plate_hmac] = entry
        return entry

    async def remove(self, plate_hmac: str) -> bool:
        removed = self._entries.pop(plate_hmac, None) is not None
        if self._repo is not None:
            removed = await self._repo.remove(plate_hmac) or removed
        return removed

    def match(self, plate_hmac: str) -> WatchlistEntry | None:
        return self._entries.get(plate_hmac)

    def entries(self) -> list[WatchlistEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


def alert(
    watchlist: Watchlist,
    notifier: WebhookNotifier | None,
    plate_hmac: str,
    *,
    source: str,
    track_id: int,
    frame_idx: int,
    country: str,
    province_code: str | None,
    confidence: float,
) -> WatchlistEntry | None:
    """Look up one confirmed plate; on a match, count it and call the webhook if one is set."""
    entry = watchlist.match(plate_hmac)
    if entry is None:
        return None
    WATCHLIST_MATCHES.labels(source=source).inc()
    _log.info("watchlist.match", source=source, track_id=track_id, label=entry.label)
    if notifier is not None:
        notifier.notify(
            match_event(
                entry,
                source=source,
                track_id=track_id,
                frame_idx=frame_idx,
                country=country,
                province_code=province_code,
                confidence=confidence,
            )
        )
    return entry


async def refresh_worker(watchlist: Watchlist, *, interval_seconds: float = 30.0) -> None:
    """Background task: reload the watchlist from the database every `interval_seconds`."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await watchlist.load()
        except Exception:
            _log.exception("watchlist.refresh_error")
