"""Async storage round-trip on an in-memory SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from anpr.storage.models import Detection
from anpr.storage.repository import DetectionRepository, init_db


@pytest_asyncio.fixture
async def repo() -> AsyncIterator[DetectionRepository]:
    engine = await init_db("sqlite+aiosqlite:///:memory:")
    yield DetectionRepository(engine)
    await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_list(repo: DetectionRepository) -> None:
    saved = await repo.save(
        Detection(
            plate_hmac="a" * 64,
            province_code="34",
            confidence=0.95,
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=110,
            bbox_y2=60,
            track_id=1,
            confirmed=True,
        )
    )
    assert saved.id is not None
    rows = await repo.list_recent()
    assert len(rows) == 1
    assert rows[0].plate_hmac == "a" * 64
    assert rows[0].confirmed is True


@pytest.mark.asyncio
async def test_confirmed_only_filter(repo: DetectionRepository) -> None:
    await repo.save(
        Detection(
            plate_hmac="a" * 64,
            confidence=0.9,
            bbox_x1=0,
            bbox_y1=0,
            bbox_x2=1,
            bbox_y2=1,
            confirmed=False,
        )
    )
    await repo.save(
        Detection(
            plate_hmac="b" * 64,
            confidence=0.9,
            bbox_x1=0,
            bbox_y1=0,
            bbox_x2=1,
            bbox_y2=1,
            confirmed=True,
        )
    )
    rows = await repo.list_recent(confirmed_only=True)
    assert len(rows) == 1
    assert rows[0].confirmed is True


@pytest.mark.asyncio
async def test_purge_older_than(repo: DetectionRepository) -> None:
    from datetime import UTC, datetime, timedelta

    old = Detection(
        plate_hmac="o" * 64,
        confidence=0.9,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
        timestamp=datetime.now(UTC) - timedelta(hours=48),
    )
    fresh = Detection(
        plate_hmac="f" * 64,
        confidence=0.9,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=1,
        bbox_y2=1,
    )
    await repo.save(old)
    await repo.save(fresh)

    purged = await repo.purge_older_than(retention_hours=24)
    assert purged == 1

    remaining = await repo.list_recent()
    assert len(remaining) == 1
    assert remaining[0].plate_hmac == "f" * 64
