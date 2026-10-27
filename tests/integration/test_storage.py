"""Async storage round-trip on an in-memory SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

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


async def test_init_db_adds_the_country_column_to_an_old_database(tmp_path: Path) -> None:
    import sqlite3
    from contextlib import closing

    path = tmp_path / "old.db"
    with closing(sqlite3.connect(path)) as db:
        # The detections table as 0.2.x created it, with one row in it.
        db.execute(
            "CREATE TABLE detections (id INTEGER PRIMARY KEY, timestamp DATETIME NOT NULL, "
            "track_id INTEGER, plate_hmac VARCHAR(64) NOT NULL, province_code VARCHAR(2), "
            "confidence FLOAT NOT NULL, bbox_x1 INTEGER NOT NULL, bbox_y1 INTEGER NOT NULL, "
            "bbox_x2 INTEGER NOT NULL, bbox_y2 INTEGER NOT NULL, confirmed BOOLEAN NOT NULL)"
        )
        db.execute(
            "INSERT INTO detections VALUES (1, '2026-10-01 10:00:00', 1, ?, '34', 0.9, 0, 0, 1, 1, 1)",
            ("o" * 64,),
        )
        db.commit()

    engine = await init_db(f"sqlite+aiosqlite:///{path}")
    try:
        repo = DetectionRepository(engine)
        await repo.save(
            Detection(
                plate_hmac="n" * 64,
                country="de",
                confidence=0.9,
                bbox_x1=0,
                bbox_y1=0,
                bbox_x2=1,
                bbox_y2=1,
                confirmed=True,
            )
        )
        rows = {r.plate_hmac[0]: r for r in await repo.list_recent()}
        assert rows["o"].country is None
        assert rows["o"].province_code == "34"
        assert rows["n"].country == "de"
    finally:
        await engine.dispose()
    # A second start finds the column and leaves it alone.
    await (await init_db(f"sqlite+aiosqlite:///{path}")).dispose()
