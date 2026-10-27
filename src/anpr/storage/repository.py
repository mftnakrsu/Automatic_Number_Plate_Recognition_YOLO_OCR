"""Async repository on top of SQLAlchemy + SQLModel.

Defaults to SQLite via aiosqlite for zero-setup demos; switch
`ANPR_DATABASE_URL` to `postgresql+asyncpg://...` for production.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy import Connection, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel, delete, select

from anpr.logging import get_logger
from anpr.storage.models import Detection, WatchlistEntry

_log = get_logger(__name__)


async def init_db(database_url: str) -> AsyncEngine:
    """Open the async engine, create tables if missing and add columns newer versions brought."""
    engine = create_async_engine(database_url, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
    return engine


# Columns added to existing tables after their first release, as (table, column, SQL type).
_ADDED_COLUMNS = (("detections", "country", "VARCHAR(2)"),)


def _add_missing_columns(conn: Connection) -> None:
    """`create_all` skips tables that already exist, so add new nullable columns by hand."""
    inspector = inspect(conn)
    for table, column, sql_type in _ADDED_COLUMNS:
        existing = {c["name"] for c in inspector.get_columns(table)}
        if column not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
            _log.info("storage.column_added", table=table, column=column)


class DetectionRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            yield session

    async def save(self, detection: Detection) -> Detection:
        async with self.session() as s:
            s.add(detection)
            await s.commit()
            await s.refresh(detection)
            return detection

    async def list_recent(
        self,
        *,
        limit: int = 100,
        confirmed_only: bool = False,
    ) -> list[Detection]:
        async with self.session() as s:
            stmt = (
                select(Detection)
                .order_by(Detection.timestamp.desc())  # type: ignore[attr-defined]
                .limit(limit)
            )
            if confirmed_only:
                stmt = stmt.where(Detection.confirmed.is_(True))  # type: ignore[attr-defined]
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def purge_older_than(self, retention_hours: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        async with self.session() as s:
            stmt = delete(Detection).where(Detection.timestamp < cutoff)  # type: ignore[arg-type]
            result = await s.execute(stmt)
            await s.commit()
            return result.rowcount or 0  # type: ignore[attr-defined]


class WatchlistRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async def list_all(self) -> list[WatchlistEntry]:
        async with self._sessionmaker() as s:
            result = await s.execute(
                select(WatchlistEntry).order_by(WatchlistEntry.created_at)  # type: ignore[arg-type]
            )
            return list(result.scalars().all())

    async def upsert(self, plate_hmac: str, label: str | None) -> WatchlistEntry:
        """Add the plate, or set its label if it is already listed. A None label keeps the old one."""
        for attempt in range(2):
            async with self._sessionmaker() as s:
                result = await s.execute(
                    select(WatchlistEntry).where(WatchlistEntry.plate_hmac == plate_hmac)
                )
                entry = result.scalars().first()
                if entry is None:
                    entry = WatchlistEntry(plate_hmac=plate_hmac, label=label)
                elif label is not None:
                    entry.label = label
                s.add(entry)
                try:
                    await s.commit()
                except IntegrityError:
                    # Another writer inserted the same plate between our select and insert.
                    if attempt:
                        raise
                    continue
                await s.refresh(entry)
                return entry
        raise AssertionError("unreachable")

    async def remove(self, plate_hmac: str) -> bool:
        async with self._sessionmaker() as s:
            stmt = delete(WatchlistEntry).where(
                WatchlistEntry.plate_hmac == plate_hmac  # type: ignore[arg-type]
            )
            result = await s.execute(stmt)
            await s.commit()
            return bool(result.rowcount)  # type: ignore[attr-defined]


async def retention_worker(
    repo: DetectionRepository,
    *,
    retention_hours: int,
    interval_seconds: int = 3600,
) -> None:
    """Background task: periodically purge detections older than the TTL."""
    while True:
        try:
            deleted = await repo.purge_older_than(retention_hours)
            if deleted:
                _log.info("storage.retention.purged", count=deleted)
        except Exception:
            _log.exception("storage.retention.error")
        await asyncio.sleep(interval_seconds)
