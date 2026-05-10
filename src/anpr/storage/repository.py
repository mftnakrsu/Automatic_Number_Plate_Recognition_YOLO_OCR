"""Async repository on top of SQLAlchemy + SQLModel.

Defaults to SQLite via aiosqlite for zero-setup demos; switch
`ANPR_DATABASE_URL` to `postgresql+asyncpg://...` for production.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel, delete, select

from anpr.logging import get_logger
from anpr.storage.models import Detection

_log = get_logger(__name__)


async def init_db(database_url: str) -> AsyncEngine:
    """Open the async engine and create tables if missing."""
    engine = create_async_engine(database_url, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


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
