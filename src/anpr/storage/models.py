"""SQLModel ORM models.

Note we never persist raw plate text — only the HMAC-SHA256 hash. This is
deliberate; see `anpr.storage.hashing` for the KVKK/GDPR rationale.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Detection(SQLModel, table=True):
    __tablename__ = "detections"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utc_now, index=True)
    track_id: int | None = Field(default=None, index=True)
    plate_hmac: str = Field(max_length=64, index=True)
    province_code: str | None = Field(default=None, max_length=2, index=True)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    confirmed: bool = Field(default=False, index=True)
