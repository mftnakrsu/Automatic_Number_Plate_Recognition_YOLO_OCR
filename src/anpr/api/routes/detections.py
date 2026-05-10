"""GET /api/v1/detections — list recently persisted plate reads."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from anpr.api.deps import get_repo
from anpr.storage.repository import DetectionRepository

router = APIRouter(prefix="/api/v1", tags=["detections"])


class DetectionOut(BaseModel):
    id: int
    timestamp: datetime
    track_id: int | None
    plate_hmac: str
    province_code: str | None
    confidence: float
    bbox: tuple[int, int, int, int]
    confirmed: bool


@router.get("/detections", response_model=list[DetectionOut])
async def list_detections(
    repo: Annotated[DetectionRepository, Depends(get_repo)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    confirmed_only: bool = False,
) -> list[DetectionOut]:
    rows = await repo.list_recent(limit=limit, confirmed_only=confirmed_only)
    return [
        DetectionOut(
            id=row.id or 0,
            timestamp=row.timestamp,
            track_id=row.track_id,
            plate_hmac=row.plate_hmac,
            province_code=row.province_code,
            confidence=row.confidence,
            bbox=(row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
            confirmed=row.confirmed,
        )
        for row in rows
    ]
