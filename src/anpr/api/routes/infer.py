"""POST /api/v1/infer — upload an image, get plate detections + OCR."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from anpr.api.auth import require_api_key
from anpr.api.deps import app_settings, get_detector, get_reader
from anpr.config import Settings
from anpr.detector.base import Detector
from anpr.imaging import ImageDecodeError, ImageTooLargeError, decode_upload
from anpr.logging import get_logger
from anpr.observability import DETECTIONS, INFERENCE_LATENCY
from anpr.ocr.base import PlateReader
from anpr.pipeline import infer_image

router = APIRouter(prefix="/api/v1", tags=["inference"], dependencies=[Depends(require_api_key)])
log = get_logger(__name__)


class BBoxOut(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ParsedOut(BaseModel):
    country: str
    # Türkiye only; null for the other countries.
    province_code: str | None
    province_name: str | None
    letters: str
    digits: str
    canonical: str


class InferenceResult(BaseModel):
    bbox: BBoxOut
    detection_confidence: float
    raw_text: str | None = None
    normalized_text: str | None = None
    parsed: ParsedOut | None = None


@router.post("/infer", response_model=list[InferenceResult])
async def infer_endpoint(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(app_settings)],
    detector: Annotated[Detector, Depends(get_detector)],
    reader: Annotated[PlateReader, Depends(get_reader)],
) -> list[InferenceResult]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="upload too large")
    try:
        bgr = decode_upload(raw, max_pixels=settings.max_image_pixels)
    except ImageTooLargeError as e:
        # Detail is deliberately generic — the exception text names internal
        # limits and is for the operator's log, not the caller.
        log.warning("infer.image_rejected", reason=str(e))
        raise HTTPException(status_code=413, detail="image too large") from e
    except ImageDecodeError as e:
        log.warning("infer.undecodable", reason=str(e))
        raise HTTPException(status_code=400, detail="cannot decode image") from e

    start = time.perf_counter()
    reads = infer_image(bgr, detector, reader, formats=settings.plate_format_list)
    INFERENCE_LATENCY.labels(route="infer").observe(time.perf_counter() - start)
    for r in reads:
        DETECTIONS.labels(route="infer", parsed="valid" if r.parsed else "invalid").inc()
    return [
        InferenceResult(
            bbox=BBoxOut(
                x1=r.detection.bbox.x1,
                y1=r.detection.bbox.y1,
                x2=r.detection.bbox.x2,
                y2=r.detection.bbox.y2,
            ),
            detection_confidence=r.detection.confidence,
            raw_text=r.raw_text,
            normalized_text=r.normalized_text,
            parsed=(
                ParsedOut(
                    country=r.parsed.country,
                    province_code=r.parsed.province_code,
                    province_name=r.parsed.province_name,
                    letters=r.parsed.letters,
                    digits=r.parsed.digits,
                    canonical=r.parsed.canonical,
                )
                if r.parsed
                else None
            ),
        )
        for r in reads
    ]
