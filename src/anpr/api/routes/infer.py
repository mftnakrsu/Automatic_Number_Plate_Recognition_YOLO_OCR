"""POST /api/v1/infer — upload an image, get plate detections + OCR."""

from __future__ import annotations

import io
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from anpr.api.deps import app_settings, get_detector, get_reader
from anpr.config import Settings
from anpr.detector.base import Detector
from anpr.observability import DETECTIONS, INFERENCE_LATENCY
from anpr.ocr.base import PlateReader
from anpr.pipeline import infer_image

router = APIRouter(prefix="/api/v1", tags=["inference"])


class BBoxOut(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ParsedOut(BaseModel):
    province_code: str
    province_name: str
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
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}") from e

    start = time.perf_counter()
    reads = infer_image(bgr, detector, reader)
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
