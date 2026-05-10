"""WebSocket /ws/stream — receives binary JPEG frames, emits JSON read events.

Producer/consumer pattern with a small bounded queue. The producer reads
frames off the WebSocket; the consumer is the streaming `Pipeline`. Frames
are dropped on queue overflow to keep latency bounded.

When a read is confirmed by the temporal voter the HMAC-hashed plate is
persisted to the configured database (raw plate text is never stored — see
`anpr.storage.hashing` for the KVKK/GDPR rationale).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from anpr.api.deps import app_settings, get_detector, get_reader, get_repo
from anpr.config import Settings
from anpr.detector.base import Detector
from anpr.logging import get_logger
from anpr.observability import ACTIVE_STREAMS, CONFIRMED_PLATES, DETECTIONS
from anpr.ocr.base import PlateReader
from anpr.pipeline import Pipeline
from anpr.storage.hashing import hash_plate
from anpr.storage.models import Detection
from anpr.storage.repository import DetectionRepository

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws/stream")
async def stream_ws(
    websocket: WebSocket,
    settings: Annotated[Settings, Depends(app_settings)],
    detector: Annotated[Detector, Depends(get_detector)],
    reader: Annotated[PlateReader, Depends(get_reader)],
    repo: Annotated[DetectionRepository, Depends(get_repo)],
) -> None:
    await websocket.accept()
    ACTIVE_STREAMS.inc()
    log.info("stream.connect")

    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=4)
    pepper = settings.plate_hmac_pepper.get_secret_value()
    persisted_track_ids: set[int] = set()

    async def producer() -> None:
        try:
            while True:
                blob = await websocket.receive_bytes()
                arr = np.frombuffer(blob, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                await queue.put(frame)
        except WebSocketDisconnect:
            await queue.put(None)

    async def frames() -> AsyncIterator[np.ndarray]:
        while True:
            f = await queue.get()
            if f is None:
                return
            yield f

    pipeline = Pipeline(
        detector,
        reader,
        detect_every_n=settings.detect_every_n_frames,
        min_track_dwell=settings.min_track_dwell,
        sr_min_plate_width=settings.sr_min_plate_width,
    )

    prod_task = asyncio.create_task(producer())

    try:
        async for event in pipeline.process(frames()):
            payload: dict[str, Any] = {
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "bbox": list(event.plate.detection.bbox.as_xyxy()),
                "detection_confidence": event.plate.detection.confidence,
                "raw_text": event.plate.raw_text,
                "normalized_text": event.plate.normalized_text,
                "parsed": (
                    {
                        "canonical": event.plate.parsed.canonical,
                        "province_code": event.plate.parsed.province_code,
                        "province_name": event.plate.parsed.province_name,
                    }
                    if event.plate.parsed
                    else None
                ),
                "confirmed": event.confirmed,
            }
            await websocket.send_json(payload)

            DETECTIONS.labels(
                route="stream",
                parsed="valid" if event.plate.parsed else "invalid",
            ).inc()

            if (
                event.confirmed is not None
                and event.plate.parsed is not None
                and event.track_id not in persisted_track_ids
            ):
                CONFIRMED_PLATES.labels(province_code=event.plate.parsed.province_code).inc()
                await repo.save(
                    Detection(
                        track_id=event.track_id,
                        plate_hmac=hash_plate(event.confirmed, pepper),
                        province_code=event.plate.parsed.province_code,
                        confidence=event.plate.detection.confidence,
                        bbox_x1=event.plate.detection.bbox.x1,
                        bbox_y1=event.plate.detection.bbox.y1,
                        bbox_x2=event.plate.detection.bbox.x2,
                        bbox_y2=event.plate.detection.bbox.y2,
                        confirmed=True,
                    )
                )
                persisted_track_ids.add(event.track_id)
                log.info(
                    "stream.persisted",
                    track_id=event.track_id,
                    province=event.plate.parsed.province_code,
                )
    except WebSocketDisconnect:
        log.info("stream.client_disconnect")
    except Exception:
        log.exception("stream.error")
        raise
    finally:
        prod_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await prod_task
        ACTIVE_STREAMS.dec()
        log.info("stream.disconnect")
