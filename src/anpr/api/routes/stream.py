"""WebSocket /ws/stream — receives binary JPEG frames, emits read events as JSON.

Producer/consumer pattern with a small bounded queue. The producer reads frames
off the WebSocket; the consumer is the streaming `Pipeline`. Frames are dropped
on queue overflow to keep latency bounded — by design, a slow consumer never
backs up the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from anpr.api.deps import app_settings, get_detector, get_reader
from anpr.config import Settings
from anpr.detector.base import Detector
from anpr.logging import get_logger
from anpr.ocr.base import PlateReader
from anpr.pipeline import Pipeline

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws/stream")
async def stream_ws(
    websocket: WebSocket,
    settings: Annotated[Settings, Depends(app_settings)],
    detector: Annotated[Detector, Depends(get_detector)],
    reader: Annotated[PlateReader, Depends(get_reader)],
) -> None:
    await websocket.accept()
    log.info("stream.connect")

    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=4)

    async def producer() -> None:
        try:
            while True:
                blob = await websocket.receive_bytes()
                arr = np.frombuffer(blob, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
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
    except WebSocketDisconnect:
        log.info("stream.client_disconnect")
    except Exception:
        log.exception("stream.error")
        raise
    finally:
        prod_task.cancel()
        try:
            await prod_task
        except (asyncio.CancelledError, Exception):
            pass
        log.info("stream.disconnect")
