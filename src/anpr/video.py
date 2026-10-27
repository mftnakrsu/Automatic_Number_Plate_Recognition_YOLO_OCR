"""Frame sources and confirmed-plate events for `anpr watch`.

A source is a video file, a network stream (`rtsp://`, `http://`, ...) or a
local camera index. Frames are read off the event loop in the default
executor, because `cv2.VideoCapture.read()` blocks. Live sources reconnect a
few times after a dropped connection; a file simply ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from anpr.detector.base import Detector
from anpr.logging import get_logger
from anpr.ocr.base import PlateReader
from anpr.pipeline import Pipeline
from anpr.postprocess.turkish import parse_turkish_plate

log = get_logger(__name__)

_LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class SourceError(RuntimeError):
    """The video source could not be opened."""


class _Capture(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802 - cv2 naming

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


def parse_source(raw: str) -> str | int:
    """A bare number is a camera index; anything else is a path or URL."""
    return int(raw) if raw.isdigit() else raw


def is_live(source: str | int) -> bool:
    return isinstance(source, int) or source.lower().startswith(_LIVE_SCHEMES)


def _open(source: str | int) -> _Capture:
    return cv2.VideoCapture(source)  # type: ignore[return-value]


async def read_frames(
    source: str | int,
    *,
    max_frames: int | None = None,
    reconnect_attempts: int = 5,
    reconnect_delay: float = 2.0,
    opener: Callable[[str | int], _Capture] = _open,
) -> AsyncIterator[np.ndarray]:
    """Yield BGR frames until the source ends or `max_frames` is reached.

    Raises `SourceError` if the source can't be opened at all.
    """
    loop = asyncio.get_running_loop()
    cap = await loop.run_in_executor(None, opener, source)
    produced = 0
    attempts = 0
    try:
        if not cap.isOpened():
            raise SourceError(f"cannot open video source: {source!r}")
        while max_frames is None or produced < max_frames:
            ok, frame = await loop.run_in_executor(None, cap.read)
            if ok and frame is not None:
                attempts = 0
                produced += 1
                yield frame
                continue
            if not is_live(source) or attempts >= reconnect_attempts:
                return
            attempts += 1
            log.warning("watch.reconnect", attempt=attempts, of=reconnect_attempts)
            cap.release()
            await asyncio.sleep(reconnect_delay)
            cap = await loop.run_in_executor(None, opener, source)
    finally:
        cap.release()


@dataclass(slots=True, frozen=True)
class ConfirmedPlate:
    frame_idx: int
    track_id: int
    plate: str
    province_code: str
    province_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


async def watch(
    frames: AsyncIterator[np.ndarray],
    detector: Detector,
    reader: PlateReader,
    *,
    detect_every_n: int,
    min_track_dwell: int,
    sr_min_plate_width: int = 0,
) -> AsyncIterator[ConfirmedPlate]:
    """Run the streaming pipeline and yield each track's plate once, when confirmed."""
    pipeline = Pipeline(
        detector,
        reader,
        detect_every_n=detect_every_n,
        min_track_dwell=min_track_dwell,
        sr_min_plate_width=sr_min_plate_width,
    )
    reported: set[int] = set()
    async for event in pipeline.process(frames):
        if event.confirmed is None or event.track_id in reported:
            continue
        parsed = parse_turkish_plate(event.confirmed) or event.plate.parsed
        if parsed is None:
            continue
        reported.add(event.track_id)
        yield ConfirmedPlate(
            frame_idx=event.frame_idx,
            track_id=event.track_id,
            plate=parsed.canonical,
            province_code=parsed.province_code,
            province_name=parsed.province_name,
            confidence=event.plate.detection.confidence,
            bbox=event.plate.detection.bbox.as_xyxy(),
        )
