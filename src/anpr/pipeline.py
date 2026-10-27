"""End-to-end inference.

- `infer_image()` — synchronous one-shot used by the CLI and tests.
- `Pipeline` — async streaming class with frame-skip + tracking + temporal voting.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anpr.detector.base import Detection, Detector
from anpr.ocr.base import OcrResult, PlateReader
from anpr.postprocess.dewarp import dewarp
from anpr.postprocess.formats import DEFAULT_FORMATS, ParsedPlate, PlateFormat, parse_plate
from anpr.postprocess.temporal import TemporalVoter
from anpr.tracker.iou import IoUTracker


@dataclass(slots=True, frozen=True)
class PlateRead:
    detection: Detection
    ocr: OcrResult | None
    raw_text: str | None
    normalized_text: str | None
    parsed: ParsedPlate | None

    @property
    def is_valid(self) -> bool:
        """The read fits one of the enabled plate formats."""
        return self.parsed is not None

    @property
    def is_valid_turkish_plate(self) -> bool:
        return self.parsed is not None and self.parsed.country == "tr"


@dataclass(slots=True, frozen=True)
class StreamReadEvent:
    """A single read emitted by the streaming pipeline."""

    frame_idx: int
    track_id: int
    plate: PlateRead
    confirmed: str | None  # set when temporal voter has reached dwell threshold


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _build_plate_read(
    det: Detection, ocr: OcrResult | None, formats: Sequence[PlateFormat]
) -> PlateRead:
    if ocr is None:
        return PlateRead(det, None, None, None, None)
    normalized = "".join(ocr.text.split()).upper()
    parsed: ParsedPlate | None = None
    result = parse_plate(normalized, formats)
    if result is not None:
        parsed, normalized = result
    return PlateRead(
        detection=det,
        ocr=ocr,
        raw_text=ocr.text,
        normalized_text=normalized,
        parsed=parsed,
    )


def infer_image(
    image: np.ndarray | str | Path,
    detector: Detector,
    reader: PlateReader,
    *,
    formats: Sequence[PlateFormat] = DEFAULT_FORMATS,
) -> list[PlateRead]:
    """Run detection + OCR + plate-format parse on a single image."""
    if not isinstance(image, np.ndarray):
        image = read_image(image)
    detections = detector.detect(image)
    crops = [d.bbox.crop(image) for d in detections]
    ocr_results = reader.read_batch(crops)
    return [_build_plate_read(d, o, formats) for d, o in zip(detections, ocr_results, strict=True)]


class Pipeline:
    """Async streaming pipeline with frame-skip + tracking + temporal voting.

    Detect every Nth frame; track between using IoU; OCR per detection on the
    detected frames; majority-vote across the per-track read history.
    """

    def __init__(
        self,
        detector: Detector,
        reader: PlateReader,
        *,
        detect_every_n: int = 3,
        min_track_dwell: int = 5,
        sr_min_plate_width: int = 0,
        formats: Sequence[PlateFormat] = DEFAULT_FORMATS,
    ) -> None:
        if detect_every_n < 1:
            raise ValueError("detect_every_n must be >= 1")
        self._detector = detector
        self._reader = reader
        self._detect_every_n = detect_every_n
        self._tracker = IoUTracker()
        self._voter = TemporalVoter(min_dwell=min_track_dwell)
        self._sr_min_w = sr_min_plate_width
        self._formats = formats

    async def process(self, frames: AsyncIterator[np.ndarray]) -> AsyncIterator[StreamReadEvent]:
        loop = asyncio.get_running_loop()
        idx = 0
        async for frame in frames:
            if idx % self._detect_every_n != 0:
                idx += 1
                continue

            detections = await loop.run_in_executor(None, self._detector.detect, frame)
            tracked = self._tracker.update(detections)

            for track_id, det in tracked:
                crop = det.bbox.crop(frame)
                if crop.size == 0:
                    continue
                if self._sr_min_w > 0 and crop.shape[1] < self._sr_min_w:
                    pass  # super-resolution stub — wire Real-ESRGAN here if available
                crop = dewarp(crop)
                ocr = await loop.run_in_executor(None, self._reader.read, crop)
                plate = _build_plate_read(det, ocr, self._formats)

                confirmed: str | None = None
                if plate.parsed is not None and plate.ocr is not None:
                    confirmed = self._voter.observe(
                        track_id, plate.parsed.canonical, plate.ocr.confidence
                    )

                yield StreamReadEvent(
                    frame_idx=idx,
                    track_id=track_id,
                    plate=plate,
                    confirmed=confirmed,
                )
            idx += 1
