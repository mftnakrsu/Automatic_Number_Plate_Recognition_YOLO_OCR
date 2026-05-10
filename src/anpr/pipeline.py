"""End-to-end inference (F1: synchronous one-shot).

`infer_image()` runs detection then OCR on one image. The async streaming
`Pipeline` class, postprocess fixes, and Turkish-plate parsing land in the
next commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anpr.detector.base import Detection, Detector
from anpr.ocr.base import OcrResult, PlateReader


@dataclass(slots=True, frozen=True)
class PlateRead:
    detection: Detection
    ocr: OcrResult | None
    raw_text: str | None
    normalized_text: str | None


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def infer_image(
    image: np.ndarray | str | Path,
    detector: Detector,
    reader: PlateReader,
) -> list[PlateRead]:
    if not isinstance(image, np.ndarray):
        image = read_image(image)
    detections = detector.detect(image)
    crops = [d.bbox.crop(image) for d in detections]
    ocr_results = reader.read_batch(crops)
    return [
        PlateRead(
            detection=det,
            ocr=ocr,
            raw_text=ocr.text if ocr else None,
            normalized_text=("".join(ocr.text.split()).upper() if ocr else None),
        )
        for det, ocr in zip(detections, ocr_results, strict=True)
    ]
