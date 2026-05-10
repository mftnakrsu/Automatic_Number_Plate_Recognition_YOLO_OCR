"""Smoke test the sync pipeline with mock detector + reader.

The real fast-alpr integration test lives behind a marker so CI without GPU/
network skips it.
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult
from anpr.pipeline import infer_image


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    def __init__(self, text: str = "34 ABC 1234", confidence: float = 0.97) -> None:
        self._text = text
        self._confidence = confidence

    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text=self._text, confidence=self._confidence)

    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in plate_crops]


def test_infer_image_with_valid_plate() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    reads = infer_image(img, _MockDetector(), _MockReader())
    assert len(reads) == 1
    r = reads[0]
    assert r.detection.confidence == pytest.approx(0.9)
    assert r.ocr is not None
    assert r.parsed is not None
    assert r.parsed.canonical == "34 ABC 1234"
    assert r.parsed.province_name == "İstanbul"


def test_infer_image_corrects_ocr_confusion() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # OCR misread "06 ABC 1234" as "O6 ABC I234" — letter O for 0, I for 1
    reader = _MockReader(text="O6 ABC I234", confidence=0.91)
    reads = infer_image(img, _MockDetector(), reader)
    assert reads[0].parsed is not None
    assert reads[0].parsed.canonical == "06 ABC 1234"


def test_infer_image_with_unparseable_ocr() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    reader = _MockReader(text="????????", confidence=0.5)
    reads = infer_image(img, _MockDetector(), reader)
    assert reads[0].parsed is None
    assert reads[0].normalized_text == "????????"
