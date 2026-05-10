"""Smoke test the sync pipeline with mocks (no model files needed)."""

from __future__ import annotations

import numpy as np

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

    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text=self._text, confidence=self._confidence)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


def test_infer_image_basic() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    reads = infer_image(img, _MockDetector(), _MockReader())
    assert len(reads) == 1
    assert reads[0].ocr is not None
    assert reads[0].normalized_text == "34ABC1234"
    assert reads[0].detection.confidence == 0.9


def test_infer_image_with_no_ocr() -> None:
    class _NoneReader:
        def read(self, c: np.ndarray) -> None:
            return None

        def read_batch(self, cs: list[np.ndarray]) -> list[None]:
            return [None for _ in cs]

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    reads = infer_image(img, _MockDetector(), _NoneReader())
    assert reads[0].normalized_text is None
    assert reads[0].ocr is None
