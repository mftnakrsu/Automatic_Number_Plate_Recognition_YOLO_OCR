"""Response-shape regression: /api/v1/infer JSON must keep stable keys + types.

This is a contract test for the public API — adding new keys is safe, but
renaming or removing keys would break downstream consumers. We assert
exact key sets so accidental drift fails loudly.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [
            Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9),
            Detection(BBox(0, 0, 100, 100), confidence=0.7),
        ]


class _MockReader:
    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="34 ABC 1234", confidence=0.96)

    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in plate_crops]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    from anpr.api import deps as deps_mod
    from anpr.api.app import create_app

    deps_mod._DETECTOR = _MockDetector()  # type: ignore[attr-defined]
    deps_mod._READER = _MockReader()  # type: ignore[attr-defined]

    class _StubRepo:
        async def save(self, d):
            return d

        async def list_recent(self, *, limit=100, confirmed_only=False):
            return []

    deps_mod._REPO = _StubRepo()  # type: ignore[attr-defined]
    return TestClient(create_app())


_INFER_KEYS = {"bbox", "detection_confidence", "raw_text", "normalized_text", "parsed"}
_BBOX_KEYS = {"x1", "y1", "x2", "y2"}
_PARSED_KEYS = {"province_code", "province_name", "letters", "digits", "canonical"}


def test_infer_response_key_contract(client: TestClient) -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    r = client.post(
        "/api/v1/infer",
        files={"file": ("t.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert set(item.keys()) == _INFER_KEYS
        assert set(item["bbox"].keys()) == _BBOX_KEYS
        assert isinstance(item["detection_confidence"], float)
        assert isinstance(item["raw_text"], str)
        assert isinstance(item["normalized_text"], str)
        assert item["parsed"] is None or set(item["parsed"].keys()) == _PARSED_KEYS
