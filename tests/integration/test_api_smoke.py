"""Smoke tests for the FastAPI app with mocked detector + reader.

Bypasses the lifespan event by skipping the TestClient context-manager pattern;
stubs the deps module's globals directly.
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
        ]


class _MockReader:
    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="34 ABC 1234", confidence=0.96)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    from anpr.api import deps as deps_mod
    from anpr.api.app import create_app

    deps_mod._DETECTOR = _MockDetector()  # type: ignore[attr-defined]
    deps_mod._READER = _MockReader()  # type: ignore[attr-defined]

    app = create_app()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_version(client: TestClient) -> None:
    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_infer(client: TestClient) -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    files = {"file": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    r = client.post("/api/v1/infer", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data) == 1
    assert data[0]["parsed"]["canonical"] == "34 ABC 1234"
    assert data[0]["parsed"]["province_name"] == "İstanbul"


def test_infer_rejects_empty(client: TestClient) -> None:
    files = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    r = client.post("/api/v1/infer", files=files)
    assert r.status_code == 400


def test_infer_rejects_garbage(client: TestClient) -> None:
    files = {"file": ("garbage.jpg", io.BytesIO(b"not an image"), "image/jpeg")}
    r = client.post("/api/v1/infer", files=files)
    assert r.status_code == 400


def test_request_id_header(client: TestClient) -> None:
    r = client.get("/health")
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) >= 8


def test_request_id_passthrough(client: TestClient) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abcdefab"})
    assert r.headers["X-Request-ID"] == "abcdefab"
