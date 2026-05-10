"""/metrics endpoint exposes Prometheus text format and counts requests."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


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


def test_metrics_endpoint_text(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "anpr_requests_total" in body
    assert "anpr_request_seconds" in body


def test_metrics_count_a_request(client: TestClient) -> None:
    # baseline
    r0 = client.get("/metrics")
    assert r0.status_code == 200
    # induce a /health request
    client.get("/health")
    r1 = client.get("/metrics")
    # health was hit at least once → count metric should be present in output
    assert 'path="/health"' in r1.text
