"""Optional API key on the inference and detection endpoints.

With `ANPR_API_KEY` unset the API stays open (covered by test_api_smoke.py).
With it set, `/api/v1/*` and `/ws/stream` need the key; health, version,
metrics and the web page stay reachable.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from anpr.config import Settings
from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult

KEY = "k3y-for-tests_0123456789abcdef"


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="34 ABC 1234", confidence=0.96)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


class _MockRepo:
    async def list_recent(self, *, limit: int, confirmed_only: bool) -> list[object]:
        return []


def _make_client(monkeypatch: pytest.MonkeyPatch, api_key: str | None) -> TestClient:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    if api_key is None:
        monkeypatch.delenv("ANPR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANPR_API_KEY", api_key)
    from anpr.api import deps as deps_mod
    from anpr.api.app import create_app

    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]
    deps_mod._DETECTOR = _MockDetector()  # type: ignore[attr-defined]
    deps_mod._READER = _MockReader()  # type: ignore[attr-defined]
    deps_mod._REPO = _MockRepo()  # type: ignore[attr-defined]
    return TestClient(create_app())


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from anpr.api import deps as deps_mod

    yield _make_client(monkeypatch, KEY)
    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def open_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from anpr.api import deps as deps_mod

    yield _make_client(monkeypatch, None)
    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]


def _jpeg() -> dict[str, tuple[str, io.BytesIO, str]]:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return {"file": ("t.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}


# --- HTTP ---------------------------------------------------------------------


def test_infer_without_key_is_401(client: TestClient) -> None:
    r = client.post("/api/v1/infer", files=_jpeg())
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_infer_with_wrong_key_is_401(client: TestClient) -> None:
    r = client.post("/api/v1/infer", files=_jpeg(), headers={"X-API-Key": KEY + "x"})
    assert r.status_code == 401


def test_infer_with_header_key(client: TestClient) -> None:
    r = client.post("/api/v1/infer", files=_jpeg(), headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text


def test_infer_with_bearer_key(client: TestClient) -> None:
    r = client.post("/api/v1/infer", files=_jpeg(), headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200, r.text


def test_detections_needs_key(client: TestClient) -> None:
    assert client.get("/api/v1/detections").status_code == 401
    r = client.get("/api/v1/detections", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.parametrize("path", ["/health", "/version", "/metrics", "/"])
def test_open_paths_need_no_key(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_open_api_when_no_key_configured(open_client: TestClient) -> None:
    assert open_client.post("/api/v1/infer", files=_jpeg()).status_code == 200


# --- WebSocket ----------------------------------------------------------------


def test_stream_without_key_is_refused(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/stream"):
        pass
    assert exc.value.code == 1008


def test_stream_with_wrong_subprotocol_key_is_refused(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/stream", subprotocols=["anpr", "anpr.key.nope"]),
    ):
        pass


def test_stream_with_subprotocol_key(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream", subprotocols=["anpr", f"anpr.key.{KEY}"]) as ws:
        assert ws.accepted_subprotocol == "anpr"


def test_stream_with_header_key(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream", headers={"X-API-Key": KEY}) as ws:
        assert ws.accepted_subprotocol is None


def test_stream_open_when_no_key_configured(open_client: TestClient) -> None:
    with open_client.websocket_connect("/ws/stream") as ws:
        assert ws.accepted_subprotocol is None


# --- Web page and settings ----------------------------------------------------


def test_index_asks_for_key_only_when_needed(client: TestClient, open_client: TestClient) -> None:
    assert 'data-auth-required="true"' in client.get("/").text
    assert 'data-auth-required="false"' in open_client.get("/").text


def test_short_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    monkeypatch.setenv("ANPR_API_KEY", "too-short")
    with pytest.raises(ValueError, match="at least 16"):
        Settings()


def test_key_with_unsafe_characters_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    monkeypatch.setenv("ANPR_API_KEY", "has spaces and = signs 0123456789")
    with pytest.raises(ValueError, match="letters, digits"):
        Settings()
