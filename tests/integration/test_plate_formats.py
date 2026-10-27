"""`ANPR_PLATE_FORMATS` end to end: settings, `/api/v1/infer`, `/ws/stream` and `anpr watch`."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from anpr.config import Settings
from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    text = "M AB 1234"

    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text=self.text, confidence=0.95)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


class _MockRepo:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    async def save(self, detection: Any) -> Any:
        self.saved.append(detection)
        return detection


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "f" * 64)
    monkeypatch.setenv("ANPR_DETECT_EVERY_N_FRAMES", "1")
    monkeypatch.setenv("ANPR_MIN_TRACK_DWELL", "3")
    monkeypatch.delenv("ANPR_API_KEY", raising=False)
    monkeypatch.delenv("ANPR_WEBHOOK_URL", raising=False)


# --- Settings -----------------------------------------------------------------


def test_default_is_turkish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANPR_PLATE_FORMATS", raising=False)
    assert [f.code for f in Settings().plate_format_list] == ["tr"]


def test_codes_are_cleaned_and_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANPR_PLATE_FORMATS", " DE, fr,de ,")
    settings = Settings()
    assert settings.plate_formats == "de,fr"
    assert [f.code for f in settings.plate_format_list] == ["de", "fr"]


@pytest.mark.parametrize(("value", "message"), [("de,xx", "unknown xx"), (" , ", "at least one")])
def test_bad_codes_are_rejected(monkeypatch: pytest.MonkeyPatch, value: str, message: str) -> None:
    monkeypatch.setenv("ANPR_PLATE_FORMATS", value)
    with pytest.raises(ValueError, match=message):
        Settings()


# --- API ----------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ANPR_PLATE_FORMATS", "tr,de")
    from anpr.api import deps as deps_mod
    from anpr.api.app import create_app
    from anpr.watchlist import Watchlist

    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setattr(deps_mod, "_DETECTOR", _MockDetector())
    monkeypatch.setattr(deps_mod, "_READER", _MockReader())
    monkeypatch.setattr(deps_mod, "_REPO", _MockRepo())
    monkeypatch.setattr(deps_mod, "_WATCHLIST", Watchlist())
    monkeypatch.setattr(deps_mod, "_NOTIFIER", None)
    yield TestClient(create_app())
    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_infer_reads_a_german_plate(client: TestClient) -> None:
    r = client.post("/api/v1/infer", files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    assert r.status_code == 200, r.text
    parsed = r.json()[0]["parsed"]
    assert parsed == {
        "country": "de",
        "province_code": None,
        "province_name": None,
        "letters": "MAB",
        "digits": "1234",
        "canonical": "MAB 1234",
    }


def test_stream_confirms_and_stores_a_german_plate(client: TestClient) -> None:
    from anpr.api import deps as deps_mod

    with client.websocket_connect("/ws/stream") as ws:
        events = []
        for _ in range(4):
            ws.send_bytes(_jpeg())
            events.append(ws.receive_json())
    assert events[2]["confirmed"] == "MAB 1234"
    assert events[2]["parsed"]["country"] == "de"
    saved = cast(_MockRepo, deps_mod._REPO).saved
    assert len(saved) == 1
    assert saved[0].province_code is None


# --- CLI ----------------------------------------------------------------------


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(20):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_watch_prints_the_country(
    video: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import anpr.cli as cli

    reader = _MockReader()
    reader.text = "GH 456 JK"
    monkeypatch.setenv("ANPR_PLATE_FORMATS", "fr")
    monkeypatch.setenv("ANPR_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'anpr.db'}")
    monkeypatch.setattr(cli, "_build_models", lambda settings: (_MockDetector(), reader))

    result = CliRunner().invoke(cli.app, ["watch", str(video), "--json"])
    assert result.exit_code == 0, result.output
    line = json.loads(result.stdout.splitlines()[0])
    assert line["plate"] == "GH-456-JK"
    assert line["country"] == "fr"
    assert line["province_code"] is None

    result = CliRunner().invoke(cli.app, ["watch", str(video)])
    assert result.exit_code == 0, result.output
    assert "GH-456-JK  FR" in result.stdout
