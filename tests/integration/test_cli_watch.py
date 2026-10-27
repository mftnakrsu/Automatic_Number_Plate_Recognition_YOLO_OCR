"""`anpr watch` on a generated video file with mock models."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult
from anpr.storage.hashing import hash_plate


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="06 AA 42", confidence=0.95)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(30):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture(autouse=True)
def _mock_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import anpr.cli as cli

    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "p" * 64)
    monkeypatch.setenv("ANPR_DETECT_EVERY_N_FRAMES", "1")
    monkeypatch.setenv("ANPR_MIN_TRACK_DWELL", "3")
    monkeypatch.setenv("ANPR_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'anpr.db'}")
    monkeypatch.setattr(cli, "_build_models", lambda settings: (_MockDetector(), _MockReader()))


def test_watch_prints_each_confirmed_plate_once(video: Path) -> None:
    from anpr.cli import app

    result = CliRunner().invoke(app, ["watch", str(video)])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("06 AA 42") == 1
    assert "Ankara" in result.stdout


def test_watch_json_lines(video: Path) -> None:
    from anpr.cli import app

    result = CliRunner().invoke(app, ["watch", str(video), "--json"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["plate"] == "06 AA 42"
    assert lines[0]["province_code"] == "06"
    assert lines[0]["track_id"] == 1


def test_watch_store_keeps_only_the_hash(video: Path, tmp_path: Path) -> None:
    from anpr.cli import app

    result = CliRunner().invoke(app, ["watch", str(video), "--store"])
    assert result.exit_code == 0, result.output
    with sqlite3.connect(tmp_path / "anpr.db") as db:
        rows = db.execute("SELECT plate_hmac, province_code, confirmed FROM detections").fetchall()
    assert len(rows) == 1
    plate_hmac, province, confirmed = rows[0]
    assert plate_hmac == hash_plate("06 AA 42", "p" * 64)
    assert province == "06"
    assert confirmed == 1


def test_watch_unopenable_source_exits_2(tmp_path: Path) -> None:
    from anpr.cli import app

    result = CliRunner().invoke(app, ["watch", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 2
