"""Frame sources and the confirmed-plate stream behind `anpr watch`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult
from anpr.video import SourceError, is_live, parse_source, read_frames, watch


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="34 ABC 1234", confidence=0.95)

    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in plate_crops]


class _FakeCapture:
    """Yields `frames` black frames, then fails like a dropped stream."""

    def __init__(self, frames: int) -> None:
        self._left = frames
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._left == 0:
            return False, None
        self._left -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _write_video(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()
    return path


async def _collect(it: AsyncIterator[np.ndarray]) -> list[np.ndarray]:
    return [f async for f in it]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("2", 2), ("clip.mp4", "clip.mp4"), ("rtsp://cam/1", "rtsp://cam/1")],
)
def test_parse_source(raw: str, expected: str | int) -> None:
    assert parse_source(raw) == expected


@pytest.mark.parametrize(
    ("source", "live"),
    [(0, True), ("rtsp://cam/1", True), ("https://cam/mjpg", True), ("clip.mp4", False)],
)
def test_is_live(source: str | int, live: bool) -> None:
    assert is_live(source) is live


async def test_reads_every_frame_of_a_file(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi", 6)
    frames = await _collect(read_frames(str(video)))
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)


async def test_max_frames_stops_early(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi", 6)
    assert len(await _collect(read_frames(str(video), max_frames=3))) == 3


async def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        await _collect(read_frames(str(tmp_path / "nope.mp4")))


async def test_live_source_reconnects_after_a_drop() -> None:
    # A frame resets the attempt count, so the drop after the second capture's frame gets its
    # own reconnect; the third capture fails straight away and uses up the single attempt.
    captures = [_FakeCapture(2), _FakeCapture(1), _FakeCapture(0)]
    opened: list[str | int] = []

    def opener(source: str | int) -> _FakeCapture:
        opened.append(source)
        return captures[len(opened) - 1]

    frames = await _collect(
        read_frames("rtsp://cam/1", reconnect_attempts=1, reconnect_delay=0, opener=opener)
    )
    assert len(frames) == 3
    assert opened == ["rtsp://cam/1"] * 3
    assert all(c.released for c in captures)


async def test_file_source_does_not_reconnect() -> None:
    opened: list[str | int] = []

    def opener(source: str | int) -> _FakeCapture:
        opened.append(source)
        return _FakeCapture(2)

    frames = await _collect(read_frames("clip.mp4", reconnect_delay=0, opener=opener))
    assert len(frames) == 2
    assert len(opened) == 1


async def test_watch_confirms_each_track_once() -> None:
    async def frames() -> AsyncIterator[np.ndarray]:
        for _ in range(12):
            yield np.zeros((240, 320, 3), dtype=np.uint8)

    plates = [
        p
        async for p in watch(
            frames(), _MockDetector(), _MockReader(), detect_every_n=1, min_track_dwell=3
        )
    ]
    assert len(plates) == 1
    plate = plates[0]
    assert plate.plate == "34 ABC 1234"
    assert plate.province_code == "34"
    assert plate.province_name == "İstanbul"
    assert plate.frame_idx == 2
