"""Regression tests for the decompression-bomb guard (GHSA-r5v2-w3w2-4c8w)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from anpr.imaging import (
    ImageDecodeError,
    ImageTooLargeError,
    decode_frame,
    decode_upload,
)


def _png(width: int, height: int) -> bytes:
    """A solid-colour PNG: tiny compressed, huge decoded."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (0, 0, 0)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def test_decode_upload_accepts_a_normal_image() -> None:
    bgr = decode_upload(_png(64, 48), max_pixels=40_000_000)
    assert bgr.shape == (48, 64, 3)


def test_decode_upload_rejects_bomb_before_decoding() -> None:
    # 16000x16000 compresses to well under 1 MB but decodes to ~768 MB.
    bomb = _png(16_000, 16_000)
    assert len(bomb) < 1024 * 1024
    with pytest.raises(ImageTooLargeError):
        decode_upload(bomb, max_pixels=40_000_000)


def test_decode_upload_rejects_the_pillow_warning_band() -> None:
    # Between max_pixels and 2x max_pixels Pillow only warns; we must still fail.
    with pytest.raises(ImageTooLargeError):
        decode_upload(_png(3000, 3000), max_pixels=4_000_000)


def test_decode_upload_rejects_garbage() -> None:
    with pytest.raises(ImageDecodeError):
        decode_upload(b"definitely not an image", max_pixels=40_000_000)


def test_decode_frame_accepts_a_normal_frame() -> None:
    frame = decode_frame(_png(32, 24), max_bytes=4 * 1024 * 1024, max_pixels=40_000_000)
    assert frame.shape == (24, 32, 3)


def test_decode_frame_rejects_oversized_bytes_before_parsing() -> None:
    with pytest.raises(ImageTooLargeError):
        decode_frame(_png(64, 64), max_bytes=8, max_pixels=40_000_000)


def test_decode_frame_rejects_bomb_before_imdecode() -> None:
    # cv2.imdecode has no bomb guard of its own — this is the WebSocket path
    # that the reported PoC used to OOM-kill the container.
    with pytest.raises(ImageTooLargeError):
        decode_frame(_png(16_000, 16_000), max_bytes=4 * 1024 * 1024, max_pixels=40_000_000)
