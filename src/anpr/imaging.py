"""Bounded decoding of untrusted image bytes (decompression-bomb guard).

Both `/api/v1/infer` and `/ws/stream` accept attacker-controlled bytes on an
unauthenticated endpoint. A byte-size cap does **not** bound the decoded
raster: a ~50 KB PNG can declare 30000x30000 px, which is 2.7 GB once decoded
to RGB. The pixel count has to be checked against the *declared* dimensions,
before any pixel data is materialized.

Pillow's `Image.open()` is lazy — it parses the container header (so `.size` is
known) but does not decode pixels until `.load()` / `.convert()` is called.
That window is where we reject oversized images. `cv2.imdecode()` has no such
window and no bomb guard at all: by the time it returns, the full raster is
already allocated. WebSocket frames are therefore probed with Pillow first and
only handed to OpenCV once the declared dimensions are known to be in budget.
"""

from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image

DEFAULT_MAX_IMAGE_PIXELS = 40_000_000

# Pillow's upstream default (~89 MP) is far more permissive than this service
# needs, and it is only advisory below 2x. Lower it at import time so that every
# Pillow entry point in the process inherits a sane ceiling even before settings
# are loaded; `configure_pillow_limits()` then applies the configured value.
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS


class ImageDecodeError(ValueError):
    """The bytes are not a decodable image."""


class ImageTooLargeError(ValueError):
    """The declared or decoded raster exceeds the configured pixel budget."""


def configure_pillow_limits(max_pixels: int) -> None:
    """Lower Pillow's process-wide decompression-bomb ceiling to `max_pixels`.

    Called once from the app factory. Note Pillow's own semantics: it raises
    `DecompressionBombError` only above `2 * MAX_IMAGE_PIXELS` and merely emits
    a `DecompressionBombWarning` in the band between. `_open_checked()` promotes
    that warning to an error and additionally checks `img.size` itself, so the
    effective cutoff is exactly `max_pixels`.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels


def _open_checked(raw: bytes, *, max_pixels: int) -> Image.Image:
    """Open lazily and verify the declared size. Pixels are not decoded here.

    Returns the still-undecoded `Image`; the caller owns closing it.
    """
    try:
        with warnings.catch_warnings():
            # Pillow only *warns* in the (MAX_IMAGE_PIXELS, 2 * MAX_IMAGE_PIXELS]
            # band, and warnings are swallowed by default — promote it so nothing
            # in that band can slip through.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ImageTooLargeError(f"pillow decompression-bomb guard tripped: {e}") from e
    except Exception as e:
        raise ImageDecodeError(f"unsupported or corrupt image data: {e!r}") from e

    width, height = img.size
    pixels = width * height
    if width <= 0 or height <= 0 or pixels > max_pixels:
        img.close()
        raise ImageTooLargeError(f"declared {width}x{height} = {pixels} px exceeds {max_pixels} px")
    return img


def decode_upload(raw: bytes, *, max_pixels: int) -> np.ndarray:
    """Decode an uploaded still image to a BGR array, bounded by `max_pixels`.

    Raises `ImageTooLargeError` or `ImageDecodeError`. The message is for the
    server log only — never return it to the client.
    """
    with _open_checked(raw, max_pixels=max_pixels) as img:
        try:
            rgb = np.asarray(img.convert("RGB"))
        except Exception as e:
            raise ImageDecodeError(f"could not decode image pixels: {e!r}") from e
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_frame(blob: bytes, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Decode one streamed frame to a BGR array, bounded by bytes *and* pixels.

    `cv2.imdecode()` allocates the whole raster before returning and has no
    decompression-bomb guard, so the budget is enforced in three steps:

    1. the compressed frame is capped at `max_bytes` before anything is parsed;
    2. the header is parsed with Pillow (lazy — no pixels decoded) and the
       declared dimensions are checked against `max_pixels`;
    3. the decoded shape is re-checked, so a header that lies about its
       dimensions cannot widen the bound.

    Frames Pillow cannot identify are rejected rather than passed to OpenCV:
    fail closed, since an unparseable header means an unbounded decode.
    """
    if len(blob) > max_bytes:
        raise ImageTooLargeError(f"frame is {len(blob)} bytes, limit is {max_bytes} bytes")

    _open_checked(blob, max_pixels=max_pixels).close()

    frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ImageDecodeError("opencv could not decode frame")
    height, width = frame.shape[:2]
    if height * width > max_pixels:
        raise ImageTooLargeError(f"decoded {width}x{height} px exceeds {max_pixels} px")
    return frame
