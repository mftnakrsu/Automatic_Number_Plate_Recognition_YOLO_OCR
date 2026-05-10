"""End-to-end test against the real fast-alpr + fast-plate-ocr models.

Gated behind `ANPR_RUN_REAL=1` because the first run downloads ~30MB of
model weights and the inference itself is much slower than the mock tests.
Skipped in the default CI matrix; enable in a dedicated entry with cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL = os.getenv("ANPR_RUN_REAL", "").lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(not REAL, reason="ANPR_RUN_REAL not set")


def test_end_to_end_on_sample(sample_plate_path: Path) -> None:
    pytest.importorskip("fast_alpr")
    pytest.importorskip("fast_plate_ocr")

    from anpr.detector.fast_alpr import FastAlprDetector
    from anpr.ocr.fast_plate import FastPlateOcr
    from anpr.pipeline import infer_image

    detector = FastAlprDetector()
    reader = FastPlateOcr()

    reads = infer_image(sample_plate_path, detector, reader)

    assert len(reads) > 0, "fast-alpr should detect at least one plate on the sample"
    assert any(r.ocr is not None for r in reads), "fast-plate-ocr should read at least one plate"
