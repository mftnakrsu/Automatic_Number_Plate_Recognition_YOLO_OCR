"""OCR abstraction.

A `PlateReader` consumes a tight plate crop and returns the characters plus a
confidence. Per-character confidences are used downstream for confusion-pair
correction and temporal majority voting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True, frozen=True)
class OcrResult:
    text: str
    confidence: float
    char_confidences: tuple[float, ...] = field(default_factory=tuple)


@runtime_checkable
class PlateReader(Protocol):
    def read(self, plate_crop: np.ndarray) -> OcrResult | None: ...
    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]: ...
