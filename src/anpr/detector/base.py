"""Detector abstraction.

Concrete detector adapters implement `Detector`. The pipeline only depends on
this Protocol, so swapping fast-alpr for a fine-tuned YOLO26 or RF-DETR is a
one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True, frozen=True)
class BBox:
    """Axis-aligned bounding box; integer pixel coords; (x2, y2) exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1 : self.y2, self.x1 : self.x2]

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.width, self.height

    def clip(self, max_w: int, max_h: int) -> BBox:
        return BBox(
            max(0, min(self.x1, max_w)),
            max(0, min(self.y1, max_h)),
            max(0, min(self.x2, max_w)),
            max(0, min(self.y2, max_h)),
        )


@dataclass(slots=True, frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int = 0


@runtime_checkable
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...
