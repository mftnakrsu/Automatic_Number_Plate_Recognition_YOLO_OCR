"""fast-alpr 0.4.0+ detector adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from anpr.detector.base import BBox, Detection


class FastAlprDetector:
    """Wraps fast-alpr's bundled YOLOv9-t 384 license-plate ONNX detector.

    Defaults to the MIT-licensed `yolo-v9-t-384-license-plate-end2end` model
    that ships with fast-alpr 0.4.x. To use a different hub model, pass one of
    the documented names (see `fast_alpr.default_detector.PlateDetectorModel`)
    via `model_name`. Custom-ONNX paths require subclassing fast-alpr's
    `BaseDetector` — out of scope here.
    """

    def __init__(
        self,
        model_name: str = "yolo-v9-t-384-license-plate-end2end",
        conf_threshold: float = 0.4,
    ) -> None:
        from fast_alpr.default_detector import LicensePlateDetector

        self._impl = LicensePlateDetector(
            detection_model=model_name,  # type: ignore[arg-type]
            conf_thresh=conf_threshold,
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection. `image` is HxWx3 BGR uint8 (OpenCV convention)."""
        raw = self._impl.predict(image)
        # predict() returns a flat list[DetectionResult] for a single image,
        # or list[list[...]] for a batched call. We only ever pass one image.
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        h, w = image.shape[:2]
        return [_to_detection(r, w, h) for r in raw]


def _to_detection(raw: Any, max_w: int, max_h: int) -> Detection:
    bbox_obj = raw.bounding_box
    confidence = raw.confidence
    bbox = BBox(
        int(bbox_obj.x1),
        int(bbox_obj.y1),
        int(bbox_obj.x2),
        int(bbox_obj.y2),
    ).clip(max_w, max_h)
    return Detection(bbox=bbox, confidence=float(confidence))
