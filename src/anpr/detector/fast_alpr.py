"""fast-alpr 0.4.0+ detector adapter."""

from __future__ import annotations

import numpy as np

from anpr.detector.base import BBox, Detection


class FastAlprDetector:
    """Wraps fast-alpr's bundled YOLOv9-t 384 license-plate ONNX detector.

    Defaults to the MIT-licensed model that ships with fast-alpr 0.4.0+. To use a
    fine-tuned plate detector, pass its registered name or a path to a custom
    ONNX file as `model_name`.
    """

    def __init__(
        self,
        model_name: str = "yolo-v9-t-384-license-plate-end2end",
        conf_threshold: float = 0.4,
    ) -> None:
        from fast_alpr.detector import LicensePlateDetector

        self._impl = LicensePlateDetector(
            detector_model=model_name,
            conf_thresh=conf_threshold,
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection. `image` is HxWx3 BGR uint8 (OpenCV convention)."""
        raw = self._impl.predict(image)
        h, w = image.shape[:2]
        return [_to_detection(r, w, h) for r in raw]


def _to_detection(raw: object, max_w: int, max_h: int) -> Detection:
    bbox_obj = getattr(raw, "bounding_box", None) or raw.bbox
    confidence = getattr(raw, "confidence", None)
    if confidence is None:
        confidence = raw.score
    bbox = BBox(
        int(bbox_obj.x1),
        int(bbox_obj.y1),
        int(bbox_obj.x2),
        int(bbox_obj.y2),
    ).clip(max_w, max_h)
    return Detection(bbox=bbox, confidence=float(confidence))
