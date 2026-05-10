"""fast-plate-ocr 1.1+ adapter (default: cct-s-v2-global-model)."""

from __future__ import annotations

from typing import Any

import numpy as np

from anpr.ocr.base import OcrResult


class FastPlateOcr:
    """Wraps fast-plate-ocr's `LicensePlateRecognizer`.

    The 2026 default is `cct-s-v2-global-model` — trained on ~220k plates from
    65+ countries with sub-millisecond GPU latency. The recogniser returns
    per-character probabilities which we average to a single confidence and
    expose for downstream confusion correction / temporal voting.
    """

    def __init__(
        self,
        model_name: str = "cct-s-v2-global-model",
        device: str = "auto",
    ) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        self._impl = LicensePlateRecognizer(
            hub_ocr_model=model_name,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        results = self._impl.run(plate_crop, return_confidence=True)
        if not results:
            return None
        return _to_ocr_result(results[0])

    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]:
        if not plate_crops:
            return []
        results = self._impl.run(plate_crops, return_confidence=True)
        out: list[OcrResult | None] = []
        for i, _ in enumerate(plate_crops):
            r = results[i] if i < len(results) else None
            out.append(_to_ocr_result(r) if r is not None else None)
        return out


def _to_ocr_result(prediction: Any) -> OcrResult | None:
    text = getattr(prediction, "plate", None)
    if not text:
        return None
    char_probs = getattr(prediction, "char_probs", None)
    if char_probs is None or len(char_probs) == 0:
        return OcrResult(text=str(text), confidence=1.0)
    char_confidences = tuple(float(p) for p in char_probs)
    avg = sum(char_confidences) / len(char_confidences) if char_confidences else 1.0
    return OcrResult(
        text=str(text),
        confidence=avg,
        char_confidences=char_confidences,
    )
