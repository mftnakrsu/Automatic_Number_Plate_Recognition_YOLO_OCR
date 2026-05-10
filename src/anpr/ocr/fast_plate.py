"""fast-plate-ocr 1.1+ adapter (default: cct-s-v2-global-model)."""

from __future__ import annotations

import numpy as np

from anpr.ocr.base import OcrResult


class FastPlateOcr:
    """Wraps fast-plate-ocr's `LicensePlateRecognizer`.

    The 2026 default is `cct-s-v2-global-model` — trained on ~220k plates from
    65+ countries with sub-millisecond latency on GPU. For best Turkish-plate
    quality you can fine-tune and pass a custom model name or path.
    """

    def __init__(
        self,
        model_name: str = "cct-s-v2-global-model",
        device: str = "auto",
    ) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        self._impl = LicensePlateRecognizer(model_name, device=device)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def read(self, plate_crop: np.ndarray) -> OcrResult | None:
        out = self._impl.run(plate_crop, return_confidence=True)
        return _coerce(out)

    def read_batch(self, plate_crops: list[np.ndarray]) -> list[OcrResult | None]:
        if not plate_crops:
            return []
        return [self.read(c) for c in plate_crops]


def _coerce(out: object) -> OcrResult | None:
    """fast-plate-ocr v1.1+ returns (list[str], list[list[float]])."""
    if not isinstance(out, tuple) or len(out) != 2:
        return None
    texts, confs = out
    if not texts:
        return None
    text = str(texts[0]).strip()
    if not text:
        return None
    char_confs = tuple(float(c) for c in confs[0]) if confs and confs[0] is not None else ()
    agg = sum(char_confs) / len(char_confs) if char_confs else 1.0
    return OcrResult(text=text, confidence=agg, char_confidences=char_confs)
