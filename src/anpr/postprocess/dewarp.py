"""Plate-region dewarp / normalization.

A full implementation would regress the four plate corners and apply
`cv2.getPerspectiveTransform` (see arXiv:2104.11649). For v1 we only normalize
height to a fixed value while preserving aspect ratio — fast-plate-ocr does
its own internal preprocessing, so this is a placeholder integration point.
"""

from __future__ import annotations

import cv2
import numpy as np


def dewarp(plate_crop: np.ndarray, *, target_height: int = 64) -> np.ndarray:
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return plate_crop
    if h == target_height:
        return plate_crop
    target_w = max(1, round(w * target_height / h))
    return cv2.resize(plate_crop, (target_w, target_height), interpolation=cv2.INTER_CUBIC)
