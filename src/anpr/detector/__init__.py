"""Plate detection adapters."""

from anpr.detector.base import Detection, Detector
from anpr.detector.fast_alpr import FastAlprDetector

__all__ = ["Detection", "Detector", "FastAlprDetector"]
