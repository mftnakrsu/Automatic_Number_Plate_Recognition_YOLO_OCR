"""OCR adapters."""

from anpr.ocr.base import OcrResult, PlateReader
from anpr.ocr.fast_plate import FastPlateOcr

__all__ = ["FastPlateOcr", "OcrResult", "PlateReader"]
