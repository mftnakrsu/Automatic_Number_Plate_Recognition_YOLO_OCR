"""Dependency-injected singletons for the API.

Heavy resources (detector, OCR reader, DB engine) are loaded once in the
lifespan event and cached in module globals. FastAPI dependencies read from
these globals so each request reuses the same instance — model files are
huge, reloading per request is unacceptable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from anpr.config import Settings, get_settings

if TYPE_CHECKING:
    from anpr.detector.base import Detector
    from anpr.ocr.base import PlateReader
    from anpr.storage.repository import DetectionRepository


@lru_cache(maxsize=1)
def _settings_singleton() -> Settings:
    return get_settings()


def app_settings() -> Settings:
    return _settings_singleton()


_DETECTOR: Detector | None = None
_READER: PlateReader | None = None
_REPO: DetectionRepository | None = None


def set_detector(d: Detector) -> None:
    global _DETECTOR
    _DETECTOR = d


def get_detector() -> Detector:
    if _DETECTOR is None:
        raise RuntimeError("Detector not initialized — lifespan event missing?")
    return _DETECTOR


def set_reader(r: PlateReader) -> None:
    global _READER
    _READER = r


def get_reader() -> PlateReader:
    if _READER is None:
        raise RuntimeError("Reader not initialized — lifespan event missing?")
    return _READER


def set_repo(r: DetectionRepository) -> None:
    global _REPO
    _REPO = r


def get_repo() -> DetectionRepository:
    if _REPO is None:
        raise RuntimeError("DetectionRepository not initialized — lifespan event missing?")
    return _REPO
