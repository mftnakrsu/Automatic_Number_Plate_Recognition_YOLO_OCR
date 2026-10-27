"""Detection storage layer (SQLModel + HMAC plate hashing)."""

from anpr.storage.hashing import hash_plate
from anpr.storage.models import Detection, WatchlistEntry
from anpr.storage.repository import DetectionRepository, WatchlistRepository, init_db

__all__ = [
    "Detection",
    "DetectionRepository",
    "WatchlistEntry",
    "WatchlistRepository",
    "hash_plate",
    "init_db",
]
