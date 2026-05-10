"""Detection storage layer (SQLModel + HMAC plate hashing)."""

from anpr.storage.hashing import hash_plate
from anpr.storage.models import Detection
from anpr.storage.repository import DetectionRepository, init_db

__all__ = ["Detection", "DetectionRepository", "hash_plate", "init_db"]
