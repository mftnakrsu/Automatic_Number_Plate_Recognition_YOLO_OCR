"""HMAC-SHA256 plate hashing for KVKK/GDPR-compliant persistence.

Plates are personal data under both GDPR (Reg. EU 2016/679 art. 4(1)) and
KVKK (Law 6698 art. 3(d) — Türkiye). Storing raw plate text in a database
creates unnecessary regulatory exposure. We hash with HMAC-SHA256 plus a
per-deployment pepper instead: the hash is stable across restarts of the
same deployment, deterministic for matching (same plate → same hash), but
not reversible without the pepper.
"""

from __future__ import annotations

import hashlib
import hmac


def hash_plate(plate_text: str, pepper: str) -> str:
    """Hex-encoded HMAC-SHA256 of the normalized plate text.

    Whitespace is stripped and casing folded before hashing so that
    `"34 abc 1234"`, `"34ABC1234"`, and `"  34   ABC   1234"` all map to the
    same hash.
    """
    if not pepper:
        raise ValueError("pepper must not be empty")
    normalized = "".join(plate_text.split()).upper().encode("utf-8")
    return hmac.new(pepper.encode("utf-8"), normalized, hashlib.sha256).hexdigest()
