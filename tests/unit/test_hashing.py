from __future__ import annotations

import pytest

from anpr.storage.hashing import hash_plate


def test_hash_is_deterministic() -> None:
    assert hash_plate("34 ABC 1234", "secret") == hash_plate("34 ABC 1234", "secret")


def test_hash_normalizes_whitespace_and_case() -> None:
    pepper = "x" * 64
    h1 = hash_plate("34 ABC 1234", pepper)
    h2 = hash_plate("34abc1234", pepper)
    h3 = hash_plate("  34   ABC   1234  ", pepper)
    assert h1 == h2 == h3


def test_hash_differs_with_different_pepper() -> None:
    a = hash_plate("34ABC1234", "pepper-a")
    b = hash_plate("34ABC1234", "pepper-b")
    assert a != b


def test_hash_differs_with_different_plate() -> None:
    assert hash_plate("34ABC1234", "pepper") != hash_plate("34ABC1235", "pepper")


def test_hash_is_64_hex_chars() -> None:
    h = hash_plate("34ABC1234", "pepper")
    assert len(h) == 64
    int(h, 16)


def test_empty_pepper_rejected() -> None:
    with pytest.raises(ValueError, match="pepper"):
        hash_plate("34ABC1234", "")
