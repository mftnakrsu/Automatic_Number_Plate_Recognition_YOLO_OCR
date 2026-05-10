from __future__ import annotations

from anpr.postprocess.confusion import correct_confusions
from anpr.postprocess.turkish import parse_turkish_plate


def _first_valid(text: str) -> str | None:
    for c in correct_confusions(text):
        if parse_turkish_plate(c) is not None:
            return c
    return None


def test_already_valid_passes_through_first() -> None:
    candidates = correct_confusions("34ABC1234")
    assert candidates[0] == "34ABC1234"


def test_letter_o_in_province_becomes_zero() -> None:
    assert _first_valid("O6ABC1234") == "06ABC1234"


def test_letter_i_in_digit_zone_becomes_one() -> None:
    assert _first_valid("34 ABC I234") == "34ABC1234"


def test_digit_zero_in_letter_zone_becomes_o() -> None:
    # "34 0BC 1234" → "34 OBC 1234"
    assert _first_valid("340BC1234") == "34OBC1234"


def test_unrecoverable_returns_none() -> None:
    assert _first_valid("XXXXXXXX") is None
