"""Hypothesis property tests for the Turkish plate parser."""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from anpr.postprocess.turkish import PROVINCE_CODES, parse_turkish_plate

_PROVINCES = st.sampled_from(sorted(PROVINCE_CODES.keys()))
_LETTERS = st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=3)
_DIGITS = st.text(alphabet=string.digits, min_size=2, max_size=4)


@st.composite
def _valid_plate(draw: st.DrawFn) -> str:
    province = draw(_PROVINCES)
    letters = draw(_LETTERS)
    digits = draw(_DIGITS)
    while len(letters) + len(digits) > 7:
        digits = draw(_DIGITS)
    return f"{province}{letters}{digits}"


@given(_valid_plate())
@settings(max_examples=200)
def test_canonical_roundtrip(text: str) -> None:
    """Every regex-valid plate parses, and its canonical form re-parses identically."""
    parsed = parse_turkish_plate(text)
    assert parsed is not None, text
    assert parsed.province_code == text[:2]

    reparsed = parse_turkish_plate(parsed.canonical)
    assert reparsed is not None
    assert reparsed == parsed


@given(st.text(alphabet=string.ascii_letters + string.digits + " ", max_size=20))
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=200)
def test_random_strings_never_crash(text: str) -> None:
    """The parser must never raise — only return TurkishPlate or None."""
    result = parse_turkish_plate(text)
    if result is not None:
        assert result.province_code in PROVINCE_CODES
        assert result.canonical.count(" ") == 2  # "NN AAA NNN" shape


@given(_valid_plate())
@settings(max_examples=100)
def test_case_and_whitespace_normalization(text: str) -> None:
    """Mixed-case and arbitrary whitespace are normalized."""
    lower = text.lower()
    spaced = "  ".join(text)  # insert double-spaces between every char
    assert parse_turkish_plate(text) == parse_turkish_plate(lower)
    assert parse_turkish_plate(text) == parse_turkish_plate(spaced)
