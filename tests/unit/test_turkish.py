from __future__ import annotations

import pytest

from anpr.postprocess.turkish import PROVINCE_CODES, parse_turkish_plate


@pytest.mark.parametrize(
    ("text", "expected_canonical"),
    [
        ("34ABC1234", "34 ABC 1234"),
        ("34 ABC 1234", "34 ABC 1234"),
        ("34abc1234", "34 ABC 1234"),
        ("06AA42", "06 AA 42"),
        ("01A1234", "01 A 1234"),
        ("81 DZ 99", "81 DZ 99"),
        ("3 4 ABC 1234", "34 ABC 1234"),
    ],
)
def test_parses_valid_plates(text: str, expected_canonical: str) -> None:
    parsed = parse_turkish_plate(text)
    assert parsed is not None
    assert parsed.canonical == expected_canonical


@pytest.mark.parametrize(
    "text",
    [
        "82ABC123",  # province > 81
        "00ABC123",  # province 00
        "99XYZ999",  # province 99
        "",  # empty
        "HELLO",  # no digits
        "12345678",  # all digits
        "ABCDEFGH",  # all letters
        "34A4",  # too short
        "34ABCDE1234",  # too long
    ],
)
def test_rejects_invalid_plates(text: str) -> None:
    assert parse_turkish_plate(text) is None


def test_all_81_provinces_present() -> None:
    expected = {f"{i:02d}" for i in range(1, 82)}
    assert set(PROVINCE_CODES.keys()) == expected
    assert all(name for name in PROVINCE_CODES.values())


def test_istanbul_lookup() -> None:
    parsed = parse_turkish_plate("34 XYZ 123")
    assert parsed is not None
    assert parsed.province_name == "İstanbul"
