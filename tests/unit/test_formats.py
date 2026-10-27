from __future__ import annotations

import pytest

from anpr.postprocess.confusion import shape_candidates
from anpr.postprocess.formats import FORMATS, get_formats, parse_plate


@pytest.mark.parametrize(
    ("code", "text", "canonical", "letters", "digits"),
    [
        ("tr", "34ABC1234", "34 ABC 1234", "ABC", "1234"),
        ("de", "MAB1234", "MAB 1234", "MAB", "1234"),
        ("de", "B MW 1", "BMW 1", "BMW", "1"),
        ("de", "HHAB123E", "HHAB 123E", "HHAB", "123"),
        ("de", "TÜAB12", "TÜAB 12", "TÜAB", "12"),
        ("fr", "AB-123-CD", "AB-123-CD", "ABCD", "123"),
        ("fr", "gh456jk", "GH-456-JK", "GHJK", "456"),
        ("it", "AB 123 CD", "AB 123 CD", "ABCD", "123"),
        ("es", "1234BCD", "1234 BCD", "BCD", "1234"),
        ("be", "1ABC123", "1-ABC-123", "ABC", "1123"),
        ("pt", "AA00AA", "AA-00-AA", "AAAA", "00"),
        ("pt", "12-AB-34", "12-AB-34", "AB", "1234"),
        ("pt", "1234AB", "12-34-AB", "AB", "1234"),
    ],
)
def test_valid_plates(code: str, text: str, canonical: str, letters: str, digits: str) -> None:
    parsed = FORMATS[code].parse(text)
    assert parsed is not None
    assert parsed.country == code
    assert parsed.canonical == canonical
    assert parsed.letters == letters
    assert parsed.digits == digits


@pytest.mark.parametrize(
    ("code", "text"),
    [
        ("de", "M1234"),  # one letter is too few
        ("de", "ABCDEF12"),  # six letters is too many
        ("de", "MAB0123"),  # digits can't start with 0
        ("de", "MAB12345"),  # five digits
        ("de", "ABCDE1234"),  # nine characters
        ("fr", "AB000CD"),  # 000 isn't issued
        ("fr", "SS123AB"),  # SS isn't issued
        ("fr", "AI123CD"),  # no I
        ("it", "AQ123CD"),  # no Q in Italy
        ("es", "1234AEI"),  # vowels
        ("es", "123BCD"),
        ("be", "0ABC123"),
        ("pt", "ABCDEF"),
        ("tr", "AB123CD"),
    ],
)
def test_invalid_plates(code: str, text: str) -> None:
    assert FORMATS[code].parse(text) is None


def test_q_is_fine_in_france_but_not_italy() -> None:
    assert FORMATS["fr"].parse("AQ123CD") is not None
    assert FORMATS["it"].parse("AQ123CD") is None


def test_only_turkish_plates_have_a_province() -> None:
    tr = FORMATS["tr"].parse("06AA42")
    assert tr is not None
    assert (tr.province_code, tr.province_name) == ("06", "Ankara")
    de = FORMATS["de"].parse("MAB1234")
    assert de is not None
    assert de.province_code is None
    assert de.province_name is None


@pytest.mark.parametrize(
    ("codes", "ocr", "country", "fixed"),
    [
        (["fr"], "A8123CD", "fr", "AB123CD"),  # 8 in a letter slot
        (["es"], "12B4BCD", "es", "1284BCD"),  # B in a digit slot
        (["be"], "IABC1Z3", "be", "1ABC123"),
        (["de"], "MA81234", "de", "MAB1234"),
        (["it"], "AB1Z3CD", "it", "AB123CD"),
        (["tr"], "O6AA42", "tr", "06AA42"),
    ],
)
def test_lookalike_characters_are_fixed(
    codes: list[str], ocr: str, country: str, fixed: str
) -> None:
    result = parse_plate(ocr, get_formats(codes))
    assert result is not None
    parsed, used = result
    assert parsed.country == country
    assert used == fixed


def test_exact_read_beats_a_corrected_one() -> None:
    # Valid in Spain as read; France would need two letters swapped in.
    result = parse_plate("1234BCD", get_formats(["fr", "es"]))
    assert result is not None
    assert result[0].country == "es"


def test_order_breaks_ties() -> None:
    fr_first = parse_plate("AB123CD", get_formats(["fr", "it"]))
    it_first = parse_plate("AB123CD", get_formats(["it", "fr"]))
    assert fr_first is not None
    assert it_first is not None
    assert fr_first[0].country == "fr"
    assert it_first[0].country == "it"


def test_default_is_turkish_only() -> None:
    assert parse_plate("AB123CD") is None
    result = parse_plate("34ABC1234")
    assert result is not None
    assert result[0].country == "tr"


def test_unknown_code() -> None:
    with pytest.raises(KeyError):
        get_formats(["xx"])


def test_shape_candidates() -> None:
    assert shape_candidates("A8-1Z3", ["LLDD"]) == ["A81Z3"]
    assert shape_candidates("A81Z3", ["LLDDD"]) == ["A81Z3", "AB123"]
    assert shape_candidates("A81Z3E", ["LLDDD*"]) == ["A81Z3E", "AB123E"]
