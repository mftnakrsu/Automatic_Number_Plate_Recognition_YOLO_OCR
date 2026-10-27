"""Plate formats by country, chosen with `ANPR_PLATE_FORMATS`.

Each format checks a read against the country's current plate layout and
gives it a canonical spelling. The OCR model reads the characters without
separators, so a format works on the bare string ("AB123CD") and puts the
separators back itself ("AB-123-CD").

When a read fits no enabled format as it stands, each format retries it with
look-alike characters swapped for the letter or digit its layout expects in
that position (see `anpr.postprocess.confusion`). An exact read in any
format wins over a corrected one.

Only Türkiye's plates name a region (the province). The other layouts carry
no region the string alone can reveal: a German district code can be one to
three letters, and telling "M AB 1234" from "MA B 1234" needs the list of
district codes, which is not built in. German plates are therefore given as
letters and digits ("MAB 1234").
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from anpr.postprocess.confusion import correct_confusions, shape_candidates
from anpr.postprocess.turkish import parse_turkish_plate


@dataclass(slots=True, frozen=True)
class ParsedPlate:
    country: str
    canonical: str
    letters: str
    digits: str
    # Türkiye only; None for the other countries.
    province_code: str | None = None
    province_name: str | None = None


@dataclass(slots=True, frozen=True)
class PlateFormat:
    code: str
    name: str
    parse: Callable[[str], ParsedPlate | None]
    candidates: Callable[[str], list[str]]


def _clean(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalnum())


def _parse_tr(text: str) -> ParsedPlate | None:
    p = parse_turkish_plate(text)
    if p is None:
        return None
    return ParsedPlate("tr", p.canonical, p.letters, p.digits, p.province_code, p.province_name)


# --- Germany ------------------------------------------------------------------
# District code (1-3 letters, umlauts allowed) + 1-2 letters + 1-4 digits
# without a leading zero, at most 8 together, then E (electric) or H (historic).

_DE = re.compile(r"(?P<letters>[A-ZÄÖÜ]{2,5})(?P<digits>[1-9]\d{0,3})(?P<suffix>[EH]?)")
_DE_MAX = 8


def _parse_de(text: str) -> ParsedPlate | None:
    m = _DE.fullmatch(_clean(text))
    if m is None or len(m["letters"]) + len(m["digits"]) > _DE_MAX:
        return None
    letters, digits, suffix = m["letters"], m["digits"], m["suffix"]
    return ParsedPlate("de", f"{letters} {digits}{suffix}", letters, digits)


def _shapes_de(n: int) -> list[str]:
    shapes = []
    for length, suffix in ((n, ""), (n - 1, "*")):
        for letters in range(2, 6):
            digits = length - letters
            if 1 <= digits <= 4 and letters + digits <= _DE_MAX:
                shapes.append("L" * letters + "D" * digits + suffix)
    return shapes


# --- France (SIV, since 2009): AB-123-CD -------------------------------------
# No I, O or U; SS is not issued; 000 is not issued.

_FR_L = "[A-HJ-NP-TV-Z]"
_FR = re.compile(rf"(?P<a>{_FR_L}{{2}})(?P<n>\d{{3}})(?P<b>{_FR_L}{{2}})")


def _parse_fr(text: str) -> ParsedPlate | None:
    m = _FR.fullmatch(_clean(text))
    if m is None or m["n"] == "000" or "SS" in (m["a"], m["b"]):
        return None
    return ParsedPlate("fr", f"{m['a']}-{m['n']}-{m['b']}", m["a"] + m["b"], m["n"])


# --- Italy (since 1994): AB 123 CD --------------------------------------------
# No I, O, Q or U.

_IT_L = "[A-HJ-NPR-TV-Z]"
_IT = re.compile(rf"(?P<a>{_IT_L}{{2}})(?P<n>\d{{3}})(?P<b>{_IT_L}{{2}})")


def _parse_it(text: str) -> ParsedPlate | None:
    m = _IT.fullmatch(_clean(text))
    if m is None:
        return None
    return ParsedPlate("it", f"{m['a']} {m['n']} {m['b']}", m["a"] + m["b"], m["n"])


# --- Spain (since 2000): 1234 BCD ---------------------------------------------
# Consonants only, without Ñ and Q.

_ES = re.compile(r"(?P<n>\d{4})(?P<l>[BCDFGHJKLMNPRSTVWXYZ]{3})")


def _parse_es(text: str) -> ParsedPlate | None:
    m = _ES.fullmatch(_clean(text))
    if m is None:
        return None
    return ParsedPlate("es", f"{m['n']} {m['l']}", m["l"], m["n"])


# --- Belgium (since 2010): 1-ABC-123 ------------------------------------------

_BE = re.compile(r"(?P<d>[1-9])(?P<l>[A-Z]{3})(?P<n>\d{3})")


def _parse_be(text: str) -> ParsedPlate | None:
    m = _BE.fullmatch(_clean(text))
    if m is None:
        return None
    return ParsedPlate("be", f"{m['d']}-{m['l']}-{m['n']}", m["l"], m["d"] + m["n"])


# --- Portugal: AA-00-AA (2020), 00-AA-00 (2005), 00-00-AA (1992), AA-00-00 ----

_PT_SHAPES = ("LLDDLL", "DDLLDD", "DDDDLL", "LLDDDD")
_PT = re.compile(r"[A-Z]{2}\d{2}[A-Z]{2}|\d{2}[A-Z]{2}\d{2}|\d{4}[A-Z]{2}|[A-Z]{2}\d{4}")


def _parse_pt(text: str) -> ParsedPlate | None:
    s = _clean(text)
    if _PT.fullmatch(s) is None:
        return None
    letters = "".join(c for c in s if c.isalpha())
    digits = "".join(c for c in s if c.isdigit())
    return ParsedPlate("pt", f"{s[:2]}-{s[2:4]}-{s[4:]}", letters, digits)


def _fixed(*shapes: str) -> Callable[[str], list[str]]:
    def candidates(text: str) -> list[str]:
        return shape_candidates(text, [s for s in shapes if len(s) == len(_clean(text))])

    return candidates


def _de_candidates(text: str) -> list[str]:
    return shape_candidates(text, _shapes_de(len(_clean(text))))


FORMATS: dict[str, PlateFormat] = {
    f.code: f
    for f in (
        PlateFormat("tr", "Türkiye", _parse_tr, correct_confusions),
        PlateFormat("de", "Germany", _parse_de, _de_candidates),
        PlateFormat("fr", "France", _parse_fr, _fixed("LLDDDLL")),
        PlateFormat("it", "Italy", _parse_it, _fixed("LLDDDLL")),
        PlateFormat("es", "Spain", _parse_es, _fixed("DDDDLLL")),
        PlateFormat("be", "Belgium", _parse_be, _fixed("DLLLDDD")),
        PlateFormat("pt", "Portugal", _parse_pt, _fixed(*_PT_SHAPES)),
    )
}

DEFAULT_FORMATS: tuple[PlateFormat, ...] = (FORMATS["tr"],)


def get_formats(codes: Iterable[str]) -> tuple[PlateFormat, ...]:
    """Look up formats by country code, in the given order. Unknown codes raise `KeyError`."""
    return tuple(FORMATS[c] for c in codes)


def parse_plate(
    text: str, formats: Sequence[PlateFormat] = DEFAULT_FORMATS
) -> tuple[ParsedPlate, str] | None:
    """Parse `text` with the first format that accepts it.

    Returns the plate and the string that parsed, which differs from `text`
    when look-alike characters had to be swapped. Exact reads are tried in
    every format before any correction is.
    """
    for fmt in formats:
        parsed = fmt.parse(text)
        if parsed is not None:
            return parsed, text
    for fmt in formats:
        for candidate in fmt.candidates(text):
            parsed = fmt.parse(candidate)
            if parsed is not None:
                return parsed, candidate
    return None
