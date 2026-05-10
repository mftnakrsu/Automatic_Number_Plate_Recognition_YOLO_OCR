"""Turkish license plate validation and parsing.

Format spec (per Wikipedia "Vehicle registration plates of Turkey"):

    NN [A]{1-3} [N]{2-4}    where the total of letters + digits is 4-6.

`NN` is the province code (01-81); the remainder follows three patterns:

    NN A NNNN     — 1 letter, 4 digits (older format, still valid)
    NN AA NNN/NNNN
    NN AAA NN/NNN

Total plate length (excluding spaces) is 6, 7, or 8 characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# fmt: off
PROVINCE_CODES: dict[str, str] = {
    "01": "Adana",          "02": "Adıyaman",       "03": "Afyonkarahisar",
    "04": "Ağrı",           "05": "Amasya",         "06": "Ankara",
    "07": "Antalya",        "08": "Artvin",         "09": "Aydın",
    "10": "Balıkesir",      "11": "Bilecik",        "12": "Bingöl",
    "13": "Bitlis",         "14": "Bolu",           "15": "Burdur",
    "16": "Bursa",          "17": "Çanakkale",      "18": "Çankırı",
    "19": "Çorum",          "20": "Denizli",        "21": "Diyarbakır",
    "22": "Edirne",         "23": "Elazığ",         "24": "Erzincan",
    "25": "Erzurum",        "26": "Eskişehir",      "27": "Gaziantep",
    "28": "Giresun",        "29": "Gümüşhane",      "30": "Hakkâri",
    "31": "Hatay",          "32": "Isparta",        "33": "Mersin",
    "34": "İstanbul",       "35": "İzmir",          "36": "Kars",
    "37": "Kastamonu",      "38": "Kayseri",        "39": "Kırklareli",
    "40": "Kırşehir",       "41": "Kocaeli",        "42": "Konya",
    "43": "Kütahya",        "44": "Malatya",        "45": "Manisa",
    "46": "Kahramanmaraş",  "47": "Mardin",         "48": "Muğla",
    "49": "Muş",            "50": "Nevşehir",       "51": "Niğde",
    "52": "Ordu",           "53": "Rize",           "54": "Sakarya",
    "55": "Samsun",         "56": "Siirt",          "57": "Sinop",
    "58": "Sivas",          "59": "Tekirdağ",       "60": "Tokat",
    "61": "Trabzon",        "62": "Tunceli",        "63": "Şanlıurfa",
    "64": "Uşak",           "65": "Van",            "66": "Yozgat",
    "67": "Zonguldak",      "68": "Aksaray",        "69": "Bayburt",
    "70": "Karaman",        "71": "Kırıkkale",      "72": "Batman",
    "73": "Şırnak",         "74": "Bartın",         "75": "Ardahan",
    "76": "Iğdır",          "77": "Yalova",         "78": "Karabük",
    "79": "Kilis",          "80": "Osmaniye",       "81": "Düzce",
}
# fmt: on

_PLATE_PATTERN = re.compile(
    r"^"
    r"(?P<province>0[1-9]|[1-7]\d|8[01])"
    r"(?P<letters>[A-Z]{1,3})"
    r"(?P<digits>\d{2,4})"
    r"$"
)

_MIN_TOTAL_LEN = 5
_MAX_TOTAL_LEN = 9


@dataclass(slots=True, frozen=True)
class TurkishPlate:
    province_code: str
    province_name: str
    letters: str
    digits: str
    canonical: str


def parse_turkish_plate(text: str) -> TurkishPlate | None:
    """Validate and parse a Turkish plate. Whitespace is ignored, case folded."""
    cleaned = "".join(text.split()).upper()
    if not _MIN_TOTAL_LEN <= len(cleaned) <= _MAX_TOTAL_LEN:
        return None
    m = _PLATE_PATTERN.match(cleaned)
    if m is None:
        return None
    province = m["province"]
    name = PROVINCE_CODES.get(province)
    if name is None:
        return None
    letters = m["letters"]
    digits = m["digits"]
    return TurkishPlate(
        province_code=province,
        province_name=name,
        letters=letters,
        digits=digits,
        canonical=f"{province} {letters} {digits}",
    )
