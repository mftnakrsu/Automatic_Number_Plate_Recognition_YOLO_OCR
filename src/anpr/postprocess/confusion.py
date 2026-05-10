"""Position-aware OCR confusion-pair correction for Turkish plates.

OCR engines often confuse visually similar character pairs (0/O, 1/I/L, 5/S,
8/B, 2/Z, 6/G, 0/D/Q). For Turkish plates the format pins which positions
must be digits vs letters:

    [digit][digit][letter][letter?][letter?][digit][digit][digit?][digit?]

Knowing this lets us correct OCR confusions deterministically. We don't know
the letter-block length up front, so we enumerate the three valid splits
(1, 2, 3 letters) and emit a candidate for each.
"""

from __future__ import annotations

# In a "letter must be here" position, digits become the visually-closest letter.
_DIGIT_TO_LETTER = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }
)

# In a "digit must be here" position, letters become the visually-closest digit.
_LETTER_TO_DIGIT = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }
)


def correct_confusions(text: str) -> list[str]:
    """Return up to four candidate strings: original + 1/2/3-letter splits.

    Caller iterates and returns the first one that passes `parse_turkish_plate`.
    """
    cleaned = "".join(text.split()).upper()
    candidates: list[str] = [cleaned]
    if len(cleaned) < 5 or len(cleaned) > 9:
        return candidates

    head = cleaned[:2].translate(_LETTER_TO_DIGIT)
    tail = cleaned[2:]
    n = len(tail)

    for letters_len in (1, 2, 3):
        digits_len = n - letters_len
        if not 2 <= digits_len <= 4:
            continue
        letters = tail[:letters_len].translate(_DIGIT_TO_LETTER)
        digits = tail[letters_len:].translate(_LETTER_TO_DIGIT)
        candidates.append(head + letters + digits)

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
