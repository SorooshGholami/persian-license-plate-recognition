"""Turn raw OCR output into a validated, human-readable Iranian plate.

The OCR network emits Latin class names, e.g. ``"12PwD35373"``. An Iranian
civilian plate is two digits, one letter, three digits and a two-digit
province code. This module validates that shape and resolves the province and
vehicle category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lpr.plate_data import CATEGORIES, LATIN_TO_PERSIAN, PROVINCES

#: Iranian plates carry seven digits in total.
PLATE_DIGIT_COUNT = 7

_EN_TO_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_FA_TO_EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_NON_DIGIT = re.compile(r"\D")

#: A plate reads as two digits, the letter, three digits, then the two-digit
#: province code. Anchoring the letter's position matters: if segmentation
#: returns the characters out of order, the text must be rejected rather than
#: silently reassembled into a different -- but valid-looking -- plate.
_PLATE_PATTERN = re.compile(r"(?P<prefix>\d{2})(?P<letter>[A-Za-z]+)(?P<suffix>\d{5})")


class InvalidPlateError(ValueError):
    """Raised when OCR output cannot be a valid Iranian plate."""


@dataclass(frozen=True)
class PlateInfo:
    """A successfully parsed plate."""

    raw: str
    """Unmodified OCR output, e.g. ``"12PwD35373"``."""

    letter: str
    """Persian letter, e.g. ``"د"`` (``"الف"`` for آ, ``"♿"`` for disabled)."""

    digits: str
    """All seven digits, in Persian, e.g. ``"۱۲۳۵۳۷۳"``."""

    template: str
    """Display form: ``۱۲`` + letter + ``۳۵۳`` + ``۷۳``."""

    province: str | None
    """Issuing province, or ``None`` if the code is unassigned."""

    category: str | None
    """Vehicle category, or ``None`` if the letter is unknown."""

    @property
    def province_code(self) -> str:
        """The trailing two-digit province code, in Persian digits."""
        return self.digits[5:7]

    def to_latin(self) -> str:
        """The digits rendered with Latin numerals, handy for searching."""
        return self.digits.translate(_FA_TO_EN_DIGITS)

    def __str__(self) -> str:
        return self.template


def to_persian_digits(text: str) -> str:
    """Convert any Latin digits in ``text`` to Persian digits."""
    return text.translate(_EN_TO_FA_DIGITS)


def to_latin_digits(text: str) -> str:
    """Convert any Persian digits in ``text`` to Latin digits."""
    return text.translate(_FA_TO_EN_DIGITS)


def parse_plate(raw: str) -> PlateInfo:
    """Parse raw OCR output into a :class:`PlateInfo`.

    Args:
        raw: The concatenated OCR class names, e.g. ``"12PwD35373"``.

    Returns:
        The parsed plate.

    Raises:
        InvalidPlateError: If the text cannot be a valid Iranian plate. The
            message explains which rule failed, so callers can log it.
    """
    if not raw:
        raise InvalidPlateError("OCR produced no characters")

    match = _PLATE_PATTERN.fullmatch(raw)
    if match is None:
        # Work out which rule broke so the caller can log something useful.
        digits = _NON_DIGIT.sub("", raw)
        if not _NON_DIGIT.findall(raw):
            raise InvalidPlateError(f"no letter found in {raw!r}")
        if len(digits) != PLATE_DIGIT_COUNT:
            raise InvalidPlateError(
                f"expected {PLATE_DIGIT_COUNT} digits, read {len(digits)} in {raw!r}"
            )
        raise InvalidPlateError(
            f"letter is not in the third position in {raw!r}"
        )

    letter_token = match.group("letter")
    digits = match.group("prefix") + match.group("suffix")

    letter = LATIN_TO_PERSIAN.get(letter_token)
    if letter is None:
        raise InvalidPlateError(f"unknown plate letter {letter_token!r}")

    # آ is written out as الف on a plate.
    if letter == "آ":
        letter = "الف"

    persian_digits = to_persian_digits(digits)
    province_code = persian_digits[5:7]
    template = (
        persian_digits[0:2] + letter + persian_digits[2:5] + province_code
    )

    return PlateInfo(
        raw=raw,
        letter=letter,
        digits=persian_digits,
        template=template,
        province=PROVINCES.get(province_code),
        category=CATEGORIES.get(letter),
    )


def try_parse_plate(raw: str) -> tuple[PlateInfo | None, str | None]:
    """Non-raising variant of :func:`parse_plate`.

    Returns:
        ``(plate, None)`` on success, or ``(None, reason)`` on failure.
    """
    try:
        return parse_plate(raw), None
    except InvalidPlateError as exc:
        return None, str(exc)
