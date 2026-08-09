"""Draw a recognised plate onto a clean plate template.

Persian needs both reshaping (contextual letter forms) and bidi reordering
before PIL can draw it, which is why the two extra dependencies exist.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lpr import config
from lpr.plate_data import DIGITS_TEMPLATE_OFFSET, LETTER_TEMPLATE_OFFSETS
from lpr.plate_format import PlateInfo

_LETTER_FONT_SIZE = 100
_EMOJI_FONT_SIZE = 64
#: Fallback position for letters missing from the offsets table.
_DEFAULT_LETTER_OFFSET = (180, 20)


def _shape_persian(text: str) -> str:
    """Apply Arabic shaping and bidi reordering so PIL renders text correctly."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Rendering Persian text needs 'arabic-reshaper' and 'python-bidi'. "
            "Install them with: pip install arabic-reshaper python-bidi"
        ) from exc

    return get_display(arabic_reshaper.reshape(text))


def _format_digits(digits: str) -> str:
    """Space the seven digits into the plate's visual grouping."""
    return f"{digits[0:2]}      {digits[2:5]} {digits[5:7]}"


def render_plate(
    plate: PlateInfo,
    template_path: Path | str | None = None,
    font_path: Path | str | None = None,
    emoji_font_path: Path | str | None = None,
) -> np.ndarray:
    """Render ``plate`` onto the blank template.

    Args:
        plate: The parsed plate to draw.
        template_path: Blank plate image; defaults to the bundled template.
        font_path: Persian font; defaults to the bundled one.
        emoji_font_path: Font used for the ♿ disabled-driver symbol.

    Returns:
        The rendered plate as a BGR array, ready for ``cv2.imwrite``.
    """
    template_path = Path(template_path or config.PLATE_TEMPLATE)
    font_path = Path(font_path or config.FONT_FARSI)
    emoji_font_path = Path(emoji_font_path or config.FONT_EMOJI)

    image = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    digit_font = ImageFont.truetype(str(font_path), _LETTER_FONT_SIZE)
    is_emoji = plate.letter == "♿"
    letter_font = (
        ImageFont.truetype(str(emoji_font_path), _EMOJI_FONT_SIZE, encoding="unic")
        if is_emoji
        else digit_font
    )

    offset = LETTER_TEMPLATE_OFFSETS.get(plate.letter, _DEFAULT_LETTER_OFFSET)
    draw.text(offset, _shape_persian(plate.letter), (0, 0, 0), font=letter_font)
    draw.text(
        DIGITS_TEMPLATE_OFFSET,
        _format_digits(plate.digits),
        (0, 0, 0),
        font=digit_font,
    )

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
