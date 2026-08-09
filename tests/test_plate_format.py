"""Tests for plate parsing and the reference tables.

These cover the pure-Python half of the pipeline, so they run without
TensorFlow, OpenCV, or the trained weights:

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lpr.config import FALLBACK_OCR_CLASSES
from lpr.plate_data import (
    CATEGORIES,
    LATIN_TO_PERSIAN,
    LETTER_TEMPLATE_OFFSETS,
    PROVINCES,
)
from lpr.plate_format import (
    InvalidPlateError,
    parse_plate,
    to_latin_digits,
    to_persian_digits,
    try_parse_plate,
)


class TestDigitConversion(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(to_persian_digits("12345"), "۱۲۳۴۵")
        self.assertEqual(to_latin_digits("۱۲۳۴۵"), "12345")
        self.assertEqual(to_latin_digits(to_persian_digits("9078")), "9078")

    def test_leaves_letters_alone(self):
        self.assertEqual(to_persian_digits("12D34"), "۱۲D۳۴")


class TestParsePlate(unittest.TestCase):
    def test_disabled_driver_plate(self):
        plate = parse_plate("12PwD35373")
        self.assertEqual(plate.template, "۱۲♿۳۵۳۷۳")
        self.assertEqual(plate.letter, "♿")
        self.assertEqual(plate.digits, "۱۲۳۵۳۷۳")
        self.assertEqual(plate.province_code, "۷۳")
        self.assertEqual(plate.province, "فارس")
        self.assertEqual(plate.category, "معلولان و جانبازان")
        self.assertEqual(plate.to_latin(), "1235373")

    def test_multi_character_class_names(self):
        for raw, letter in (
            ("12Gh35373", "ق"),
            ("12Sin35373", "س"),
            ("12Taxi35373", "ت"),
            ("12PuV35373", "ع"),
            ("12Sad35373", "ص"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_plate(raw).letter, letter)

    def test_alef_is_spelled_out(self):
        # آ is written الف on a real plate.
        self.assertEqual(parse_plate("12A35373").letter, "الف")

    def test_str_is_the_template(self):
        self.assertEqual(str(parse_plate("12D35373")), "۱۲د۳۵۳۷۳")

    def test_unassigned_province_code_is_none(self):
        # 50 is not an issued province code.
        self.assertIsNone(parse_plate("12D35350").province)


class TestRejection(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(InvalidPlateError):
            parse_plate("")

    def test_no_letter(self):
        with self.assertRaisesRegex(InvalidPlateError, "no letter"):
            parse_plate("1234567")

    def test_wrong_digit_count(self):
        with self.assertRaisesRegex(InvalidPlateError, "expected 7 digits"):
            parse_plate("12D3537")
        with self.assertRaisesRegex(InvalidPlateError, "expected 7 digits"):
            parse_plate("12D353731")

    def test_letter_out_of_position(self):
        # Misordered OCR output must not be silently reassembled.
        with self.assertRaisesRegex(InvalidPlateError, "third position"):
            parse_plate("1235373D")

    def test_unknown_letter(self):
        with self.assertRaisesRegex(InvalidPlateError, "unknown plate letter"):
            parse_plate("12ZZ35373")

    def test_try_parse_returns_reason(self):
        plate, reason = try_parse_plate("12ZZ35373")
        self.assertIsNone(plate)
        self.assertIn("unknown plate letter", reason)

        plate, reason = try_parse_plate("12D35373")
        self.assertIsNotNone(plate)
        self.assertIsNone(reason)


class TestReferenceTables(unittest.TestCase):
    def setUp(self):
        self.letters = [c for c in FALLBACK_OCR_CLASSES if not c.isdigit()]

    def test_ocr_classes(self):
        self.assertEqual(len(FALLBACK_OCR_CLASSES), 28)
        self.assertEqual(len(self.letters), 18)

    def test_every_letter_class_maps_to_persian(self):
        for cls in self.letters:
            with self.subTest(cls=cls):
                self.assertIn(cls, LATIN_TO_PERSIAN)

    def test_every_letter_can_be_rendered_and_categorised(self):
        for latin, persian in LATIN_TO_PERSIAN.items():
            key = "الف" if persian == "آ" else persian
            with self.subTest(letter=latin):
                self.assertIn(key, LETTER_TEMPLATE_OFFSETS)
                self.assertIn(key, CATEGORIES)

    def test_every_letter_class_parses_end_to_end(self):
        for cls in self.letters:
            with self.subTest(cls=cls):
                plate = parse_plate(f"12{cls}35373")
                self.assertTrue(plate.category)
                self.assertTrue(plate.province)

    def test_province_codes_are_two_persian_digits(self):
        for code in PROVINCES:
            with self.subTest(code=code):
                self.assertEqual(len(code), 2)
                self.assertTrue(to_latin_digits(code).isdigit())


if __name__ == "__main__":
    unittest.main()
