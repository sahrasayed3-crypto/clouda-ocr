from __future__ import annotations

import unittest

from clouda_data.validation.image_quality import (
    detect_blank_from_pixels,
    validate_dimensions,
)
from clouda_data.validation.layout_preservation import validate_crop_ratio


class ValidationTests(unittest.TestCase):
    def test_blank_detection(self) -> None:
        self.assertFalse(detect_blank_from_pixels([255, 255, 255]).ok)
        self.assertTrue(detect_blank_from_pixels([255, 0, 255]).ok)

    def test_dimensions(self) -> None:
        self.assertTrue(validate_dimensions(10, 10).ok)
        self.assertFalse(validate_dimensions(0, 10).ok)

    def test_crop_ratio(self) -> None:
        validate_crop_ratio(0.01, 0.02)
        with self.assertRaises(ValueError):
            validate_crop_ratio(0.03, 0.02)


if __name__ == "__main__":
    unittest.main()
