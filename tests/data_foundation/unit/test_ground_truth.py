from __future__ import annotations

import unittest

from clouda_data.ground_truth.checksums import sha256_text
from clouda_data.ground_truth.normalization import normalize_for_comparison
from clouda_data.ground_truth.validators import require_text_preserved


class GroundTruthTests(unittest.TestCase):
    def test_arabic_comparison_normalization(self) -> None:
        self.assertEqual(normalize_for_comparison("أحْمــد  ١"), "احمد ١")
        self.assertEqual(normalize_for_comparison("على"), "علي")

    def test_checksum_preservation(self) -> None:
        text = "نص عربي"
        self.assertEqual(sha256_text(text), sha256_text(text))
        require_text_preserved(text, text)
        with self.assertRaises(ValueError):
            require_text_preserved(text, text + "!")


if __name__ == "__main__":
    unittest.main()
