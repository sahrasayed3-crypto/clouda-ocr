from __future__ import annotations

import unittest

from clouda_data.evaluation.cer import cer
from clouda_data.evaluation.wer import wer


class EvaluationTests(unittest.TestCase):
    def test_cer(self) -> None:
        self.assertEqual(cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(cer("abc", "axc"), 1 / 3)

    def test_wer(self) -> None:
        self.assertEqual(wer("هذا نص", "هذا نص"), 0.0)
        self.assertAlmostEqual(wer("هذا نص قصير", "هذا نص طويل"), 1 / 3)


if __name__ == "__main__":
    unittest.main()
