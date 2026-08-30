import unittest

from pdfword.accuracy import (
    compute_accuracy_metrics,
    estimate_quality_components,
    evaluate_pages_against_references,
    normalize_for_accuracy,
)
from pdfword.models import PageResult


class TestAccuracy(unittest.TestCase):
    def test_normalization_keeps_semantic_arabic(self) -> None:
        normalized = normalize_for_accuracy("أحمد\nإلى\nمدرسة")
        self.assertEqual(normalized, "أحمد\nإلى\nمدرسة")

    def test_normalization_preserves_punctuation_letters_and_spacing(self) -> None:
        source = "آمنة  2026-075.\r\nإلى مدرسة"
        self.assertEqual(
            normalize_for_accuracy(source),
            "آمنة  2026-075.\nإلى مدرسة",
        )

    def test_metrics_perfect_match(self) -> None:
        metrics = compute_accuracy_metrics("هذا نص عربي", "هذا نص عربي")
        self.assertEqual(metrics["word_accuracy"], 100.0)
        self.assertEqual(metrics["char_accuracy"], 100.0)

    def test_metrics_detect_errors(self) -> None:
        metrics = compute_accuracy_metrics("هذا نص عربي", "هذا نص")
        self.assertLess(metrics["word_accuracy"], 100.0)
        self.assertLess(metrics["char_accuracy"], 100.0)

    def test_reference_accuracy_is_separate_from_heuristic_quality(self) -> None:
        page = PageResult(
            page_no=1,
            model_used="local:direct_pdf_text",
            markdown="hello brave world",
            text_quality_score=42.0,
            route_used="direct_pdf_text",
            requires_manual_review=True,
        )

        report = evaluate_pages_against_references([page], ["hello world"])

        self.assertGreater(report["pages"][0]["wer"], 0)
        self.assertEqual(report["pages"][0]["heuristic_quality_score"], 42.0)
        self.assertEqual(report["pages"][0]["engine_used"], "local:direct_pdf_text")
        self.assertEqual(report["pages"][0]["route_used"], "direct_pdf_text")
        self.assertLess(report["document"]["word_accuracy"], 100)

    def test_heuristic_quality_label_is_not_real_accuracy(self) -> None:
        parts = estimate_quality_components("Readable text without a reference")

        self.assertEqual(parts["label"], "درجة جودة تقديرية")
        self.assertNotIn("accuracy", parts["label"].lower())


if __name__ == "__main__":
    unittest.main()
