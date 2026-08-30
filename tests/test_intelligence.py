import io
import unittest

from pypdf import PdfWriter

from pdfword.intelligence import (
    analyze_document_intelligence,
    analyze_layout_intelligence,
)
from pdfword.models import PageResult


class TestIntelligenceHelpers(unittest.TestCase):
    def test_document_intelligence_blank_pdf(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        out = io.BytesIO()
        writer.write(out)
        payload = out.getvalue()
        meta = analyze_document_intelligence(payload, pages=[1])
        self.assertEqual(meta.page_count, 1)
        self.assertGreaterEqual(meta.extractable_ratio, 0.0)
        self.assertLessEqual(meta.extractable_ratio, 1.0)

    def test_layout_summary_detects_table_hint(self) -> None:
        info = analyze_layout_intelligence("Title:\nA | B | C\n1 | 2 | 3")
        self.assertGreaterEqual(info.tables, 1)

    def test_page_result_defaults(self) -> None:
        row = PageResult(page_no=1, model_used="x", markdown="m")
        self.assertFalse(row.requires_manual_review)
        self.assertIsNone(row.quality_score)


if __name__ == "__main__":
    unittest.main()
