import io
import unittest

from pypdf import PdfWriter

from pdfword.document_analysis import _estimate_category, analyze_pdf_document


class TestDocumentAnalysis(unittest.TestCase):
    def test_estimate_category_low_text(self) -> None:
        cat = _estimate_category(
            avg_text_chars=10, arabic_ratio=0.6, landscape_ratio=0.1, has_text=False
        )
        self.assertEqual(cat, "low_quality_scans")

    def test_estimate_category_dense_arabic(self) -> None:
        cat = _estimate_category(
            avg_text_chars=1200, arabic_ratio=0.7, landscape_ratio=0.0, has_text=True
        )
        self.assertEqual(cat, "born_digital_modern")

    def test_analyze_blank_pdf(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        out = io.BytesIO()
        writer.write(out)
        data = out.getvalue()

        meta = analyze_pdf_document(data)
        self.assertEqual(meta.page_count, 1)
        self.assertEqual(meta.selected_pages, 1)
        self.assertEqual(meta.category, "low_quality_scans")


if __name__ == "__main__":
    unittest.main()
