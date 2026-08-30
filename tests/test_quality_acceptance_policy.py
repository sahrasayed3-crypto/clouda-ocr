import io

from pypdf import PdfWriter

from pdfword.models import PageResult
from pdfword.ocr_pipeline import (
    MIN_ACCEPT_QUALITY_SCORE,
    TARGET_QUALITY_SCORE,
    process_pdf,
)
from pdfword.accuracy import final_acceptance_decision
from pdfword.docx_export import markdown_to_docx


def _blank_pdf(page_count: int = 1) -> bytes:
    out = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    writer.write(out)
    return out.getvalue()


def test_acceptance_threshold_floor_is_ninety() -> None:
    assert MIN_ACCEPT_QUALITY_SCORE == 90.0
    assert TARGET_QUALITY_SCORE == 97.0


def test_final_acceptance_boundaries_and_indeterminate_quality() -> None:
    text = "Readable Arabic and English text " * 20

    low = final_acceptance_decision(
        text, estimated_text_quality=89.99, threshold=90.0, expected_non_empty=True
    )
    exact = final_acceptance_decision(
        text, estimated_text_quality=90.0, threshold=90.0, expected_non_empty=True
    )
    high = final_acceptance_decision(
        text, estimated_text_quality=90.01, threshold=90.0, expected_non_empty=True
    )
    missing = final_acceptance_decision(
        text, estimated_text_quality=None, threshold=90.0, expected_non_empty=True
    )

    assert low["accepted"] is False
    assert low["requires_manual_review"] is True
    assert exact["accepted"] is True
    assert high["accepted"] is True
    assert missing["accepted"] is False
    assert missing["requires_manual_review"] is True
    assert missing["review_reason"] == "indeterminate_quality_score"


def test_blank_page_has_an_explicit_non_ocr_state() -> None:
    rows, text = process_pdf(
        pdf_bytes=_blank_pdf(),
        from_page=1,
        to_page=1,
        api_key="",
        fast_model="",
        accurate_model="",
        progress_bar=None,
        status_placeholder=None,
    )
    assert rows[0].requires_manual_review is False
    assert rows[0].route_used == "blank_page"
    assert rows[0].model_used == "system:blank_page"
    assert text == ""


def test_manual_review_marker_is_added_to_docx() -> None:
    payload = markdown_to_docx(
        [
            PageResult(
                page_no=1,
                model_used="pending:future_ocr_engine",
                markdown="Needs OCR later.",
                quality_score=0.0,
                text_quality_score=0.0,
                requires_manual_review=True,
                review_reason="pending_ocr_model",
            )
        ]
    )
    assert payload[:2] == b"PK"


def test_images_and_tables_are_not_added_to_docx() -> None:
    payload = markdown_to_docx(
        [
            PageResult(
                page_no=1,
                model_used="local:direct_pdf_text",
                markdown="Text only\n\n| A | B |\n| - | - |\n| 1 | 2 |",
            )
        ]
    )
    assert payload[:2] == b"PK"
