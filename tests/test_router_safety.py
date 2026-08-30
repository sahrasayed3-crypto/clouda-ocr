import io

import pytest
from pypdf import PdfWriter

from pdfword.ocr_pipeline import process_pdf


def _blank_pdf(page_count: int = 1) -> bytes:
    out = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    writer.write(out)
    return out.getvalue()


def test_invalid_page_range_is_rejected_before_processing() -> None:
    with pytest.raises(ValueError, match="between 1 and 1"):
        process_pdf(
            pdf_bytes=_blank_pdf(1),
            from_page=None,
            to_page=None,
            page_numbers=[2],
            progress_bar=None,
            status_placeholder=None,
        )


def test_empty_page_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        process_pdf(
            pdf_bytes=_blank_pdf(1),
            from_page=None,
            to_page=None,
            page_numbers=[],
            progress_bar=None,
            status_placeholder=None,
        )


def test_blank_pdf_does_not_call_cloud_or_unsupported_ocr() -> None:
    attempts = []

    def cloud_attempt_allowed(cost: float = 0.0) -> bool:
        attempts.append(cost)
        return True

    rows, _ = process_pdf(
        pdf_bytes=_blank_pdf(1),
        from_page=1,
        to_page=1,
        api_key="secret",  # pragma: allowlist secret
        fast_model="model-a",
        accurate_model="model-b",
        progress_bar=None,
        status_placeholder=None,
        cloud_attempt_allowed=cloud_attempt_allowed,
    )

    assert attempts == []
    assert rows[0].route_used == "blank_page"
    assert rows[0].requires_manual_review is False
