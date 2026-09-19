from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter

import pdfword.page_routing as page_routing
from pdfword.page_routing import (
    AnalysisWarningCode,
    DocumentRoutingContext,
    analyze_pdf_page,
)

FIXTURES = Path(__file__).with_name("fixtures")
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _blank_pdf(page_count: int = 1) -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=595, height=842)
    writer.write(output)
    return output.getvalue()


def _text_pdf(text: str, *, y: float = 160) -> bytes:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((220, y), text, fontsize=18, fontname="helv")
    payload = document.tobytes(garbage=4, deflate=True)
    document.close()
    return payload


def _analyze_bytes(payload: bytes, page_number: int = 1):
    document = DocumentRoutingContext.from_pdf(payload)
    reader = PdfReader(io.BytesIO(payload))
    return analyze_pdf_page(reader.pages[page_number - 1], page_number, document)


def test_document_context_hashes_pdf_once_and_reuses_digest(monkeypatch) -> None:
    calls = 0
    real = page_routing._sha256_hex

    def counted(payload: bytes) -> str:
        nonlocal calls
        calls += 1
        return real(payload)

    monkeypatch.setattr(page_routing, "_sha256_hex", counted)
    context = DocumentRoutingContext.from_pdf(_blank_pdf(page_count=2))
    reader = PdfReader(io.BytesIO(context.pdf_bytes))

    first = analyze_pdf_page(reader.pages[0], 1, context)
    second = analyze_pdf_page(reader.pages[1], 2, context)

    assert calls == 1
    assert first.document_sha256 == second.document_sha256 == context.pdf_sha256


def test_page_diagnostics_never_include_extracted_text() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())

    payload = analysis.to_diagnostics()

    assert "Digital PDF text" not in json.dumps(payload)
    assert payload["normalized_character_count"] > 20
    assert payload["span_count"] >= 1


def test_digital_page_reports_positioned_text_and_content_operations() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())

    assert analysis.embedded_text_present is True
    assert analysis.text_coverage_ratio is not None
    assert analysis.text_coverage_ratio > 0
    assert any(span.bbox is not None for span in analysis.spans)
    assert analysis.visible_text_operations is not None
    assert analysis.visible_text_operations > 0


def test_scanned_and_blank_pages_have_distinct_visible_evidence() -> None:
    scanned = _analyze_bytes((FIXTURES / "scanned.pdf").read_bytes())
    blank = _analyze_bytes((FIXTURES / "blank.pdf").read_bytes())

    assert scanned.embedded_text_present is False
    assert scanned.image_count == 1
    assert scanned.visible_image_operations is not None
    assert scanned.visible_image_operations > 0
    assert scanned.blank_evidence is False
    assert blank.image_count == 0
    assert blank.blank_evidence is True


def test_arabic_sample_reports_arabic_structure_without_exposing_text() -> None:
    analysis = _analyze_bytes((SAMPLES / "sample_clear_ar.pdf").read_bytes())

    assert analysis.arabic_character_count > 0
    assert analysis.arabic_ratio > 0
    assert analysis.embedded_text not in json.dumps(analysis.to_diagnostics())


def test_short_centered_title_is_not_near_blank_from_length_alone() -> None:
    analysis = _analyze_bytes(_text_pdf("Short Title"))

    assert analysis.normalized_character_count == len("Short Title")
    assert analysis.near_blank_evidence is False


class _UnreadablePage:
    @property
    def mediabox(self):
        raise ValueError("bad geometry")

    @property
    def images(self):
        raise ValueError("bad images")

    def extract_text(self, **_kwargs):
        raise ValueError("bad text")

    def get_contents(self):
        raise ValueError("bad content")


def test_unavailable_page_evidence_is_explicit_not_fabricated() -> None:
    context = DocumentRoutingContext.from_pdf(_blank_pdf())

    analysis = analyze_pdf_page(_UnreadablePage(), 1, context)

    assert analysis.page_width_pt == 0
    assert analysis.text_coverage_ratio is None
    assert analysis.image_count is None
    assert analysis.visible_text_operations is None
    assert analysis.blank_evidence is False
    assert {
        AnalysisWarningCode.TEXT_EXTRACTION_FAILED,
        AnalysisWarningCode.UNSUPPORTED_PAGE_GEOMETRY,
        AnalysisWarningCode.IMAGE_METADATA_UNAVAILABLE,
        AnalysisWarningCode.CONTENT_STREAM_UNAVAILABLE,
    } <= set(analysis.warnings)
