from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

import pdfword.page_routing as page_routing
from pdfword.page_routing import (
    AnalysisWarningCode,
    DigitalTextGateVerdict,
    DigitalTextReasonCode,
    DocumentRoutingContext,
    PageDecision,
    PageNextPath,
    analyze_pdf_page,
    decide_page_route,
    evaluate_digital_text_trust,
    validate_trusted_context_before_extraction,
    validate_trusted_text_after_extraction,
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


def _image_bytes(size: tuple[int, int] = (60, 60)) -> bytes:
    image = Image.new("RGB", size, "white")
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((1, 1, size[0] - 2, size[1] - 2), outline="black", width=2)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _hybrid_pdf(text: str, *, full_page_image: bool) -> bytes:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    rectangle = (
        fitz.Rect(0, 0, 595, 842) if full_page_image else fitz.Rect(510, 40, 550, 80)
    )
    image_size = (1190, 1684) if full_page_image else (40, 40)
    page.insert_image(rectangle, stream=_image_bytes(image_size))
    page.insert_textbox(
        fitz.Rect(72, 100, 523, 500), text, fontsize=12, fontname="helv"
    )
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


def test_complete_digital_text_gate_is_trusted() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.TRUSTED
    assert DigitalTextReasonCode.COMPLETE_DIGITAL_TEXT in result.reason_codes


def test_gate_rejects_a_page_without_embedded_text() -> None:
    analysis = _analyze_bytes((FIXTURES / "scanned.pdf").read_bytes())

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.NO_EMBEDDED_TEXT in result.reason_codes


def test_hidden_sparse_text_over_page_image_is_untrusted() -> None:
    analysis = _analyze_bytes(_hybrid_pdf("x", full_page_image=True))

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.IMAGE_DOMINANT_PARTIAL_TEXT in result.reason_codes


def test_incomplete_hybrid_text_is_untrusted() -> None:
    analysis = _analyze_bytes(_hybrid_pdf("Header only", full_page_image=True))

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.HYBRID_PAGE_INCOMPLETE_TEXT in result.reason_codes


def test_complete_digital_page_with_decorative_image_is_trusted() -> None:
    text = (
        "This complete born digital page contains enough meaningful text to "
        "demonstrate that the small corner image is decorative rather than a scan."
    )
    analysis = _analyze_bytes(_hybrid_pdf(text, full_page_image=False))

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.TRUSTED
    assert DigitalTextReasonCode.HARMLESS_DECORATIVE_IMAGES in result.reason_codes


def test_fragmented_arabic_text_is_untrusted_with_explicit_reason() -> None:
    base = _analyze_bytes((SAMPLES / "sample_clear_ar.pdf").read_bytes())
    analysis = replace(
        base,
        suspicious_fragmentation=True,
        isolated_arabic_character_ratio=0.9,
        arabic_integrity_risk=True,
    )

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.ARABIC_ISOLATED_CHARACTER_RUNS in result.reason_codes


def test_pathological_arabic_spacing_is_untrusted_with_explicit_reason() -> None:
    base = _analyze_bytes((SAMPLES / "sample_clear_ar.pdf").read_bytes())
    analysis = replace(
        base,
        pathological_arabic_spacing_count=2,
        arabic_integrity_risk=True,
    )

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.ARABIC_PATHOLOGICAL_SPACING in result.reason_codes


def test_unicode_corruption_is_untrusted_with_explicit_reason() -> None:
    base = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    analysis = replace(base, unicode_corruption_count=1)

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.UNICODE_CORRUPTION_DETECTED in result.reason_codes


def test_complex_layout_risk_is_uncertain_not_silently_trusted() -> None:
    base = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    analysis = replace(base, multi_column_risk=True)

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNCERTAIN
    assert DigitalTextReasonCode.COMPLEX_LAYOUT_RISK in result.reason_codes


def test_unavailable_gate_evidence_is_uncertain() -> None:
    context = DocumentRoutingContext.from_pdf(_blank_pdf())
    analysis = analyze_pdf_page(_UnreadablePage(), 1, context)

    result = evaluate_digital_text_trust(analysis)

    assert result.verdict is DigitalTextGateVerdict.UNCERTAIN
    assert DigitalTextReasonCode.REQUIRED_EVIDENCE_UNAVAILABLE in result.reason_codes


def test_short_centered_title_gate_is_not_near_blank() -> None:
    analysis = _analyze_bytes(_text_pdf("Short Title"))

    result = evaluate_digital_text_trust(analysis)

    assert analysis.near_blank_evidence is False
    assert result.verdict in {
        DigitalTextGateVerdict.TRUSTED,
        DigitalTextGateVerdict.UNCERTAIN,
    }


def test_trusted_gate_decision_authorizes_direct_text() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)

    result = decide_page_route(analysis, gate, ocr_available=False)

    assert result.decision is PageDecision.TRUSTED_DIGITAL_TEXT
    assert result.next_path is PageNextPath.DIRECT_PDF_TEXT
    assert result.ocr_required is False
    assert result.review_required is False
    assert result.trusted_context is not None


def test_untrusted_page_with_unavailable_ocr_preserves_pending_path() -> None:
    analysis = _analyze_bytes((FIXTURES / "scanned.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)

    result = decide_page_route(analysis, gate, ocr_available=False)

    assert result.decision is PageDecision.OCR_REQUIRED
    assert result.next_path is PageNextPath.PENDING_OCR_MODEL
    assert result.ocr_required is True
    assert result.ocr_available is False


def test_untrusted_page_with_available_ocr_selects_local_ocr() -> None:
    analysis = _analyze_bytes((FIXTURES / "scanned.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)

    result = decide_page_route(analysis, gate, ocr_available=True)

    assert result.decision is PageDecision.OCR_REQUIRED
    assert result.next_path is PageNextPath.LOCAL_OCR
    assert result.ocr_available is True


def test_ambiguous_short_title_requires_review() -> None:
    analysis = _analyze_bytes(_text_pdf("Short Title"))
    gate = evaluate_digital_text_trust(analysis)

    result = decide_page_route(analysis, gate, ocr_available=False)

    assert result.decision is PageDecision.REVIEW_REQUIRED
    assert result.next_path is PageNextPath.MANUAL_REVIEW
    assert result.review_required is True
    assert result.trusted_context is None


def test_blank_and_structural_near_blank_share_closed_blank_decision() -> None:
    blank = _analyze_bytes((FIXTURES / "blank.pdf").read_bytes())
    near_blank = _analyze_bytes((FIXTURES / "near_blank_page_number.pdf").read_bytes())

    blank_result = decide_page_route(
        blank, evaluate_digital_text_trust(blank), ocr_available=False
    )
    near_blank_result = decide_page_route(
        near_blank,
        evaluate_digital_text_trust(near_blank),
        ocr_available=False,
    )

    assert blank_result.decision is PageDecision.BLANK_OR_NEAR_BLANK
    assert blank_result.next_path is PageNextPath.NO_EXTRACTION
    assert near_blank_result.decision is PageDecision.BLANK_OR_NEAR_BLANK
    assert near_blank_result.next_path is PageNextPath.NO_EXTRACTION


def test_trusted_context_rejects_another_document_or_page() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)
    decision = decide_page_route(analysis, gate, ocr_available=False)
    assert decision.trusted_context is not None

    assert validate_trusted_context_before_extraction(
        decision.trusted_context,
        document_sha256=analysis.document_sha256,
        page_number=analysis.page_number,
        gate=gate,
    )
    assert not validate_trusted_context_before_extraction(
        decision.trusted_context,
        document_sha256="0" * 64,
        page_number=analysis.page_number,
        gate=gate,
    )
    assert not validate_trusted_context_before_extraction(
        decision.trusted_context,
        document_sha256=analysis.document_sha256,
        page_number=analysis.page_number + 1,
        gate=gate,
    )


def test_post_extraction_digest_mismatch_is_rejected() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)
    decision = decide_page_route(analysis, gate, ocr_available=False)
    assert decision.trusted_context is not None

    assert validate_trusted_text_after_extraction(
        decision.trusted_context, analysis.embedded_text
    )
    assert not validate_trusted_text_after_extraction(
        decision.trusted_context,
        "different extracted text",
    )


def test_page_decision_diagnostics_are_categorical_and_hide_context_digests() -> None:
    analysis = _analyze_bytes((FIXTURES / "digital_text.pdf").read_bytes())
    gate = evaluate_digital_text_trust(analysis)
    decision = decide_page_route(analysis, gate, ocr_available=False)

    payload = decision.to_diagnostics()
    serialized = json.dumps(payload).lower()

    assert payload["decision"] == "trusted_digital_text"
    assert "sha256" not in serialized
    assert "accuracy" not in serialized
    assert "confidence" not in serialized
    assert "quality" not in serialized
