from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PIL import Image

from .engines import OCRResult


class OCRReviewVerdict(StrEnum):
    ACCEPTED = "accepted"
    REREAD_REQUIRED = "reread_required"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class OCRIssueCode(StrEnum):
    EMPTY_OCR_OUTPUT = "empty_ocr_output"
    SUSPICIOUSLY_SPARSE_OUTPUT = "suspiciously_sparse_output"
    DUPLICATE_LINE_SEQUENCE = "duplicate_line_sequence"
    REPEATED_TEXT_BLOCK = "repeated_text_block"
    READING_ORDER_RISK = "reading_order_risk"
    LAYOUT_COVERAGE_GAP = "layout_coverage_gap"
    LOW_ENGINE_CONFIDENCE = "low_engine_confidence"
    MISSING_EXPECTED_REGION = "missing_expected_region"
    ARABIC_FRAGMENTATION = "arabic_fragmentation"
    ARABIC_PATHOLOGICAL_SPACING = "arabic_pathological_spacing"
    UNICODE_CORRUPTION = "unicode_corruption"
    CONFLICTING_OCR_EVIDENCE = "conflicting_ocr_evidence"
    ENGINE_ERROR = "engine_error"
    REREAD_DISAGREEMENT = "reread_disagreement"
    REQUIRED_EVIDENCE_UNAVAILABLE = "required_evidence_unavailable"


class OCRPageState(StrEnum):
    ACCEPTED_FIRST_PASS = "accepted_first_pass"
    ACCEPTED_AFTER_SELECTIVE_REREAD = "accepted_after_selective_reread"
    REVIEW_REQUIRED = "review_required"
    PENDING_OCR_MODEL = "pending_ocr_model"
    OCR_FAILED = "ocr_failed"


@dataclass(frozen=True)
class RenderedPageContext:
    page_no: int
    render_identity: str = field(repr=False)
    width_px: int
    height_px: int
    coordinate_space: str = "image_pixels"


@dataclass(frozen=True)
class ReviewRegion:
    region_id: str
    page_no: int
    bbox_px: tuple[int, int, int, int] = field(repr=False)
    render_identity: str = field(repr=False)
    image_width_px: int
    image_height_px: int
    reason_codes: tuple[OCRIssueCode, ...]
    priority: str

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_number": self.page_no,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "priority": self.priority,
        }


@dataclass(frozen=True)
class OCRReviewResult:
    verdict: OCRReviewVerdict
    issue_codes: tuple[OCRIssueCode, ...]
    suspicious_regions: tuple[ReviewRegion, ...]
    safe_diagnostics: tuple[tuple[str, int | str | bool | None], ...]
    selective_reread_justified: bool
    manual_review_required: bool

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "review_verdict": self.verdict.value,
            "issue_codes": [code.value for code in self.issue_codes],
            "suspicious_region_count": len(self.suspicious_regions),
            "safe_diagnostics": dict(self.safe_diagnostics),
            "selective_reread_justified": self.selective_reread_justified,
            "manual_review_required": self.manual_review_required,
        }


@dataclass(frozen=True)
class ReReadResult:
    region_id: str
    engine_name: str
    success: bool
    text: str = field(repr=False)
    model_revision: str | None = None
    issue_codes: tuple[OCRIssueCode, ...] = ()


@dataclass(frozen=True)
class OCRReconciliationResult:
    state: OCRPageState
    text: str = field(repr=False)
    reason_codes: tuple[OCRIssueCode, ...]
    accepted_replacements: tuple[str, ...] = ()
    unresolved_regions: tuple[str, ...] = ()
    review_required: bool = False
    provenance: tuple[str, ...] = ()

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "final_state": self.state.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "accepted_reread_count": len(self.accepted_replacements),
            "unresolved_region_count": len(self.unresolved_regions),
            "review_required": self.review_required,
            "provenance": list(self.provenance),
        }


def make_rendered_page_context(page_no: int, image_bytes: bytes) -> RenderedPageContext:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    if page_no < 1 or width < 1 or height < 1:
        raise ValueError("Rendered page context requires positive page and dimensions")
    return RenderedPageContext(
        page_no=page_no,
        render_identity=hashlib.sha256(image_bytes).hexdigest(),
        width_px=width,
        height_px=height,
    )


def _text_unicode_corruption_count(text: str) -> int:
    return sum(
        character == "\ufffd"
        or 0xE000 <= ord(character) <= 0xF8FF
        or (character not in {"\n", "\t"} and ord(character) < 32)
        for character in text
    )


def review_first_pass(
    result: OCRResult,
    analysis: Any,
    render: RenderedPageContext,
) -> OCRReviewResult:
    del render
    issues: list[OCRIssueCode] = []
    text = (result.text or "").strip()
    if not result.success:
        issues.append(OCRIssueCode.ENGINE_ERROR)
    elif not text:
        issues.append(OCRIssueCode.EMPTY_OCR_OUTPUT)
    elif _text_unicode_corruption_count(text) or bool(
        getattr(analysis, "unicode_corruption_count", 0)
    ):
        issues.append(OCRIssueCode.UNICODE_CORRUPTION)
    if bool(getattr(analysis, "suspicious_fragmentation", False)) or bool(
        getattr(analysis, "arabic_integrity_risk", False)
    ):
        issues.append(OCRIssueCode.ARABIC_FRAGMENTATION)
    if getattr(analysis, "pathological_arabic_spacing_count", 0):
        issues.append(OCRIssueCode.ARABIC_PATHOLOGICAL_SPACING)
    if getattr(analysis, "reading_order_risk", False) is True:
        issues.append(OCRIssueCode.READING_ORDER_RISK)
    if not issues:
        return OCRReviewResult(
            verdict=OCRReviewVerdict.ACCEPTED,
            issue_codes=(),
            suspicious_regions=(),
            safe_diagnostics=(("first_pass_characters", len(text)),),
            selective_reread_justified=False,
            manual_review_required=False,
        )
    return OCRReviewResult(
        verdict=(
            OCRReviewVerdict.FAILED
            if OCRIssueCode.ENGINE_ERROR in issues
            else OCRReviewVerdict.REVIEW_REQUIRED
        ),
        issue_codes=tuple(dict.fromkeys(issues)),
        suspicious_regions=(),
        safe_diagnostics=(("first_pass_characters", len(text)),),
        selective_reread_justified=False,
        manual_review_required=True,
    )
