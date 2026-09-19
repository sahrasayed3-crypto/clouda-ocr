from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PIL import Image

from .engines import OCRBox, OCRResult


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
class ReReadBudget:
    max_regions: int = 4
    max_attempts_per_region: int = 1
    max_total_attempts: int = 4
    max_total_pixels: int = 4_000_000
    max_total_bytes: int = 8 * 1024 * 1024
    min_width_px: int = 8
    min_height_px: int = 8


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
    source_text: str = field(default="", repr=False)

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
    regions = regions_from_ocr_boxes(result.boxes, render, tuple(dict.fromkeys(issues)))
    reread_justified = bool(regions) and OCRIssueCode.ENGINE_ERROR not in issues
    return OCRReviewResult(
        verdict=(
            OCRReviewVerdict.FAILED
            if OCRIssueCode.ENGINE_ERROR in issues
            else (
                OCRReviewVerdict.REREAD_REQUIRED
                if reread_justified
                else OCRReviewVerdict.REVIEW_REQUIRED
            )
        ),
        issue_codes=tuple(dict.fromkeys(issues)),
        suspicious_regions=regions,
        safe_diagnostics=(("first_pass_characters", len(text)),),
        selective_reread_justified=reread_justified,
        manual_review_required=not reread_justified,
    )


def regions_from_ocr_boxes(
    boxes: tuple[OCRBox, ...],
    render: RenderedPageContext,
    reasons: tuple[OCRIssueCode, ...],
) -> tuple[ReviewRegion, ...]:
    candidates: list[ReviewRegion] = []
    for index, box in enumerate(boxes, start=1):
        metadata = box.metadata
        if (
            box.bbox is None
            or metadata.get("coordinate_space") != render.coordinate_space
            or metadata.get("render_identity") != render.render_identity
            or metadata.get("image_width_px") != render.width_px
            or metadata.get("image_height_px") != render.height_px
        ):
            continue
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box.bbox):
            continue
        x0, y0, x1, y1 = box.bbox
        region = ReviewRegion(
            region_id=f"ocr-region-{index}",
            page_no=render.page_no,
            bbox_px=(math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)),
            render_identity=render.render_identity,
            image_width_px=render.width_px,
            image_height_px=render.height_px,
            reason_codes=reasons,
            priority="high",
            source_text=box.text,
        )
        candidates.append(region)
    return validate_and_bound_regions(tuple(candidates), render, ReReadBudget())


def _intersection_over_union(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _valid_region(
    region: ReviewRegion, render: RenderedPageContext, budget: ReReadBudget
) -> bool:
    if (
        region.page_no != render.page_no
        or region.render_identity != render.render_identity
        or region.image_width_px != render.width_px
        or region.image_height_px != render.height_px
    ):
        return False
    coordinates = region.bbox_px
    if any(not isinstance(value, int) or isinstance(value, bool) for value in coordinates):
        return False
    x0, y0, x1, y1 = coordinates
    if x0 < 0 or y0 < 0 or x1 > render.width_px or y1 > render.height_px:
        return False
    return x1 - x0 >= budget.min_width_px and y1 - y0 >= budget.min_height_px


def validate_and_bound_regions(
    regions: tuple[ReviewRegion, ...],
    render: RenderedPageContext,
    budget: ReReadBudget,
) -> tuple[ReviewRegion, ...]:
    accepted: list[ReviewRegion] = []
    pixels = 0
    for region in regions:
        if len(accepted) >= budget.max_regions or not _valid_region(region, render, budget):
            continue
        area = (region.bbox_px[2] - region.bbox_px[0]) * (
            region.bbox_px[3] - region.bbox_px[1]
        )
        if pixels + area > budget.max_total_pixels or any(
            _intersection_over_union(region.bbox_px, known.bbox_px) >= 0.85
            for known in accepted
        ):
            continue
        accepted.append(region)
        pixels += area
    return tuple(accepted)


def crop_review_region(
    image_bytes: bytes, region: ReviewRegion, render: RenderedPageContext
) -> bytes:
    if not _valid_region(region, render, ReReadBudget()):
        raise ValueError("Review region is not bound to the current rendered image")
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.size != (render.width_px, render.height_px):
            raise ValueError("Rendered image dimensions no longer match region context")
        crop = image.crop(region.bbox_px)
        output = io.BytesIO()
        crop.save(output, format="PNG")
    return output.getvalue()


def reconcile_ocr_results(
    first_pass_text: str,
    review: OCRReviewResult,
    rereads: tuple[ReReadResult, ...],
) -> OCRReconciliationResult:
    if review.verdict is OCRReviewVerdict.ACCEPTED:
        return OCRReconciliationResult(
            OCRPageState.ACCEPTED_FIRST_PASS,
            first_pass_text,
            (),
            provenance=("first_pass",),
        )
    if review.verdict is not OCRReviewVerdict.REREAD_REQUIRED:
        return OCRReconciliationResult(
            OCRPageState.REVIEW_REQUIRED,
            "",
            review.issue_codes,
            unresolved_regions=tuple(region.region_id for region in review.suspicious_regions),
            review_required=True,
        )
    by_region = {item.region_id: item for item in rereads}
    reconciled = first_pass_text
    accepted: list[str] = []
    for region in review.suspicious_regions:
        reread = by_region.get(region.region_id)
        if (
            reread is None
            or not reread.success
            or not reread.text.strip()
            or not region.source_text
            or reconciled.count(region.source_text) != 1
            or _text_unicode_corruption_count(reread.text)
        ):
            return OCRReconciliationResult(
                OCRPageState.REVIEW_REQUIRED,
                "",
                tuple(dict.fromkeys((*review.issue_codes, OCRIssueCode.REREAD_DISAGREEMENT))),
                unresolved_regions=(region.region_id,),
                review_required=True,
            )
        reconciled = reconciled.replace(region.source_text, reread.text, 1)
        accepted.append(region.region_id)
    return OCRReconciliationResult(
        OCRPageState.ACCEPTED_AFTER_SELECTIVE_REREAD,
        reconciled,
        review.issue_codes,
        accepted_replacements=tuple(accepted),
        provenance=tuple(f"selective_reread:{region_id}" for region_id in accepted),
    )
