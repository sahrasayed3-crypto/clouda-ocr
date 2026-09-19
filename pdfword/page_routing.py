from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pypdf.generic import ContentStream

_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3]")
_ARABIC_SPACING_RE = re.compile(
    r"(?:[\u0621-\u063a\u0641-\u064a\u066e-\u06d3]\s+){3,}"
    r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3]"
)
_PAGE_NUMBER_RE = re.compile(r"(?:\d{1,4}|[ivxlcdm]{1,8})", re.IGNORECASE)
_TEXT_OPERATORS = {b"Tj", b"TJ", b"'", b'"'}
_PATH_PAINT_OPERATORS = {
    b"S",
    b"s",
    b"f",
    b"F",
    b"f*",
    b"B",
    b"B*",
    b"b",
    b"b*",
}
_NEAR_BLANK_TEXT_LIMIT = 12
_SMALL_IMAGE_DENSITY_LIMIT = 0.10


class EvidenceAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class AnalysisWarningCode(StrEnum):
    TEXT_EXTRACTION_FAILED = "text_extraction_failed"
    TEXT_GEOMETRY_UNAVAILABLE = "text_geometry_unavailable"
    IMAGE_METADATA_UNAVAILABLE = "image_metadata_unavailable"
    CONTENT_STREAM_UNAVAILABLE = "content_stream_unavailable"
    UNSUPPORTED_PAGE_GEOMETRY = "unsupported_page_geometry"


class DigitalTextGateVerdict(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    UNCERTAIN = "uncertain"


class PageDecision(StrEnum):
    TRUSTED_DIGITAL_TEXT = "trusted_digital_text"
    OCR_REQUIRED = "ocr_required"
    REVIEW_REQUIRED = "review_required"
    BLANK_OR_NEAR_BLANK = "blank_or_near_blank"


class PageNextPath(StrEnum):
    DIRECT_PDF_TEXT = "direct_pdf_text"
    LOCAL_OCR = "local_ocr"
    PENDING_OCR_MODEL = "pending_ocr_model"
    MANUAL_REVIEW = "manual_review"
    NO_EXTRACTION = "no_extraction"


class DigitalTextReasonCode(StrEnum):
    COMPLETE_DIGITAL_TEXT = "complete_digital_text"
    SUFFICIENT_TEXT_DISTRIBUTION = "sufficient_text_distribution"
    HARMLESS_DECORATIVE_IMAGES = "harmless_decorative_images"
    NO_EMBEDDED_TEXT = "no_embedded_text"
    INSUFFICIENT_USABLE_TEXT = "insufficient_usable_text"
    SUSPICIOUS_SPARSE_TEXT = "suspicious_sparse_text"
    SUSPICIOUS_FRAGMENTATION = "suspicious_fragmentation"
    ARABIC_ISOLATED_CHARACTER_RUNS = "arabic_isolated_character_runs"
    ARABIC_PATHOLOGICAL_SPACING = "arabic_pathological_spacing"
    UNICODE_CORRUPTION_DETECTED = "unicode_corruption_detected"
    IMAGE_DOMINANT_PARTIAL_TEXT = "image_dominant_partial_text"
    HYBRID_PAGE_INCOMPLETE_TEXT = "hybrid_page_incomplete_text"
    LAYOUT_ORDER_RISK = "layout_order_risk"
    COMPLEX_LAYOUT_RISK = "complex_layout_risk"
    REQUIRED_EVIDENCE_UNAVAILABLE = "required_evidence_unavailable"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass(frozen=True)
class DocumentRoutingContext:
    pdf_bytes: bytes = field(repr=False)
    pdf_sha256: str

    @classmethod
    def from_pdf(cls, pdf_bytes: bytes) -> DocumentRoutingContext:
        return cls(pdf_bytes=pdf_bytes, pdf_sha256=_sha256_hex(pdf_bytes))


@dataclass(frozen=True)
class TextSpan:
    text: str = field(repr=False)
    bbox: tuple[float, float, float, float] | None
    font_size: float | None
    sequence_index: int


@dataclass(frozen=True)
class PageAnalysis:
    document_sha256: str
    page_number: int
    page_width_pt: float
    page_height_pt: float
    embedded_text: str = field(repr=False)
    normalized_text: str = field(repr=False)
    embedded_text_present: bool
    raw_character_count: int
    normalized_character_count: int
    non_whitespace_character_count: int
    alphanumeric_character_count: int
    word_count: int
    line_count: int
    span_count: int
    spans: tuple[TextSpan, ...] = field(repr=False)
    text_coverage_ratio: float | None
    horizontal_distribution: tuple[int, ...] | None
    vertical_distribution: tuple[int, ...] | None
    image_count: int | None
    largest_image_pixels: int | None
    largest_image_density_estimate: float | None
    visible_text_operations: int | None
    visible_image_operations: int | None
    visible_path_operations: int | None
    blank_evidence: bool
    near_blank_evidence: bool
    suspicious_sparse_text: bool
    suspicious_fragmentation: bool
    hybrid_page: bool
    arabic_character_count: int
    arabic_ratio: float
    isolated_arabic_character_ratio: float | None
    pathological_arabic_spacing_count: int
    unicode_corruption_count: int
    arabic_integrity_risk: bool
    reading_order_risk: bool | None
    multi_column_risk: bool | None
    warnings: tuple[AnalysisWarningCode, ...]
    evidence_codes: tuple[str, ...]

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "page_width_pt": round(self.page_width_pt, 3),
            "page_height_pt": round(self.page_height_pt, 3),
            "embedded_text_present": self.embedded_text_present,
            "raw_character_count": self.raw_character_count,
            "normalized_character_count": self.normalized_character_count,
            "non_whitespace_character_count": self.non_whitespace_character_count,
            "alphanumeric_character_count": self.alphanumeric_character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "span_count": self.span_count,
            "text_coverage_ratio": _rounded(self.text_coverage_ratio),
            "horizontal_distribution": self.horizontal_distribution,
            "vertical_distribution": self.vertical_distribution,
            "image_count": self.image_count,
            "largest_image_pixels": self.largest_image_pixels,
            "largest_image_density_estimate": _rounded(
                self.largest_image_density_estimate
            ),
            "visible_text_operations": self.visible_text_operations,
            "visible_image_operations": self.visible_image_operations,
            "visible_path_operations": self.visible_path_operations,
            "blank_evidence": self.blank_evidence,
            "near_blank_evidence": self.near_blank_evidence,
            "suspicious_sparse_text": self.suspicious_sparse_text,
            "suspicious_fragmentation": self.suspicious_fragmentation,
            "hybrid_page": self.hybrid_page,
            "arabic_character_count": self.arabic_character_count,
            "arabic_ratio": _rounded(self.arabic_ratio),
            "isolated_arabic_character_ratio": _rounded(
                self.isolated_arabic_character_ratio
            ),
            "pathological_arabic_spacing_count": (
                self.pathological_arabic_spacing_count
            ),
            "unicode_corruption_count": self.unicode_corruption_count,
            "arabic_integrity_risk": self.arabic_integrity_risk,
            "reading_order_risk": self.reading_order_risk,
            "multi_column_risk": self.multi_column_risk,
            "warnings": [warning.value for warning in self.warnings],
            "evidence_codes": list(self.evidence_codes),
        }


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    reason_code: DigitalTextReasonCode
    evidence: tuple[tuple[str, int | float | bool | None], ...] = ()


@dataclass(frozen=True)
class DigitalTextGateResult:
    verdict: DigitalTextGateVerdict
    checks: tuple[GateCheck, ...]
    reason_codes: tuple[DigitalTextReasonCode, ...]

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "reason_code": check.reason_code.value,
                    "evidence": dict(check.evidence),
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class TrustedDigitalTextContext:
    document_sha256: str
    page_number: int
    normalized_text_sha256: str
    gate_result: DigitalTextGateResult


@dataclass(frozen=True)
class PageDecisionResult:
    decision: PageDecision
    next_path: PageNextPath
    gate_verdict: DigitalTextGateVerdict
    reason_codes: tuple[str, ...]
    evidence: tuple[tuple[str, int | float | bool | str | None], ...]
    ocr_required: bool
    ocr_available: bool
    review_required: bool
    trusted_context: TrustedDigitalTextContext | None = field(default=None, repr=False)

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "next_path": self.next_path.value,
            "gate_verdict": self.gate_verdict.value,
            "reason_codes": list(self.reason_codes),
            "evidence": dict(self.evidence),
            "ocr_required": self.ocr_required,
            "ocr_available": self.ocr_available,
            "ocr_pending": (self.next_path is PageNextPath.PENDING_OCR_MODEL),
            "review_required": self.review_required,
        }


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_embedded_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200e", "").replace("\u200f", "")
    return re.sub(r"\s+", " ", normalized).strip()


def digest_normalized_text(text: str) -> str:
    return _sha256_hex(normalize_embedded_text(text).encode("utf-8"))


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _page_dimensions(
    page: Any, warnings: list[AnalysisWarningCode]
) -> tuple[float, float]:
    try:
        width = max(0.0, float(page.mediabox.width))
        height = max(0.0, float(page.mediabox.height))
        if width <= 0 or height <= 0:
            raise ValueError("non-positive page geometry")
        return width, height
    except Exception:
        warnings.append(AnalysisWarningCode.UNSUPPORTED_PAGE_GEOMETRY)
        return 0.0, 0.0


def _extract_text_and_spans(
    page: Any,
    warnings: list[AnalysisWarningCode],
) -> tuple[str, tuple[TextSpan, ...]]:
    spans: list[TextSpan] = []

    def visitor(
        text: str,
        _current_matrix: list[float],
        text_matrix: list[float],
        _font_dictionary: dict[str, Any] | None,
        font_size: float,
    ) -> None:
        normalized = normalize_embedded_text(text)
        if not normalized:
            return
        bbox: tuple[float, float, float, float] | None = None
        try:
            x = float(text_matrix[4])
            y = float(text_matrix[5])
            size = max(0.0, float(font_size))
            width = max(size * 0.45 * len(normalized), size * 0.45)
            bbox = (x, y, x + width, y + size)
        except (IndexError, TypeError, ValueError):
            pass
        spans.append(
            TextSpan(
                text=normalized,
                bbox=bbox,
                font_size=float(font_size) if font_size is not None else None,
                sequence_index=len(spans),
            )
        )

    try:
        text = page.extract_text(visitor_text=visitor) or ""
    except Exception:
        warnings.append(AnalysisWarningCode.TEXT_EXTRACTION_FAILED)
        return "", ()
    if text.strip() and not spans:
        warnings.append(AnalysisWarningCode.TEXT_GEOMETRY_UNAVAILABLE)
    return text, tuple(spans)


def _text_geometry(spans: tuple[TextSpan, ...], width: float, height: float) -> tuple[
    float | None,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    bool | None,
    bool | None,
]:
    positioned = [span for span in spans if span.bbox is not None]
    if not positioned or width <= 0 or height <= 0:
        return None, None, None, None, None
    horizontal = [0, 0, 0, 0]
    vertical = [0, 0, 0, 0]
    area = 0.0
    centers: list[tuple[float, float]] = []
    for span in positioned:
        assert span.bbox is not None
        x0, y0, x1, y1 = span.bbox
        clipped_width = max(0.0, min(width, x1) - max(0.0, x0))
        clipped_height = max(0.0, min(height, y1) - max(0.0, y0))
        area += clipped_width * clipped_height
        x_center = min(0.999999, max(0.0, (x0 + x1) / 2 / width))
        y_center = min(0.999999, max(0.0, (y0 + y1) / 2 / height))
        horizontal[int(x_center * 4)] += 1
        vertical[int(y_center * 4)] += 1
        centers.append((x_center, y_center))
    occupied_horizontal = [index for index, count in enumerate(horizontal) if count]
    multi_column = bool(
        len(positioned) >= 4
        and len(occupied_horizontal) >= 2
        and max(occupied_horizontal) - min(occupied_horizontal) >= 2
    )
    y_moves = [
        centers[index + 1][1] - centers[index][1] for index in range(len(centers) - 1)
    ]
    reading_order_risk = bool(
        len(y_moves) >= 4 and sum(1 for move in y_moves if abs(move) > 0.35) >= 2
    )
    return (
        min(1.0, area / (width * height)),
        tuple(horizontal),
        tuple(vertical),
        reading_order_risk,
        multi_column,
    )


def _image_evidence(
    page: Any,
    page_area: float,
    warnings: list[AnalysisWarningCode],
) -> tuple[int | None, int | None, float | None]:
    try:
        image_sizes: list[tuple[int, int]] = []
        for image in page.images:
            size = getattr(getattr(image, "image", None), "size", None)
            if size and len(size) == 2:
                image_sizes.append((int(size[0]), int(size[1])))
        largest = max((w * h for w, h in image_sizes), default=0)
        density = largest / max(1.0, page_area)
        return len(image_sizes), largest, density
    except Exception:
        warnings.append(AnalysisWarningCode.IMAGE_METADATA_UNAVAILABLE)
        return None, None, None


def _content_operations(
    page: Any, warnings: list[AnalysisWarningCode]
) -> tuple[int | None, int | None, int | None]:
    try:
        content = page.get_contents()
        if content is None:
            return 0, 0, 0
        stream = ContentStream(content, page.pdf)
        operators = [operator for _operands, operator in stream.operations]
        return (
            sum(operator in _TEXT_OPERATORS for operator in operators),
            sum(operator == b"Do" for operator in operators),
            sum(operator in _PATH_PAINT_OPERATORS for operator in operators),
        )
    except Exception:
        warnings.append(AnalysisWarningCode.CONTENT_STREAM_UNAVAILABLE)
        return None, None, None


def _unicode_corruption_count(text: str) -> int:
    total = 0
    for character in text:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if character == "\ufffd" or 0xE000 <= codepoint <= 0xF8FF:
            total += 1
        elif category in {"Cc", "Cf"} and character not in {"\n", "\t"}:
            total += 1
    return total


def _isolated_arabic_ratio(text: str) -> float | None:
    arabic_tokens = [
        token for token in re.findall(r"\S+", text) if _ARABIC_RE.search(token)
    ]
    if not arabic_tokens:
        return None
    isolated = sum(
        1 for token in arabic_tokens if len(_ARABIC_LETTER_RE.findall(token)) == 1
    )
    return isolated / len(arabic_tokens)


def analyze_pdf_page(
    page: Any,
    page_number: int,
    document: DocumentRoutingContext,
) -> PageAnalysis:
    warnings: list[AnalysisWarningCode] = []
    width, height = _page_dimensions(page, warnings)
    embedded_text, spans = _extract_text_and_spans(page, warnings)
    normalized_text = normalize_embedded_text(embedded_text)
    page_area = width * height
    (
        coverage,
        horizontal_distribution,
        vertical_distribution,
        reading_order_risk,
        multi_column_risk,
    ) = _text_geometry(spans, width, height)
    image_count, largest_image_pixels, image_density = _image_evidence(
        page, page_area, warnings
    )
    text_ops, image_ops, path_ops = _content_operations(page, warnings)

    non_whitespace = sum(not character.isspace() for character in embedded_text)
    alphanumeric = sum(character.isalnum() for character in embedded_text)
    words = re.findall(r"\S+", normalized_text)
    lines = [line for line in embedded_text.splitlines() if line.strip()]
    arabic_count = len(_ARABIC_RE.findall(normalized_text))
    arabic_ratio = arabic_count / max(1, alphanumeric)
    isolated_ratio = _isolated_arabic_ratio(normalized_text)
    pathological_spacing = len(_ARABIC_SPACING_RE.findall(embedded_text))
    corruption = _unicode_corruption_count(embedded_text)
    fragmentation_ratio = (
        sum(len(word) == 1 for word in words) / len(words) if words else 0.0
    )

    all_structural_evidence_available = all(
        value is not None for value in (image_count, text_ops, image_ops, path_ops)
    )
    blank = bool(
        all_structural_evidence_available
        and not normalized_text
        and image_count == 0
        and text_ops == 0
        and image_ops == 0
        and path_ops == 0
    )
    is_page_number = bool(_PAGE_NUMBER_RE.fullmatch(normalized_text))
    small_isolated_image = bool(
        not normalized_text
        and image_count
        and image_density is not None
        and image_density <= _SMALL_IMAGE_DENSITY_LIMIT
        and (path_ops or 0) == 0
    )
    near_blank = bool(
        (
            is_page_number
            and len(normalized_text) <= _NEAR_BLANK_TEXT_LIMIT
            and (image_count or 0) == 0
            and (path_ops or 0) == 0
        )
        or small_isolated_image
    )
    sparse = bool(
        normalized_text
        and (len(normalized_text) < 40 or (coverage is not None and coverage < 0.002))
    )
    fragmented = bool(
        len(words) >= 8 and (fragmentation_ratio >= 0.60 or len(spans) > len(words) * 2)
    )
    hybrid = bool(image_count and normalized_text)
    arabic_integrity_risk = bool(
        arabic_count
        and (
            (isolated_ratio is not None and isolated_ratio >= 0.60)
            or pathological_spacing > 0
            or corruption > 0
        )
    )

    evidence_codes: list[str] = []
    if normalized_text:
        evidence_codes.append("embedded_text_present")
    if image_count:
        evidence_codes.append("embedded_images_present")
    if blank:
        evidence_codes.append("blank_content_stream")
    if near_blank:
        evidence_codes.append("near_blank_structural_evidence")
    if sparse:
        evidence_codes.append("suspicious_sparse_text")
    if fragmented:
        evidence_codes.append("suspicious_fragmentation")
    if hybrid:
        evidence_codes.append("hybrid_page")
    if arabic_integrity_risk:
        evidence_codes.append("arabic_integrity_risk")

    return PageAnalysis(
        document_sha256=document.pdf_sha256,
        page_number=page_number,
        page_width_pt=width,
        page_height_pt=height,
        embedded_text=embedded_text,
        normalized_text=normalized_text,
        embedded_text_present=bool(normalized_text),
        raw_character_count=len(embedded_text),
        normalized_character_count=len(normalized_text),
        non_whitespace_character_count=non_whitespace,
        alphanumeric_character_count=alphanumeric,
        word_count=len(words),
        line_count=len(lines),
        span_count=len(spans),
        spans=spans,
        text_coverage_ratio=coverage,
        horizontal_distribution=horizontal_distribution,
        vertical_distribution=vertical_distribution,
        image_count=image_count,
        largest_image_pixels=largest_image_pixels,
        largest_image_density_estimate=image_density,
        visible_text_operations=text_ops,
        visible_image_operations=image_ops,
        visible_path_operations=path_ops,
        blank_evidence=blank,
        near_blank_evidence=near_blank,
        suspicious_sparse_text=sparse,
        suspicious_fragmentation=fragmented,
        hybrid_page=hybrid,
        arabic_character_count=arabic_count,
        arabic_ratio=arabic_ratio,
        isolated_arabic_character_ratio=isolated_ratio,
        pathological_arabic_spacing_count=pathological_spacing,
        unicode_corruption_count=corruption,
        arabic_integrity_risk=arabic_integrity_risk,
        reading_order_risk=reading_order_risk,
        multi_column_risk=multi_column_risk,
        warnings=tuple(dict.fromkeys(warnings)),
        evidence_codes=tuple(evidence_codes),
    )


def _gate_check(
    name: str,
    passed: bool,
    reason: DigitalTextReasonCode,
    **evidence: int | float | bool | None,
) -> GateCheck:
    return GateCheck(
        name=name,
        passed=passed,
        reason_code=reason,
        evidence=tuple(sorted(evidence.items())),
    )


def evaluate_digital_text_trust(analysis: PageAnalysis) -> DigitalTextGateResult:
    missing_evidence = bool(analysis.warnings)
    has_text = analysis.embedded_text_present and bool(analysis.normalized_text)
    usable_text = (
        analysis.normalized_character_count >= 40
        and analysis.word_count >= 6
        and analysis.alphanumeric_character_count >= 30
    )
    distributed_text = bool(
        analysis.text_coverage_ratio is not None
        and analysis.text_coverage_ratio >= 0.002
        and analysis.horizontal_distribution is not None
        and sum(value > 0 for value in analysis.horizontal_distribution) >= 1
    )
    isolated_arabic = bool(
        analysis.arabic_character_count
        and analysis.isolated_arabic_character_ratio is not None
        and analysis.isolated_arabic_character_ratio >= 0.60
    )
    pathological_arabic_spacing = bool(
        analysis.arabic_character_count
        and analysis.pathological_arabic_spacing_count > 0
    )
    unicode_corruption = analysis.unicode_corruption_count > 0
    fragmented = analysis.suspicious_fragmentation
    image_dominant_partial = bool(
        analysis.hybrid_page
        and analysis.largest_image_density_estimate is not None
        and analysis.largest_image_density_estimate > _SMALL_IMAGE_DENSITY_LIMIT
        and not usable_text
    )
    incomplete_hybrid = bool(analysis.hybrid_page and not usable_text)
    decorative_image = bool(
        analysis.hybrid_page
        and analysis.largest_image_density_estimate is not None
        and analysis.largest_image_density_estimate <= _SMALL_IMAGE_DENSITY_LIMIT
        and usable_text
    )
    layout_order_risk = analysis.reading_order_risk is True
    complex_layout_risk = analysis.multi_column_risk is True

    checks = (
        _gate_check(
            "embedded_text",
            has_text,
            (
                DigitalTextReasonCode.COMPLETE_DIGITAL_TEXT
                if has_text
                else DigitalTextReasonCode.NO_EMBEDDED_TEXT
            ),
            characters=analysis.normalized_character_count,
        ),
        _gate_check(
            "usable_text",
            usable_text,
            (
                DigitalTextReasonCode.COMPLETE_DIGITAL_TEXT
                if usable_text
                else DigitalTextReasonCode.INSUFFICIENT_USABLE_TEXT
            ),
            characters=analysis.normalized_character_count,
            words=analysis.word_count,
        ),
        _gate_check(
            "text_distribution",
            distributed_text,
            (
                DigitalTextReasonCode.SUFFICIENT_TEXT_DISTRIBUTION
                if distributed_text
                else DigitalTextReasonCode.SUSPICIOUS_SPARSE_TEXT
            ),
            coverage=analysis.text_coverage_ratio,
        ),
        _gate_check(
            "fragmentation",
            not fragmented,
            DigitalTextReasonCode.SUSPICIOUS_FRAGMENTATION,
            fragmented=fragmented,
        ),
        _gate_check(
            "arabic_isolation",
            not isolated_arabic,
            DigitalTextReasonCode.ARABIC_ISOLATED_CHARACTER_RUNS,
            isolated_ratio=analysis.isolated_arabic_character_ratio,
        ),
        _gate_check(
            "arabic_spacing",
            not pathological_arabic_spacing,
            DigitalTextReasonCode.ARABIC_PATHOLOGICAL_SPACING,
            occurrences=analysis.pathological_arabic_spacing_count,
        ),
        _gate_check(
            "unicode_integrity",
            not unicode_corruption,
            DigitalTextReasonCode.UNICODE_CORRUPTION_DETECTED,
            corrupt_codepoints=analysis.unicode_corruption_count,
        ),
        _gate_check(
            "image_dominance",
            not image_dominant_partial,
            DigitalTextReasonCode.IMAGE_DOMINANT_PARTIAL_TEXT,
            image_density=analysis.largest_image_density_estimate,
        ),
        _gate_check(
            "hybrid_completeness",
            not incomplete_hybrid,
            DigitalTextReasonCode.HYBRID_PAGE_INCOMPLETE_TEXT,
            hybrid=analysis.hybrid_page,
        ),
        _gate_check(
            "reading_order",
            not layout_order_risk,
            DigitalTextReasonCode.LAYOUT_ORDER_RISK,
            risk=analysis.reading_order_risk,
        ),
        _gate_check(
            "layout_complexity",
            not complex_layout_risk,
            DigitalTextReasonCode.COMPLEX_LAYOUT_RISK,
            risk=analysis.multi_column_risk,
        ),
        _gate_check(
            "required_evidence",
            not missing_evidence,
            DigitalTextReasonCode.REQUIRED_EVIDENCE_UNAVAILABLE,
            warning_count=len(analysis.warnings),
        ),
    )

    reasons: list[DigitalTextReasonCode] = []
    if usable_text:
        reasons.append(DigitalTextReasonCode.COMPLETE_DIGITAL_TEXT)
    if distributed_text:
        reasons.append(DigitalTextReasonCode.SUFFICIENT_TEXT_DISTRIBUTION)
    if decorative_image:
        reasons.append(DigitalTextReasonCode.HARMLESS_DECORATIVE_IMAGES)
    if not has_text:
        reasons.append(DigitalTextReasonCode.NO_EMBEDDED_TEXT)
    if has_text and not usable_text:
        reasons.append(DigitalTextReasonCode.INSUFFICIENT_USABLE_TEXT)
    if analysis.suspicious_sparse_text:
        reasons.append(DigitalTextReasonCode.SUSPICIOUS_SPARSE_TEXT)
    if fragmented:
        reasons.append(DigitalTextReasonCode.SUSPICIOUS_FRAGMENTATION)
    if isolated_arabic:
        reasons.append(DigitalTextReasonCode.ARABIC_ISOLATED_CHARACTER_RUNS)
    if pathological_arabic_spacing:
        reasons.append(DigitalTextReasonCode.ARABIC_PATHOLOGICAL_SPACING)
    if unicode_corruption:
        reasons.append(DigitalTextReasonCode.UNICODE_CORRUPTION_DETECTED)
    if image_dominant_partial:
        reasons.append(DigitalTextReasonCode.IMAGE_DOMINANT_PARTIAL_TEXT)
    if incomplete_hybrid:
        reasons.append(DigitalTextReasonCode.HYBRID_PAGE_INCOMPLETE_TEXT)
    if layout_order_risk:
        reasons.append(DigitalTextReasonCode.LAYOUT_ORDER_RISK)
    if complex_layout_risk:
        reasons.append(DigitalTextReasonCode.COMPLEX_LAYOUT_RISK)
    if missing_evidence:
        reasons.append(DigitalTextReasonCode.REQUIRED_EVIDENCE_UNAVAILABLE)

    definitive_untrusted = any(
        (
            fragmented,
            isolated_arabic,
            pathological_arabic_spacing,
            unicode_corruption,
            image_dominant_partial,
            incomplete_hybrid,
        )
    )
    uncertain = bool(
        missing_evidence
        or layout_order_risk
        or complex_layout_risk
        or (has_text and (not usable_text or not distributed_text))
    )
    if definitive_untrusted or (not has_text and not missing_evidence):
        verdict = DigitalTextGateVerdict.UNTRUSTED
    elif uncertain:
        verdict = DigitalTextGateVerdict.UNCERTAIN
    elif usable_text and distributed_text:
        verdict = DigitalTextGateVerdict.TRUSTED
    else:
        verdict = DigitalTextGateVerdict.UNCERTAIN

    return DigitalTextGateResult(
        verdict=verdict,
        checks=checks,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _decision_reasons(
    analysis: PageAnalysis,
    gate: DigitalTextGateResult,
    extra: str,
) -> tuple[str, ...]:
    reasons = [reason.value for reason in gate.reason_codes]
    reasons.extend(analysis.evidence_codes)
    reasons.append(extra)
    return tuple(dict.fromkeys(reasons))


def decide_page_route(
    analysis: PageAnalysis,
    gate: DigitalTextGateResult,
    *,
    ocr_available: bool,
) -> PageDecisionResult:
    common_evidence: tuple[tuple[str, int | float | bool | str | None], ...] = (
        ("page_number", analysis.page_number),
        ("embedded_text_present", analysis.embedded_text_present),
        ("blank_evidence", analysis.blank_evidence),
        ("near_blank_evidence", analysis.near_blank_evidence),
    )
    if analysis.blank_evidence or analysis.near_blank_evidence:
        reason = (
            "blank_content_stream"
            if analysis.blank_evidence
            else "near_blank_structural_evidence"
        )
        return PageDecisionResult(
            decision=PageDecision.BLANK_OR_NEAR_BLANK,
            next_path=PageNextPath.NO_EXTRACTION,
            gate_verdict=gate.verdict,
            reason_codes=_decision_reasons(analysis, gate, reason),
            evidence=common_evidence,
            ocr_required=False,
            ocr_available=ocr_available,
            review_required=analysis.near_blank_evidence,
        )
    if gate.verdict is DigitalTextGateVerdict.TRUSTED:
        trusted_context = TrustedDigitalTextContext(
            document_sha256=analysis.document_sha256,
            page_number=analysis.page_number,
            normalized_text_sha256=digest_normalized_text(analysis.embedded_text),
            gate_result=gate,
        )
        return PageDecisionResult(
            decision=PageDecision.TRUSTED_DIGITAL_TEXT,
            next_path=PageNextPath.DIRECT_PDF_TEXT,
            gate_verdict=gate.verdict,
            reason_codes=_decision_reasons(analysis, gate, "trusted_gate_passed"),
            evidence=common_evidence,
            ocr_required=False,
            ocr_available=ocr_available,
            review_required=False,
            trusted_context=trusted_context,
        )
    if gate.verdict is DigitalTextGateVerdict.UNTRUSTED:
        return PageDecisionResult(
            decision=PageDecision.OCR_REQUIRED,
            next_path=(
                PageNextPath.LOCAL_OCR
                if ocr_available
                else PageNextPath.PENDING_OCR_MODEL
            ),
            gate_verdict=gate.verdict,
            reason_codes=_decision_reasons(
                analysis,
                gate,
                "ocr_available" if ocr_available else "ocr_unavailable",
            ),
            evidence=common_evidence,
            ocr_required=True,
            ocr_available=ocr_available,
            review_required=not ocr_available,
        )
    return PageDecisionResult(
        decision=PageDecision.REVIEW_REQUIRED,
        next_path=PageNextPath.MANUAL_REVIEW,
        gate_verdict=gate.verdict,
        reason_codes=_decision_reasons(analysis, gate, "gate_uncertain"),
        evidence=common_evidence,
        ocr_required=False,
        ocr_available=ocr_available,
        review_required=True,
    )


def validate_trusted_context_before_extraction(
    context: TrustedDigitalTextContext,
    *,
    document_sha256: str,
    page_number: int,
    gate: DigitalTextGateResult,
) -> bool:
    return bool(
        gate.verdict is DigitalTextGateVerdict.TRUSTED
        and context.document_sha256 == document_sha256
        and context.page_number == page_number
        and context.gate_result == gate
    )


def validate_trusted_text_after_extraction(
    context: TrustedDigitalTextContext,
    extracted_text: str,
) -> bool:
    return context.normalized_text_sha256 == digest_normalized_text(extracted_text)


__all__ = [
    "AnalysisWarningCode",
    "DigitalTextGateResult",
    "DigitalTextGateVerdict",
    "DigitalTextReasonCode",
    "DocumentRoutingContext",
    "EvidenceAvailability",
    "GateCheck",
    "PageAnalysis",
    "PageDecision",
    "PageDecisionResult",
    "PageNextPath",
    "TextSpan",
    "TrustedDigitalTextContext",
    "analyze_pdf_page",
    "decide_page_route",
    "digest_normalized_text",
    "evaluate_digital_text_trust",
    "normalize_embedded_text",
    "validate_trusted_context_before_extraction",
    "validate_trusted_text_after_extraction",
]
