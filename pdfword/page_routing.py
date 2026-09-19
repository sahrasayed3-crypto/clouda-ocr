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


__all__ = [
    "AnalysisWarningCode",
    "DocumentRoutingContext",
    "EvidenceAvailability",
    "PageAnalysis",
    "TextSpan",
    "analyze_pdf_page",
    "digest_normalized_text",
    "normalize_embedded_text",
]
