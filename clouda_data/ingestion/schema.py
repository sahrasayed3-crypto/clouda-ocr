from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

SourceType = Literal[
    "pdf",
    "docx",
    "image",
    "text",
    "json_layout",
    "page_xml",
    "alto_xml",
    "ground_truth_json",
]
RegionType = Literal[
    "body",
    "title",
    "heading",
    "footnote",
    "margin",
    "header",
    "footer",
    "page_number",
    "table",
    "image",
    "empty",
]


@dataclass(frozen=True)
class LayoutRegion:
    region_id: str
    region_type: RegionType
    bbox: tuple[int, int, int, int]
    reading_order: int
    language: str = "ar"
    text: str | None = None


@dataclass(frozen=True)
class PageRecord:
    document_id: str
    page_id: str
    source_path: str
    source_type: SourceType
    language: str
    page_number: int
    clean_text: str
    text_checksum: str
    image_checksum: str | None = None
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    footnote_regions: list[str] = field(default_factory=list)
    margin_regions: list[str] = field(default_factory=list)
    title_regions: list[str] = field(default_factory=list)
    table_regions: list[str] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    source_license: str = "unknown"
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_path: str
    source_type: SourceType
    language: str = "ar"
    source_license: str = "unknown"
    title: str | None = None


@dataclass(frozen=True)
class SourcePage:
    document_id: str
    page_id: str
    page_number: int
    source_path: str
    source_type: SourceType
    language: str = "ar"
    ground_truth_path: str | None = None
    clean_text: str | None = None
    text_checksum: str | None = None
    layout_path: str | None = None
    source_license: str = "unknown"
    reading_order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FileRegistration:
    original_path: str
    canonical_path: str
    file_role: str
    source_type: SourceType
    checksum: str
    size_bytes: int
    duplicate_of: str | None = None


@dataclass(frozen=True)
class SourceManifest:
    documents: list[SourceDocument]
    pages: list[SourcePage]
