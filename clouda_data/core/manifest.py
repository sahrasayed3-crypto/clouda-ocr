from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .status import PageStatus


@dataclass(frozen=True)
class GeneratedPageManifestEntry:
    source_document_id: str
    source_page_id: str
    generated_page_id: str
    source_image_path: str
    output_image_path: str
    ground_truth_path: str
    source_checksum: str
    output_checksum: str
    text_checksum: str
    distortion_profile: str
    distortion_operations: list[str]
    operation_order: list[str]
    parameters: dict[str, Any]
    random_seed: int
    render_dpi: int
    image_width: int
    image_height: int
    source_language: str
    layout_metadata: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    validation_status: PageStatus = PageStatus.QUEUED
    rejection_reason: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software_version: str = "0.1.0"
