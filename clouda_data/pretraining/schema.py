"""Versioned domain schema for OCR dataset samples.

A :class:`DatasetSample` is one logical OCR training unit: one image (when
present) paired with its ground-truth text. The schema is stable, versioned,
and JSON-serializable. Rows are plain dictionaries when written to manifests;
``schema_version`` travels with every row.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Any

SCHEMA_VERSION = "clouda.pretraining.sample.v1"


class SplitName(str, Enum):
    UNASSIGNED = "unassigned"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"


class DuplicateState(str, Enum):
    UNIQUE = "unique"
    CANONICAL = "canonical"
    DUPLICATE = "duplicate"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"


class ValidationStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    EXCLUDED = "excluded"


EXPORTABLE_SPLITS = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
EXPORTABLE_STATUSES = (ValidationStatus.OK, ValidationStatus.WARNING)
EXPORTABLE_DUPLICATE_STATES = (
    DuplicateState.UNIQUE,
    DuplicateState.CANONICAL,
    DuplicateState.CONFLICTING_DUPLICATE,
)


def stable_sample_id(source_id: str, source_path: str, record_key: str = "") -> str:
    """Derive a deterministic sample id from provenance coordinates."""

    digest = hashlib.sha256(
        "\x00".join((source_id, source_path, record_key)).encode("utf-8")
    ).hexdigest()
    return f"smp_{digest[:20]}"


@dataclass(frozen=True)
class DatasetSample:
    """One logical OCR dataset sample with full provenance."""

    sample_id: str
    source_id: str
    source_dataset: str = ""
    source_path: str = ""
    source_record_id: str | None = None
    source_url: str | None = None
    source_license: str | None = None
    source_split: str | None = None
    document_id: str | None = None
    page_id: str | None = None
    page_index: int | None = None
    group_id: str | None = None
    image_path: str | None = None
    text: str | None = None
    raw_text: str | None = None
    language: str = "ar"
    script: str = "arabic"
    document_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    file_extension: str | None = None
    mime_type: str | None = None
    file_sha256: str | None = None
    normalized_text_sha256: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    transformations: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.OK
    validation_findings: list[dict[str, Any]] = field(default_factory=list)
    exclusion_reason: str | None = None
    duplicate_state: DuplicateState = DuplicateState.UNIQUE
    duplicate_of: str | None = None
    target_split: SplitName = SplitName.UNASSIGNED
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation_status"] = self.validation_status.value
        payload["duplicate_state"] = self.duplicate_state.value
        payload["target_split"] = self.target_split.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSample:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["validation_status"] = ValidationStatus(
            kwargs.get("validation_status", ValidationStatus.OK.value)
        )
        kwargs["duplicate_state"] = DuplicateState(
            kwargs.get("duplicate_state", DuplicateState.UNIQUE.value)
        )
        kwargs["target_split"] = SplitName(
            kwargs.get("target_split", SplitName.UNASSIGNED.value)
        )
        return cls(**kwargs)

    def evolve(self, **changes: Any) -> DatasetSample:
        return replace(self, **changes)


def sort_key(sample: DatasetSample) -> tuple[str, str, str]:
    """Canonical deterministic ordering for samples and manifest rows."""

    return (
        sample.source_id,
        sample.source_path,
        sample.sample_id,
    )
