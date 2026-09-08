"""Versioned domain schema for OCR dataset samples.

A :class:`DatasetSample` is one logical OCR training unit: one image (when
present) paired with its ground-truth text. The schema is stable, versioned,
and JSON-serializable. Rows are plain dictionaries when written to manifests;
``schema_version`` travels with every row.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Any

SCHEMA_VERSION = "clouda.pretraining.sample.v1"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:($|/)")


def _contains_unsafe_format_character(value: str) -> bool:
    """Detect invisible controls and direction-changing format characters."""

    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


def canonical_relative_path(value: str, *, allow_empty: bool = False) -> str:
    """Return a platform-neutral, root-relative provenance path.

    This is deliberately lexical: callers that access the filesystem must also
    resolve the result beneath their configured root to defend against symlinks.
    """

    if not isinstance(value, str):
        raise TypeError("Provenance paths must be strings.")
    if _contains_unsafe_format_character(value):
        raise ValueError(
            "Provenance paths cannot contain control or invisible format characters."
        )
    normalized_separators = value.replace("\\", "/")
    if (
        normalized_separators.startswith("/")
        or normalized_separators.startswith("//")
        or _WINDOWS_DRIVE_RE.match(normalized_separators)
    ):
        raise ValueError(f"Provenance path must be relative: {value!r}")
    normalized = posixpath.normpath(normalized_separators)
    if normalized == ".":
        normalized = ""
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"Provenance path escapes its root: {value!r}")
    if not normalized and not allow_empty:
        raise ValueError("Provenance path cannot be empty.")
    return normalized


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
    """Derive an id from source id plus canonical root-relative coordinates."""

    if not source_id or _contains_unsafe_format_character(source_id):
        raise ValueError("source_id must be a non-empty string without controls.")
    canonical_source_path = canonical_relative_path(source_path)
    canonical_record_key = canonical_relative_path(record_key, allow_empty=True)

    digest = hashlib.sha256(
        "\x00".join((source_id, canonical_source_path, canonical_record_key)).encode(
            "utf-8"
        )
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
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown sample fields: {sorted(unknown)}")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sample schema version: {version!r}; expected "
                f"{SCHEMA_VERSION!r}."
            )
        kwargs = dict(data)
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
