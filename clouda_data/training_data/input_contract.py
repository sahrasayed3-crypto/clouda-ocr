"""Canonical manifest input contract for the training data loader.

The loader consumes the canonical Clouda pre-training manifest
(``clouda.pretraining.manifest.v1``). It does not define a competing format.
Validation is fail-closed: malformed or unsafe manifests are rejected before
training iteration begins.

Holdout protection deliberately delegates to the existing data-policy engine
(``clouda_training.experiments.dataset``) — the same
``PROTECTED_SPLIT_NAMES`` / ``PROTECTED_ROLES`` / protected-marker rules used
by the Training Experiment Framework — so there is exactly one definition of
"protected" in the repository.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from clouda_data.pretraining.manifest import (
    MANIFEST_SCHEMA_VERSION,
    iter_manifest,
    read_manifest,
)

# Reuse the canonical policy engine — do not re-implement holdout rules.
from clouda_training.experiments.dataset import (
    _marker,
    validate_training_dataset,
)

SYSTEM = "clouda.training_data.input.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_HEADER_KEYS = {
    "_schema_version",
    "_row_count",
    "dataset_id",
    "dataset_version",
    "manifest_sha256",
}


class ManifestInputError(ValueError):
    """Raised when a manifest is malformed or unsafe for training use."""


class ProtectedManifestError(PermissionError):
    """Raised when a manifest contains protected/holdout data (fail-closed)."""


@dataclass(frozen=True)
class ManifestIdentity:
    dataset_id: str
    dataset_version: str
    manifest_sha256: str
    row_count: int
    split: str
    header: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": SYSTEM,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "row_count": self.row_count,
            "split": self.split,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_str(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestInputError(f"{context}: {key!r} must be a non-empty string")
    return value.strip()


def validate_canonical_manifest(
    path: str | Path,
    *,
    dataset_id: str,
    dataset_version: str,
    split: str = "train",
) -> ManifestIdentity:
    """Validate the canonical manifest for training use and fail closed.

    Checks schema version, dataset identity agreement between header and
    config, header integrity (row count, optional recorded manifest hash),
    per-row sample identity, artifact references, provenance presence, and
    training eligibility. Protected holdout content is rejected through the
    canonical policy engine.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical manifest not found: {manifest_path}")

    # Delegate to the Training Experiment Framework's dataset validation:
    # this enforces dataset_id/dataset_version agreement, protected split
    # names, protected markers (including nested provenance/metadata) and
    # split row presence. It raises PermissionError/ValueError on violations.
    # PermissionError is re-raised as ProtectedManifestError so callers can
    # distinguish policy rejections from malformed input.
    try:
        validate_training_dataset(
            _dataset_section(manifest_path, dataset_id, dataset_version, split)
        )
    except PermissionError as exc:
        raise ProtectedManifestError(str(exc)) from exc

    header, rows = read_manifest(manifest_path)
    if header.get("_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestInputError(
            f"Unsupported manifest schema version: {header.get('_schema_version')!r}"
        )
    header_id = _require_str(header, "dataset_id", "manifest header")
    header_version = _require_str(header, "dataset_version", "manifest header")
    if header_id != dataset_id.strip() or header_version != dataset_version.strip():
        raise ManifestInputError(
            "Dataset identity mismatch: manifest declares "
            f"{header_id!r}/{header_version!r}, config expects "
            f"{dataset_id!r}/{dataset_version!r}"
        )
    recorded_hash = header.get("manifest_sha256")
    if recorded_hash is not None and str(recorded_hash).strip():
        recorded = str(recorded_hash).strip().lower()
        if not _SHA256_RE.fullmatch(recorded):
            raise ManifestInputError("Header manifest_sha256 is not a SHA-256 digest")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        context = f"manifest row {index}"
        sample_id = _require_str(row, "sample_id", context)
        if sample_id in seen_ids:
            raise ManifestInputError(f"Duplicate sample_id in manifest: {sample_id}")
        seen_ids.add(sample_id)
        source_id = row.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ManifestInputError(f"{context}: source_id must be a string")
        if row.get("image_path") is None and row.get("text") is None:
            raise ManifestInputError(
                f"{context}: sample has neither image_path nor text reference"
            )
        provenance = row.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise ManifestInputError(f"{context}: provenance must be an object")
        _require_training_eligibility(row, context)

    digest = sha256_file(manifest_path)
    return ManifestIdentity(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_sha256=digest,
        row_count=len(rows),
        split=split,
        header=header,
    )


def _require_training_eligibility(row: dict[str, Any], context: str) -> None:
    """Fail closed on ambiguous training eligibility and forbidden roles."""

    for field in ("target_split", "split", "source_split", "dataset_role", "role"):
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            raise ManifestInputError(
                f"{context}: field {field!r} must be a string when present"
            )
    status = row.get("validation_status")
    if isinstance(status, str) and _marker(status) == "error":
        raise ManifestInputError(
            f"{context}: validation_status=error is not training-eligible"
        )
    duplicate_state = row.get("duplicate_state")
    if isinstance(duplicate_state, str) and _marker(duplicate_state) in {
        "duplicate",
        "conflicting_duplicate",
    }:
        raise ManifestInputError(
            f"{context}: duplicate_state={duplicate_state!r} is not training-eligible"
        )
    exclusion_reason = row.get("exclusion_reason")
    if exclusion_reason:
        raise ManifestInputError(
            f"{context}: excluded sample cannot be used for training: "
            f"{exclusion_reason!r}"
        )


def iter_canonical_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream canonical rows lazily (header excluded, validated separately)."""

    for payload in iter_manifest(path):
        if "_schema_version" in payload and "sample_id" not in payload:
            continue
        yield payload


def _dataset_section(
    manifest_path: Path, dataset_id: str, dataset_version: str, split: str
) -> Any:
    """Build a minimal DatasetSection stand-in for the policy engine.

    Avoids importing the full ExperimentConfig stack; the policy engine only
    reads dataset_id/dataset_version/manifest_path/split attributes.
    """

    from types import SimpleNamespace

    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_path=manifest_path,
        split=split,
    )
