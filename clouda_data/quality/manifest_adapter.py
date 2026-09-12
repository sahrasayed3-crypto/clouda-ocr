"""Manifest adapter between the quality gate and pretraining manifests.

Single boundary module: every read of a canonical pretraining manifest by the
quality gate, every file-hash identity, and every on-disk artifact path
resolution flows through here so the safety rules (canonical relative paths,
containment under the configured root, symlink refusal before any open) are
applied exactly once.

Loader adapter boundary: the clean derived manifest produced by
:mod:`clouda_data.quality.derived` is accepted directly by the canonical
Training Data Loader input contract; see :func:`training_stream_contract`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from clouda_contracts.storage import validate_relative_components
from clouda_data.pretraining import hashing
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.pretraining.schema import (
    DatasetSample,
    canonical_relative_path,
)
from clouda_data.quality.models import canonical_json

QUALITY_RUN_ID_PREFIX = "QRUN_"
RUN_ID_DIGEST_LENGTH = 16


class UnsafeArtifactPath(ValueError):
    """Raised when a sample's artifact path fails a safety check."""


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[DatasetSample]]:
    """Read a canonical manifest and parse rows strictly into samples."""

    header, rows = read_manifest(path)
    samples = [DatasetSample.from_dict(row) for row in rows]
    return header, samples


def manifest_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of the manifest file (pretraining hashing rules)."""

    return hashing.sha256_file(path)


def resolve_artifact_path(sample: DatasetSample, root: str | Path) -> Path:
    """Resolve ``sample.image_path`` safely beneath ``root``.

    Defense in depth, in order:

    1. lexical canonical-relative-path check (schema rules);
    2. per-component safety validation (contracts storage rules);
    3. resolved containment under the resolved root (symlink-proof);
    4. refusal of symlinked final paths **before any open** (PATH_SAFE).
    """

    root_path = Path(root)
    if sample.image_path is None:
        raise UnsafeArtifactPath(
            f"Sample {sample.sample_id!r} has no image_path to resolve."
        )
    try:
        canonical = canonical_relative_path(sample.image_path)
    except (TypeError, ValueError) as exc:
        raise UnsafeArtifactPath(
            f"Sample {sample.sample_id!r} image_path is not a canonical "
            f"relative path: {sample.image_path!r} ({exc})"
        ) from exc

    relative = Path(canonical)
    try:
        validate_relative_components(relative)
    except ValueError as exc:
        raise UnsafeArtifactPath(
            f"Sample {sample.sample_id!r} image_path has unsafe components: "
            f"{canonical!r} ({exc})"
        ) from exc

    resolved_root = root_path.expanduser().resolve(strict=False)
    resolved = (resolved_root / relative).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeArtifactPath(
            f"Sample {sample.sample_id!r} image_path escapes the artifact "
            f"root: {sample.image_path!r}"
        ) from exc

    if resolved.is_symlink() or (resolved_root / relative).is_symlink():
        raise UnsafeArtifactPath(
            f"Sample {sample.sample_id!r} image_path is a symbolic link; "
            f"refusing to open: {canonical!r} (PATH_SAFE)"
        )
    return resolved


def run_identity(
    manifest_sha: str,
    config_identity: str,
    algorithm_versions: dict[str, str],
) -> dict[str, str]:
    """Deterministic run identity derived from manifest, config, algorithms."""

    payload = {
        "manifest_sha256": manifest_sha,
        "config_identity": config_identity,
        "algorithm_versions": {
            str(key): str(value) for key, value in sorted(algorithm_versions.items())
        },
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    run_id = f"{QUALITY_RUN_ID_PREFIX}{digest[:RUN_ID_DIGEST_LENGTH]}"
    return {
        "run_id": run_id,
        "manifest_sha256": manifest_sha,
        "config_identity": config_identity,
        "algorithm_versions": json.dumps(payload["algorithm_versions"]),
        "identity_fingerprint": digest,
    }


TRAINING_STREAM_CONTRACT = """\
Training Data Loader connection: CANONICAL.

The input contract for a training run is the clean derived manifest written
by clouda_data.quality.derived:

1. The derived manifest is a canonical ``clouda.pretraining.manifest.v1``
   JSONL file written by ``pretraining.manifest.write_manifest`` (atomic,
   canonically sorted rows).
2. Its header carries ``dataset_id`` and the derived ``dataset_version`` plus
   full lineage: ``source_dataset_version``, ``source_manifest_sha256``,
   ``quality_run_id``, ``config_identity``, ``derived_dataset_version``,
   ``exclusion_report_sha256``, and the gate ``verdict``.
3. It contains no excluded sample IDs, no protected rows (fail-closed
   re-validation via ``revalidate_derived``), and is disjoint from the
   quarantine manifest.
4. Consumers MUST re-check protection at read time. The canonical
   ``validate_canonical_manifest`` check in
   ``clouda_data.training_data.input_contract`` enforces identity, integrity,
   explicit train eligibility, and fail-closed protection before
   ``StreamingTrainingDataLoader`` yields any row.
"""


def training_stream_contract() -> str:
    """Return the documented deferred Training Data Loader contract."""

    return TRAINING_STREAM_CONTRACT


__all__ = [
    "RUN_ID_DIGEST_LENGTH",
    "QUALITY_RUN_ID_PREFIX",
    "UnsafeArtifactPath",
    "load_manifest",
    "manifest_sha256",
    "resolve_artifact_path",
    "run_identity",
    "training_stream_contract",
]
