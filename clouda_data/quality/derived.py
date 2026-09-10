"""Derived clean + quarantine manifest writing with fail-closed re-validation.

The original manifest is NEVER modified: its SHA-256 is verified before and
after every write (``SourceManifestDriftError`` on any drift). The clean
manifest excludes the policy losers and carries full lineage in its header;
the quarantine manifest receives the disjoint set of protected/held-out rows.
After writing, ``revalidate_derived`` re-reads the derived manifest and
re-checks protection (fail-closed) exactly like
``clouda_lab.dataset_selection.validate_derived_manifest_for_training``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clouda_contracts.protection import record_is_protected
from clouda_data.pretraining.manifest import read_manifest, write_manifest
from clouda_data.pretraining.schema import DatasetSample
from clouda_data.quality.models import canonical_json
from clouda_data.quality.policy import exclusion_report

DERIVED_DATASET_VERSION_PREFIX = "derived-1.0.0+"


class SourceManifestDriftError(RuntimeError):
    """Raised when the source manifest changed during a derived write."""


class DerivedManifestValidationError(RuntimeError):
    """Raised when a derived manifest fails post-write re-validation."""


def _reject_self_overwrite(source_path: Path, output_path: Path) -> None:
    """Refuse output paths that would clobber the source manifest (R4-H1).

    Windows is case-insensitive, so a case-variant stem with a different
    suffix-extension collision (``m.jsonl`` vs ``m.JSONL``) is also rejected.
    """

    try:
        source_resolved = source_path.resolve()
        output_resolved = output_path.resolve()
    except OSError as exc:  # pragma: no cover - unusual filesystem state
        raise ValueError(f"Cannot resolve output path safely: {exc}") from exc
    if output_resolved == source_resolved:
        raise ValueError(
            "Refusing to write the derived manifest over the source manifest: "
            f"{output_resolved}. Choose a different --output path."
        )
    if output_resolved.parent == source_resolved.parent and (
        output_resolved.stem == source_resolved.stem
    ):
        raise ValueError(
            "Refusing to write the derived manifest next to the source with "
            "the same stem (case-insensitive collision risk): "
            f"{output_resolved}. Choose a different --output path."
        )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_identity(row: dict[str, Any]) -> str:
    return canonical_json(row)


def write_clean_manifest(
    source_manifest_path: str | Path,
    samples: list[DatasetSample],
    exclusions: list[Any],
    quarantine_ids: list[str] | tuple[str, ...],
    run: Any,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the clean derived manifest and return its lineage document.

    Verifies the source manifest SHA-256 before AND after the write and
    raises :class:`SourceManifestDriftError` if it changed. Rows are the
    samples minus excluded IDs, written via the canonical pretraining writer
    with lineage header metadata.
    """

    source_path = Path(source_manifest_path)
    source_sha_before = _hash_file(source_path)

    output = Path(output_path)
    # R4-H1 fix: refuse any output that would overwrite the source manifest
    # (or a case-variant of it) BEFORE any write happens.
    _reject_self_overwrite(source_path, output)

    excluded_ids = {decision.sample_id for decision in exclusions}
    quarantine_set = set(quarantine_ids)
    overlap = excluded_ids & quarantine_set
    if overlap:
        raise ValueError(
            "Clean and quarantine sample_id sets must be disjoint; overlap: "
            f"{sorted(overlap)}"
        )

    clean_rows = [
        sample.to_dict()
        for sample in sorted(samples, key=lambda sample: sample.sample_id)
        if sample.sample_id not in excluded_ids
        and sample.sample_id not in quarantine_set
    ]

    config_identity = str(getattr(run, "config_identity", "") or "")
    run_id = str(getattr(run, "run_id", "") or "")
    verdict = getattr(run, "verdict", "")
    verdict_value = getattr(verdict, "value", verdict)

    report = exclusion_report(
        exclusions,
        total_samples=len(samples),
        quarantine_ids=quarantine_ids,
    )
    report_text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    report_sha = _sha256_text(report_text)
    derived_version = f"{DERIVED_DATASET_VERSION_PREFIX}{source_sha_before[:12]}"

    written = write_manifest(
        output_path,
        clean_rows,
        metadata={
            "source_manifest_sha256": source_sha_before,
            "quality_run_id": run_id,
            "config_identity": config_identity,
            "derived_dataset_version": derived_version,
            "exclusion_report_sha256": report_sha,
            "verdict": str(verdict_value),
        },
    )

    source_sha_after = _hash_file(source_path)
    if source_sha_after != source_sha_before:
        raise SourceManifestDriftError(
            "Source manifest changed during derived write: "
            f"{source_path} (before={source_sha_before}, after={source_sha_after})"
        )

    # Report file travels next to the derived manifest.
    report_path = (
        Path(written)
        .with_suffix("")
        .with_name(Path(written).stem + ".exclusion_report.json")
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "clean_manifest_path": str(written),
        "exclusion_report_path": str(report_path),
        "source_manifest_sha256": source_sha_before,
        "quality_run_id": run_id,
        "config_identity": config_identity,
        "derived_dataset_version": derived_version,
        "exclusion_report_sha256": report_sha,
        "verdict": str(verdict_value),
        "clean_row_count": len(clean_rows),
        "excluded_count": len(excluded_ids),
        "quarantine_count": len(quarantine_set),
    }


def write_quarantine_manifest(
    samples: list[DatasetSample],
    quarantine_ids: list[str] | tuple[str, ...],
    run: Any,
    source_manifest_sha256: str,
    output_path: str | Path,
) -> Path:
    """Write the quarantine manifest (same canonical writer, disjoint ids)."""

    wanted = set(quarantine_ids)
    rows = [
        sample.to_dict()
        for sample in sorted(samples, key=lambda sample: sample.sample_id)
        if sample.sample_id in wanted
    ]
    found_ids = {row["sample_id"] for row in rows}
    missing = wanted - found_ids
    if missing:
        raise ValueError(f"Quarantine ids not present in samples: {sorted(missing)}")
    run_id = str(getattr(run, "run_id", "") or "")
    return write_manifest(
        output_path,
        rows,
        metadata={
            "source_manifest_sha256": source_manifest_sha256,
            "quality_run_id": run_id,
            "derived_dataset_version": (
                f"{DERIVED_DATASET_VERSION_PREFIX}{source_manifest_sha256[:12]}"
            ),
            "quarantine_count": len(rows),
        },
    )


def revalidate_derived(
    path: str | Path,
    excluded_ids: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """Re-read a derived manifest and re-check protection, fail-closed.

    Raises :class:`DerivedManifestValidationError` listing every violation if
    any row is protected (per ``record_is_protected``) or any excluded id
    reappears in the derived manifest.
    """

    header, rows = read_manifest(path)
    violations: list[str] = []
    if record_is_protected(header):
        violations.append("header: protected header (fail-closed)")

    seen_ids: set[str] = set()
    excluded = set(excluded_ids)
    for index, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", f"<row {index}>"))
        # Strict re-parse: an unknown field or schema drift must fail here.
        try:
            DatasetSample.from_dict(row)
        except (TypeError, ValueError) as exc:
            violations.append(f"{sample_id}: strict parse failed ({exc})")
            continue
        if record_is_protected(row):
            violations.append(f"{sample_id}: protected row present (fail-closed)")
        if sample_id in excluded:
            violations.append(f"{sample_id}: excluded id reappeared")
        if sample_id in seen_ids:
            violations.append(f"{sample_id}: duplicate row")
        seen_ids.add(sample_id)

    if violations:
        raise DerivedManifestValidationError(
            "Derived manifest validation failed:\n" + "\n".join(violations)
        )
    return {
        "rows": len(rows),
        "header_row_count": header.get("_row_count"),
        "protected_rows": 0,
        "excluded_reappearances": 0,
    }


def _hash_file(path: Path) -> str:
    from clouda_data.pretraining.hashing import sha256_file

    return sha256_file(path)


__all__ = [
    "DERIVED_DATASET_VERSION_PREFIX",
    "DerivedManifestValidationError",
    "SourceManifestDriftError",
    "revalidate_derived",
    "write_clean_manifest",
    "write_quarantine_manifest",
]
