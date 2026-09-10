"""Clouda Data Factory ↔ pre-training / training-framework adapters.

This module is the canonical integration seam between the synthetic data
factory and the rest of Clouda:

    Data Factory run manifest (factory/v1 rows)
      → :func:`factory_rows_to_samples`      (lossless provenance mapping)
      → :func:`write_pretraining_manifest`   (canonical
        ``clouda.pretraining.manifest.v1`` JSONL, atomic write)
      → :func:`assign_splits`                (leakage-safe splitting)
      → Training Experiment Framework dry-run
        (``clouda_training.experiments.run_experiment``, mock adapter only)

Contract rules:

- provenance is preserved, never dropped: source identity (``source_ref``),
  source hash (``source_sha256``), output hash (``output_sha256``), seed,
  seed mode, profile, transform steps, renderer, QC record, and ground-truth
  reference all travel inside the sample's ``provenance`` dict;
- every factory sample is marked ``factory_generated: true`` and its split is
  reset so the standard leakage-safe splitter assigns it;
- ground-truth text, when the run recorded one, becomes the sample text
  (raw and normalized — the factory contract forbids text rewriting, so the
  normalized copy equals the raw copy);
- ``holdout`` data can never originate here: the factory consumes raw inputs
  directly; the adapter additionally refuses rows that reference a protected
  split marker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..pretraining.hashing import sha256_file, sha256_text
from ..pretraining.schema import (
    DatasetSample,
    SplitName,
    canonical_relative_path,
    stable_sample_id,
)

FACTORY_ADAPTER_VERSION = "clouda.factory.adapter.v1"


class FactoryAdapterError(ValueError):
    """Raised when a factory manifest cannot be converted safely."""


_PROTECTED_SPLIT_MARKERS = {
    "holdout",
    "protected_holdout",
    "benchmark_holdout",
    "private_holdout",
}
_PROTECTED_ROLES = {
    "holdout",
    "protected_holdout",
    "benchmark",
    "evaluation_only",
}
_TRAINING_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _normalized_marker(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FactoryAdapterError(f"Factory row has malformed {field} metadata")
    return value.strip().casefold()


def _mapping_is_protected(mapping: Any, *, nested: bool = False) -> bool:
    if not isinstance(mapping, dict):
        if nested:
            raise FactoryAdapterError("Factory row has malformed protection metadata")
        return False
    protected = mapping.get("protected")
    if "protected" in mapping and not (
        isinstance(protected, bool)
        or (
            isinstance(protected, str)
            and protected.strip().casefold()
            in {"true", "yes", "1", "protected", "false", "no", "0"}
        )
    ):
        raise FactoryAdapterError("Factory row has malformed protection metadata")
    if protected is True or (
        isinstance(protected, str)
        and protected.strip().casefold() in {"true", "yes", "1", "protected"}
    ):
        return True
    for field in ("target_split", "split", "source_split"):
        if field in mapping:
            marker = _normalized_marker(mapping[field], field=field)
            if marker in _PROTECTED_SPLIT_MARKERS or "holdout" in marker:
                return True
    for field in ("dataset_role", "role", "purpose"):
        if field in mapping:
            marker = _normalized_marker(mapping[field], field=field)
            if marker in _PROTECTED_ROLES or "holdout" in marker:
                return True
    return False


def _portable_source_ref(value: Any) -> str:
    """Preserve a source label without leaking a machine-local absolute path."""
    text = str(value or "")
    windows_path = PureWindowsPath(text)
    posix_path = PurePosixPath(text)
    if windows_path.is_absolute():
        return windows_path.name
    if posix_path.is_absolute():
        return posix_path.name
    return canonical_relative_path(text, allow_empty=True)


@dataclass(frozen=True)
class AdapterReport:
    """Counts describing one conversion."""

    rows_total: int
    rows_ok: int
    rows_error: int
    rows_skipped: int
    samples: int
    adapter_version: str = FACTORY_ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "rows_ok": self.rows_ok,
            "rows_error": self.rows_error,
            "rows_skipped": self.rows_skipped,
            "samples": self.samples,
            "adapter_version": self.adapter_version,
        }


def read_factory_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a Data Factory run manifest (JSONL rows, one per artifact)."""
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FactoryAdapterError(
                    f"Malformed factory manifest line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise FactoryAdapterError(
                    f"Factory manifest line {line_number} is not an object"
                )
            rows.append(payload)
    return rows


def factory_rows_to_samples(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    run_dir: str | Path | None = None,
) -> tuple[list[DatasetSample], AdapterReport]:
    """Convert factory manifest rows into canonical dataset samples.

    Only rows with ``status == "ok"`` become samples; error rows are counted
    and skipped (they carry no output artifact). Rows whose provenance marks a
    protected split are rejected outright.
    """
    samples: list[DatasetSample] = []
    ok = error = skipped = 0
    for row in rows:
        status = str(row.get("status", ""))
        if status == "error":
            error += 1
            continue
        if status != "ok":
            skipped += 1
            continue
        profile = str(row.get("profile", ""))
        if _mapping_is_protected(row) or any(
            _mapping_is_protected(row[field], nested=True)
            for field in ("provenance", "metadata")
            if field in row and row[field] is not None
        ):
            raise FactoryAdapterError(
                "Factory row references protected data and cannot be converted"
            )
        run_id = str(row.get("run_id", ""))
        document_id = str(row.get("document_id", ""))
        page_index = row.get("page_index")
        variant_id = str(row.get("variant_id", ""))
        record_key = f"{run_id}:{document_id}:{page_index}:{variant_id}"
        output_rel = canonical_relative_path(str(row.get("output_path", "")))
        if PurePosixPath(output_rel).suffix.casefold() not in _TRAINING_IMAGE_SUFFIXES:
            raise FactoryAdapterError(
                "Factory row must reference a raster image for training conversion"
            )
        run_root = Path(run_dir) if run_dir is not None else None
        gt_path_value = row.get("gt_path")
        gt_hash_value = row.get("gt_sha256")
        gt_path_missing = gt_path_value is None or gt_path_value == ""
        gt_hash_missing = gt_hash_value is None or gt_hash_value == ""
        if gt_path_missing != gt_hash_missing:
            raise FactoryAdapterError(
                "Factory row has unpaired ground-truth path and checksum"
            )
        if not gt_path_missing and (
            not isinstance(gt_path_value, str) or not isinstance(gt_hash_value, str)
        ):
            raise FactoryAdapterError(
                "Factory row has malformed ground-truth provenance"
            )
        gt_path = str(gt_path_value or "")
        gt_text: str | None = None
        if gt_path:
            if run_root is None:
                raise FactoryAdapterError(
                    "Factory ground-truth path requires the run directory"
                )
            root = run_root.resolve()
            raw_gt_file = Path(gt_path)
            gt_file = (
                raw_gt_file.resolve()
                if raw_gt_file.is_absolute()
                else (root / raw_gt_file).resolve()
            )
            try:
                gt_file.relative_to(root)
            except ValueError as exc:
                raise FactoryAdapterError(
                    "Factory ground-truth path escapes the run directory"
                ) from exc
            expected_gt_hash = gt_hash_value
            if (
                not isinstance(expected_gt_hash, str)
                or not expected_gt_hash
                or not gt_file.is_file()
                or sha256_file(gt_file) != expected_gt_hash
            ):
                raise FactoryAdapterError(
                    "Factory ground-truth artifact is missing or corrupt"
                )
            try:
                gt_text = gt_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise FactoryAdapterError(
                    "Factory ground-truth artifact is not valid UTF-8 text"
                ) from exc
        provenance: dict[str, Any] = {
            "adapter_version": FACTORY_ADAPTER_VERSION,
            "factory_run_id": run_id,
            "source_ref": _portable_source_ref(row.get("source_ref")),
            "seed": row.get("seed"),
            "seed_mode": row.get("seed_mode"),
            "base_seed": row.get("base_seed"),
            "profile": profile,
            "profile_schema": row.get("profile_schema"),
            "renderer": row.get("renderer"),
            "dpi": row.get("dpi"),
            "color": row.get("color"),
            "jpeg_quality": row.get("jpeg_quality"),
            "transform_steps": row.get("transform_steps", []),
            "render_config": row.get("render_config"),
            "distortion_config": row.get("distortion_config"),
            "config_hash": row.get("config_hash"),
            "qc_passed": row.get("qc_passed"),
            "qc": row.get("qc"),
            "source_sha256": row.get("source_sha256"),
            "output_sha256": row.get("output_sha256"),
            "clean_sha256": row.get("clean_sha256"),
            "gt_sha256": row.get("gt_sha256"),
            "source_type": row.get("source_type"),
            "created_utc": row.get("created_utc"),
            "factory_version": row.get("factory_version"),
            "factory_generated": True,
        }
        samples.append(
            DatasetSample(
                sample_id=stable_sample_id(source_id, output_rel, record_key),
                source_id=source_id,
                source_dataset="clouda_data_factory",
                source_path=output_rel,
                source_record_id=record_key,
                document_id=document_id or None,
                page_index=int(page_index) if page_index is not None else None,
                group_id=f"{source_id}:{document_id}" if document_id else None,
                image_path=output_rel or None,
                text=gt_text,
                raw_text=gt_text,
                language="ar",
                script="arabic",
                file_sha256=str(row.get("output_sha256")) or None,
                normalized_text_sha256=sha256_text(gt_text) if gt_text else None,
                provenance=provenance,
                transformations=[
                    str(step.get("distortion", step.get("stage", "")))
                    for step in row.get("transform_steps", [])
                    if isinstance(step, dict)
                ],
                target_split=SplitName.UNASSIGNED,
            )
        )
        ok += 1
    report = AdapterReport(
        rows_total=len(rows),
        rows_ok=ok,
        rows_error=error,
        rows_skipped=skipped,
        samples=len(samples),
    )
    return samples, report


def write_pretraining_manifest(
    path: str | Path,
    samples: list[DatasetSample | dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write samples as a canonical ``clouda.pretraining.manifest.v1`` file."""
    from ..pretraining.manifest import write_manifest

    rows = [
        sample.to_dict() if isinstance(sample, DatasetSample) else sample
        for sample in samples
    ]
    return write_manifest(path, rows, metadata=metadata)


def run_dir_to_dataset_manifest(
    run_dir: str | Path,
    output_manifest: str | Path,
    *,
    source_id: str = "clouda_data_factory",
) -> tuple[Path, AdapterReport, str]:
    """One-call conversion: factory run dir → canonical dataset manifest.

    Returns ``(manifest_path, report, manifest_sha256)``. The SHA-256 links the
    training experiment's provenance to the exact factory output.
    """
    run_root = Path(run_dir)
    rows = read_factory_manifest(run_root / "manifest.jsonl")
    for row in rows:
        if row.get("status") != "ok":
            continue
        for path_field, hash_field, required in (
            ("output_path", "output_sha256", True),
            ("png_path", "png_sha256", False),
            ("pdf_path", "pdf_sha256", False),
        ):
            output_path = row.get(path_field)
            expected_hash = row.get(hash_field)
            if not output_path and not expected_hash and not required:
                continue
            if not isinstance(output_path, str) or not output_path or not expected_hash:
                raise FactoryAdapterError(
                    "Factory row has incomplete artifact provenance"
                )
            artifact = (run_root / output_path).resolve()
            try:
                artifact.relative_to(run_root.resolve())
            except ValueError as exc:
                raise FactoryAdapterError(
                    "Factory artifact path escapes the run directory"
                ) from exc
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise FactoryAdapterError(
                    f"Factory artifact is missing or corrupt: {output_path}"
                )
    run_metadata: dict[str, Any] = {}
    for metadata_name in ("metadata.json", "run_config.json"):
        metadata_path = run_root / metadata_name
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FactoryAdapterError(
                    f"Malformed factory run metadata: {metadata_path}"
                ) from exc
            if isinstance(payload, dict):
                run_metadata.update(payload)
    for row in rows:
        row.setdefault("config_hash", run_metadata.get("config_hash"))
        row.setdefault("factory_version", run_metadata.get("factory_version"))
    samples, report = factory_rows_to_samples(
        rows, source_id=source_id, run_dir=run_root
    )
    if not samples:
        raise FactoryAdapterError(
            f"Factory run {run_root} produced no convertible ok rows "
            f"({report.rows_total} rows, {report.rows_error} errors)"
        )
    # Leakage-safe split assignment so the manifest is directly consumable by
    # the Training Experiment Framework (which requires a non-empty, non-
    # protected selected split). The seed is derived from the sample content
    # hashes + source id, so identical factory content always converts to the
    # identical split assignment regardless of run timestamps.
    import hashlib as _hashlib

    from ..pretraining.splitting import assign_splits

    content_material = "\x1f".join(
        [source_id]
        + sorted(
            ":".join(
                (
                    str(sample.provenance.get("output_sha256") or sample.sample_id),
                    str(sample.provenance.get("gt_sha256") or ""),
                )
            )
            for sample in samples
        )
    )
    split_seed = int.from_bytes(
        _hashlib.sha256(content_material.encode("utf-8")).digest()[:4], "big"
    )
    dataset_version_material = "\x1f".join(
        [str(run_metadata.get("config_hash") or ""), content_material]
    )
    dataset_version = _hashlib.sha256(
        dataset_version_material.encode("utf-8")
    ).hexdigest()
    samples, split_report = assign_splits(samples, seed=split_seed)
    if not split_report.passed:
        raise FactoryAdapterError(
            "Split leakage checks failed for the converted factory run"
        )
    target = write_pretraining_manifest(
        output_manifest,
        [sample.to_dict() for sample in samples],
        metadata={
            "dataset_role": "training",
            "dataset_id": source_id,
            "dataset_version": dataset_version,
            "factory_adapter_version": FACTORY_ADAPTER_VERSION,
            "factory_run_id": str(samples[0].provenance.get("factory_run_id", "")),
            "source_ids": [source_id],
            "split_seed": split_seed,
        },
    )
    return target, report, sha256_file(target)
