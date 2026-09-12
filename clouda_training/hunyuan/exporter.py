"""Canonical manifest -> HunyuanOCR-1.5 raw OCR JSONL exporter.

Safety: fail-closed — reuses the canonical protection guard
(`clouda_contracts.protection`) and dataset validation; protected/holdout
rows can never be exported. Portability: canonical manifests are never
rewritten; absolute Hunyuan paths exist only in generated artifacts, and
every artifact row keeps lineage to its canonical sample id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from clouda_contracts.checksums import sha256_file
from clouda_contracts.protection import (
    normalize_marker,
    protection_metadata_is_malformed,
    record_is_protected,
)
from clouda_data.pretraining.manifest import read_manifest

from clouda_training.hunyuan.models import (
    HUNYUAN_IMAGE_PLACEHOLDER,
    HunyuanConversationTurn,
    HunyuanExportConfig,
    HunyuanExportReport,
    HunyuanRawSample,
)

GT_FIELDS = ("ground_truth", "gt", "text", "transcript")
IMAGE_FIELDS = ("image_path", "image", "source_path")


class HunyuanExportError(RuntimeError):
    """Raised when a manifest row cannot be exported safely."""


def _first_field(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for name in fields:
        if row.get(name):
            return row[name]
    return None


def _absolute_image_path(image_root: str, canonical_path: str) -> str:
    """Absolutize a canonical relative path inside the generated artifact only."""
    candidate = Path(canonical_path)
    if candidate.is_absolute():
        return str(candidate)
    root = Path(image_root)
    if not root.is_absolute():
        raise HunyuanExportError(
            f"image_root must be absolute for Hunyuan export: {image_root!r}"
        )
    # resolve without requiring the file to exist (validation happens later)
    return os.path.abspath(str(root / candidate))


def build_raw_sample(
    row: dict[str, Any],
    config: HunyuanExportConfig,
) -> HunyuanRawSample:
    """Convert one canonical manifest row into a Hunyuan raw sample.

    Fails closed on anything ambiguous.
    """
    sample_id = row.get("sample_id")
    if not sample_id or not isinstance(sample_id, str):
        raise HunyuanExportError(f"manifest row missing string sample_id: {row!r}")

    # Protection: reuse the canonical guard, never duplicate it.
    if protection_metadata_is_malformed(row):
        raise HunyuanExportError(f"row {sample_id}: malformed protection metadata")
    if record_is_protected(row):
        raise HunyuanExportError(f"row {sample_id}: protected rows cannot be exported")

    raw_gt = _first_field(row, GT_FIELDS)
    if not raw_gt or not str(raw_gt).strip():
        raise HunyuanExportError(f"row {sample_id}: missing/empty ground truth")

    raw_image = _first_field(row, IMAGE_FIELDS)
    if not raw_image or not str(raw_image).strip():
        raise HunyuanExportError(f"row {sample_id}: missing/empty image path")

    image_abs = _absolute_image_path(config.image_root, str(raw_image))

    prompt = config.prompt_profile.prompt_text
    conversations = [
        HunyuanConversationTurn(
            from_role="human",
            value=f"{HUNYUAN_IMAGE_PLACEHOLDER}\n{prompt}",
        ),
        HunyuanConversationTurn(from_role="gpt", value=str(raw_gt)),
    ]
    return HunyuanRawSample(
        image_path=[image_abs],
        conversations=conversations,
        canonical_sample_id=sample_id,
        prompt_identity=config.prompt_profile.identity,
    )


def _validate_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Filter rows to the requested split, rejecting protected ones (fail-closed)."""
    wanted = split.strip().lower()
    if (
        wanted in {"holdout", "protected_holdout", "benchmark_holdout"}
        or "holdout" in wanted
    ):
        raise PermissionError(
            f"Protected split cannot be exported for training: {split}"
        )
    selected = []
    for row in rows:
        marker = row.get("target_split", row.get("split"))
        if marker is not None and normalize_marker(str(marker)) != wanted:
            continue
        selected.append(row)
    if not selected:
        raise HunyuanExportError(f"manifest has no rows for split {split!r}")
    # Per-row protection is enforced in build_raw_sample (fail-closed, counted).
    # Whole-manifest protection (header marker) hard-fails in export_raw_jsonl.
    return selected


def export_raw_jsonl(
    config: HunyuanExportConfig,
    output_path: str | Path,
) -> HunyuanExportReport:
    """Export canonical manifest rows to Hunyuan raw OCR JSONL.

    Returns a lineage-bearing report; never mutates the canonical manifest.
    """
    manifest = Path(config.manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if sha256_file(manifest) != config.manifest_hash:
        raise ValueError(
            "manifest_hash mismatch — refusing to export a changed manifest"
        )

    _header, rows = read_manifest(manifest)
    # Fail closed on whole-manifest protection markers (e.g. evaluation_only).
    if protection_metadata_is_malformed(_header):
        raise HunyuanExportError("manifest header has malformed protection metadata")
    if record_is_protected(_header):
        raise PermissionError(
            "manifest header marks the whole dataset protected — refusing to export"
        )
    selected = _validate_split(rows, config.split)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = HunyuanExportReport(config=config)
    seen_ids: set[str] = set()

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            try:
                sample = build_raw_sample(row, config)
            except HunyuanExportError as exc:
                key = (
                    "protected"
                    if "protected" in str(exc)
                    else (
                        "missing_gt"
                        if "ground truth" in str(exc)
                        else (
                            "missing_image" if "image path" in str(exc) else "malformed"
                        )
                    )
                )
                report.skipped_counts[key] = report.skipped_counts.get(key, 0) + 1
                continue
            if sample.canonical_sample_id in seen_ids:
                report.skipped_counts["duplicate_sample_id"] = (
                    report.skipped_counts.get("duplicate_sample_id", 0) + 1
                )
                continue
            seen_ids.add(sample.canonical_sample_id)
            handle.write(
                json.dumps(sample.to_upstream_dict(), ensure_ascii=False) + "\n"
            )
            report.exported_count += 1
            report.sample_ids.append(sample.canonical_sample_id)

    if report.exported_count == 0:
        raise HunyuanExportError("export produced zero valid samples")

    report.output_path = str(out_path)
    report.output_sha256 = sha256_file(out_path)
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return report
