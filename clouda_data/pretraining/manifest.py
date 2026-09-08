"""Versioned dataset manifests.

The canonical manifest is line-oriented JSONL: a deterministic header line
followed by one row per sample in canonical order. Writes are atomic
(temp file + ``os.replace``), so an interrupted run never corrupts an
existing manifest. Image bytes are never embedded; paths are stored relative
to each sample's registered source root.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

MANIFEST_SCHEMA_VERSION = "clouda.pretraining.manifest.v1"


def iter_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Malformed JSONL at {manifest_path}, line {line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Manifest row at {manifest_path}, line {line_number} "
                        "must be an object."
                    )
                yield payload


def read_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    for line_index, payload in enumerate(iter_manifest(path), start=1):
        if "_schema_version" in payload and "sample_id" not in payload:
            if header or line_index != 1:
                raise ValueError("Manifest must contain exactly one leading header.")
            header = payload
        else:
            rows.append(payload)
    if header and "_row_count" in header and header["_row_count"] != len(rows):
        raise ValueError(
            f"Manifest row count mismatch: header says {header['_row_count']}, "
            f"found {len(rows)}."
        )
    return header, rows


def read_samples(path: str | Path) -> list:
    from .schema import DatasetSample

    _, rows = read_manifest(path)
    return [DatasetSample.from_dict(row) for row in rows]


def write_manifest(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically write the canonical manifest (header + sorted rows)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("source_id", "")),
            str(row.get("source_path", "")),
            str(row.get("sample_id", "")),
            json.dumps(row, ensure_ascii=False, sort_keys=True),
        ),
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            header = {
                "_schema_version": MANIFEST_SCHEMA_VERSION,
                "_row_count": len(ordered),
            }
            for key, value in (metadata or {}).items():
                if key in header or key.startswith("_schema_"):
                    raise ValueError(f"Reserved manifest metadata key: {key}")
                header[key] = value
            handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            for row in ordered:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def write_csv_export(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Optional human-readable CSV mirror of a manifest subset."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "source_id",
        "document_id",
        "page_id",
        "image_path",
        "language",
        "file_sha256",
        "normalized_text_sha256",
        "validation_status",
        "duplicate_state",
        "target_split",
        "exclusion_reason",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    os.replace(tmp_path, target)
    return target
