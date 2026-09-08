"""Versioned dataset manifests.

The canonical manifest is line-oriented JSONL: a deterministic header line
followed by one row per sample in canonical order. Writes are atomic
(temp file + ``os.replace``), so an interrupted run never corrupts an
existing manifest. Image bytes are never embedded; paths are stored
relative to the dataset root.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

MANIFEST_SCHEMA_VERSION = "clouda.pretraining.manifest.v1"


def iter_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    for payload in iter_manifest(path):
        if "_schema_version" in payload and "sample_id" not in payload:
            header = payload
        else:
            rows.append(payload)
    return header, rows


def read_samples(path: str | Path) -> list:
    from .schema import DatasetSample

    _, rows = read_manifest(path)
    return [DatasetSample.from_dict(row) for row in rows]


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Atomically write the canonical manifest (header + sorted rows)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        header = {
            "_schema_version": MANIFEST_SCHEMA_VERSION,
            "_row_count": len(rows),
        }
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_path, target)
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
