"""Run manifest writers (JSONL + CSV) — System A semantics, extended fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..provenance.integrity import atomic_target

_DEFAULT_FIELDS = [
    "run_id",
    "document_id",
    "page_index",
    "variant_id",
    "source_type",
    "source_ref",
    "source_sha256",
    "clean_sha256",
    "output_sha256",
    "renderer",
    "profile",
    "profile_schema",
    "seed",
    "seed_mode",
    "base_seed",
    "dpi",
    "color",
    "jpeg_quality",
    "gt_path",
    "gt_sha256",
    "status",
    "error",
    "qc_passed",
    "created_utc",
    "output_path",
]


def write_manifest_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    with atomic_target(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
                )
    return path


def write_manifest_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = _DEFAULT_FIELDS
    with atomic_target(path) as tmp:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                flat = {
                    k: (
                        v
                        if not isinstance(v, (dict, list))
                        else json.dumps(v, ensure_ascii=False)
                    )
                    for k, v in row.items()
                }
                writer.writerow(flat)
    return path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
