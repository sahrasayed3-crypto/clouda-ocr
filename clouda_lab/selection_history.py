"""Lightweight selection history registry.

Tracks which samples were previously used in training subsets, evaluation
subsets, hard-example batches, and active-learning batches.

Storage: one JSONL registry file (append-only) per lab artifacts root —
``selections/history.jsonl``. Each record: purpose, selection_id, sample_ids,
timestamp, source manifest hash. No database; no duplicated truth (the
authoritative dataset remains the manifest; this registry only records
*usage events*).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_SCHEMA_VERSION = "clouda.lab.selection_history.v1"

VALID_PURPOSES = (
    "training_subset",
    "evaluation_subset",
    "hard_example_batch",
    "active_learning_batch",
)


class SelectionHistory:
    """Append-only usage registry backed by a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        purpose: str,
        selection_id: str,
        sample_ids: list[str] | tuple[str, ...],
        source_manifest: str = "",
        source_manifest_sha256: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if purpose not in VALID_PURPOSES:
            raise ValueError(
                f"Unknown selection purpose: {purpose} (have: {VALID_PURPOSES})"
            )
        record = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "recorded_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "purpose": purpose,
            "selection_id": selection_id,
            "sample_ids": list(sample_ids),
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_manifest_sha256,
            "metadata": dict(metadata or {}),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def used_sample_ids(self, purposes: tuple[str, ...] | None = None) -> set[str]:
        """All sample ids ever recorded (optionally filtered by purpose)."""
        used: set[str] = set()
        for record in self.load():
            if purposes is None or record.get("purpose") in purposes:
                used.update(record.get("sample_ids", []))
        return used

    def usage_count(self, sample_id: str) -> int:
        """Number of recorded batches containing this sample."""
        return sum(
            1 for record in self.load() if sample_id in record.get("sample_ids", [])
        )

    def history_for(self, sample_id: str) -> list[dict[str, Any]]:
        """All usage records for one sample (audit / provenance view)."""
        return [
            record
            for record in self.load()
            if sample_id in record.get("sample_ids", [])
        ]


__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "VALID_PURPOSES",
    "SelectionHistory",
]
