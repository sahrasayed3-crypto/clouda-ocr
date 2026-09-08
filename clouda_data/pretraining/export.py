"""Training-ready export interface.

An :class:`TrainingExporter` turns validated, deduplicated, split samples
into a concrete training format. Exports are deterministic, filter out
excluded/duplicate/holdout samples by default, and never copy image bytes —
they reference images by dataset-root-relative path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .hashing import atomic_write_text
from .schema import (
    DatasetSample,
    EXPORTABLE_DUPLICATE_STATES,
    EXPORTABLE_SPLITS,
    EXPORTABLE_STATUSES,
    SplitName,
    sort_key,
)


@dataclass
class ExportConfig:
    include_holdout: bool = False
    include_duplicates: bool = False
    include_raw_text: bool = False
    text_field: str = "normalized"  # or "raw"
    splits: tuple[str, ...] = tuple(split.value for split in EXPORTABLE_SPLITS)


@dataclass
class ExportResult:
    exporter: str
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exporter": self.exporter,
            "files": self.files,
            "counts": self.counts,
            "schema_version": "clouda.pretraining.export.v1",
        }


class TrainingExporter(Protocol):
    """Adapter interface for future formats (HF datasets, Parquet, ...)."""

    name: str

    def export(
        self,
        samples: list[DatasetSample],
        out_dir: Path,
        config: ExportConfig,
    ) -> ExportResult: ...


def select_exportable(
    samples: list[DatasetSample], config: ExportConfig
) -> list[DatasetSample]:
    """Apply the standard export filters deterministically."""

    allowed_splits = set(config.splits)
    if config.include_holdout:
        allowed_splits.add(SplitName.HOLDOUT.value)
    selected = []
    for sample in sorted(samples, key=sort_key):
        if sample.validation_status not in EXPORTABLE_STATUSES:
            continue
        if sample.duplicate_state not in EXPORTABLE_DUPLICATE_STATES:
            if not (
                config.include_duplicates and sample.duplicate_state.name == "DUPLICATE"
            ):
                continue
        if sample.target_split.value not in allowed_splits:
            continue
        selected.append(sample)
    return selected


class JsonlTrainingExporter:
    """Generic JSONL export: ``{"image": ..., "text": ..., ...provenance}``."""

    name = "jsonl"

    def export(
        self,
        samples: list[DatasetSample],
        out_dir: Path,
        config: ExportConfig,
    ) -> ExportResult:
        selected = select_exportable(samples, config)
        by_split: dict[str, list[DatasetSample]] = {}
        for sample in selected:
            by_split.setdefault(sample.target_split.value, []).append(sample)

        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        counts: dict[str, int] = {}
        for split in sorted(by_split):
            rows = [self._row(sample, config) for sample in by_split[split]]
            path = atomic_write_text(
                out_dir / f"{split}.jsonl",
                "".join(row + "\n" for row in rows),
            )
            files.append(str(path))
            counts[split] = len(rows)
        return ExportResult(exporter=self.name, files=files, counts=counts)

    def _row(self, sample: DatasetSample, config: ExportConfig) -> str:
        import json

        text = (
            sample.raw_text
            if config.text_field == "raw" and sample.raw_text is not None
            else sample.text
        )
        payload: dict[str, Any] = {
            "image": sample.image_path or "",
            "text": text or "",
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "document_id": sample.document_id,
            "page_id": sample.page_id,
            "language": sample.language,
            "split": sample.target_split.value,
            "file_sha256": sample.file_sha256,
            "normalized_text_sha256": sample.normalized_text_sha256,
            "duplicate_state": sample.duplicate_state.value,
            "schema_version": sample.schema_version,
        }
        if config.include_raw_text:
            payload["raw_text"] = sample.raw_text
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


EXPORTERS: dict[str, TrainingExporter] = {}
_exporter = JsonlTrainingExporter()
EXPORTERS[_exporter.name] = _exporter


def register_exporter(exporter: TrainingExporter) -> None:
    EXPORTERS[exporter.name] = exporter


def get_exporter(name: str) -> TrainingExporter:
    if name not in EXPORTERS:
        raise KeyError(f"Unknown exporter: {name} (available: {sorted(EXPORTERS)})")
    return EXPORTERS[name]
