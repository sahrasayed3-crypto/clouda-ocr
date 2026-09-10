"""I/O adapters: load OCR samples from existing project result formats.

The lab consumes **existing** data — it never defines a second benchmark
format. Supported inputs:

- ``load_samples_jsonl``: JSONL/JSON list of records with ground-truth +
  prediction text (the benchmark evidence shape: ``reference_text`` /
  ``ground_truth_text`` + ``prediction_text`` / ``ocr_text``, matching
  ``clouda_data.evaluation.execution.evaluate_records``).
- ``load_samples_from_evaluation``: the dict produced by
  ``clouda_data.evaluation.execution.evaluate_records`` (re-uses its pages).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import OCRSample

_TEXT_FIELDS_GT = ("reference_text", "ground_truth_text", "gt_text", "text")
_TEXT_FIELDS_PRED = ("prediction_text", "ocr_text", "predicted_text", "hypothesis")
_ID_FIELDS = ("sample_id", "page_id", "generated_page_id", "id")


def _first(record: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str):
            return value
    return None


def record_to_sample(
    record: dict[str, Any],
    *,
    default_model: str = "unspecified",
    default_dataset: str = "unspecified",
) -> OCRSample:
    """Convert one evaluation/manifest record into an :class:`OCRSample`.

    Metadata is copied verbatim from known manifest keys; nothing inferred.
    """
    ground_truth = _first(record, _TEXT_FIELDS_GT)
    prediction = _first(record, _TEXT_FIELDS_PRED)
    if ground_truth is None:
        raise ValueError(
            f"Record has no ground-truth text field (tried {_TEXT_FIELDS_GT})"
        )
    sample_id = _first(record, _ID_FIELDS)
    if not sample_id:
        raise ValueError("Record has no sample/page id field")
    metadata_keys = (
        "profile",
        "profile_id",
        "severity",
        "overall_severity",
        "distortion",
        "document_type",
        "page_type",
        "source",
        "source_document_id",
        "source_id",
        "split",
        "target_split",
        "language",
        "processing_time",
        "estimated_visual_difficulty",
    )
    metadata = {key: record[key] for key in metadata_keys if key in record}
    return OCRSample(
        sample_id=str(sample_id),
        ground_truth=ground_truth,
        prediction=prediction or "",
        model_id=str(record.get("model_id", default_model)),
        run_id=str(record.get("run_id", "unspecified")),
        dataset_id=str(record.get("dataset_id", default_dataset)),
        page_id=str(record.get("page_id", "")),
        metadata=metadata,
    )


def load_samples_jsonl(
    path: str | Path,
    *,
    default_model: str = "unspecified",
    default_dataset: str = "unspecified",
) -> list[OCRSample]:
    """Load samples from a JSONL file or a JSON array file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    records: list[dict[str, Any]]
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [
        record_to_sample(
            record,
            default_model=default_model,
            default_dataset=default_dataset,
        )
        for record in records
    ]


def load_samples_from_evaluation(
    evaluation: dict[str, Any],
    *,
    model_id: str = "unspecified",
) -> list[OCRSample]:
    """Re-use pages from ``clouda_data.evaluation.execution.evaluate_records``.

    The evaluate_records output does not carry raw text (only metrics), so
    this loader requires the original records list when available.
    """
    pages = evaluation.get("pages", [])
    if not pages:
        return []
    # evaluate_records pages lack raw text; callers needing full analysis
    # should use load_samples_jsonl on the raw evidence instead. This loader
    # serves comparison flows where metrics suffice.
    samples: list[OCRSample] = []
    for page in pages:
        samples.append(
            OCRSample(
                sample_id=str(page.get("page_id", "")),
                ground_truth="",
                prediction="",
                model_id=str(page.get("model", model_id)),
                run_id="unspecified",
                dataset_id=str(page.get("dataset", "unspecified")),
                page_id=str(page.get("page_id", "")),
                metadata={
                    key: value
                    for key, value in page.items()
                    if key not in {"page_id", "model"}
                },
            )
        )
    return samples


def export_json(data: Any, path: str | Path) -> Path:
    """Write JSON (machine-readable, ensure_ascii=False) atomically-ish."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def export_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def export_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


__all__ = [
    "export_csv",
    "export_json",
    "export_jsonl",
    "load_samples_from_evaluation",
    "load_samples_jsonl",
    "record_to_sample",
]
