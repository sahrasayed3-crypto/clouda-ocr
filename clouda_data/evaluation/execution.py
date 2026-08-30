from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from clouda_contracts.storage import StorageRoots
from clouda_data.evaluation.cer import cer
from clouda_data.evaluation.normalization import normalize_ocr_text
from clouda_data.evaluation.wer import wer


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for record in records:
        reference = str(
            record.get("reference_text") or record.get("ground_truth_text") or ""
        )
        prediction = str(record.get("prediction_text") or record.get("ocr_text") or "")
        normalized_reference = normalize_ocr_text(reference)
        normalized_prediction = normalize_ocr_text(prediction)
        page = {
            "page_id": record.get("generated_page_id") or record.get("page_id"),
            "dataset": record.get("dataset_id", "unspecified"),
            "source": record.get("source_document_id", "unspecified"),
            "profile": record.get("profile_id", "clean"),
            "severity": record.get("overall_severity", "none"),
            "difficulty": record.get("estimated_visual_difficulty", "unspecified"),
            "document": record.get("source_document_id", "unspecified"),
            "page_type": record.get("page_type", "body"),
            "language": record.get("language", "ar"),
            "model": record.get("model_id", "unspecified"),
            "model_revision": record.get("model_revision", "unspecified"),
            "cer": cer(reference, prediction),
            "wer": wer(reference, prediction),
            "exact_match": reference == prediction,
            "normalized_exact_match": normalized_reference == normalized_prediction,
            "character_accuracy": max(0.0, 1.0 - cer(reference, prediction)),
            "word_accuracy": max(0.0, 1.0 - wer(reference, prediction)),
            "missing_text": bool(reference and not prediction),
            "hallucination": bool(not reference and prediction),
            "empty_output": not bool(prediction.strip()),
            "success": bool(prediction.strip()),
            "processing_time": float(record.get("processing_time", 0.0)),
        }
        pages.append(page)

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"pages": 0}
        return {
            "pages": len(items),
            "cer": mean(item["cer"] for item in items),
            "wer": mean(item["wer"] for item in items),
            "exact_match_rate": mean(item["exact_match"] for item in items),
            "normalized_exact_match_rate": mean(
                item["normalized_exact_match"] for item in items
            ),
            "character_accuracy": mean(item["character_accuracy"] for item in items),
            "word_accuracy": mean(item["word_accuracy"] for item in items),
            "missing_text_rate": mean(item["missing_text"] for item in items),
            "hallucination_rate": mean(item["hallucination"] for item in items),
            "empty_output_rate": mean(item["empty_output"] for item in items),
            "page_success_rate": mean(item["success"] for item in items),
            "processing_time": sum(item["processing_time"] for item in items),
            "throughput_pages_per_second": len(items)
            / max(0.001, sum(item["processing_time"] for item in items)),
        }

    dimensions = [
        "dataset",
        "source",
        "profile",
        "severity",
        "difficulty",
        "document",
        "page_type",
        "language",
        "model",
        "model_revision",
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for page in pages:
            buckets[str(page[dimension])].append(page)
        grouped[dimension] = {key: summary(value) for key, value in buckets.items()}
    return {
        "schema_version": 1,
        "normalization_policy": "clouda_normalization_v1",
        "summary": summary(pages),
        "groups": grouped,
        "pages": pages,
        "generated_at_monotonic": time.monotonic(),
    }


def evaluate_manifest(path: str | Path, output: str | Path | None = None) -> Path:
    input_path = Path(path).expanduser().resolve()
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_records(records)
    target = (
        Path(output).expanduser().resolve()
        if output
        else StorageRoots.from_env().artifact_root
        / "reports"
        / "evaluation"
        / f"{input_path.stem}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target.with_suffix(".md").write_text(
        "# OCR evaluation\n\n"
        f"- Pages: {report['summary'].get('pages', 0)}\n"
        f"- CER: {report['summary'].get('cer', 0):.6f}\n"
        f"- WER: {report['summary'].get('wer', 0):.6f}\n",
        encoding="utf-8",
    )
    return target
