from __future__ import annotations

import csv
import json
from pathlib import Path

from clouda_contracts.security import sanitize_spreadsheet_cell

from .aggregations import (
    PageMetric,
    aggregate_metrics,
    group_by_dimension,
    group_by_profile,
)


def write_json_report(metrics: list[PageMetric], path: str | Path) -> None:
    payload = {
        "summary": aggregate_metrics(metrics),
        "by_profile": group_by_profile(metrics),
        "by_dataset": group_by_dimension(metrics, "dataset_id"),
        "by_document_type": group_by_dimension(metrics, "document_type"),
        "by_quality_class": group_by_dimension(metrics, "quality_class"),
        "by_language": group_by_dimension(metrics, "language"),
        "pages": [metric.__dict__ for metric in metrics],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_csv_report(metrics: list[PageMetric], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page_id",
                "profile",
                "severity",
                "cer",
                "wer",
                "dataset_id",
                "document_type",
                "quality_class",
                "language",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    key: sanitize_spreadsheet_cell(value)
                    for key, value in metric.__dict__.items()
                }
            )


def markdown_summary(metrics: list[PageMetric]) -> str:
    summary = aggregate_metrics(metrics)
    return f"# OCR Evaluation Summary\n\nPages: {summary['pages']}\n\nCER mean: {summary['cer_mean']:.4f}\n\nWER mean: {summary['wer_mean']:.4f}\n"
