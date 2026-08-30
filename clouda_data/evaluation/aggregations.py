from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class PageMetric:
    page_id: str
    profile: str
    severity: str
    cer: float
    wer: float
    dataset_id: str = "unspecified"
    document_type: str = "unspecified"
    quality_class: str = "unspecified"
    language: str = "ar"


def aggregate_metrics(metrics: list[PageMetric]) -> dict[str, float]:
    if not metrics:
        return {"pages": 0, "cer_mean": 0.0, "wer_mean": 0.0}
    return {
        "pages": len(metrics),
        "cer_mean": mean(m.cer for m in metrics),
        "wer_mean": mean(m.wer for m in metrics),
    }


def group_by_profile(metrics: list[PageMetric]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PageMetric]] = {}
    for metric in metrics:
        grouped.setdefault(metric.profile, []).append(metric)
    return {profile: aggregate_metrics(items) for profile, items in grouped.items()}


def group_by_dimension(
    metrics: list[PageMetric],
    dimension: str,
) -> dict[str, dict[str, float]]:
    allowed = {"dataset_id", "document_type", "quality_class", "language"}
    if dimension not in allowed:
        raise ValueError(f"Unsupported evaluation dimension: {dimension}")
    grouped: dict[str, list[PageMetric]] = {}
    for metric in metrics:
        grouped.setdefault(str(getattr(metric, dimension)), []).append(metric)
    return {key: aggregate_metrics(items) for key, items in grouped.items()}
