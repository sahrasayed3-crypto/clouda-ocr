"""Batch OCR analysis across many samples with dimension aggregation.

Composes :mod:`clouda_lab.error_analysis` — it does not recompute metrics.
Samples are supplied as :class:`clouda_lab.models.OCRSample` objects; loaders
turn existing benchmark/manifest/result files into samples (see
:mod:`clouda_lab.io`).
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from .error_analysis import analyze_sample
from .models import BatchReport, BatchSummary, ErrorAnalysis, OCRSample

# Dimensions every sample can be grouped by. Values come from sample
# metadata only; missing keys fall back to "unspecified" (never inferred).
DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "model_id",
    "run_id",
    "dataset_id",
    "document_type",
    "profile",
    "distortion",
    "source",
    "split",
)

_METADATA_DIMENSIONS = frozenset(
    {"document_type", "profile", "distortion", "source", "split"}
)

_PERCENTILES = (50.0, 90.0, 95.0)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _metadata_value(sample: OCRSample, dimension: str) -> str:
    if dimension == "model_id":
        return sample.model_id
    if dimension == "run_id":
        return sample.run_id
    if dimension == "dataset_id":
        return sample.dataset_id
    value = sample.metadata.get(dimension)
    return str(value) if value is not None else "unspecified"


def _page_summary(group: str, key: str, analysis: ErrorAnalysis) -> BatchSummary:
    return BatchSummary(
        group=group,
        key=key,
        count=1,
        cer_mean=analysis.cer,
        wer_mean=analysis.wer,
        ncer_mean=analysis.normalized_cer,
        exact_match_rate=1.0 if analysis.exact_match else 0.0,
        cer_p50=analysis.cer,
        cer_p90=analysis.cer,
        cer_p95=analysis.cer,
        cer_max=analysis.cer,
    )


def _group_summary(group: str, key: str, analyses: list[ErrorAnalysis]) -> BatchSummary:
    cers = sorted(a.cer for a in analyses)
    return BatchSummary(
        group=group,
        key=key,
        count=len(analyses),
        cer_mean=mean(a.cer for a in analyses),
        wer_mean=mean(a.wer for a in analyses),
        ncer_mean=mean(a.normalized_cer for a in analyses),
        exact_match_rate=mean(1.0 if a.exact_match else 0.0 for a in analyses),
        cer_p50=_percentile(cers, 50.0),
        cer_p90=_percentile(cers, 90.0),
        cer_p95=_percentile(cers, 95.0),
        cer_max=cers[-1] if cers else 0.0,
    )


def analyze_batch(
    samples: Iterable[OCRSample],
    *,
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    worst_n: int = 10,
    best_n: int = 10,
) -> BatchReport:
    """Analyze many samples and aggregate by the requested dimensions."""
    analyses: list[ErrorAnalysis] = []
    page_rows: list[tuple[dict[str, str], BatchSummary]] = []
    error_type_counts: dict[str, int] = {}
    substitutions: dict[tuple[str, str], int] = {}

    for sample in samples:
        analysis = analyze_sample(sample)
        analyses.append(analysis)
        keys = {dim: _metadata_value(sample, dim) for dim in dimensions}
        page_rows.append((keys, _page_summary("page", sample.sample_id, analysis)))
        for error_type, count in analysis.error_type_counts.items():
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + count
        for gt, pred, count in analysis.substitutions:
            substitutions[(gt, pred)] = substitutions.get((gt, pred), 0) + count

    groups: dict[str, dict[str, list[ErrorAnalysis]]] = {}
    for (keys, _row), analysis in zip(page_rows, analyses):
        for dim, key in keys.items():
            groups.setdefault(dim, {}).setdefault(key, []).append(analysis)

    grouped_summaries = {
        dim: {
            key: _group_summary(dim, key, members)
            for key, members in sorted(keys.items())
        }
        for dim, keys in groups.items()
    }

    ordered_pages = sorted(
        page_rows,
        key=lambda item: (-item[1].cer_mean, item[1].key),
    )
    worst = [row for _keys, row in ordered_pages[: max(0, worst_n)]]
    best = [row for _keys, row in ordered_pages[::-1][: max(0, best_n)]]
    best.sort(key=lambda row: (row.cer_mean, row.key))

    return BatchReport(
        total_samples=len(analyses),
        overall=(
            _group_summary("overall", "all", analyses)
            if analyses
            else BatchSummary(
                "overall", "all", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        ),
        groups=grouped_summaries,
        error_type_counts=dict(sorted(error_type_counts.items())),
        common_substitutions=_top_substitutions(substitutions),
        worst_pages=tuple(worst),
        best_pages=tuple(best),
    )


def _top_substitutions(
    substitutions: dict[tuple[str, str], int], limit: int = 20
) -> tuple[tuple[str, str, int], ...]:
    ranked = sorted(
        substitutions.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )
    return tuple((gt, pred, count) for (gt, pred), count in ranked[:limit])


def export_batch_csv(report: BatchReport, path) -> None:
    """Write per-page rows (worst-first) as CSV for spreadsheet consumption."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "key",
                "count",
                "cer_mean",
                "wer_mean",
                "ncer_mean",
                "exact_match_rate",
                "cer_p50",
                "cer_p90",
                "cer_p95",
                "cer_max",
            ]
        )
        for row in report.worst_pages:
            writer.writerow(
                [
                    row.group,
                    row.key,
                    row.count,
                    f"{row.cer_mean:.6f}",
                    f"{row.wer_mean:.6f}",
                    f"{row.ncer_mean:.6f}",
                    f"{row.exact_match_rate:.4f}",
                    f"{row.cer_p50:.6f}",
                    f"{row.cer_p90:.6f}",
                    f"{row.cer_p95:.6f}",
                    f"{row.cer_max:.6f}",
                ]
            )


__all__ = [
    "DEFAULT_DIMENSIONS",
    "analyze_batch",
    "export_batch_csv",
]
