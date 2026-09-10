"""Evaluation Service — the canonical backend entry point for lab evaluation.

Composes existing metric code and the new analysis engines. The future
API/UI calls *this*, never the low-level pieces directly:

- evaluate one page/sample → detailed error analysis;
- evaluate a set of samples (dataset slice / model output set) → batch report;
- compare two models/runs → failure analysis report;
- failure buckets, hard examples, recommendations, exports.

Low-level metric code is reused from ``clouda_data.evaluation`` — nothing
duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .active_learning import recommend_next_batch
from .batch_analysis import analyze_batch
from .error_analysis import analyze_sample
from .failure_analysis import SampleMetrics, compare_failure, summarize_comparison
from .failure_buckets import assign_buckets, primary_bucket
from .hard_examples import select_hard_examples
from .io import export_csv, export_json, export_jsonl, load_samples_jsonl
from .models import (
    BatchReport,
    ErrorAnalysis,
    FailureReport,
    OCRSample,
    TrainingBatchRecommendation,
)


class EvaluationService:
    """Backend evaluation facade. Stateless; safe to share across requests."""

    # -- single sample -----------------------------------------------------

    def evaluate_sample(self, sample: OCRSample) -> ErrorAnalysis:
        return analyze_sample(sample)

    # -- batch / slice -----------------------------------------------------

    def evaluate_batch(
        self,
        samples: Sequence[OCRSample],
        *,
        dimensions: Sequence[str] | None = None,
        worst_n: int = 10,
        best_n: int = 10,
    ) -> BatchReport:
        if dimensions is None:
            from .batch_analysis import DEFAULT_DIMENSIONS

            dimensions = DEFAULT_DIMENSIONS
        return analyze_batch(
            samples,
            dimensions=dimensions,
            worst_n=worst_n,
            best_n=best_n,
        )

    def evaluate_file(
        self,
        path: str | Path,
        *,
        dimensions: Sequence[str] | None = None,
    ) -> BatchReport:
        samples = load_samples_jsonl(path)
        return self.evaluate_batch(samples, dimensions=dimensions)

    # -- comparison --------------------------------------------------------

    def compare_models(
        self,
        baseline: Sequence[SampleMetrics],
        candidate: Sequence[SampleMetrics],
        *,
        baseline_id: str,
        candidate_id: str,
        thresholds: Mapping[str, float] | None = None,
    ) -> FailureReport:
        return compare_failure(
            baseline,
            candidate,
            baseline_id=baseline_id,
            candidate_id=candidate_id,
            thresholds=thresholds,
        )

    # -- buckets / hard examples / recommendations -------------------------

    def failure_buckets(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        thresholds: Mapping[str, float] | None = None,
    ) -> dict[str, list]:
        return assign_buckets(rows, thresholds=thresholds)

    def hard_examples(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        weights: Mapping[str, float] | None = None,
        top_n: int | None = None,
        percentile: float | None = None,
        min_score: float | None = None,
        failure_cer: float = 0.5,
    ) -> list:
        return select_hard_examples(
            rows,
            weights=weights,
            top_n=top_n,
            percentile=percentile,
            min_score=min_score,
            failure_cer=failure_cer,
        )

    def recommend_next(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        strategy: str = "balanced_hard",
        batch_size: int = 32,
        seed: int = 20260723,
        history: Sequence[str] = (),
    ) -> TrainingBatchRecommendation:
        return recommend_next_batch(
            rows,
            strategy=strategy,
            batch_size=batch_size,
            seed=seed,
            history=history,
        )

    # -- exports -----------------------------------------------------------

    @staticmethod
    def export_analysis(analysis: ErrorAnalysis, path: str | Path) -> Path:
        return export_json(analysis.to_dict(), path)

    @staticmethod
    def export_batch(report: BatchReport, path: str | Path, *, csv: bool = False) -> Path:
        if csv:
            rows = report.worst_pages and [row.to_dict() for row in report.worst_pages]
            return export_csv(rows or [], path)
        return export_json(report.to_dict(), path)

    @staticmethod
    def export_failure_report(report: FailureReport, path: str | Path) -> Path:
        return export_json(
            {"summary": summarize_comparison(report), "report": report.to_dict()}, path
        )

    @staticmethod
    def export_recommendation(
        recommendation: TrainingBatchRecommendation, path: str | Path
    ) -> Path:
        return export_json(recommendation.to_dict(), path)

    @staticmethod
    def export_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.casefold() == ".csv":
            return export_csv([dict(row) for row in rows], target)
        return export_jsonl([dict(row) for row in rows], target)


__all__ = [
    "EvaluationService",
]
