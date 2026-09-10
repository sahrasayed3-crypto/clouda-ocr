"""Post-run analysis pipeline (backend services only).

    run
     ↓
  evaluation results      (caller-provided per-sample metrics from actual eval)
     ↓
  error analysis          (clouda_lab.error_analysis)
     ↓
  failure analysis        (baseline run vs candidate run)
     ↓
  hard-example mining     (clouda_lab.hard_examples)
     ↓
  next-batch recommendation (clouda_lab.active_learning)

Mock/dry-run training may drive the lifecycle in tests, but no OCR
improvement is ever fabricated from MockTrainer output — comparisons use
only metrics supplied by the caller (in tests: synthetic fixtures).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .active_learning import recommend_next_batch
from .failure_analysis import SampleMetrics, compare_failure, summarize_comparison
from .hard_examples import select_hard_examples
from .models import RunAnalysis
from .training_orchestrator import TrainingOrchestrator


def analyze_run(
    *,
    run_id: str,
    orchestrator: TrainingOrchestrator,
    baseline_metrics: Sequence[SampleMetrics],
    candidate_metrics: Sequence[SampleMetrics],
    baseline_id: str = "baseline",
    rows: Sequence[Mapping[str, Any]] | None = None,
    thresholds: Mapping[str, float] | None = None,
    hard_example_top_n: int = 20,
    recommendation_strategy: str = "balanced_hard",
    recommendation_batch_size: int = 16,
    seed: int = 20260723,
    history: Sequence[str] = (),
) -> RunAnalysis:
    """Full post-run backend analysis for one training run.

    ``baseline_metrics``/``candidate_metrics`` are per-sample metric sets for
    the baseline and the analyzed run (from actual evaluation). ``rows`` are
    the analysis rows feeding hard-example mining (sample_id + cer/wer +
    error_type_counts + metadata).
    """
    failure_report = compare_failure(
        baseline_metrics,
        candidate_metrics,
        baseline_id=baseline_id,
        candidate_id=run_id,
        thresholds=thresholds,
    )
    status = orchestrator.run_status(run_id)
    metrics_summary = _run_metrics_summary(orchestrator, run_id)

    hard = select_hard_examples(
        rows or _rows_from_metrics(candidate_metrics),
        top_n=hard_example_top_n,
    )
    recommendation = recommend_next_batch(
        rows or _rows_from_metrics(candidate_metrics),
        strategy=recommendation_strategy,
        batch_size=recommendation_batch_size,
        seed=seed,
        history=history,
    )
    return RunAnalysis(
        run_id=run_id,
        status=status,
        metrics_summary=metrics_summary,
        failure_report=failure_report,
        hard_examples=tuple(hard),
        recommendation=recommendation,
    )


def _run_metrics_summary(
    orchestrator: TrainingOrchestrator, run_id: str
) -> dict[str, Any]:
    try:
        handle = orchestrator.inspect_run(run_id)
    except FileNotFoundError:
        return {"available": False}
    summary = handle.get("summary", {})
    return {
        "available": True,
        "final_metrics": summary.get("final_metrics", {}),
        "best_metrics": summary.get("best_metrics", {}),
        "status": summary.get("status", ""),
    }


def _rows_from_metrics(
    metrics: Sequence[SampleMetrics],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in metrics:
        row: dict[str, Any] = {
            "sample_id": item.sample_id,
            "cer": item.cer,
            "wer": item.wer,
            "ncer": item.ncer,
            "model_id": "unspecified",
        }
        if item.error_types:
            row["error_type_counts"] = dict(item.error_types)
        if item.metadata:
            row["metadata"] = dict(item.metadata)
        rows.append(row)
    return rows


__all__ = [
    "analyze_run",
]
