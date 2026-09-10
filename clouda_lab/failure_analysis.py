"""Failure Analysis Engine.

Compares per-sample evaluation results between two models/runs/checkpoints
and classifies each sample as improved / regressed / unchanged /
newly_failed / recovered with configurable thresholds.

All deltas are computed from actual per-sample metrics supplied by the
caller — nothing is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from .models import (
    FailureComparison,
    FailureReport,
    IMPROVED,
    NEWLY_FAILED,
    RECOVERED,
    REGRESSED,
    UNCHANGED,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Absolute change in an error metric required to reclassify a sample.
    "improve": 0.05,
    "regress": 0.05,
    # CER beyond which a sample counts as "failed" for newly_failed/recovered.
    "failure_cer": 0.5,
}


@dataclass(frozen=True)
class SampleMetrics:
    """Per-sample metric snapshot for one model/run."""

    sample_id: str
    cer: float
    wer: float = 0.0
    ncer: float = 0.0
    error_types: Mapping[str, int] | None = None
    metadata: Mapping[str, Any] | None = None


def _resolve_id(source: Any) -> str:
    if isinstance(source, str):
        return source
    return str(
        getattr(source, "model_id", "")
        or getattr(source, "run_id", "")
        or "unspecified"
    )


def _metrics_map(samples: Sequence[SampleMetrics]) -> dict[str, SampleMetrics]:
    result: dict[str, SampleMetrics] = {}
    for sample in samples:
        if sample.sample_id in result:
            raise ValueError(
                f"Duplicate sample_id in metrics input: {sample.sample_id}"
            )
        result[sample.sample_id] = sample
    return result


def compare_failure(
    baseline: Sequence[SampleMetrics],
    candidate: Sequence[SampleMetrics],
    *,
    baseline_id: str,
    candidate_id: str,
    thresholds: Mapping[str, float] | None = None,
) -> FailureReport:
    """Compare per-sample metrics and produce the aggregated failure report.

    Classification (per sample, CER = error rate so lower is better):
    - ``improved``:      cer_after < cer_before - improve_threshold
    - ``regressed``:     cer_after > cer_before + regress_threshold
    - ``newly_failed``:  regressed-or-flat and candidate CER >= failure_cer
                         while baseline CER < failure_cer
    - ``recovered``:     improved-or-flat and baseline CER >= failure_cer
                         while candidate CER < failure_cer
    - ``unchanged``:     otherwise
    ``newly_failed``/``recovered`` take precedence over
    regressed/improved/unchanged when their stricter conditions hold.
    """
    resolved = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(f"Unknown thresholds: {sorted(unknown)}")
        resolved.update(thresholds)

    base_map = _metrics_map(baseline)
    cand_map = _metrics_map(candidate)
    common_ids = sorted(set(base_map) & set(cand_map))

    comparisons: list[FailureComparison] = []
    counts = {
        IMPROVED: 0,
        REGRESSED: 0,
        UNCHANGED: 0,
        NEWLY_FAILED: 0,
        RECOVERED: 0,
    }
    error_type_delta_totals: dict[str, int] = {}

    for sample_id in common_ids:
        base = base_map[sample_id]
        cand = cand_map[sample_id]
        cer_delta = cand.cer - base.cer
        if cand.cer < base.cer - resolved["improve"]:
            classification = IMPROVED
        elif cand.cer > base.cer + resolved["regress"]:
            classification = REGRESSED
        else:
            classification = UNCHANGED
        if base.cer < resolved["failure_cer"] <= cand.cer:
            classification = NEWLY_FAILED
        elif base.cer >= resolved["failure_cer"] > cand.cer:
            classification = RECOVERED

        error_deltas: dict[str, int] = {}
        base_errors = dict(base.error_types or {})
        cand_errors = dict(cand.error_types or {})
        for category in sorted(set(base_errors) | set(cand_errors)):
            delta = cand_errors.get(category, 0) - base_errors.get(category, 0)
            if delta:
                error_deltas[category] = delta
                error_type_delta_totals[category] = (
                    error_type_delta_totals.get(category, 0) + delta
                )

        comparisons.append(
            FailureComparison(
                sample_id=sample_id,
                classification=classification,
                cer_before=base.cer,
                cer_after=cand.cer,
                cer_delta=cer_delta,
                wer_before=base.wer,
                wer_after=cand.wer,
                wer_delta=cand.wer - base.wer,
                ncer_before=base.ncer,
                ncer_after=cand.ncer,
                ncer_delta=cand.ncer - base.ncer,
                error_type_deltas=error_deltas,
                metadata=dict(cand.metadata or {}),
            )
        )
        counts[classification] += 1

    regressed_sorted = sorted(
        (c for c in comparisons if c.classification in (REGRESSED, NEWLY_FAILED)),
        key=lambda c: (-c.cer_delta, c.sample_id),
    )
    improved_sorted = sorted(
        (c for c in comparisons if c.classification in (IMPROVED, RECOVERED)),
        key=lambda c: (c.cer_delta, c.sample_id),
    )
    persistent = sorted(
        c.sample_id
        for c in comparisons
        if c.cer_before >= resolved["failure_cer"]
        and c.cer_after >= resolved["failure_cer"]
    )

    return FailureReport(
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        thresholds=dict(resolved),
        counts=counts,
        comparisons=tuple(comparisons),
        worst_regressions=tuple(c.sample_id for c in regressed_sorted[:10]),
        best_improvements=tuple(c.sample_id for c in improved_sorted[:10]),
        persistent_failures=tuple(persistent),
        error_type_delta_totals=dict(sorted(error_type_delta_totals.items())),
    )


def summarize_comparison(report: FailureReport) -> dict[str, Any]:
    """Small machine-readable rollup used by exports and the E2E flow."""
    cer_deltas = [c.cer_delta for c in report.comparisons]
    return {
        "baseline_id": report.baseline_id,
        "candidate_id": report.candidate_id,
        "counts": dict(report.counts),
        "common_samples": len(report.comparisons),
        "mean_cer_delta": mean(cer_deltas) if cer_deltas else 0.0,
        "worst_regressions": list(report.worst_regressions),
        "best_improvements": list(report.best_improvements),
        "persistent_failures": list(report.persistent_failures),
        "error_type_delta_totals": dict(report.error_type_delta_totals),
    }


__all__ = [
    "DEFAULT_THRESHOLDS",
    "SampleMetrics",
    "compare_failure",
    "summarize_comparison",
]
