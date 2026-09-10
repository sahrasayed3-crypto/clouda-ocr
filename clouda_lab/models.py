"""Clouda Lab backend — analysis engine contracts.

Frozen, JSON-serializable data models shared by the Clouda Lab backend
services (error analysis, batch analysis, failure analysis, hard-example
mining, active learning, distortion experiments, evaluation service, and the
training orchestrator facade).

Design rules
------------
- Dataclasses only: no framework imports, no UI code, no I/O.
- Every model has a ``to_dict`` that produces JSON-safe values so future
  API/UI layers can serialize without transformation.
- IDs reference existing repository entities (factory run rows, pretraining
  manifest ``sample_id`` values, training run ids) — the lab never mints a
  second identity for an object that already has one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Error analysis models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorRecord:
    """One aligned difference between ground truth and prediction.

    ``position`` indexes into the *normalized* character list used for
    alignment; ``word_index`` is the word ordinal when the record comes from
    word-level alignment (``None`` otherwise). ``context`` carries a small
    window of surrounding ground-truth text for future UI highlighting.
    """

    level: str  # "char" | "word"
    operation: str  # "match" | "substitution" | "insertion" | "deletion"
    error_type: str  # Arabic-aware classification label
    gt: str | None
    predicted: str | None
    position: int  # gt-side index (0-based); hyp-side for insertions
    predicted_position: int | None
    word_index: int | None
    context: str = ""
    normalized: bool = False  # record produced during normalized alignment

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "operation": self.operation,
            "error_type": self.error_type,
            "gt": self.gt,
            "predicted": self.predicted,
            "position": self.position,
            "predicted_position": self.predicted_position,
            "word_index": self.word_index,
            "context": self.context,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class ErrorAnalysis:
    """Full error analysis for a single OCR sample."""

    page_id: str
    model_id: str
    run_id: str
    dataset_id: str
    cer: float
    wer: float
    normalized_cer: float
    exact_match: bool
    character_count: int
    word_count: int
    char_records: tuple[ErrorRecord, ...]
    word_records: tuple[ErrorRecord, ...]
    error_type_counts: dict[str, int]
    error_type_rates: dict[str, float]
    substitutions: tuple[tuple[str, str, int], ...]  # (gt, pred, count)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "cer": self.cer,
            "wer": self.wer,
            "normalized_cer": self.normalized_cer,
            "exact_match": self.exact_match,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "char_errors": [
                r.to_dict() for r in self.char_records if r.operation != "match"
            ],
            "char_matches": sum(1 for r in self.char_records if r.operation == "match"),
            "word_errors": [
                r.to_dict() for r in self.word_records if r.operation != "match"
            ],
            "error_type_counts": dict(self.error_type_counts),
            "error_type_rates": dict(self.error_type_rates),
            "substitutions": [
                {"gt": gt, "predicted": pred, "count": count}
                for gt, pred, count in self.substitutions
            ],
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Sample / prediction inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRSample:
    """One OCR evaluation sample: ground truth plus one model prediction.

    ``metadata`` carries only what is actually present in source manifests
    (profile, severity, distortion, document_type, source, split, tags, ...).
    Nothing is inferred.
    """

    sample_id: str
    ground_truth: str
    prediction: str
    model_id: str = "unspecified"
    run_id: str = "unspecified"
    dataset_id: str = "unspecified"
    page_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "page_id": self.page_id,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Batch analysis models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSummary:
    """Aggregated metrics for one group of analyzed samples."""

    group: str
    key: str
    count: int
    cer_mean: float
    wer_mean: float
    ncer_mean: float
    exact_match_rate: float
    cer_p50: float
    cer_p90: float
    cer_p95: float
    cer_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "key": self.key,
            "count": self.count,
            "cer_mean": self.cer_mean,
            "wer_mean": self.wer_mean,
            "ncer_mean": self.ncer_mean,
            "exact_match_rate": self.exact_match_rate,
            "cer_p50": self.cer_p50,
            "cer_p90": self.cer_p90,
            "cer_p95": self.cer_p95,
            "cer_max": self.cer_max,
        }


@dataclass(frozen=True)
class BatchReport:
    """Result of batch analysis over many OCR samples."""

    total_samples: int
    overall: BatchSummary
    groups: dict[str, dict[str, BatchSummary]]  # dimension -> key -> summary
    error_type_counts: dict[str, int]
    common_substitutions: tuple[tuple[str, str, int], ...]
    worst_pages: tuple[BatchSummary, ...]  # per-page rows sorted worst-first
    best_pages: tuple[BatchSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "overall": self.overall.to_dict(),
            "groups": {
                dim: {key: s.to_dict() for key, s in keys.items()}
                for dim, keys in self.groups.items()
            },
            "error_type_counts": dict(self.error_type_counts),
            "common_substitutions": [
                {"gt": gt, "predicted": pred, "count": n}
                for gt, pred, n in self.common_substitutions
            ],
            "worst_pages": [s.to_dict() for s in self.worst_pages],
            "best_pages": [s.to_dict() for s in self.best_pages],
        }


# ---------------------------------------------------------------------------
# Failure analysis models
# ---------------------------------------------------------------------------

# Classifications for per-sample before/after comparison.
IMPROVED = "improved"
REGRESSED = "regressed"
UNCHANGED = "unchanged"
NEWLY_FAILED = "newly_failed"
RECOVERED = "recovered"


@dataclass(frozen=True)
class FailureComparison:
    """Per-sample comparison of one model/run against another."""

    sample_id: str
    classification: str
    cer_before: float
    cer_after: float
    cer_delta: float  # after - before (positive = worse for CER)
    wer_before: float
    wer_after: float
    wer_delta: float
    ncer_before: float
    ncer_after: float
    ncer_delta: float
    error_type_deltas: dict[str, int]  # +gained / -resolved per category
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "classification": self.classification,
            "cer_before": self.cer_before,
            "cer_after": self.cer_after,
            "cer_delta": self.cer_delta,
            "wer_before": self.wer_before,
            "wer_after": self.wer_after,
            "wer_delta": self.wer_delta,
            "ncer_before": self.ncer_before,
            "ncer_after": self.ncer_after,
            "ncer_delta": self.ncer_delta,
            "error_type_deltas": dict(self.error_type_deltas),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FailureReport:
    """Aggregated before/after comparison across samples."""

    baseline_id: str
    candidate_id: str
    thresholds: dict[str, float]
    counts: dict[str, int]  # classification -> count
    comparisons: tuple[FailureComparison, ...]
    worst_regressions: tuple[str, ...]  # sample ids
    best_improvements: tuple[str, ...]
    persistent_failures: tuple[str, ...]
    error_type_delta_totals: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "thresholds": dict(self.thresholds),
            "counts": dict(self.counts),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "worst_regressions": list(self.worst_regressions),
            "best_improvements": list(self.best_improvements),
            "persistent_failures": list(self.persistent_failures),
            "error_type_delta_totals": dict(self.error_type_delta_totals),
        }


# ---------------------------------------------------------------------------
# Failure buckets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureBucket:
    """One structured failure bucket assignment for a sample."""

    bucket: str
    sample_id: str
    evidence: dict[str, Any]  # measured values that triggered the bucket

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "sample_id": self.sample_id,
            "evidence": dict(self.evidence),
        }


# ---------------------------------------------------------------------------
# Hard example / active learning models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardExampleScore:
    """Transparent, reproducible score for one sample."""

    sample_id: str
    score: float
    signals: dict[str, float]  # raw signal values before weighting
    weights: dict[str, float]  # weights actually applied
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "score": self.score,
            "signals": dict(self.signals),
            "weights": dict(self.weights),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class TrainingBatchRecommendation:
    """Deterministic next-batch recommendation for active learning."""

    strategy: str
    seed: int
    selections: tuple[HardExampleScore, ...]
    rationale: dict[str, str]  # sample_id -> human-readable reason
    balance: dict[str, dict[str, int]]  # dimension -> key -> count
    excluded: tuple[str, ...]  # already-used / filtered sample ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "selections": [s.to_dict() for s in self.selections],
            "rationale": dict(self.rationale),
            "balance": {dim: dict(keys) for dim, keys in self.balance.items()},
            "excluded": list(self.excluded),
        }


# ---------------------------------------------------------------------------
# Selection models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSelection:
    """The result of a dataset selection query (ids + criteria)."""

    selection_id: str
    sample_ids: tuple[str, ...]
    criteria: dict[str, Any]
    seed: int
    created_utc: str
    excluded_protected: int  # rows rejected by the holdout guard

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "sample_ids": list(self.sample_ids),
            "criteria": dict(self.criteria),
            "seed": self.seed,
            "created_utc": self.created_utc,
            "excluded_protected": self.excluded_protected,
        }


# ---------------------------------------------------------------------------
# Distortion experiment models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistortionVariantPlan:
    """One planned variant inside a distortion experiment."""

    variant_id: str
    profile: str
    distortions: tuple[str, ...]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "profile": self.profile,
            "distortions": list(self.distortions),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class DistortionExperiment:
    """A tracked, deterministic distortion experiment definition."""

    experiment_id: str
    source_manifest: str
    source_sample_ids: tuple[str, ...]
    variants: tuple[DistortionVariantPlan, ...]
    created_utc: str
    factory_run_dir: str | None = None  # set after generation
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "source_manifest": self.source_manifest,
            "source_sample_ids": list(self.source_sample_ids),
            "variants": [v.to_dict() for v in self.variants],
            "created_utc": self.created_utc,
            "factory_run_dir": self.factory_run_dir,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Run analysis (post-run pipeline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunAnalysis:
    """Backend analysis artifact for one training run lifecycle."""

    run_id: str
    status: str
    metrics_summary: dict[str, Any]
    failure_report: FailureReport | None
    hard_examples: tuple[HardExampleScore, ...]
    recommendation: TrainingBatchRecommendation | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "metrics_summary": dict(self.metrics_summary),
            "failure_report": (
                self.failure_report.to_dict() if self.failure_report else None
            ),
            "hard_examples": [s.to_dict() for s in self.hard_examples],
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
        }
