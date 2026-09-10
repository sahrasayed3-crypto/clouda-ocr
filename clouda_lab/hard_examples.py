"""Hard Example Mining engine.

Ranks samples with a **transparent, configurable** scoring function: every
signal, weight and the final score are recorded in the output — no hidden
magic constants. Signals are normalized to [0, 1] per batch before weighting.

Available signals (only computed when input data provides them):
- ``cer``, ``wer``, ``ncer``: raw error rates (already in [0, ∞), clipped).
- ``regression_magnitude``: CER delta vs a baseline model/run.
- ``error_diversity``: distinct error categories on the sample / max seen.
- ``persistent_failure_count``: times the sample failed across runs / max.
- ``distortion_severity``: metadata severity mapped low<mid<high.
- ``error_type_rarity``: 1 - (category frequency share), rewarding samples
  whose errors are uncommon in the batch.
- ``cross_model_failure``: fraction of models failing the sample (CER ≥ threshold).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import HardExampleScore

DEFAULT_WEIGHTS: dict[str, float] = {
    "cer": 0.30,
    "wer": 0.15,
    "ncer": 0.15,
    "regression_magnitude": 0.10,
    "error_diversity": 0.10,
    "persistent_failure_count": 0.10,
    "distortion_severity": 0.05,
    "error_type_rarity": 0.05,
}

_SEVERITY_ORDER = {"none": 0.0, "low": 0.33, "light": 0.33, "medium": 0.66, "mid": 0.66, "high": 1.0, "heavy": 1.0, "severe": 1.0}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class _SignalExtractor:
    """Builds per-sample raw signals from one batch of inputs."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        failure_cer: float,
    ) -> None:
        self.failure_cer = failure_cer
        self.samples = list(samples)
        max_cer = max((float(s.get("cer", 0.0) or 0.0) for s in self.samples), default=0.0)
        self._cer_norm = (max_cer or 1.0)
        self._max_categories = max(
            (
                len((s.get("error_type_counts") or {}))
                for s in self.samples
            ),
            default=1,
        ) or 1
        self._max_persistent = max(
            (float(s.get("persistent_failure_count", 0) or 0) for s in self.samples),
            default=0.0,
        ) or 1.0
        self._max_regression = max(
            (float(s.get("regression_magnitude", 0) or 0) for s in self.samples),
            default=0.0,
        ) or 1.0
        # Category frequency across the batch for rarity.
        category_totals: dict[str, int] = {}
        for sample in self.samples:
            for category, count in (sample.get("error_type_counts") or {}).items():
                category_totals[category] = category_totals.get(category, 0) + int(count)
        total = sum(category_totals.values()) or 1
        self._category_share = {
            category: count / total for category, count in category_totals.items()
        }
        models_per_sample: dict[str, set[str]] = {}
        for sample in self.samples:
            model = str(sample.get("model_id", "unspecified"))
            models_per_sample.setdefault(str(sample.get("sample_id")), set()).add(model)
        self._models_per_sample = models_per_sample

    def signals(self, sample: Mapping[str, Any]) -> dict[str, float]:
        counts = dict(sample.get("error_type_counts") or {})
        cer = _clip01(float(sample.get("cer", 0.0) or 0.0))
        wer = _clip01(float(sample.get("wer", 0.0) or 0.0))
        ncer = _clip01(float(sample.get("ncer", 0.0) or 0.0))
        categories = list(counts)
        rarity = 0.0
        if categories:
            rarity = 1.0 - sum(self._category_share.get(c, 0.0) for c in categories) / len(categories)
        metadata = sample.get("metadata") or {}
        severity = str(sample.get("severity", metadata.get("severity", "none")))
        models = self._models_per_sample.get(str(sample.get("sample_id")), set())
        return {
            "cer": cer,
            "wer": wer,
            "ncer": ncer,
            "regression_magnitude": _clip01(
                float(sample.get("regression_magnitude", 0) or 0) / self._max_regression
            ),
            "error_diversity": len(categories) / self._max_categories if categories else 0.0,
            "persistent_failure_count": _clip01(
                float(sample.get("persistent_failure_count", 0) or 0) / self._max_persistent
            ),
            "distortion_severity": _SEVERITY_ORDER.get(severity.casefold(), 0.0),
            "error_type_rarity": _clip01(rarity),
            "cross_model_failure": (
                sum(
                    1
                    for m in models
                    if float(sample.get("cer", 0.0) or 0.0) >= self.failure_cer
                )
                / max(1, len(models))
            )
            if models
            else 0.0,
        }


def rank_hard_examples(
    samples: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
    failure_cer: float = 0.5,
) -> list[HardExampleScore]:
    """Rank samples hardest-first with the transparent weighted score.

    Each result records the raw signals, the weights applied, and the final
    score. Deterministic: ties break by sample_id.
    """
    resolved_weights = {
        key: float((weights or {}).get(key, DEFAULT_WEIGHTS.get(key, 0.0)))
        for key in DEFAULT_WEIGHTS
    }
    if weights:
        for key in weights:
            if key not in DEFAULT_WEIGHTS:
                raise ValueError(f"Unknown hard-example signal: {key}")
    extractor = _SignalExtractor(samples, failure_cer=failure_cer)
    scored: list[HardExampleScore] = []
    for sample in samples:
        signals = extractor.signals(sample)
        score = sum(
            resolved_weights[name] * signals.get(name, 0.0) for name in resolved_weights
        )
        scored.append(
            HardExampleScore(
                sample_id=str(sample.get("sample_id")),
                score=round(score, 6),
                signals={k: round(v, 6) for k, v in signals.items()},
                weights=dict(resolved_weights),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.sample_id))
    return [
        HardExampleScore(
            sample_id=item.sample_id,
            score=item.score,
            signals=item.signals,
            weights=item.weights,
            rank=rank,
        )
        for rank, item in enumerate(scored, 1)
    ]


def select_hard_examples(
    samples: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
    failure_cer: float = 0.5,
    top_n: int | None = None,
    min_score: float | None = None,
    percentile: float | None = None,
) -> list[HardExampleScore]:
    """Rank then cut by top-N / score floor / percentile (hardest fraction)."""
    ranked = rank_hard_examples(samples, weights=weights, failure_cer=failure_cer)
    if percentile is not None:
        if not 0 < percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")
        cutoff_index = max(1, int(round(len(ranked) * percentile / 100.0)))
        ranked = ranked[:cutoff_index]
    if min_score is not None:
        ranked = [item for item in ranked if item.score >= min_score]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


__all__ = [
    "DEFAULT_WEIGHTS",
    "rank_hard_examples",
    "select_hard_examples",
]
