"""Active Learning / next-batch recommender.

Deterministic recommendation of candidate pages for the next training batch.
Strategies:

- ``hardest_only``: pure hard-example ranking, no balancing.
- ``balanced_hard``: hard ranking with per-group quotas (round-robin over
  group keys, hardest first inside each group).
- ``diversity_first``: maximize error-type coverage before raw hardness.
- ``mixed_curriculum``: alternate hard/diverse/easy thirds.
- ``deterministic_random``: seed-hashed random baseline.

Every selection records a per-sample rationale string. History (previously
used samples) is excluded when provided.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

from .hard_examples import rank_hard_examples
from .models import HardExampleScore, TrainingBatchRecommendation

STRATEGIES = (
    "hardest_only",
    "balanced_hard",
    "diversity_first",
    "mixed_curriculum",
    "deterministic_random",
)


def _hash_rank(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"lab-recommend:{seed}:{sample_id}".encode("utf-8")).digest()


def _group_key(sample: Mapping[str, Any], dimension: str) -> str:
    value = sample.get(dimension)
    if value is None and isinstance(sample.get("metadata"), Mapping):
        value = sample["metadata"].get(dimension)
    return str(value) if value is not None else "unspecified"


def _balance_counts(
    selected: Sequence[str], samples_by_id: Mapping[str, Mapping[str, Any]], dimension: str
) -> dict[str, int]:
    counter = Counter(
        _group_key(samples_by_id[sid], dimension) for sid in selected if sid in samples_by_id
    )
    return dict(sorted(counter.items()))


def recommend_next_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    strategy: str = "balanced_hard",
    batch_size: int = 32,
    seed: int = 20260723,
    weights: Mapping[str, float] | None = None,
    history: Sequence[str] = (),
    balance_dimensions: Sequence[str] = ("profile", "distortion"),
    quota_dimension: str = "profile",
    failure_cer: float = 0.5,
) -> TrainingBatchRecommendation:
    """Recommend the next training batch. Fully deterministic per inputs."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy} (have: {STRATEGIES})")
    samples_by_id = {str(s.get("sample_id")): s for s in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("Duplicate sample_id in recommendation input")
    excluded = [sid for sid in history if sid in samples_by_id]
    candidates = [s for s in samples if str(s.get("sample_id")) not in set(history)]

    ranked = rank_hard_examples(candidates, weights=weights, failure_cer=failure_cer)
    rank_map = {item.sample_id: item for item in ranked}

    selected: list[str] = []
    rationale: dict[str, str] = {}

    if strategy == "hardest_only":
        for item in ranked[:batch_size]:
            selected.append(item.sample_id)
            rationale[item.sample_id] = (
                f"hardest rank {item.rank} (score={item.score:.4f})"
            )
    elif strategy == "deterministic_random":
        ordered = sorted(
            candidates,
            key=lambda s: (_hash_rank(str(s.get("sample_id")), seed), str(s.get("sample_id"))),
        )
        for sample in ordered[:batch_size]:
            sid = str(sample.get("sample_id"))
            selected.append(sid)
            rationale[sid] = "deterministic random baseline (seed-hashed)"
    elif strategy == "balanced_hard":
        groups: dict[str, list[HardExampleScore]] = {}
        for item in ranked:
            key = _group_key(samples_by_id[item.sample_id], quota_dimension)
            groups.setdefault(key, []).append(item)
        group_cycle = sorted(groups)
        while len(selected) < batch_size and any(groups[g] for g in group_cycle):
            for group in group_cycle:
                if len(selected) >= batch_size:
                    break
                if groups[group]:
                    item = groups[group].pop(0)
                    selected.append(item.sample_id)
                    rationale[item.sample_id] = (
                        f"hardest in {quota_dimension}={group} "
                        f"(rank {item.rank}, score={item.score:.4f})"
                    )
    elif strategy == "diversity_first":
        category_of: dict[str, str] = {}
        for sample in candidates:
            counts = sample.get("error_type_counts") or {}
            top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else "none"
            category_of[str(sample.get("sample_id"))] = str(top)
        seen_categories: set[str] = set()
        for item in ranked:
            if len(selected) >= batch_size:
                break
            sid = item.sample_id
            category = category_of.get(sid, "none")
            if category not in seen_categories:
                seen_categories.add(category)
                selected.append(sid)
                rationale[sid] = f"first representative of error category '{category}' (rank {item.rank})"
        for item in ranked:
            if len(selected) >= batch_size:
                break
            if item.sample_id not in selected:
                selected.append(item.sample_id)
                rationale[item.sample_id] = (
                    f"fill by hardness (rank {item.rank}, score={item.score:.4f})"
                )
    else:  # mixed_curriculum
        third = max(1, batch_size // 3)
        hard_part = [item.sample_id for item in ranked[:third]]
        diverse_tail = [item.sample_id for item in ranked[third : 2 * third]]
        easy_pool = sorted(ranked[2 * third :], key=lambda i: (i.score, i.sample_id))
        easy_part = [item.sample_id for item in easy_pool[:batch_size]]
        selected = (hard_part + diverse_tail + easy_part)[:batch_size]
        for sid in selected:
            rank = rank_map[sid].rank
            if sid in hard_part:
                rationale[sid] = f"curriculum-hard (rank {rank})"
            elif sid in diverse_tail:
                rationale[sid] = f"curriculum-mid (rank {rank})"
            else:
                rationale[sid] = f"curriculum-easy (rank {rank})"

    balance = {
        dim: _balance_counts(selected, samples_by_id, dim) for dim in balance_dimensions
    }
    selections = tuple(rank_map[sid] for sid in selected if sid in rank_map)
    if strategy == "deterministic_random":
        selections = tuple(
            HardExampleScore(
                sample_id=sid, score=0.0, signals={}, weights={}, rank=0
            )
            for sid in selected
        )

    return TrainingBatchRecommendation(
        strategy=strategy,
        seed=seed,
        selections=selections,
        rationale=rationale,
        balance=balance,
        excluded=tuple(excluded),
    )


__all__ = [
    "STRATEGIES",
    "recommend_next_batch",
]
