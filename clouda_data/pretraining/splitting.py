"""Deterministic, leakage-safe train/validation/test/holdout splitting.

Rules:

1. Samples are grouped by ``group_id`` (falling back to ``document_id``,
   then ``source_record_id``, then ``sample_id``). Pages of one document
   always share a group, so a document can never straddle splits.
2. Groups that share a file hash or a normalized-text hash are merged with
   union-find before assignment, so duplicate and conflicting content
   cannot leak across splits either.
3. Each merged group is assigned exactly one split by hashing
   ``seed + smallest group key`` — deterministic, reproducible, and stable
   when new groups arrive later (no global reshuffle).
4. The ``holdout`` split is structurally protected: exports exclude it
   unless explicitly and deliberately enabled.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .schema import DatasetSample, SplitName, ValidationStatus

DEFAULT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.05,
    "holdout": 0.05,
}
SPLIT_ORDER = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST, SplitName.HOLDOUT)
PROTECTED_SPLITS = (SplitName.HOLDOUT,)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Merge deterministically toward the lexicographically smaller root.
            small, large = sorted((left_root, right_root))
            self._parent[large] = small


@dataclass
class SplitReport:
    counts: dict[str, int] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    leakage_checks: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    ratios: dict[str, float] = field(default_factory=dict)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "group_counts": self.group_counts,
            "leakage_checks": self.leakage_checks,
            "seed": self.seed,
            "ratios": self.ratios,
            "passed": self.passed,
            "schema_version": "clouda.pretraining.split.v1",
        }


def resolve_group_key(sample: DatasetSample) -> str:
    if sample.group_id:
        return f"{sample.source_id}:group:{sample.group_id}"
    if sample.document_id:
        return f"{sample.source_id}:{sample.document_id}"
    if sample.source_record_id:
        return f"{sample.source_id}:record:{sample.source_record_id}"
    return sample.sample_id


def _split_for_key(key: str, seed: int, boundaries: list[tuple[str, int]]) -> SplitName:
    digest = hashlib.sha256(f"{seed}:split:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest, "big")
    for name, cumulative in boundaries:
        if bucket < cumulative:
            return SplitName(name)
    return SplitName.HOLDOUT


def assign_splits(
    samples: list[DatasetSample],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> tuple[list[DatasetSample], SplitReport]:
    """Assign target splits deterministically and check for leakage."""

    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != {name.value for name in SPLIT_ORDER}:
        raise ValueError("Split ratios must define train, validation, test, holdout.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
        for value in ratios.values()
    ):
        raise ValueError("Split ratios must be finite numbers between 0 and 1.")
    exact_ratios = {name: Fraction(str(value)) for name, value in ratios.items()}
    if sum(exact_ratios.values()) != 1:
        raise ValueError("Split ratios must sum to 1.0.")

    active = [
        s
        for s in samples
        if s.validation_status
        not in (ValidationStatus.ERROR, ValidationStatus.EXCLUDED)
    ]

    group_members: dict[str, list[str]] = {}
    samples_by_group: dict[str, list[DatasetSample]] = {}
    for sample in active:
        key = resolve_group_key(sample)
        group_members.setdefault(key, []).append(sample.sample_id)
        samples_by_group.setdefault(key, []).append(sample)

    union = _UnionFind()
    for key in group_members:
        union.add(key)
    hash_owner: dict[tuple[str, str], str] = {}
    sample_groups: dict[str, set[str]] = {}
    for key in sorted(group_members):
        for sample in sorted(
            samples_by_group[key],
            key=lambda item: (item.sample_id, item.source_path),
        ):
            sample_groups.setdefault(sample.sample_id, set()).add(key)
            for kind, hash_value in (
                ("file", sample.file_sha256),
                ("text", sample.normalized_text_sha256),
            ):
                if not hash_value:
                    continue
                owner = hash_owner.setdefault((kind, hash_value), key)
                if owner != key:
                    union.union(owner, key)

    for sample in active:
        if not sample.duplicate_of:
            continue
        own_key = resolve_group_key(sample)
        for canonical_key in sorted(sample_groups.get(sample.duplicate_of, set())):
            union.union(own_key, canonical_key)

    merged_root: dict[str, str] = {key: union.find(key) for key in group_members}
    merged_members: dict[str, list[str]] = {}
    for key, root in merged_root.items():
        merged_members.setdefault(root, []).append(key)

    cumulative = Fraction(0)
    domain_size = 1 << 256
    boundaries: list[tuple[str, int]] = []
    for name in ("train", "validation", "test", "holdout"):
        cumulative += exact_ratios[name]
        boundaries.append((name, int(cumulative * domain_size)))

    split_of_group: dict[str, SplitName] = {}
    for root, members in merged_members.items():
        anchor = min(members)
        split = _split_for_key(anchor, seed, boundaries)
        for member in members:
            split_of_group[member] = split

    updated: list[DatasetSample] = []
    for sample in sorted(samples, key=lambda s: s.sample_id):
        active_status = sample.validation_status not in (
            ValidationStatus.ERROR,
            ValidationStatus.EXCLUDED,
        )
        split = (
            split_of_group.get(resolve_group_key(sample), SplitName.UNASSIGNED)
            if active_status
            else SplitName.UNASSIGNED
        )
        updated.append(sample.evolve(target_split=split))

    report = _build_report(updated, group_members, seed, ratios)
    return updated, report


def _build_report(
    samples: list[DatasetSample],
    group_members: dict[str, list[str]],
    seed: int,
    ratios: dict[str, float],
) -> SplitReport:
    active = [
        s
        for s in samples
        if s.validation_status
        not in (ValidationStatus.ERROR, ValidationStatus.EXCLUDED)
        and s.target_split != SplitName.UNASSIGNED
    ]
    counts = {name.value: 0 for name in SPLIT_ORDER}
    group_counts: dict[str, int] = {name.value: 0 for name in SPLIT_ORDER}
    seen_groups: dict[str, set[str]] = {name.value: set() for name in SPLIT_ORDER}
    for sample in active:
        split = sample.target_split.value
        counts[split] += 1
        seen_groups[split].add(resolve_group_key(sample))
    for split, groups in seen_groups.items():
        group_counts[split] = len(groups)

    checks: list[dict[str, Any]] = []
    passed = True

    def record(name: str, violations: set[str]) -> None:
        nonlocal passed
        ok = not violations
        passed = passed and ok
        checks.append({"check": name, "passed": ok, "violations": len(violations)})

    file_hash_split: dict[str, set[str]] = {}
    text_hash_split: dict[str, set[str]] = {}
    document_split: dict[str, set[str]] = {}
    for sample in active:
        split = sample.target_split.value
        if sample.file_sha256:
            file_hash_split.setdefault(sample.file_sha256, set()).add(split)
        if sample.normalized_text_sha256:
            text_hash_split.setdefault(sample.normalized_text_sha256, set()).add(split)
        if sample.document_id:
            document_split.setdefault(
                f"{sample.source_id}:{sample.document_id}", set()
            ).add(split)
    record(
        "file_hash_not_shared_across_splits",
        {h for h, splits in file_hash_split.items() if len(splits) > 1},
    )
    record(
        "normalized_text_hash_not_shared_across_splits",
        {h for h, splits in text_hash_split.items() if len(splits) > 1},
    )
    record(
        "document_not_shared_across_splits",
        {d for d, splits in document_split.items() if len(splits) > 1},
    )
    group_split: dict[str, set[str]] = {}
    sample_id_splits: dict[str, set[str]] = {}
    for sample in active:
        group_split.setdefault(resolve_group_key(sample), set()).add(
            sample.target_split.value
        )
        sample_id_splits.setdefault(sample.sample_id, set()).add(
            sample.target_split.value
        )
    record(
        "group_not_shared_across_splits",
        {group for group, splits in group_split.items() if len(splits) > 1},
    )
    duplicate_cluster_splits: dict[str, set[str]] = {}
    for sample in active:
        if sample.duplicate_of:
            duplicate_cluster_splits.setdefault(sample.duplicate_of, set()).add(
                sample.target_split.value
            )
            duplicate_cluster_splits[sample.duplicate_of].update(
                sample_id_splits.get(sample.duplicate_of, set())
            )
    record(
        "duplicate_cluster_not_shared_across_splits",
        {
            canonical
            for canonical, splits in duplicate_cluster_splits.items()
            if len(splits) > 1
        },
    )

    # Groups are assigned exactly one split, so the holdout group set is
    # disjoint from every other split's group set by construction; verify.
    training_groups = (
        seen_groups.get("train", set())
        | seen_groups.get("validation", set())
        | seen_groups.get("test", set())
    )
    record(
        "holdout_disjoint_from_training",
        seen_groups.get("holdout", set()) & training_groups,
    )

    return SplitReport(
        counts=counts,
        group_counts=group_counts,
        leakage_checks=checks,
        seed=seed,
        ratios=ratios,
        passed=passed,
    )
