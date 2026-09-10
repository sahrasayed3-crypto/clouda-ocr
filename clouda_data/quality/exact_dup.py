"""Exact-duplicate classification for the dataset quality gate.

Wraps :func:`clouda_data.pretraining.dedupe.classify_duplicates` for the exact
signals (repeated ``sample_id``, repeated ``(source_id, source_record_id)``,
repeated ``file_sha256``) and extends it with two quality-gate passes:

- raw-text evidence: samples whose raw text hashes equal (domain-separated
  SHA-256, never ``hash()``) are grouped as EVIDENCE ONLY. The same raw text
  over different ``file_sha256`` values is a ``CONFLICTING_DUPLICATE`` (kept in
  the dataset, never a ``DUPLICATE``); the same raw text over the same file
  hash stays an exact ``DUPLICATE``.
- confirmed near-duplicate pairs (precomputed ``CONFIRMED_NEAR_DUPLICATE``
  pairs) are unioned with the same union-find semantics, reason
  ``near_duplicate_image``.

Deduplication never deletes data: samples are classified and cross-linked with
``duplicate_of`` provenance. Canonical selection reuses the pretraining dedupe
canonical ordering (lowest canonical key wins).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from clouda_data.pretraining.dedupe import classify_duplicates
from clouda_data.pretraining.schema import (
    DatasetSample,
    DuplicateState,
    ValidationStatus,
    sort_key,
)

EXACT_DUP_SCHEMA_VERSION = "clouda.quality.dedupe.v2"

RAW_TEXT_HASH_DOMAIN = b"clouda.text.raw.v1"

NEAR_DUPLICATE_IMAGE_REASON = "near_duplicate_image"

_EXACT_DUPLICATE_REASONS = {
    "duplicate_sample_id",
    "duplicate_source_record",
    "duplicate_file_hash",
}


def raw_text_sha256(text: str) -> str:
    """Domain-separated SHA-256 of raw sample text (never ``hash()``)."""

    return hashlib.sha256(
        RAW_TEXT_HASH_DOMAIN + b"\x00" + text.encode("utf-8")
    ).hexdigest()


def _canonical_key(sample: DatasetSample) -> tuple[object, ...]:
    """Pretraining dedupe canonical ordering (lowest key wins)."""

    payload = sample.to_dict()
    for key in (
        "duplicate_state",
        "duplicate_of",
        "target_split",
        "validation_findings",
    ):
        payload.pop(key, None)
    return (
        sample.validation_status in (ValidationStatus.ERROR, ValidationStatus.EXCLUDED),
        sample.source_id,
        sample.source_path,
        sample.sample_id,
        sample.source_record_id or "",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


@dataclass
class ExactDupReport:
    """Counts-only report for exact-duplicate classification."""

    unique: int = 0
    duplicate_file_hash: int = 0
    duplicate_sample_id: int = 0
    duplicate_source_record: int = 0
    conflicting_duplicate: int = 0
    near_duplicate_image: int = 0
    families: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duplicate_count(self) -> int:
        """Total samples classified as duplicates or conflicting duplicates."""

        return (
            self.duplicate_file_hash
            + self.duplicate_sample_id
            + self.duplicate_source_record
            + self.near_duplicate_image
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                "unique": self.unique,
                "duplicate_file_hash": self.duplicate_file_hash,
                "duplicate_sample_id": self.duplicate_sample_id,
                "duplicate_source_record": self.duplicate_source_record,
                "conflicting_duplicate": self.conflicting_duplicate,
                "near_duplicate_image": self.near_duplicate_image,
            },
            "families": self.families,
            "schema_version": EXACT_DUP_SCHEMA_VERSION,
        }


def _pair_endpoints(pair: Any) -> tuple[str, str] | None:
    """Accept ``(sample_id_a, sample_id_b)`` tuples or candidate objects."""

    if isinstance(pair, tuple) and len(pair) == 2:
        left, right = pair
    else:
        left = getattr(pair, "sample_id_a", None)
        right = getattr(pair, "sample_id_b", None)
    if not isinstance(left, str) or not isinstance(right, str) or left == right:
        return None
    return (left, right) if left < right else (right, left)


def _apply_raw_text_conflicts(
    ordered: list[DatasetSample],
    updated: list[DatasetSample],
    report: ExactDupReport,
) -> None:
    """Group identical raw text as evidence; different file hashes conflict.

    The same raw text over different ``file_sha256`` values becomes
    ``CONFLICTING_DUPLICATE`` (kept, never ``DUPLICATE``). The same raw text
    over the same file hash keeps its exact-duplicate classification.
    """

    by_raw_hash: dict[str, list[int]] = {}
    for index, sample in enumerate(updated):
        text = sample.raw_text if sample.raw_text is not None else sample.text
        if not text:
            continue
        by_raw_hash.setdefault(raw_text_sha256(text), []).append(index)

    for _raw_hash, indexes in sorted(by_raw_hash.items()):
        if len(indexes) < 2:
            continue
        hashes = {
            updated[index].file_sha256
            for index in indexes
            if updated[index].file_sha256
        }
        if len(hashes) < 2:
            continue  # same raw text over the same file hash: stays DUPLICATE
        canonical_index = min(indexes, key=lambda value: _canonical_key(ordered[value]))
        canonical_id = updated[canonical_index].sample_id
        canonical_hash = updated[canonical_index].file_sha256
        members: list[str] = []
        for index in indexes:
            if index == canonical_index:
                continue
            sample = updated[index]
            if sample.file_sha256 and sample.file_sha256 == canonical_hash:
                continue  # identical artifact: exact-duplicate semantics win
            members.append(sample.sample_id)
            updated[index] = sample.evolve(
                duplicate_state=DuplicateState.CONFLICTING_DUPLICATE,
                duplicate_of=canonical_id,
            )
            report.conflicting_duplicate += 1
        if members:
            report.families.append(
                {
                    "kind": "raw_text",
                    "canonical_sample_id": canonical_id,
                    "member_sample_ids": sorted(members),
                }
            )


def _apply_confirmed_near_pairs(
    ordered: list[DatasetSample],
    updated: list[DatasetSample],
    confirmed_near_pairs: Sequence[Any],
    report: ExactDupReport,
) -> None:
    """Union confirmed near-duplicate pairs with dedupe union-find semantics."""

    index_by_id = {sample.sample_id: index for index, sample in enumerate(ordered)}
    count = len(ordered)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in sorted(
        (
            endpoints
            for endpoints in map(_pair_endpoints, confirmed_near_pairs)
            if endpoints
        ),
    ):
        left, right = pair
        if left in index_by_id and right in index_by_id:
            union(index_by_id[left], index_by_id[right])

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)

    for indexes in sorted(
        components.values(), key=lambda values: _canonical_key(ordered[min(values)])
    ):
        if len(indexes) < 2:
            continue
        canonical_index = min(indexes, key=lambda value: _canonical_key(ordered[value]))
        converted: list[str] = []
        for index in indexes:
            if index == canonical_index:
                continue
            sample = updated[index]
            if sample.duplicate_state is not DuplicateState.UNIQUE:
                continue  # already exact/conflicting-linked: keep that evidence
            converted.append(sample.sample_id)
            updated[index] = sample.evolve(
                duplicate_state=DuplicateState.DUPLICATE,
                duplicate_of=updated[canonical_index].sample_id,
                exclusion_reason=(
                    sample.exclusion_reason or NEAR_DUPLICATE_IMAGE_REASON
                ),
            )
            report.near_duplicate_image += 1
            report.unique = max(0, report.unique - 1)
        if converted:
            canonical = updated[canonical_index]
            if canonical.duplicate_state is DuplicateState.UNIQUE:
                updated[canonical_index] = canonical.evolve(
                    duplicate_state=DuplicateState.CANONICAL,
                    duplicate_of=None,
                    exclusion_reason=(
                        None
                        if canonical.exclusion_reason in _EXACT_DUPLICATE_REASONS
                        else canonical.exclusion_reason
                    ),
                )
                report.unique = max(0, report.unique - 1)
            report.families.append(
                {
                    "kind": "near_image",
                    "canonical_sample_id": updated[canonical_index].sample_id,
                    "member_sample_ids": sorted(converted),
                }
            )


def classify_exact_duplicates(
    samples: list[DatasetSample],
    confirmed_near_pairs: Sequence[Any] = (),
) -> tuple[list[DatasetSample], ExactDupReport]:
    """Classify exact duplicates, raw-text conflicts, and confirmed near pairs.

    Returns the samples (deterministically ordered, never deleted) plus an
    :class:`ExactDupReport`.
    """

    pre_samples, pre_report = classify_duplicates(list(samples))
    ordered = sorted(pre_samples, key=_canonical_key)
    updated = list(ordered)

    report = ExactDupReport(
        unique=pre_report.unique,
        duplicate_file_hash=pre_report.duplicate_file_hash,
        duplicate_sample_id=pre_report.duplicate_sample_id,
        duplicate_source_record=pre_report.duplicate_source_record,
        conflicting_duplicate=pre_report.conflicting_duplicate,
    )
    report.families.extend(pre_report.families)

    _apply_raw_text_conflicts(ordered, updated, report)
    _apply_confirmed_near_pairs(ordered, updated, confirmed_near_pairs, report)

    return sorted(updated, key=sort_key), report


def families_to_clusters(
    samples: list[DatasetSample],
    report: ExactDupReport,
) -> list[Any]:
    """Convert duplicate families into DuplicateCluster models.

    Deterministic: clusters sorted by cluster_id, members sorted. Cluster id
    is ``CLU_`` + sha256 of the sorted member ids (stable across runs).
    """

    import hashlib

    from clouda_data.quality.models import DuplicateCluster

    by_id = {sample.sample_id: sample for sample in samples}
    clusters: list[DuplicateCluster] = []
    seen: set[tuple[str, ...]] = set()
    for family in report.families:
        canonical_id = str(family.get("canonical_sample_id", ""))
        members = sorted(
            mid
            for mid in [*family.get("member_sample_ids", []), canonical_id]
            if mid and mid in by_id
        )
        if len(members) < 2 or tuple(members) in seen:
            continue
        seen.add(tuple(members))
        digest = hashlib.sha256("\\x00".join(members).encode("utf-8")).hexdigest()
        level = (
            "CONFIRMED_NEAR_DUPLICATE"
            if family.get("kind") == "near_image"
            else "LIKELY_DUPLICATE"
        )
        clusters.append(
            DuplicateCluster(
                cluster_id=f"CLU_{digest[:12]}",
                member_ids=tuple(members),
                level=level,
                evidence={"kind": str(family.get("kind", "exact"))},
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)
