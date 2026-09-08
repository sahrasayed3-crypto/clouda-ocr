"""Duplicate detection and classification.

Deduplication never deletes data. Samples are classified and cross-linked:

- ``canonical``: the deterministic representative of an exact-duplicate
  family (first sample in canonical sort order);
- ``duplicate``: exact file duplicate, repeated sample id, or repeated
  source record — excluded from exports but preserved in the manifest with
  a ``duplicate_of`` provenance link;
- ``conflicting_duplicate``: identical normalized text over different
  images — kept in the dataset (they are real samples) and marked so the
  splitter can merge their groups to prevent text leakage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .schema import DatasetSample, DuplicateState, ValidationStatus, sort_key


@dataclass
class DuplicateReport:
    unique: int = 0
    duplicate_file_hash: int = 0
    duplicate_sample_id: int = 0
    duplicate_source_record: int = 0
    conflicting_duplicate: int = 0
    families: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                "unique": self.unique,
                "duplicate_file_hash": self.duplicate_file_hash,
                "duplicate_sample_id": self.duplicate_sample_id,
                "duplicate_source_record": self.duplicate_source_record,
                "conflicting_duplicate": self.conflicting_duplicate,
            },
            "families": self.families,
            "schema_version": "clouda.pretraining.dedupe.v1",
        }


def classify_duplicates(
    samples: list[DatasetSample],
) -> tuple[list[DatasetSample], DuplicateReport]:
    """Classify connected duplicate families with stable canonical selection."""

    duplicate_reasons = {
        "duplicate_sample_id",
        "duplicate_source_record",
        "duplicate_file_hash",
    }

    def canonical_key(sample: DatasetSample) -> tuple[object, ...]:
        payload = sample.to_dict()
        for key in (
            "duplicate_state",
            "duplicate_of",
            "target_split",
            "validation_findings",
        ):
            payload.pop(key, None)
        return (
            sample.validation_status
            in (ValidationStatus.ERROR, ValidationStatus.EXCLUDED),
            sample.source_id,
            sample.source_path,
            sample.sample_id,
            sample.source_record_id or "",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    ordered = sorted(samples, key=canonical_key)
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

    signal_owners: dict[tuple[str, object], int] = {}
    for index, sample in enumerate(ordered):
        signals: list[tuple[str, object]] = [("sample_id", sample.sample_id)]
        if sample.source_record_id:
            signals.append(
                ("source_record", (sample.source_id, sample.source_record_id))
            )
        if sample.file_sha256:
            signals.append(("file_hash", sample.file_sha256))
        for signal in signals:
            owner = signal_owners.setdefault(signal, index)
            union(owner, index)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)

    report = DuplicateReport()
    updated = list(ordered)
    for indexes in sorted(
        components.values(), key=lambda values: canonical_key(ordered[min(values)])
    ):
        if len(indexes) == 1:
            index = indexes[0]
            sample = ordered[index]
            exclusion = (
                None
                if sample.exclusion_reason in duplicate_reasons
                else sample.exclusion_reason
            )
            updated[index] = sample.evolve(
                duplicate_state=DuplicateState.UNIQUE,
                duplicate_of=None,
                exclusion_reason=exclusion,
            )
            report.unique += 1
            continue

        canonical_index = min(indexes, key=lambda value: canonical_key(ordered[value]))
        canonical = ordered[canonical_index]
        criteria: set[str] = set()
        ids = [ordered[index].sample_id for index in indexes]
        if len(set(ids)) < len(ids):
            criteria.add("sample_id")
        records = [
            (ordered[index].source_id, ordered[index].source_record_id)
            for index in indexes
            if ordered[index].source_record_id
        ]
        if len(set(records)) < len(records):
            criteria.add("source_record")
        hashes = [
            ordered[index].file_sha256
            for index in indexes
            if ordered[index].file_sha256
        ]
        if len(set(hashes)) < len(hashes):
            criteria.add("file_hash")
        id_counts = Counter(ids)
        record_counts = Counter(records)
        hash_counts = Counter(hashes)

        updated[canonical_index] = canonical.evolve(
            duplicate_state=DuplicateState.CANONICAL,
            duplicate_of=None,
            exclusion_reason=(
                None
                if canonical.exclusion_reason in duplicate_reasons
                else canonical.exclusion_reason
            ),
        )
        for index in indexes:
            if index == canonical_index:
                continue
            sample = ordered[index]
            if id_counts[sample.sample_id] > 1:
                reason = "duplicate_sample_id"
                report.duplicate_sample_id += 1
            elif (
                sample.source_record_id
                and record_counts[(sample.source_id, sample.source_record_id)] > 1
            ):
                reason = "duplicate_source_record"
                report.duplicate_source_record += 1
            elif sample.file_sha256 and hash_counts[sample.file_sha256] > 1:
                reason = "duplicate_file_hash"
                report.duplicate_file_hash += 1
            else:  # connected transitively; retain an explicit cluster reason
                reason = "duplicate_cluster"
            updated[index] = sample.evolve(
                duplicate_state=DuplicateState.DUPLICATE,
                duplicate_of=canonical.sample_id,
                exclusion_reason=sample.exclusion_reason or reason,
            )
        report.families.append(
            {
                "kind": "exact",
                "criteria": sorted(criteria),
                "canonical_sample_id": canonical.sample_id,
                "member_sample_ids": sorted(
                    ordered[index].sample_id
                    for index in indexes
                    if index != canonical_index
                ),
            }
        )

    by_text_hash: dict[str, list[int]] = {}
    for index, sample in enumerate(updated):
        if (
            sample.normalized_text_sha256
            and sample.duplicate_state != DuplicateState.DUPLICATE
        ):
            by_text_hash.setdefault(sample.normalized_text_sha256, []).append(index)

    for _text_hash, indexes in sorted(by_text_hash.items()):
        if len(indexes) < 2:
            continue
        if len({updated[index].file_sha256 for index in indexes}) < 2:
            continue
        canonical_index = min(indexes, key=lambda value: canonical_key(updated[value]))
        for index in indexes:
            if index == canonical_index:
                continue
            updated[index] = updated[index].evolve(
                duplicate_state=DuplicateState.CONFLICTING_DUPLICATE,
                duplicate_of=updated[canonical_index].sample_id,
            )
            report.conflicting_duplicate += 1
        report.families.append(
            {
                "kind": "normalized_text",
                "canonical_sample_id": updated[canonical_index].sample_id,
                "member_sample_ids": sorted(
                    updated[index].sample_id
                    for index in indexes
                    if index != canonical_index
                ),
            }
        )

    return sorted(updated, key=sort_key), report
