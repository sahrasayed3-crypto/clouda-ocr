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

from dataclasses import dataclass, field
from typing import Any

from .schema import DatasetSample, DuplicateState, sort_key


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
    """Classify duplicates over a deterministically ordered sample list."""

    report = DuplicateReport()
    ordered = sorted(samples, key=sort_key)

    by_id: dict[str, str] = {}
    by_record: dict[tuple[str, str], str] = {}
    canonical_for_file_hash: dict[str, str] = {}
    file_hash_families: dict[str, list[str]] = {}

    for sample in ordered:
        by_id.setdefault(sample.sample_id, sample.sample_id)
        if sample.source_record_id:
            by_record.setdefault(
                (sample.source_id, sample.source_record_id), sample.sample_id
            )
        if sample.file_sha256:
            family = file_hash_families.setdefault(sample.file_sha256, [])
            family.append(sample.sample_id)

    for file_hash, members in file_hash_families.items():
        if len(members) > 1:
            canonical_for_file_hash[file_hash] = members[0]
            report.families.append(
                {
                    "kind": "file_hash",
                    "canonical_sample_id": members[0],
                    "member_sample_ids": members[1:],
                }
            )

    updated: list[DatasetSample] = []
    seen_ids: set[str] = set()
    seen_records: set[tuple[str, str]] = set()

    for sample in ordered:
        state = DuplicateState.UNIQUE
        canonical_id: str | None = None
        reason: str | None = None

        if sample.sample_id in seen_ids:
            state = DuplicateState.DUPLICATE
            canonical_id = by_id[sample.sample_id]
            reason = "duplicate_sample_id"
            report.duplicate_sample_id += 1
        elif (
            sample.source_record_id
            and (sample.source_id, sample.source_record_id) in seen_records
        ):
            state = DuplicateState.DUPLICATE
            canonical_id = by_record[(sample.source_id, sample.source_record_id)]
            reason = "duplicate_source_record"
            report.duplicate_source_record += 1
        elif (
            sample.file_sha256
            and sample.file_sha256 in canonical_for_file_hash
            and canonical_for_file_hash[sample.file_sha256] != sample.sample_id
        ):
            state = DuplicateState.DUPLICATE
            canonical_id = canonical_for_file_hash[sample.file_sha256]
            reason = "duplicate_file_hash"
            report.duplicate_file_hash += 1
        else:
            report.unique += 1

        seen_ids.add(sample.sample_id)
        if sample.source_record_id:
            seen_records.add((sample.source_id, sample.source_record_id))

        exclusion_reason = sample.exclusion_reason
        if state == DuplicateState.DUPLICATE and exclusion_reason is None:
            exclusion_reason = reason
        updated.append(
            sample.evolve(
                duplicate_state=state,
                duplicate_of=canonical_id,
                exclusion_reason=exclusion_reason,
            )
        )

    # Conflicting duplicates: identical normalized text, different content.
    by_text_hash: dict[str, list[int]] = {}
    for index, sample in enumerate(updated):
        if sample.normalized_text_sha256:
            by_text_hash.setdefault(sample.normalized_text_sha256, []).append(index)

    for text_hash, indexes in sorted(by_text_hash.items()):
        if len(indexes) < 2:
            continue
        text_members = [updated[i] for i in indexes]
        if len({m.file_sha256 for m in text_members}) < 2:
            continue
        conflicting = [
            m
            for m in text_members
            if m.duplicate_state != DuplicateState.DUPLICATE
            and m.sample_id != text_members[0].sample_id
        ]
        if not conflicting:
            continue
        conflicting_ids = {m.sample_id for m in conflicting}
        for i in indexes:
            if updated[i].sample_id in conflicting_ids:
                updated[i] = updated[i].evolve(
                    duplicate_state=DuplicateState.CONFLICTING_DUPLICATE
                )
        report.conflicting_duplicate += len(conflicting)
        report.families.append(
            {
                "kind": "normalized_text",
                "canonical_sample_id": text_members[0].sample_id,
                "member_sample_ids": [m.sample_id for m in text_members[1:]],
            }
        )

    return updated, report
