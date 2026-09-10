"""Cross-split leakage detection (L0-L6).

Consumes the canonical fail-closed protection policy
(``clouda_contracts.protection`` + ``clouda_lab.holdout_guard``) — protection
logic is never re-implemented here. Protected row CONTENT is never emitted:
findings carry sample ids and codes only.

Severity: any CRITICAL finding fails the gate (``LeakageReport.passed``
False). eval-vs-eval overlaps and standalone normalized-GT matches are WARN.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from clouda_contracts.protection import (
    PROTECTED_SPLIT_NAMES,
    is_training_split_eligible,
    normalize_marker,
    protection_metadata_is_malformed,
    record_is_protected,
    string_marks_protected,
)
from clouda_data.pretraining.hashing import SHA256_RE
from clouda_data.pretraining.schema import DatasetSample, SplitName
from clouda_data.pretraining.splitting import resolve_group_key
from clouda_data.quality.models import (
    IssueCode,
    IssueSeverity,
    LeakageFinding,
    QualityIssue,
)

LEAKAGE_REPORT_SCHEMA_VERSION = "clouda.quality.leakage_report.v1"

PARTITION_TRAIN = "TRAIN"
PARTITION_EVAL = "EVAL"
PARTITION_PROTECTED = "PROTECTED"
PARTITION_UNPARTITIONED = "UNPARTITIONED"

EVAL_SPLITS = frozenset({"validation", "test"})

_KIND_TO_CODE = {
    "malformed_protection": IssueCode.LEAK_MALFORMED_PROTECTION,
    "exact_hash_cross_split": IssueCode.LEAK_EXACT_HASH,
    "page_identity_cross_split": IssueCode.LEAK_PAGE_IDENTITY,
    "near_image_cross_split": IssueCode.LEAK_NEAR_IMAGE,
    "derived_page_cross_boundary": IssueCode.LEAK_DERIVED_PAGE,
    "gt_text_cross_split": IssueCode.LEAK_GT_TEXT,
    "group_cross_split": IssueCode.LEAK_GROUP_STRADDLE,
}


@dataclass
class LeakageReport:
    """Aggregated leakage scan result."""

    findings: list[LeakageFinding] = field(default_factory=list)
    passed: bool = True
    scanned_samples: int = 0
    quarantined_protected: int = 0
    partition_counts: dict[str, int] = field(default_factory=dict)
    schema_version: str = LEAKAGE_REPORT_SCHEMA_VERSION


def _finding_id(kind: str, canonical_key: str, sample_ids: Sequence[str]) -> str:
    payload = "|".join([kind, canonical_key, *sorted(sample_ids)])
    return "LKG_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _split_value(sample: DatasetSample) -> str:
    """Best-effort split string from target/source fields (evidence only)."""

    if sample.target_split != SplitName.UNASSIGNED:
        return str(sample.target_split.value)
    if sample.source_split:
        return sample.source_split
    return ""


def effective_partition(sample: DatasetSample) -> str:
    """Fail-closed partition for one sample.

    Precedence: (1) ``record_is_protected`` quarantines regardless of split;
    (2) assigned ``target_split`` via canonical eligibility helpers;
    (3) free-form ``source_split`` through ``string_marks_protected``;
    (4) otherwise UNPARTITIONED.
    """

    if record_is_protected(sample.to_dict()):
        return PARTITION_PROTECTED
    if protection_metadata_is_malformed(sample.to_dict()):
        return PARTITION_PROTECTED

    if sample.target_split != SplitName.UNASSIGNED:
        value = normalize_marker(sample.target_split.value)
        if value in PROTECTED_SPLIT_NAMES:
            return PARTITION_PROTECTED
        if is_training_split_eligible(value):
            return PARTITION_TRAIN
        if value in EVAL_SPLITS:
            return PARTITION_EVAL
        return PARTITION_UNPARTITIONED

    if sample.source_split:
        value = normalize_marker(sample.source_split)
        if string_marks_protected(value):
            return PARTITION_PROTECTED
        if is_training_split_eligible(value):
            return PARTITION_TRAIN
        if value in EVAL_SPLITS:
            return PARTITION_EVAL

    return PARTITION_UNPARTITIONED


def _page_identity(sample: DatasetSample) -> str | None:
    if sample.document_id and sample.page_id:
        return f"{sample.source_id}|{sample.document_id}|{sample.page_id}"
    return None


def finding_to_issue(finding: LeakageFinding) -> QualityIssue:
    """Map a leakage finding onto the unified issue model."""

    code = _KIND_TO_CODE[finding.kind]
    severity = (
        IssueSeverity.CRITICAL
        if finding.severity == "critical"
        else IssueSeverity.WARNING
    )
    return QualityIssue(
        code=code,
        severity=severity,
        sample_ids=tuple(finding.sample_ids),
        canonical_key=finding.canonical_key,
        message=finding.detail,
        evidence={"partitions": sorted(finding.partitions)},
    )


def detect_leakage(
    samples: Sequence[DatasetSample],
    confirmed_near_pairs: Sequence[tuple[str, str]] = (),
) -> LeakageReport:
    """Run the L0-L6 leakage checks over a classified sample set."""

    report = LeakageReport(scanned_samples=len(samples))
    partitions: dict[str, str] = {}
    by_id: dict[str, DatasetSample] = {}
    for sample in samples:
        partitions[sample.sample_id] = effective_partition(sample)
        by_id[sample.sample_id] = sample
    for partition in partitions.values():
        report.partition_counts[partition] = (
            report.partition_counts.get(partition, 0) + 1
        )
    report.quarantined_protected = report.partition_counts.get(PARTITION_PROTECTED, 0)

    findings: list[LeakageFinding] = []

    def _add(
        kind: str,
        severity: str,
        canonical_key: str,
        sample_ids: Sequence[str],
        detail: str,
        corroborating: Sequence[str] = (),
        raw_splits: Sequence[str] = (),
    ) -> None:
        severity_value = (
            IssueSeverity.CRITICAL if severity == "critical" else IssueSeverity.WARNING
        )
        raw_map = (
            {sid: _split_value(by_id[sid]) for sid in sorted(set(sample_ids))}
            if raw_splits
            else {}
        )
        findings.append(
            LeakageFinding(
                finding_id=_finding_id(kind, canonical_key, sample_ids),
                kind=kind,
                severity=severity_value,
                partitions=tuple(
                    sorted(
                        {
                            partitions.get(sid, PARTITION_UNPARTITIONED)
                            for sid in sample_ids
                        }
                    )
                ),
                sample_ids=tuple(sorted(sample_ids)),
                canonical_key=canonical_key,
                corroborating_signals=tuple(corroborating),
                raw_split_values=raw_map,
                detail=detail,
            )
        )

    def _cross_of(ids: Iterable[str]) -> tuple[set[str], set[str]]:
        """Partitions spanned by an id group: (train-side, other-side)."""

        parts = {partitions.get(sid, PARTITION_UNPARTITIONED) for sid in ids}
        train_side = parts & {PARTITION_TRAIN}
        other_side = parts & {PARTITION_EVAL, PARTITION_PROTECTED}
        return train_side, other_side

    # L0: malformed protection metadata (fail-closed).
    for sample in samples:
        if protection_metadata_is_malformed(sample.to_dict()):
            _add(
                "malformed_protection",
                "critical",
                sample.sample_id,
                [sample.sample_id],
                "Malformed protection metadata fails closed (row treated as protected).",
            )

    # L1: exact artifact hash across partitions.
    hash_groups: dict[str, list[str]] = {}
    for sample in samples:
        if sample.file_sha256:
            if not SHA256_RE.fullmatch(sample.file_sha256):
                _add(
                    "malformed_protection",
                    "critical",
                    f"file_sha256:{sample.sample_id}",
                    [sample.sample_id],
                    "file_sha256 is not a valid SHA-256 digest (fail-closed).",
                )
                continue
            hash_groups.setdefault(sample.file_sha256, []).append(sample.sample_id)
    for file_hash, ids in hash_groups.items():
        train_side, other_side = _cross_of(ids)
        if train_side and other_side:
            _add(
                "exact_hash_cross_split",
                "critical",
                file_hash,
                ids,
                "Identical artifact bytes appear in both training and evaluation/protected partitions.",
                corroborating=("file_sha256",),
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )
        elif len({partitions.get(sid) for sid in ids} - {PARTITION_UNPARTITIONED}) > 1:
            _add(
                "exact_hash_cross_split",
                "warning",
                file_hash,
                ids,
                "Identical artifact bytes shared across two evaluation partitions.",
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )

    # L2: canonical page identity across partitions.
    identity_groups: dict[str, list[str]] = {}
    for sample in samples:
        identity = _page_identity(sample)
        if identity:
            identity_groups.setdefault(identity, []).append(sample.sample_id)
    for identity, ids in identity_groups.items():
        train_side, other_side = _cross_of(ids)
        if train_side and other_side:
            _add(
                "page_identity_cross_split",
                "critical",
                identity,
                ids,
                "Same canonical page identity appears in both training and evaluation/protected partitions.",
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )
        elif len({partitions.get(sid) for sid in ids} - {PARTITION_UNPARTITIONED}) > 1:
            _add(
                "page_identity_cross_split",
                "warning",
                identity,
                ids,
                "Same page identity shared across two evaluation partitions.",
            )

    # L3: near-image duplicates across train vs eval (confirmed = critical).
    for id_a, id_b in confirmed_near_pairs:
        if id_a not in partitions or id_b not in partitions:
            continue
        pair = (id_a, id_b)
        train_side, other_side = _cross_of(pair)
        if train_side and other_side:
            _add(
                "near_image_cross_split",
                "critical",
                f"near:{id_a}|{id_b}",
                pair,
                "Confirmed near-image duplicate crosses the training/evaluation boundary.",
                corroborating=("near_image_confirmed",),
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in pair)),
            )

    # L4: derived/distorted versions of the same source page across boundary.
    # (covered with L2/L5 corroboration below for samples with transformations)
    text_groups: dict[str, list[str]] = {}
    for sample in samples:
        if sample.normalized_text_sha256:
            text_groups.setdefault(sample.normalized_text_sha256, []).append(
                sample.sample_id
            )
    for text_hash, ids in text_groups.items():
        train_side, other_side = _cross_of(ids)
        if not (train_side and other_side):
            continue
        derived = any(by_id[sid].transformations for sid in ids)
        if derived:
            _add(
                "derived_page_cross_boundary",
                "critical",
                text_hash,
                ids,
                "Distorted/derived versions of the same source page cross the training/evaluation boundary.",
                corroborating=("normalized_text_sha256", "transformations"),
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )
        else:
            _add(
                "gt_text_cross_split",
                "warning",
                text_hash,
                ids,
                "Identical normalized GT text across partitions (supporting signal only).",
                corroborating=("normalized_text_sha256",),
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )

    # L6: document/group straddling train vs eval.
    group_members: dict[str, list[str]] = {}
    for sample in samples:
        key = str(resolve_group_key(sample))
        group_members.setdefault(key, []).append(sample.sample_id)
    for group_key, ids in group_members.items():
        train_side, other_side = _cross_of(ids)
        if train_side and other_side:
            _add(
                "group_cross_split",
                "critical",
                group_key,
                ids,
                "A document/group spans both training and evaluation/protected partitions.",
                raw_splits=tuple(sorted(_split_value(by_id[sid]) for sid in ids)),
            )

    findings.sort(key=lambda f: (f.severity, f.kind, f.finding_id))
    report.findings = findings
    report.passed = not any(f.severity is IssueSeverity.CRITICAL for f in findings)
    return report
