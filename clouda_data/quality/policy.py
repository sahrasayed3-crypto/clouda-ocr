"""Deterministic keep/exclude policy for the quality gate.

Never lets a training row override a protected/eval row inside a duplicate
cluster: protected rows are always kept and returned in the quarantine list,
never excluded. Within a duplicate cluster the keep preference is:

1. protected (always kept + quarantined);
2. canonical-valid — lowest pretraining ``sort_key`` among rows whose
   ``duplicate_state`` is canonical and whose validation status is clean;
3. clean-over-distorted — fewer recorded transformations;
4. stable sample id — lowest ``sample_id``.

All iteration is over sorted keys so the decision list is byte-stable across
runs regardless of input order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from clouda_contracts.protection import record_is_protected
from clouda_data.pretraining.schema import (
    DatasetSample,
    DuplicateState,
    ValidationStatus,
    sort_key,
)
from clouda_data.quality.models import ExclusionDecision

EXCLUSION_REPORT_SCHEMA_VERSION = "clouda.pretraining.exclusion.v1"

DUPLICATE_REASON = "duplicate"
NEAR_DUPLICATE_IMAGE_REASON = "near_duplicate_image"

REASON_SOURCES = frozenset({"validation", "dedupe", "split", "quality_gate"})

_QUARANTINE_REASON = "holdout"


def _as_status(value: Any) -> ValidationStatus:
    if isinstance(value, ValidationStatus):
        return value
    return ValidationStatus(str(value))


def _as_duplicate_state(value: Any) -> DuplicateState:
    if isinstance(value, DuplicateState):
        return value
    return DuplicateState(str(value))


def _sample_index(samples: Sequence[DatasetSample]) -> dict[str, DatasetSample]:
    return {sample.sample_id: sample for sample in samples}


def _issue_codes_by_sample(issues: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    """Map each sample id to its sorted issue codes."""

    codes: dict[str, list[str]] = {}
    for issue in issues:
        sample_ids = getattr(issue, "sample_ids", ())
        code = getattr(issue, "code", "quality_gate")
        for sample_id in sample_ids:
            codes.setdefault(sample_id, [])
            if code not in codes[sample_id]:
                codes[sample_id].append(code)
    return {sample_id: tuple(sorted(values)) for sample_id, values in codes.items()}


def _is_protected(sample: DatasetSample) -> bool:
    """Fail-closed protection check over the sample's canonical row dict."""

    return record_is_protected(sample.to_dict())


def _is_canonical_valid(sample: DatasetSample) -> bool:
    return _as_duplicate_state(
        sample.duplicate_state
    ) == DuplicateState.CANONICAL and _as_status(sample.validation_status) in (
        ValidationStatus.OK,
        ValidationStatus.WARNING,
    )


def _keep_key(sample: DatasetSample) -> tuple[int, str]:
    return (len(sample.transformations), sample.sample_id)


def _cluster_keep_id(members: Sequence[DatasetSample]) -> tuple[str, bool]:
    """Pick the kept member of one duplicate cluster deterministically.

    Preference: protected first (lowest sample_id) > canonical-valid (lowest
    pretraining sort_key) > clean-over-distorted (fewer transformations) >
    lowest sample_id. Returns ``(keep_sample_id, any_protected)``.
    """

    ordered = sorted(members, key=lambda sample: sample.sample_id)
    any_protected = any(_is_protected(sample) for sample in ordered)

    protected = [sample for sample in ordered if _is_protected(sample)]
    if protected:
        return protected[0].sample_id, any_protected

    canonical_valid = [sample for sample in ordered if _is_canonical_valid(sample)]
    if canonical_valid:
        return min(canonical_valid, key=sort_key).sample_id, any_protected

    return min(ordered, key=_keep_key).sample_id, any_protected


def decide_exclusions(
    samples: Sequence[DatasetSample],
    clusters: Sequence[Any],
    issues: Iterable[Any],
    config: Any,
) -> list[ExclusionDecision]:
    """Compute the deterministic exclusion list for a quality-gate run.

    Protected rows are NEVER excluded: they are always kept and returned as
    quarantine entries (``reason_source='split'``) so callers can route them
    to the quarantine manifest. Duplicate-cluster losers get ``duplicate`` /
    ``near_duplicate_image`` / the validation issue code, with
    ``reason_source`` in ``{'validation','dedupe','split','quality_gate'}``.
    """

    keep = getattr(config, "keep_exclude", None)
    exclude_duplicate = bool(getattr(keep, "exclude_duplicate", True))
    exclude_error = bool(getattr(keep, "exclude_error", True))
    exclude_conflicting = bool(getattr(keep, "exclude_conflicting_duplicate", False))

    by_id = _sample_index(samples)
    issue_codes = _issue_codes_by_sample(issues)

    excluded: dict[str, ExclusionDecision] = {}
    quarantined: dict[str, ExclusionDecision] = {}

    # Protected rows: always kept, always quarantined (fail-closed).
    for sample_id in sorted(by_id):
        if _is_protected(by_id[sample_id]):
            quarantined[sample_id] = ExclusionDecision(
                sample_id=sample_id,
                reason_code=_QUARANTINE_REASON,
                reason_source="split",
                evidence={"protected": True},
            )

    # Validation issues -> exclude error rows (never protected ones).
    if exclude_error:
        for sample_id in sorted(issue_codes):
            if sample_id in excluded or sample_id in quarantined:
                continue
            sample = by_id.get(sample_id)
            if sample is None:
                continue
            if _as_status(sample.validation_status) == ValidationStatus.ERROR:
                excluded[sample_id] = ExclusionDecision(
                    sample_id=sample_id,
                    reason_code=issue_codes[sample_id][0],
                    reason_source="validation",
                    evidence={"issue_codes": list(issue_codes[sample_id])},
                )

    # Duplicate clusters: keep exactly one member per cluster.
    if exclude_duplicate:
        for cluster in sorted(
            clusters, key=lambda item: str(getattr(item, "cluster_id", ""))
        ):
            member_ids = tuple(
                mid
                for mid in sorted(getattr(cluster, "member_ids", ()))
                if mid in by_id
            )
            if len(member_ids) < 2:
                continue
            level = str(getattr(cluster, "level", "CONFIRMED_NEAR_DUPLICATE"))
            near = "NEAR" in level.upper() or "CANDIDATE" in level.upper()
            reason_code = NEAR_DUPLICATE_IMAGE_REASON if near else DUPLICATE_REASON
            members = [by_id[mid] for mid in member_ids]
            keep_id, _ = _cluster_keep_id(members)
            for member in sorted(members, key=lambda item: item.sample_id):
                mid = member.sample_id
                if mid == keep_id or mid in quarantined or mid in excluded:
                    continue
                if (
                    _as_duplicate_state(member.duplicate_state)
                    == DuplicateState.CONFLICTING_DUPLICATE
                    and not exclude_conflicting
                ):
                    continue
                excluded[mid] = ExclusionDecision(
                    sample_id=mid,
                    reason_code=reason_code,
                    reason_source="dedupe",
                    evidence={
                        "cluster_id": str(getattr(cluster, "cluster_id", "")),
                        "kept_sample_id": keep_id,
                    },
                )

    decisions = sorted(
        list(excluded.values()) + list(quarantined.values()),
        key=lambda decision: (decision.sample_id, decision.reason_code),
    )
    for decision in decisions:
        if decision.reason_source not in REASON_SOURCES:
            raise ValueError(
                f"ExclusionDecision reason_source {decision.reason_source!r} "
                f"outside closed vocabulary {sorted(REASON_SOURCES)}."
            )
    return decisions


def quarantine_sample_ids(
    samples: Sequence[DatasetSample],
    exclusions: Sequence[ExclusionDecision],
) -> tuple[str, ...]:
    """Sorted ids of protected rows (kept + quarantined, never excluded)."""

    excluded = {
        decision.sample_id
        for decision in exclusions
        if decision.reason_source != "split"
    }
    return tuple(
        sorted(
            sample.sample_id
            for sample in samples
            if sample.sample_id not in excluded and _is_protected(sample)
        )
    )


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    """Call the dynamically-attached ``to_dict`` on a models.py artifact."""

    to_dict = getattr(artifact, "to_dict")
    return to_dict()


def exclusion_report(
    exclusions: Sequence[ExclusionDecision],
    *,
    total_samples: int = 0,
    quarantine_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the ``clouda.pretraining.exclusion.v1`` report document."""

    counts: Counter[str] = Counter(decision.reason_code for decision in exclusions)
    source_counts: Counter[str] = Counter(
        decision.reason_source for decision in exclusions
    )
    return {
        "schema_version": EXCLUSION_REPORT_SCHEMA_VERSION,
        "total_samples": total_samples,
        "excluded_count": len(exclusions),
        "quarantine_count": len(set(quarantine_ids)),
        "counts_by_reason": {reason: counts[reason] for reason in sorted(counts)},
        "counts_by_source": {
            source: source_counts[source] for source in sorted(source_counts)
        },
        "exclusions": [
            _artifact_to_dict(decision)
            for decision in sorted(exclusions, key=lambda decision: decision.sample_id)
        ],
        "quarantine_ids": sorted(set(quarantine_ids)),
    }


__all__ = [
    "DUPLICATE_REASON",
    "EXCLUSION_REPORT_SCHEMA_VERSION",
    "NEAR_DUPLICATE_IMAGE_REASON",
    "REASON_SOURCES",
    "decide_exclusions",
    "exclusion_report",
    "quarantine_sample_ids",
]
