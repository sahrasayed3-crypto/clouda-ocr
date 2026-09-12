"""Counts-only dataset health summary.

``compute_health_summary`` aggregates a list of ``DatasetSample`` values over
eleven count-only dimensions plus two small closed cross-tabs (``source`` x
``split`` and ``split`` x ``clean_distorted``). Every dimension key is taken
from the sample's own fields or one of the pinned bucket edges declared in
``BUCKET_DEFINITIONS``; no labels are invented and no sample content
(meaningful text, paths, image bytes) is echoed. The result is a frozen
``DatasetHealthSummary`` with schema ``clouda.dataset.health.v1``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from clouda_data.pretraining.schema import DatasetSample
from clouda_data.quality.models import DatasetHealthSummary

HEALTH_DIMENSIONS: tuple[str, ...] = (
    "source",
    "document_type",
    "language",
    "script",
    "split",
    "clean_distorted",
    "quality_flags",
    "validation_status",
    "duplicate_state",
    "resolution_bucket",
    "gt_length_bucket",
)

CROSS_TABS: tuple[str, ...] = (
    "source_x_split",
    "split_x_clean_distorted",
)

RESOLUTION_EDGES: tuple[int, ...] = (500, 1000, 2000, 4000)
GT_LENGTH_EDGES: tuple[int, ...] = (0, 50, 200, 1000, 10000)


def _resolution_bucket_label(max_side: int) -> str:
    """Pinned resolution bucket label for a ``max(w, h)`` side length.

    Convention: buckets are half-open ``[lower, upper)``, so an edge value
    belongs to the NEXT bucket up (500 -> "500-1000", 4000 -> ">=4000").
    """

    if max_side < RESOLUTION_EDGES[0]:
        return f"<{RESOLUTION_EDGES[0]}"
    if max_side < RESOLUTION_EDGES[1]:
        return f"{RESOLUTION_EDGES[0]}-{RESOLUTION_EDGES[1]}"
    if max_side < RESOLUTION_EDGES[2]:
        return f"{RESOLUTION_EDGES[1]}-{RESOLUTION_EDGES[2]}"
    if max_side < RESOLUTION_EDGES[3]:
        return f"{RESOLUTION_EDGES[2]}-{RESOLUTION_EDGES[3]}"
    return f">={RESOLUTION_EDGES[3]}"


def _gt_length_bucket_label(length: int) -> str:
    """Pinned ground-truth-length bucket label for ``len(sample.text or '')``."""

    if length <= GT_LENGTH_EDGES[0]:
        return "0"
    if length <= GT_LENGTH_EDGES[1]:
        return f"{GT_LENGTH_EDGES[0]}-{GT_LENGTH_EDGES[1]}"
    if length <= GT_LENGTH_EDGES[2]:
        return f"{GT_LENGTH_EDGES[1]}-{GT_LENGTH_EDGES[2]}"
    if length <= GT_LENGTH_EDGES[3]:
        return f"{GT_LENGTH_EDGES[2]}-{GT_LENGTH_EDGES[3]}"
    return f">{GT_LENGTH_EDGES[3]}"


def _clean_distorted_label(sample: DatasetSample) -> str:
    return "distorted" if sample.transformations else "clean"


def bucket_definitions() -> dict[str, Any]:
    """Pinned bucket-edge definitions (hashed into config identity)."""

    return {
        "resolution_bucket": {
            "basis": "max(width, height)",
            "edges": list(RESOLUTION_EDGES),
            "labels": [
                f"<{RESOLUTION_EDGES[0]}",
                *[
                    f"{RESOLUTION_EDGES[i]}-{RESOLUTION_EDGES[i + 1]}"
                    for i in range(len(RESOLUTION_EDGES) - 1)
                ],
                f">={RESOLUTION_EDGES[-1]}",
            ],
            "missing_label": "unknown",
        },
        "gt_length_bucket": {
            "basis": "len(sample.text or '')",
            "edges": list(GT_LENGTH_EDGES),
            "labels": [
                "0",
                *[
                    f"{GT_LENGTH_EDGES[i]}-{GT_LENGTH_EDGES[i + 1]}"
                    for i in range(len(GT_LENGTH_EDGES) - 1)
                ],
                f">{GT_LENGTH_EDGES[-1]}",
            ],
            "missing_label": "missing",
        },
    }


def _increment(counter: Counter[str], key: str) -> None:
    counter[key] += 1


def compute_health_summary(
    samples: list[DatasetSample],
) -> DatasetHealthSummary:
    """Aggregate count-only health statistics over the 11 dimensions.

    Missing/unknown values map to deterministic ``unknown`` / ``missing``
    labels instead of inventing categories: ``document_type`` may be
    ``None`` (labelled ``unknown``), image dimensions may be ``None``
    (labelled ``unknown`` resolution bucket), and missing ground-truth text
    is labelled ``missing`` in the GT-length bucket.
    """

    dimensions: dict[str, Counter[str]] = {
        name: Counter() for name in HEALTH_DIMENSIONS
    }
    cross_tab_source_split: dict[str, Counter[str]] = {}
    cross_tab_split_clean: dict[str, Counter[str]] = {}

    for sample in samples:
        max_side: int | None = None
        if sample.width is not None and sample.height is not None:
            max_side = max(sample.width, sample.height)
        if max_side is None:
            resolution_label = "unknown"
        else:
            resolution_label = _resolution_bucket_label(max_side)

        text = sample.text or ""
        if sample.text is None:
            gt_label = "missing"
        else:
            gt_label = _gt_length_bucket_label(len(text))

        dimensions["source"][sample.source_id] += 1
        dimensions["document_type"][sample.document_type or "unknown"] += 1
        dimensions["language"][sample.language or "unknown"] += 1
        dimensions["script"][sample.script or "unknown"] += 1
        dimensions["split"][sample.target_split.value] += 1
        dimensions["clean_distorted"][_clean_distorted_label(sample)] += 1
        for flag in sample.quality_flags:
            dimensions["quality_flags"][flag] += 1
        dimensions["validation_status"][sample.validation_status.value] += 1
        dimensions["duplicate_state"][sample.duplicate_state.value] += 1
        dimensions["resolution_bucket"][resolution_label] += 1
        dimensions["gt_length_bucket"][gt_label] += 1

        split = sample.target_split.value
        source = sample.source_id
        clean_label = _clean_distorted_label(sample)

        cross_tab_source_split.setdefault(source, Counter())[split] += 1
        cross_tab_split_clean.setdefault(split, Counter())[clean_label] += 1

    dimensions_payload: dict[str, dict[str, Any]] = {
        name: dict(sorted(counter.items())) for name, counter in dimensions.items()
    }
    cross_tabs_payload: dict[str, dict[str, Any]] = {
        "source_x_split": {
            row_key: dict(sorted(row_counts.items()))
            for row_key, row_counts in sorted(cross_tab_source_split.items())
        },
        "split_x_clean_distorted": {
            row_key: dict(sorted(row_counts.items()))
            for row_key, row_counts in sorted(cross_tab_split_clean.items())
        },
    }
    return DatasetHealthSummary(
        dimensions=dimensions_payload,
        cross_tabs=cross_tabs_payload,
        bucket_definitions=bucket_definitions(),
    )
