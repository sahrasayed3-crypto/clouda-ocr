"""Unit tests for clouda_data.quality.health."""

from __future__ import annotations

from clouda_data.pretraining.schema import (
    DatasetSample,
    DuplicateState,
    SplitName,
    ValidationStatus,
)
from clouda_data.quality.health import (
    RESOLUTION_EDGES,
    compute_health_summary,
)


def _sample(
    sample_id: str,
    *,
    source_id: str = "src_a",
    split: SplitName = SplitName.TRAIN,
    document_type: str | None = "page",
    language: str = "ar",
    script: str = "arabic",
    text: str | None = "hello world sample text",
    width: int | None = 800,
    height: int | None = 600,
    transformations: list[str] | None = None,
    quality_flags: list[str] | None = None,
    validation_status: ValidationStatus = ValidationStatus.OK,
    duplicate_state: DuplicateState = DuplicateState.UNIQUE,
) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        source_id=source_id,
        target_split=split,
        document_type=document_type,
        language=language,
        script=script,
        text=text,
        width=width,
        height=height,
        transformations=list(transformations or []),
        quality_flags=list(quality_flags or []),
        validation_status=validation_status,
        duplicate_state=duplicate_state,
    )


class TestComputeHealthSummary:
    def test_schema_version(self) -> None:
        summary = compute_health_summary([])
        assert summary.schema_version == "clouda.dataset.health.v1"

    def test_empty_input_zero_counts(self) -> None:
        summary = compute_health_summary([])
        assert all(sum(d.values()) == 0 for d in summary.dimensions.values())
        assert summary.cross_tabs["source_x_split"] == {}
        assert summary.cross_tabs["split_x_clean_distorted"] == {}

    def test_exact_counts_synthetic_mix(self) -> None:
        samples = [
            _sample("s1", source_id="src_a", split=SplitName.TRAIN),
            _sample(
                "s2",
                source_id="src_a",
                split=SplitName.TRAIN,
                transformations=["resized"],
            ),
            _sample(
                "s3",
                source_id="src_b",
                split=SplitName.TEST,
                validation_status=ValidationStatus.WARNING,
                duplicate_state=DuplicateState.CANONICAL,
            ),
            _sample(
                "s4",
                source_id="src_b",
                split=SplitName.TEST,
                document_type=None,
                text=None,
                quality_flags=["BLANK_PAGE"],
            ),
        ]
        summary = compute_health_summary(samples)
        dims = summary.dimensions

        assert dims["source"] == {"src_a": 2, "src_b": 2}
        assert dims["document_type"] == {"page": 3, "unknown": 1}
        assert dims["language"] == {"ar": 4}
        assert dims["script"] == {"arabic": 4}
        assert dims["split"] == {"train": 2, "test": 2}
        assert dims["clean_distorted"] == {"clean": 3, "distorted": 1}
        assert dims["quality_flags"] == {"BLANK_PAGE": 1}
        assert dims["validation_status"] == {"ok": 3, "warning": 1}
        assert dims["duplicate_state"] == {"unique": 3, "canonical": 1}

    def test_every_dimension_total_equals_sample_count(self) -> None:
        # quality_flags is per-flag (a sample can carry multiple flags), so
        # every OTHER dimension's total must equal the sample count.
        samples = [
            _sample("s1", quality_flags=["A", "B"]),
            _sample("s2", source_id="src_b", split=SplitName.VALIDATION),
            _sample("s3", source_id="src_b", split=SplitName.TEST),
        ]
        summary = compute_health_summary(samples)
        for name, counts in summary.dimensions.items():
            total = sum(counts.values())
            if name == "quality_flags":
                assert total == 2  # 1 sample with 2 flags (per-flag counting)
            else:
                assert total == 3, f"dimension {name} total {total} != 3"

    def test_resolution_bucket_edges(self) -> None:
        # max(w,h) pinned edges: <500, 500-1000, 1000-2000, 2000-4000, >=4000.
        # Half-open buckets [lower, upper): an edge value rolls UP to the
        # next bucket (1000 -> "1000-2000"); the first bucket is < 500.
        edge_cases = [
            (400, 300, "<500"),
            (499, 100, "<500"),
            (500, 100, "500-1000"),
            (1000, 100, "1000-2000"),  # max side == edge rolls up
            (1001, 100, "1000-2000"),
            (2000, 100, "2000-4000"),
            (2001, 100, "2000-4000"),
            (4000, 100, ">=4000"),
            (4001, 100, ">=4000"),
        ]
        samples = [
            _sample(f"r{i}", width=w, height=h)
            for i, (w, h, _) in enumerate(edge_cases)
        ]
        summary = compute_health_summary(samples)
        expected: dict[str, int] = {}
        for _w, _h, label in edge_cases:
            expected[label] = expected.get(label, 0) + 1
        assert summary.dimensions["resolution_bucket"] == expected
        assert list(summary.bucket_definitions["resolution_bucket"]["edges"]) == list(
            RESOLUTION_EDGES
        )

    def test_gt_length_bucket_edges(self) -> None:
        # Pinned edges 0/50/200/1000/10000 by len(text or '').
        lengths = [0, 1, 50, 51, 200, 201, 1000, 1001]
        samples = [_sample(f"g{i}", text="x" * n) for i, n in enumerate(lengths)]
        summary = compute_health_summary(samples)
        counts = summary.dimensions["gt_length_bucket"]
        assert counts["0"] == 1  # len 0
        assert counts["0-50"] == 2  # len 1, 50
        assert counts["50-200"] == 2  # len 51, 200
        assert counts["200-1000"] == 2  # len 201, 1000
        assert counts[">1000"] == 1  # len 1001

    def test_gt_missing_labelled_missing(self) -> None:
        summary = compute_health_summary([_sample("m1", text=None)])
        assert summary.dimensions["gt_length_bucket"] == {"missing": 1}

    def test_unknown_dimensions_labelled_unknown(self) -> None:
        summary = compute_health_summary(
            [_sample("u1", document_type=None, width=None, height=None)]
        )
        assert summary.dimensions["document_type"] == {"unknown": 1}
        assert summary.dimensions["resolution_bucket"] == {"unknown": 1}

    def test_cross_tab_source_x_split(self) -> None:
        samples = [
            _sample("c1", source_id="src_a", split=SplitName.TRAIN),
            _sample("c2", source_id="src_a", split=SplitName.TRAIN),
            _sample("c3", source_id="src_a", split=SplitName.TEST),
            _sample("c4", source_id="src_b", split=SplitName.TEST),
        ]
        summary = compute_health_summary(samples)
        tab = summary.cross_tabs["source_x_split"]
        assert tab == {"src_a": {"train": 2, "test": 1}, "src_b": {"test": 1}}
        # Closed: cross-tab cell sums == total samples.
        assert sum(sum(row.values()) for row in tab.values()) == len(samples)

    def test_cross_tab_split_x_clean_distorted(self) -> None:
        samples = [
            _sample("d1", split=SplitName.TRAIN),
            _sample("d2", split=SplitName.TRAIN, transformations=["rotated"]),
            _sample("d3", split=SplitName.TEST),
        ]
        summary = compute_health_summary(samples)
        tab = summary.cross_tabs["split_x_clean_distorted"]
        assert tab == {"train": {"clean": 1, "distorted": 1}, "test": {"clean": 1}}
        assert sum(sum(row.values()) for row in tab.values()) == len(samples)

    def test_bucket_definitions_present(self) -> None:
        summary = compute_health_summary([])
        assert "resolution_bucket" in summary.bucket_definitions
        assert "gt_length_bucket" in summary.bucket_definitions

    def test_round_trip(self) -> None:
        summary = compute_health_summary([_sample("t1")])
        restored = type(summary).from_dict(summary.to_dict())
        assert restored == summary
