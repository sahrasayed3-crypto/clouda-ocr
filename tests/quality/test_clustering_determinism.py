"""Deterministic clustering tests for the canonical quality pipeline."""

from __future__ import annotations

import random

from clouda_data.quality import exact_dup
from tests.quality.conftest import make_row


def _cluster_samples(rows):  # type: ignore[no-untyped-def]
    classified, report = exact_dup.classify_exact_duplicates(rows)
    return exact_dup.families_to_clusters(classified, report)


class TestClusteringDeterminism:
    def test_shuffled_input_identical_classification(self) -> None:  # type: ignore[no-untyped-def]
        rows = [
            make_row(f"smp_{i:03d}", source_id=f"src_{i % 3}", text=f"نص {i}")
            for i in range(20)
        ]
        forward = _cluster_samples(rows)
        shuffled = list(rows)
        random.Random(42).shuffle(shuffled)
        backward = _cluster_samples(shuffled)
        assert forward == backward

    def test_cluster_ids_stable(self) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص"), make_row("smp_b", text="نص")]
        first = _cluster_samples(rows)
        second = _cluster_samples(list(reversed(rows)))
        assert [c.cluster_id for c in first] == [c.cluster_id for c in second]
