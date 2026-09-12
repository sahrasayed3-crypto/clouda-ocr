"""Deterministic clustering tests (Wave2-F confirm contract, pending)."""

from __future__ import annotations

import random

import pytest

confirm = pytest.importorskip("clouda_data.quality.confirm")

from tests.quality.conftest import make_row  # noqa: E402


class TestClusteringDeterminism:
    def test_shuffled_input_identical_classification(self) -> None:  # type: ignore[no-untyped-def]
        rows = [
            make_row(f"smp_{i:03d}", source_id=f"src_{i % 3}", text=f"نص {i}")
            for i in range(20)
        ]
        forward = confirm.cluster_samples(rows)
        shuffled = list(rows)
        random.Random(42).shuffle(shuffled)
        backward = confirm.cluster_samples(shuffled)
        assert forward == backward

    def test_cluster_ids_stable(self) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص"), make_row("smp_b", text="نص")]
        first = confirm.cluster_samples(rows)
        second = confirm.cluster_samples(list(reversed(rows)))
        assert [c.cluster_id for c in first] == [c.cluster_id for c in second]
