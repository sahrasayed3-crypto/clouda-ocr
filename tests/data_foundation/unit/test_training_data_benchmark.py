"""Benchmark tests (small sizes for CI; large sizes manual/optional)."""

from __future__ import annotations

import pytest

from clouda_data.training_data.benchmark import run_benchmark


@pytest.mark.parametrize("count", [100])
def test_benchmark_100_deterministic_and_bounded(tmp_path, count):
    report = run_benchmark(count, tmp_path)
    assert report["samples"] == count
    assert report["deterministic_replay"] is True
    # Memory must stay bounded far below dataset-size linear growth
    assert report["peak_traced_memory_mb"] < 50
    assert report["batches"] >= 1


@pytest.mark.slow
@pytest.mark.parametrize("count", [1000])
def test_benchmark_1000(tmp_path, count):
    report = run_benchmark(count, tmp_path)
    assert report["samples"] == count
    assert report["deterministic_replay"] is True
    assert report["peak_traced_memory_mb"] < 100
