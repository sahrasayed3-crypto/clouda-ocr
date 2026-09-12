"""Unit tests for clouda_data.quality.benchmarks (Wave2-P)."""

from __future__ import annotations

import pytest

from clouda_data.quality.benchmarks import (
    assert_linear_scaling,
    build_synthetic_manifest,
    measure_tier,
)
from clouda_data.pretraining.manifest import read_manifest


class TestBuildSyntheticManifest:
    def test_record_count(self, tmp_path) -> None:
        manifest = build_synthetic_manifest(tmp_path, 30, seed=7)
        _header, rows = read_manifest(manifest)
        assert len(rows) == 30

    def test_dup_counts_match_rates(self, tmp_path) -> None:
        n, dup_rate, near_rate = 40, 0.10, 0.15
        manifest = build_synthetic_manifest(
            tmp_path, n, dup_rate=dup_rate, near_rate=near_rate, seed=3
        )
        _header, rows = read_manifest(manifest)
        assert len(rows) == n
        dup_paths = [r["source_path"] for r in rows if "dup_" in r["source_path"]]
        near_paths = [r["source_path"] for r in rows if "near_" in r["source_path"]]
        assert len(dup_paths) == int(round(n * dup_rate))
        assert len(near_paths) == int(round(n * near_rate))

    def test_exact_dups_are_byte_identical(self, tmp_path) -> None:
        manifest = build_synthetic_manifest(tmp_path, 20, dup_rate=0.2, seed=11)
        _header, rows = read_manifest(manifest)
        import hashlib

        dup_paths = [r["source_path"] for r in rows if "dup_" in r["source_path"]]
        assert dup_paths, "expected at least one dup row"
        for dup_path in dup_paths:
            ref = dup_path.replace("dup_", "unique_")
            dup_bytes = (tmp_path / dup_path).read_bytes()
            ref_bytes = (tmp_path / ref).read_bytes()
            assert (
                hashlib.sha256(dup_bytes).digest() == hashlib.sha256(ref_bytes).digest()
            )

    def test_deterministic_across_runs(self, tmp_path) -> None:
        m1 = build_synthetic_manifest(tmp_path / "a", 15, dup_rate=0.1, seed=5)
        m2 = build_synthetic_manifest(tmp_path / "b", 15, dup_rate=0.1, seed=5)
        import hashlib

        def _digest(path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        assert _digest(m1) == _digest(m2)

    def test_unique_rows_have_distinct_hashes(self, tmp_path) -> None:
        manifest = build_synthetic_manifest(tmp_path, 12, seed=21)
        _header, rows = read_manifest(manifest)
        import hashlib

        digests = {
            hashlib.sha256((tmp_path / r["source_path"]).read_bytes()).hexdigest()
            for r in rows
            if "unique_" in r["source_path"]
        }
        assert len(digests) == 12


class TestMeasureTier:
    def test_100_tier_produces_metrics(self, tmp_path) -> None:
        manifest = build_synthetic_manifest(tmp_path, 100, dup_rate=0.05, seed=42)
        metrics = measure_tier(manifest, tmp_path)
        assert metrics["n_records"] == 100
        assert metrics["scan_time_s"] > 0
        assert metrics["fingerprint_rate"] > 0
        assert metrics["candidate_count"] >= 0
        assert metrics["decode_count"] > 0
        assert metrics["peak_memory_bytes"] > 0
        # One decode per record (single-decode discipline).
        assert metrics["decode_count"] <= 2 * metrics["n_records"]

    def test_candidate_count_matches_dup_rates(self, tmp_path) -> None:
        n, dup_rate = 50, 0.20
        manifest = build_synthetic_manifest(tmp_path, n, dup_rate=dup_rate, seed=9)
        metrics = measure_tier(manifest, tmp_path)
        expected_exact_dup_pairs = int(round(n * dup_rate))
        assert metrics["candidate_count"] == expected_exact_dup_pairs

    def test_near_dups_produce_zero_exact_candidates(self, tmp_path) -> None:
        manifest = build_synthetic_manifest(tmp_path, 20, near_rate=0.25, seed=13)
        metrics = measure_tier(manifest, tmp_path)
        assert metrics["candidate_count"] == 0


class TestAssertLinearScaling:
    def test_passes_on_linear_synthetic_data(self) -> None:
        metrics = {
            "100": {
                "n_records": 100,
                "scan_time_s": 0.1,
                "candidate_count": 5,
                "decode_count": 100,
                "peak_memory_bytes": 1_000_000,
            },
            "1000": {
                "n_records": 1000,
                "scan_time_s": 1.0,
                "candidate_count": 50,
                "decode_count": 1000,
                "peak_memory_bytes": 10_000_000,
            },
        }
        assert_linear_scaling(metrics)

    def test_fails_on_superlinear_candidates(self) -> None:
        metrics = {
            "100": {"candidate_count": 5, "scan_time_s": 0.1, "peak_memory_bytes": 1},
            "1000": {
                "candidate_count": 500,
                "scan_time_s": 0.2,
                "peak_memory_bytes": 2,
            },
        }
        with pytest.raises(AssertionError, match="candidate"):
            assert_linear_scaling(metrics)

    def test_fails_on_superlinear_scan_time(self) -> None:
        metrics = {
            "100": {"candidate_count": 5, "scan_time_s": 0.1, "peak_memory_bytes": 1},
            "1000": {
                "candidate_count": 10,
                "scan_time_s": 50.0,
                "peak_memory_bytes": 2,
            },
        }
        with pytest.raises(AssertionError, match="scan"):
            assert_linear_scaling(metrics)

    def test_fails_on_excessive_decodes(self) -> None:
        metrics = {
            "100": {"candidate_count": 5, "scan_time_s": 0.1, "peak_memory_bytes": 1},
            "1000": {
                "candidate_count": 50,
                "scan_time_s": 1.0,
                "n_records": 1000,
                "decode_count": 5000,
                "peak_memory_bytes": 2,
            },
        }
        with pytest.raises(AssertionError, match="decode"):
            assert_linear_scaling(metrics)

    def test_single_tier_is_noop(self) -> None:
        assert_linear_scaling(
            {"100": {"candidate_count": 0, "scan_time_s": 0.01, "peak_memory_bytes": 1}}
        )


@pytest.mark.slow
class TestThousandTier:
    def test_1000_tier_scales(self, tmp_path) -> None:
        manifest_100 = build_synthetic_manifest(
            tmp_path / "t100", 100, dup_rate=0.05, seed=1
        )
        manifest_1000 = build_synthetic_manifest(
            tmp_path / "t1000", 1000, dup_rate=0.05, seed=1
        )
        m100 = measure_tier(manifest_100, tmp_path / "t100")
        m1000 = measure_tier(manifest_1000, tmp_path / "t1000")
        assert_linear_scaling({"100": m100, "1000": m1000})
