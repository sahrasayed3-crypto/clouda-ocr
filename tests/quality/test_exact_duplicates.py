"""Exact-duplicate detection tests."""

from __future__ import annotations

import pytest

exact_dup = pytest.importorskip("clouda_data.quality.exact_dup")

from tests.quality.conftest import make_row  # noqa: E402


class TestExactDuplicates:
    def test_identical_file_hash_groups(self) -> None:
        rows = [
            make_row("smp_a", image_path="imgs/a.png", file_sha256="aa" * 32),
            make_row("smp_b", image_path="imgs/b.png", file_sha256="aa" * 32),
            make_row("smp_c", image_path="imgs/c.png", file_sha256="bb" * 32),
        ]
        samples, report = exact_dup.classify_exact_duplicates(rows)
        assert len(samples) == 3
        assert report.duplicate_count >= 1

    def test_same_raw_text_different_hash_stays_conflicting(self) -> None:
        rows = [
            make_row("smp_a", text="نص متطابق تماما", file_sha256="aa" * 32),
            make_row("smp_b", text="نص متطابق تماما", file_sha256="bb" * 32),
        ]
        samples, report = exact_dup.classify_exact_duplicates(rows)
        assert len(samples) == 2
        assert report.duplicate_count == 0

    def test_unique_rows_untouched(self) -> None:
        rows = [
            make_row(f"smp_{i}", image_path=f"imgs/{i}.png", file_sha256=f"{i:064x}")
            for i in range(4)
        ]
        samples, report = exact_dup.classify_exact_duplicates(rows)
        assert report.duplicate_count == 0
        assert [s.sample_id for s in samples] == [r.sample_id for r in rows]

    def test_duplicate_count_excludes_conflicting_duplicates(self) -> None:
        # Conflicting duplicates are kept by the pipeline and reported via
        # their own counter; they must not inflate duplicate_count.
        report = exact_dup.ExactDupReport(
            duplicate_file_hash=2,
            conflicting_duplicate=5,
            near_duplicate_image=3,
        )
        assert report.duplicate_count == 5
        assert report.conflicting_duplicate == 5

    def test_distinct_families_get_distinct_cluster_ids(self) -> None:
        # Adversarial ids embedding the literal 4-char sequence "\x00" must not
        # let two unrelated duplicate families share one cluster id.
        literal = chr(92) + "x00"
        rows = [
            make_row("a", file_sha256="11" * 32),
            make_row("b" + literal + "c", file_sha256="11" * 32),
            make_row("a" + literal + "b", file_sha256="22" * 32),
            make_row("c", file_sha256="22" * 32),
        ]
        samples, report = exact_dup.classify_exact_duplicates(rows)
        clusters = exact_dup.families_to_clusters(samples, report)
        assert len(clusters) == len(report.families) == 2
        assert len({c.cluster_id for c in clusters}) == 2
