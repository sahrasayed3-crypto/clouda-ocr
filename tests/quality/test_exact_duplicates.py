"""Exact-duplicate detection tests (Wave2-C contract, module pending)."""

from __future__ import annotations

import pytest

exact_dup = pytest.importorskip("clouda_data.quality.exact_dup")

from conftest import make_row  # noqa: E402


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
