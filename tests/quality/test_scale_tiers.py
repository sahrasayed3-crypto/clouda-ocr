"""Scale-tier tests: 100 default, 1000 behind the ``slow`` marker."""

from __future__ import annotations

import pytest

from conftest import make_manifest, make_row, render_arabic_page


def _synthetic_rows(n: int) -> list:
    return [
        make_row(
            f"smp_{i:06d}",
            source_id=f"src_{i % 5}",
            image_path=f"imgs/page_{i:06d}.png",
            text=f"نص تجريبي رقم {i}",
        )
        for i in range(n)
    ]


class TestScaleTiers:
    def test_tier_100_default(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, _synthetic_rows(100))
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 101  # header + 100 rows

    @pytest.mark.slow
    def test_tier_1000_slow(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, _synthetic_rows(1000))
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1001  # header + 1000 rows

    def test_render_cost_bounded(self) -> None:
        a = render_arabic_page(1, "plain", "نص")
        b = render_arabic_page(2, "plain", "نص")
        assert a.size == b.size == (128, 96)
