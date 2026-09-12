"""Image near-duplicate fingerprint + confirmation tests."""

from __future__ import annotations

import pytest

from tests.quality.conftest import (
    add_noise,
    make_row,
    recompress,
    render_arabic_page,
)  # noqa: E402

near_index = pytest.importorskip("clouda_data.quality.near_index")


class TestImageNearDuplicates:
    def test_recompressed_image_is_near_duplicate(self) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        near = recompress(base, quality=40)
        assert near.tobytes() != base.tobytes()
        assert near.size == base.size

    def test_small_mutation_stays_candidate(self) -> None:
        base = render_arabic_page(21, "plain", "صفحة")
        mutated = add_noise(base, seed=9, count=4)
        assert mutated.tobytes() != base.tobytes()

    def test_distinct_pages_not_flagged(self) -> None:
        a = render_arabic_page(21, "plain", "صفحة")
        b = render_arabic_page(99, "dense", "مختلف")
        assert a.tobytes() != b.tobytes()

    def test_rows_with_image_paths_buildable(self) -> None:
        rows = [
            make_row("smp_a", image_path="imgs/a.png"),
            make_row("smp_b", image_path="imgs/b.png"),
        ]
        assert [r.sample_id for r in rows] == ["smp_a", "smp_b"]
