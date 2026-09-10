"""Determinism guarantees for the tests/quality fixture helpers (Wave2-Q)."""

from __future__ import annotations

import json

from conftest import (
    add_noise,
    blank_like,
    brightness,
    make_manifest,
    recompress,
    render_arabic_page,
    resize_img,
)


class TestRenderDeterminism:
    def test_same_inputs_identical_bytes(self) -> None:
        a = render_arabic_page(7, "plain", "سلام")
        b = render_arabic_page(7, "plain", "سلام")
        assert a.tobytes() == b.tobytes()

    def test_different_seed_differs(self) -> None:
        a = render_arabic_page(7, "plain", "سلام")
        b = render_arabic_page(8, "plain", "سلام")
        assert a.tobytes() != b.tobytes()

    def test_different_layout_differs(self) -> None:
        a = render_arabic_page(7, "plain", "سلام")
        b = render_arabic_page(7, "dense", "سلام")
        assert a.tobytes() != b.tobytes()

    def test_non_arabic_text_still_deterministic(self) -> None:
        a = render_arabic_page(3, "mixed", "hello world")
        b = render_arabic_page(3, "mixed", "hello world")
        assert a.tobytes() == b.tobytes()


class TestMutationHelpers:
    def test_helpers_are_deterministic(self) -> None:
        base = render_arabic_page(11, "plain", "بيانات")
        assert (
            recompress(base, quality=60).tobytes()
            == recompress(base, quality=60).tobytes()
        )
        assert resize_img(base, 0.5).tobytes() == resize_img(base, 0.5).tobytes()
        assert brightness(base, 10).tobytes() == brightness(base, 10).tobytes()
        assert (
            add_noise(base, seed=5, count=20).tobytes()
            == add_noise(base, seed=5, count=20).tobytes()
        )
        assert blank_like(base).tobytes() == blank_like(base).tobytes()

    def test_helpers_actually_change_pixels(self) -> None:
        base = render_arabic_page(11, "plain", "بيانات")
        assert add_noise(base, seed=5, count=50).tobytes() != base.tobytes()
        assert brightness(base, 30).tobytes() != base.tobytes()
        assert resize_img(base, 0.5).size != base.size


class TestManifestDeterminism:
    def test_same_records_identical_file_bytes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        records = [
            {"sample_id": f"smp_{i:03d}", "source_id": "src_a", "text": f"نص {i}"}
            for i in range(5)
        ]
        path_a = make_manifest(tmp_path / "a", records)
        path_b = make_manifest(tmp_path / "b", records)
        assert path_a.read_bytes() == path_b.read_bytes()
        header = json.loads(path_a.read_text(encoding="utf-8").splitlines()[0])
        assert header["_schema_version"] == "clouda.pretraining.manifest.v1"

    def test_manifest_rows_are_canonical_sorted(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        records = [
            {"sample_id": f"smp_{i:03d}", "source_id": f"src_{i % 2}", "text": "نص"}
            for i in range(6)
        ]
        path = make_manifest(tmp_path, records)
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines[1:]]
        keys = [(row["source_id"], row["sample_id"]) for row in rows]
        assert keys == sorted(keys)
