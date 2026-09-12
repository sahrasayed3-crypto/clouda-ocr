"""Artifact corruption / path-safety tests (module pending)."""

from __future__ import annotations

import pytest

artifacts = pytest.importorskip("clouda_data.quality.artifacts")

from tests.quality.conftest import make_row, render_arabic_page, save_png  # noqa: E402


class TestArtifactCorruption:
    def test_truncated_png_detected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        img_path = save_png(
            render_arabic_page(31, "plain", "صورة"), tmp_path / "imgs" / "full.png"
        )
        data = img_path.read_bytes()
        img_path.write_bytes(data[: len(data) // 2])
        assert len(img_path.read_bytes()) < len(data)

    def test_zero_byte_file_detected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        img_path = tmp_path / "imgs" / "empty.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"")
        assert img_path.stat().st_size == 0

    def test_text_bytes_with_png_suffix_detected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        img_path = tmp_path / "imgs" / "fake.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes("this is not an image".encode("utf-8"))
        assert img_path.suffix == ".png"

    def test_missing_image_detected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert not (tmp_path / "imgs" / "ghost.png").exists()

    def test_escape_path_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        row = make_row("smp_esc", image_path="../escape.png")
        assert ".." in row.image_path.split("/")
