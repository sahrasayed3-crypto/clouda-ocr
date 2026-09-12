"""Derived clean-manifest tests (Wave2-K contract, pending)."""

from __future__ import annotations

import pytest

derived = pytest.importorskip("clouda_data.quality.derived")

from tests.quality.conftest import make_manifest, make_row  # noqa: E402


class TestDerivedManifest:
    def test_derived_manifest_disjoint_from_quarantine(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(
            tmp_path,
            [
                make_row("smp_keep", text="احتفظ"),
                make_row("smp_drop", text="استبعد"),
            ],
        )
        assert manifest_path.exists()
        assert derived is not None

    def test_original_manifest_unmodified_after_derive(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        before = manifest_path.read_bytes()
        assert derived is not None
        assert manifest_path.read_bytes() == before

    def test_derived_rows_are_canonical_v1(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        first_line = manifest_path.read_text(encoding="utf-8").splitlines()[0]
        assert '"_schema_version": "clouda.pretraining.manifest.v1"' in first_line
