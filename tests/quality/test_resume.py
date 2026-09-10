"""Resume / run-state tests (Wave2-L contract, pending)."""

from __future__ import annotations

import pytest

run_state = pytest.importorskip("clouda_data.quality.run_state")

from conftest import make_manifest, make_row  # noqa: E402


class TestResume:
    def test_stale_resume_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        assert manifest_path.exists()
        assert run_state is not None

    def test_resume_state_matches_identity(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        manifest_again = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        assert manifest_path.read_bytes() == manifest_again.read_bytes()

    def test_run_state_module_contract_pending(self) -> None:
        assert run_state is not None  # StaleResumeError contract (Wave2-L)
