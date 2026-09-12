"""CLI tests (Wave2-N contract, pending)."""

from __future__ import annotations

import pytest

cli = pytest.importorskip("clouda_data.quality.cli")

from tests.quality.conftest import make_manifest, make_row  # noqa: E402


class TestCli:
    def test_manifest_fixture_buildable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        assert manifest_path.name == "manifest.jsonl"

    def test_cli_module_contract_pending(self) -> None:
        assert cli is not None  # clouda-quality entry point (Wave2-N)

    def test_exit_code_vocabulary_pending(self) -> None:
        assert cli is not None  # 0 PASS / 1 FAIL / 2 config error (Wave2-N)
