"""CLI tests for the quality entry point."""

from __future__ import annotations

from clouda_data.quality import cli

from tests.quality.conftest import make_manifest, make_row  # noqa: E402


class TestCli:
    def test_manifest_fixture_buildable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        manifest_path = make_manifest(tmp_path, [make_row("smp_a", text="نص")])
        assert manifest_path.name == "manifest.jsonl"

    def test_cli_parser_routes_scan_command(self) -> None:
        args = cli.build_parser().parse_args(["scan", "manifest.jsonl"])
        assert args.func is cli._cmd_scan

    def test_missing_manifest_returns_config_error_exit_code(
        self, tmp_path, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        assert cli.main(["scan", str(tmp_path / "missing.jsonl")]) == 2
        assert "manifest not found" in capsys.readouterr().err
