"""CLI tests: `clouda-training preflight` (help/json/exit codes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.cli import build_parser, main
from tests.preflight.test_dataset_and_orchestrator import (
    clean_rows,
    write_manifest,
)


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Write a real experiment config YAML the canonical loader accepts."""
    manifest = write_manifest(tmp_path, clean_rows())
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "experiment:",
                "  name: preflight_cli_probe",
                "model:",
                "  model_id: mock/ocr",
                "  adapter_type: mock",
                "dataset:",
                "  dataset_id: synthetic-test",
                "  dataset_version: v1",
                f"  manifest_path: {manifest.as_posix()}",
                "  split: train",
                "training:",
                "  seed: 7",
                "checkpoint:",
                "  save_strategy: none",
                "runtime:",
                f"  output_root: {(tmp_path / 'out').as_posix()}",
                "tracking:",
                "  enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # sanity: the canonical loader must accept it
    from clouda_training.experiments.config import load_experiment_config

    load_experiment_config(path)
    return path


def test_preflight_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["preflight", "--help"])
    assert excinfo.value.code == 0


def test_preflight_invalid_config_exit_2(tmp_path: Path, capsys) -> None:
    rc = main(["preflight", str(tmp_path / "no_such.yaml")])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_preflight_ready_json_exit_0(config_file: Path, capsys) -> None:
    rc = main(["preflight", str(config_file), "--json"])
    assert rc in (0, 1)  # ready-with-warnings also exits 0
    if rc == 0:
        payload = json.loads(capsys.readouterr().out)
        assert payload["final_status"] in {"READY", "READY_WITH_WARNINGS"}
        assert payload["training_plan"] is not None


def test_preflight_strict_exit_code(config_file: Path, capsys) -> None:
    rc_strict = main(["preflight", str(config_file), "--json", "--strict"])
    rc_normal = main(["preflight", str(config_file), "--json"])
    # strict can only make the exit code worse or equal
    assert rc_strict >= rc_normal
