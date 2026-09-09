from __future__ import annotations

import json
from pathlib import Path

from clouda_training.cli import main


def _write_fixture(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"_schema_version": "clouda.pretraining.manifest.v1", "_row_count": 1}
        )
        + "\n"
        + json.dumps({"sample_id": "one", "target_split": "train"})
        + "\n",
        encoding="utf-8",
    )
    config = {
        "experiment": {"name": "cli_mock", "tags": ["cli"]},
        "model": {"model_id": "mock/ocr", "revision": "v1", "adapter_type": "mock"},
        "dataset": {
            "dataset_id": "fixture",
            "dataset_version": "v1",
            "manifest_path": str(manifest),
            "split": "train",
        },
        "training": {"seed": 4, "max_steps": 3},
        "runtime": {
            "output_root": str(tmp_path / "runs"),
            "dry_run": True,
            "offline": True,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cli_validate_dry_run_list_show_compare_resume_and_checkpoints(
    tmp_path: Path, capsys
) -> None:
    config = _write_fixture(tmp_path)
    assert main(["validate-config", str(config), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["dry-run", str(config), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "COMPLETED"
    assert (
        main(["dry-run", str(config), "--override", "training.seed=9", "--json"]) == 0
    )
    second = json.loads(capsys.readouterr().out)

    root = str(tmp_path / "runs")
    assert main(["list", "--runs-root", root, "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert main(["show", first["run_id"], "--runs-root", root, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == first["run_id"]
    assert (
        main(
            [
                "compare",
                first["run_id"],
                second["run_id"],
                "--runs-root",
                root,
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert "training.seed" in json.loads(capsys.readouterr().out)["config_differences"]
    assert main(["checkpoints", first["run_id"], "--runs-root", root, "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_cli_returns_useful_error_code_for_invalid_config(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert main(["validate-config", str(path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
