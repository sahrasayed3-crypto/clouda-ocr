"""CLI integration tests for the training-data-* commands."""

from __future__ import annotations

import json

import pytest

from clouda_data.pipeline.cli import main
from tests.data_foundation.fixtures.training_data_fixtures import (
    build_synthetic_dataset,
)


@pytest.fixture
def sharded(tmp_path):
    manifest, root = build_synthetic_dataset(tmp_path / "ds", count=12)
    out = tmp_path / "sharded"
    result = main(
        [
            "training-data-shard",
            str(manifest),
            "--output",
            str(out),
            "--samples-per-shard",
            "4",
        ]
    )
    assert result == 0
    return manifest, root, out, out / "shard_index.json"


class TestTrainingDataCLI:
    def test_help_lists_commands(self, capsys):
        with pytest.raises(SystemExit):
            main(["training-data-shard", "--help"])
        captured = capsys.readouterr()
        assert "shard" in captured.out

    def test_shard_creates_index_and_shards(self, tmp_path):
        manifest, _root = build_synthetic_dataset(tmp_path / "ds", count=12)
        out = tmp_path / "s"
        code = main(
            [
                "training-data-shard",
                str(manifest),
                "--output",
                str(out),
                "--samples-per-shard",
                "4",
            ]
        )
        assert code == 0
        assert (out / "shard_index.json").is_file()
        assert (out / "shards").is_dir()

    def test_inspect_reports_index(self, sharded, capsys):
        _m, _r, _o, index_path = sharded
        assert main(["training-data-inspect", str(index_path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dataset_id"] == "synthetic_training_fixture"
        assert payload["total_samples"] == 12
        assert payload["total_shards"] == 3

    def test_inspect_filter_single_shard(self, sharded, capsys):
        _m, _r, _o, index_path = sharded
        assert main(["training-data-inspect", str(index_path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        shard_id = payload["shards"][0]["shard_id"]
        assert (
            main(["training-data-inspect", str(index_path), "--shard", shard_id]) == 0
        )
        filtered = json.loads(capsys.readouterr().out)
        assert len(filtered["shards"]) == 1

    def test_verify_ok(self, sharded, capsys):
        _m, _r, _o, index_path = sharded
        assert main(["training-data-verify", str(index_path)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is True
        assert report["verified_samples"] == 12

    def test_verify_fails_on_tampered_shard(self, sharded):
        _m, _r, out, index_path = sharded
        first = sorted(out.joinpath("shards").glob("*.jsonl"))[0]
        first.write_text('{"tampered": true}\n', encoding="utf-8")
        with pytest.raises(Exception, match="hash mismatch"):
            main(["training-data-verify", str(index_path)])

    def test_sample_outputs_references(self, sharded, tmp_path, capsys):
        manifest, root, _out, index_path = sharded
        assert (
            main(
                [
                    "training-data-sample",
                    str(index_path),
                    "--count",
                    "3",
                    "--manifest",
                    str(manifest),
                    "--root",
                    str(root),
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 3
        assert all(s["sample_id"].startswith("smp_") for s in payload["samples"])

    def test_dry_run_replay_identical(self, sharded, tmp_path, capsys):
        manifest, root, _out, index_path = sharded
        code = main(
            [
                "training-data-dry-run",
                str(index_path),
                "--batches",
                "2",
                "--batch-size",
                "5",
                "--manifest",
                str(manifest),
                "--root",
                str(root),
            ]
        )
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["produced_batches"] == 2
        assert report["delivered_samples"] == 10
        assert report["replay_identical"] is True

    def test_dry_run_validation_mode_flag(self, sharded, tmp_path, capsys):
        manifest, root, _out, index_path = sharded
        code = main(
            [
                "training-data-dry-run",
                str(index_path),
                "--validation-mode",
                "strict",
                "--bad-sample-policy",
                "skip_and_record",
                "--manifest",
                str(manifest),
                "--root",
                str(root),
            ]
        )
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["replay_identical"] is True


def _run(args: list[str]) -> str:
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(args)
    return buffer.getvalue()
