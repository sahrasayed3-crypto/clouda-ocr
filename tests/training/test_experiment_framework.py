from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.experiments.checkpoints import CheckpointManager
from clouda_training.experiments import runs as runs_module
from clouda_training.experiments import (
    ExperimentRegistry,
    ConfigError,
    RunStatus,
    compare_runs,
    list_checkpoints,
    list_runs,
    load_experiment_config,
    resume_run,
    run_experiment,
    verify_run_integrity,
)


def _manifest(path: Path, *, split: str = "train", protected: bool = False) -> Path:
    rows = [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": 1,
            "dataset_role": "protected_holdout" if protected else "training",
        },
        {
            "sample_id": "sample-1",
            "target_split": split,
            "source_id": "synthetic-test",
            "source_path": "page.png",
            "source_license": "Apache-2.0",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(path: Path, manifest: Path, runs: Path, **changes: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": {
            "name": "arabic_mock_baseline",
            "description": "offline lifecycle fixture",
            "tags": ["test", "arabic"],
        },
        "model": {
            "model_id": "mock/ocr",
            "revision": "test-v1",
            "model_family": "mock",
            "adapter_type": "mock",
        },
        "dataset": {
            "dataset_id": "synthetic-test",
            "dataset_version": "v1",
            "manifest_path": str(manifest),
            "split": "train",
            "preprocessing_version": "fixture-v1",
        },
        "training": {"seed": 17, "max_steps": 5, "batch_size": 2},
        "checkpoint": {
            "save_strategy": "steps",
            "save_steps": 2,
            "save_total_limit": 2,
            "keep_best": True,
            "metric_for_best": "loss",
            "greater_is_better": False,
        },
        "evaluation": {
            "enabled": True,
            "eval_split": "validation",
            "eval_steps": 2,
            "metrics": ["cer", "wer"],
        },
        "runtime": {
            "device": "cpu",
            "output_root": str(runs),
            "dry_run": True,
            "offline": True,
            "deterministic": True,
        },
        "tracking": {"enabled": True, "backend": "jsonl", "log_steps": 1},
    }
    for key, value in changes.items():
        section, field = key.split("__", 1)
        assert isinstance(payload[section], dict)
        payload[section][field] = value  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_is_strict_canonical_and_supports_overrides(tmp_path: Path) -> None:
    path = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl"),
        tmp_path / "runs",
    )
    first = load_experiment_config(path)
    second = load_experiment_config(path, overrides=["training.learning_rate=0.001"])
    assert first.hash == load_experiment_config(path).hash
    assert second.training.learning_rate == 0.001
    assert second.hash != first.hash
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment_config(path, overrides=["training.learnin_rate=3"])


def test_config_rejects_unknown_fields_and_invalid_types(tmp_path: Path) -> None:
    path = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl"),
        tmp_path / "runs",
        training__batch_size="two",
    )
    with pytest.raises(ConfigError, match="batch_size"):
        load_experiment_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training"]["batch_size"] = 2
    payload["runtime"]["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment_config(path)


def test_config_rejects_missing_required_sections(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    path.write_text(
        json.dumps({"experiment": {"name": "incomplete"}}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="missing required"):
        load_experiment_config(path)


def test_experiment_name_cannot_escape_run_root(tmp_path: Path) -> None:
    path = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl"),
        tmp_path / "runs",
        experiment__name="../../escape",
    )
    with pytest.raises(ConfigError, match="experiment.name"):
        load_experiment_config(path)


@pytest.mark.parametrize("split", ["holdout", "protected_holdout", "benchmark_holdout"])
def test_protected_holdout_cannot_start_training(tmp_path: Path, split: str) -> None:
    config = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl", split=split),
        tmp_path / "runs",
        dataset__split=split,
    )
    with pytest.raises(PermissionError, match="holdout"):
        run_experiment(load_experiment_config(config))
    assert not (tmp_path / "runs").exists()


def test_protected_manifest_marker_is_rejected(tmp_path: Path) -> None:
    config = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl", protected=True),
        tmp_path / "runs",
    )
    with pytest.raises(PermissionError, match="protected"):
        run_experiment(load_experiment_config(config))


def test_mixed_manifest_with_holdout_row_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["_row_count"] = 2
    rows.append(
        {
            "sample_id": "sample-protected",
            "target_split": " Holdout ",
            "source_id": "synthetic-test",
            "source_path": "protected.png",
        }
    )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = _config(tmp_path / "experiment.json", manifest, tmp_path / "runs")
    with pytest.raises(PermissionError, match="holdout"):
        run_experiment(load_experiment_config(config))


def test_nested_protected_row_marker_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["provenance"] = {"protected": True}
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = _config(tmp_path / "experiment.json", manifest, tmp_path / "runs")
    with pytest.raises(PermissionError, match="protected"):
        run_experiment(load_experiment_config(config))


@pytest.mark.parametrize(
    ("header_field", "configured_field"),
    [("dataset_id", "other-id"), ("dataset_version", "other-version")],
)
def test_manifest_dataset_identity_must_match_config(
    tmp_path: Path, header_field: str, configured_field: str
) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[0].update({"dataset_id": "synthetic-test", "dataset_version": "v1"})
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    change = {
        "dataset__dataset_id": "synthetic-test",
        "dataset__dataset_version": "v1",
    }
    change[f"dataset__{header_field}"] = configured_field
    config = _config(
        tmp_path / "experiment.json", manifest, tmp_path / "runs", **change
    )
    with pytest.raises(ValueError, match=header_field):
        run_experiment(load_experiment_config(config))


def test_malformed_row_split_metadata_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["target_split"] = ["train", "holdout"]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = _config(tmp_path / "experiment.json", manifest, tmp_path / "runs")
    with pytest.raises(ValueError, match="split metadata"):
        run_experiment(load_experiment_config(config))


@pytest.mark.parametrize(
    "nested",
    [
        {"provenance": {"target_split": ["holdout"]}},
        {"metadata": {"protected": "maybe"}},
        {"provenance": ["not", "a", "mapping"]},
    ],
)
def test_malformed_nested_protection_metadata_is_rejected(
    tmp_path: Path, nested: dict
) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[1].update(nested)
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = _config(tmp_path / "experiment.json", manifest, tmp_path / "runs")
    with pytest.raises(ValueError, match="protection metadata"):
        run_experiment(load_experiment_config(config))


def test_evaluation_cannot_target_protected_holdout(tmp_path: Path) -> None:
    config = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl"),
        tmp_path / "runs",
        evaluation__eval_split="holdout",
    )
    with pytest.raises(ConfigError, match="eval_split"):
        load_experiment_config(config)


def test_mock_run_records_contract_and_is_reproducible(tmp_path: Path) -> None:
    config_path = _config(
        tmp_path / "experiment.json",
        _manifest(tmp_path / "manifest.jsonl"),
        tmp_path / "runs",
    )
    config = load_experiment_config(config_path)
    left = run_experiment(config)
    right = run_experiment(config)
    assert left.run_id != right.run_id
    assert left.status is RunStatus.COMPLETED
    for name in (
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.jsonl",
        "summary.json",
        "environment.json",
        "artifacts/integrity.json",
    ):
        assert (left.path / name).is_file()
    metadata = json.loads((left.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_hash"] == config.hash
    assert metadata["dataset_manifest_hash"]
    assert metadata["dataset_version"] == "v1"
    assert "git_commit" in metadata
    environment = json.loads(
        (left.path / "environment.json").read_text(encoding="utf-8")
    )
    assert "environment_variables" not in environment
    status = json.loads((left.path / "status.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in status["history"]] == [
        "CREATED",
        "RUNNING",
        "COMPLETED",
    ]

    def losses(run):
        return [row["value"] for row in run.metrics() if row["metric_name"] == "loss"]

    assert losses(left) == losses(right)
    assert verify_run_integrity(left.path)["valid"] is True


def test_config_hash_is_portable_across_workspace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left_manifest = _manifest(left_root / "manifest.jsonl")
    right_manifest = _manifest(right_root / "manifest.jsonl")
    left = load_experiment_config(
        _config(left_root / "experiment.json", left_manifest, left_root / "runs")
    )
    right = load_experiment_config(
        _config(right_root / "experiment.json", right_manifest, right_root / "runs")
    )

    def refuse_read_bytes(_path: Path) -> bytes:
        raise AssertionError("config hashing must stream manifest bytes")

    monkeypatch.setattr(Path, "read_bytes", refuse_read_bytes)
    assert left.hash == right.hash


def test_run_integrity_detects_tampered_artifact(tmp_path: Path) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )
    run = run_experiment(config)
    (run.path / "summary.json").write_text("{}", encoding="utf-8")
    report = verify_run_integrity(run.path)
    assert report["valid"] is False
    assert "summary.json" in report["mismatches"]


def test_changed_seed_changes_mock_metrics(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    left = run_experiment(
        load_experiment_config(
            _config(tmp_path / "a.json", manifest, tmp_path / "runs")
        )
    )
    right = run_experiment(
        load_experiment_config(
            _config(
                tmp_path / "b.json",
                manifest,
                tmp_path / "runs",
                training__seed=18,
            )
        )
    )
    assert (
        left.summary()["final_metrics"]["loss"]
        != right.summary()["final_metrics"]["loss"]
    )


def test_failed_and_interrupted_runs_preserve_truthful_status(tmp_path: Path) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )
    with pytest.raises(RuntimeError, match="injected"):
        run_experiment(config, fail_at_step=2)
    assert (
        list_runs(tmp_path / "runs", status=RunStatus.FAILED)[0].status
        is RunStatus.FAILED
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config, interrupt_at_step=3)
    assert (
        list_runs(tmp_path / "runs", status=RunStatus.INTERRUPTED)[0].status
        is RunStatus.INTERRUPTED
    )


def test_initialization_failure_leaves_a_failed_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )

    def fail_capture() -> dict:
        raise RuntimeError("environment capture injected failure")

    monkeypatch.setattr(runs_module, "capture_environment", fail_capture)
    with pytest.raises(RuntimeError, match="environment capture"):
        run_experiment(config)
    failed = list_runs(tmp_path / "runs", status=RunStatus.FAILED)
    assert len(failed) == 1


def test_checkpoint_retention_best_and_resume(tmp_path: Path) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config, interrupt_at_step=5)
    interrupted = list_runs(tmp_path / "runs", status=RunStatus.INTERRUPTED)[0]
    checkpoints = list_checkpoints(interrupted.path)
    assert [item.step for item in checkpoints] == [2, 4]
    assert all(item.integrity_sha256 for item in checkpoints)
    resumed = resume_run(interrupted.run_id, tmp_path / "runs")
    assert resumed.run_id == interrupted.run_id
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.summary()["resumed_from_step"] == 4


def test_checkpoint_retention_limit_includes_best_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )
    run_path = tmp_path / "manual-run"
    manager = CheckpointManager(run_path, "manual-run", config)
    for step, loss in ((1, 0.1), (2, 0.2), (3, 0.3)):
        manager.save(step=step, epoch=float(step), metrics={"loss": loss})
    checkpoints = list_checkpoints(run_path)
    assert len(checkpoints) == 2
    assert [item.step for item in checkpoints] == [1, 3]
    assert checkpoints[0].is_best


def test_corrupt_or_incompatible_checkpoint_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    config = load_experiment_config(
        _config(tmp_path / "a.json", manifest, tmp_path / "runs")
    )
    run = run_experiment(config)
    checkpoint = list_checkpoints(run.path)[-1]
    checkpoint.metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        resume_run(run.run_id, tmp_path / "runs")


def test_incompatible_checkpoint_identity_is_rejected(tmp_path: Path) -> None:
    config = load_experiment_config(
        _config(
            tmp_path / "experiment.json",
            _manifest(tmp_path / "manifest.jsonl"),
            tmp_path / "runs",
        )
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config, interrupt_at_step=5)
    interrupted = list_runs(tmp_path / "runs", status=RunStatus.INTERRUPTED)[0]
    checkpoint = list_checkpoints(interrupted.path)[-1]
    metadata = json.loads(checkpoint.metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_version"] = "different"
    checkpoint.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        resume_run(interrupted.run_id, tmp_path / "runs")


def test_registry_and_comparison_expose_config_and_metric_differences(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl")
    left = run_experiment(
        load_experiment_config(
            _config(tmp_path / "a.json", manifest, tmp_path / "runs")
        )
    )
    right = run_experiment(
        load_experiment_config(
            _config(
                tmp_path / "b.json",
                manifest,
                tmp_path / "runs",
                training__seed=33,
            )
        )
    )
    comparison = compare_runs(left, right)
    assert comparison["runs"][0]["run_id"] == left.run_id
    assert comparison["config_differences"]["training.seed"] == [17, 33]
    assert "loss" in comparison["metric_differences"]
    assert len(list_runs(tmp_path / "runs", tags={"arabic"})) == 2
    registry = ExperimentRegistry(tmp_path / "runs")
    assert registry.latest("arabic_mock_baseline") is not None
    assert registry.best("arabic_mock_baseline", "loss") is not None
