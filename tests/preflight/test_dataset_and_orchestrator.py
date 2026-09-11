"""Tests for dataset/protection + resume checks + the preflight orchestrator."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from clouda_training.experiments.config import (
    CheckpointSection,
    DatasetSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    RuntimeSection,
    TrackingSection,
    TrainingSection,
)
from clouda_training.preflight.checks_dataset import (
    check_data_contract,
    check_dataset,
    check_resume,
)
from clouda_training.preflight.models import (
    PreflightFinalStatus,
    PreflightStatus,
)
from clouda_training.preflight.orchestrator import run_preflight


def make_config(**sections: object) -> ExperimentConfig:
    overrides = {
        "experiment": ExperimentSection(name="preflight_probe"),
        "model": ModelSection(model_id="mock/ocr"),
        "dataset": DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=Path("manifest.jsonl"),
        ),
        "training": TrainingSection(),
        "checkpoint": CheckpointSection(),
        "runtime": RuntimeSection(),
        "tracking": TrackingSection(),
    }
    overrides.update(sections)  # type: ignore[arg-type]
    return ExperimentConfig(**overrides)  # type: ignore[arg-type]


HEADER = {
    "_schema_version": "clouda.pretraining.manifest.v1",
    "_row_count": 2,
    "dataset_id": "synthetic-test",
    "dataset_version": "v1",
}


def write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    payload = [HEADER, *rows]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload),
        encoding="utf-8",
    )
    return manifest


def clean_rows() -> list[dict]:
    return [
        {
            "sample_id": "ar-1",
            "target_split": "train",
            "source_id": "synthetic-test",
            "image_path": "p/1.png",
            "ground_truth": "نص",
            "source_license": "Apache-2.0",
        },
        {
            "sample_id": "ar-2",
            "target_split": "train",
            "source_id": "synthetic-test",
            "image_path": "p/2.png",
            "ground_truth": "نص ثانٍ",
            "source_license": "Apache-2.0",
        },
    ]


# ---------------------------------------------------------------- dataset checks


def test_valid_manifest_passes(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        )
    )
    checks = check_dataset(config)
    assert all(c.status is PreflightStatus.PASS for c in checks), checks


def test_missing_manifest_blocks(tmp_path: Path) -> None:
    config = make_config(
        dataset=DatasetSection(
            dataset_id="x",
            dataset_version="v1",
            manifest_path=tmp_path / "nope.jsonl",
        )
    )
    checks = check_dataset(config)
    assert any(c.status is PreflightStatus.FAIL and c.blocker for c in checks)


def test_holdout_split_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="holdout",
        )
    )
    checks = check_dataset(config)
    assert any(c.status is PreflightStatus.FAIL and c.blocker for c in checks)


def test_protected_row_blocks(tmp_path: Path) -> None:
    rows = clean_rows()
    rows[0]["protected"] = True
    manifest = write_manifest(tmp_path, rows)
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        )
    )
    checks = check_dataset(config)
    assert any(
        c.name == "dataset.protection"
        and c.status is PreflightStatus.FAIL
        and c.blocker
        for c in checks
    )


def test_protected_header_blocks(tmp_path: Path) -> None:
    rows = clean_rows()
    manifest = write_manifest(tmp_path, rows)
    # Rewrite with a protected header
    payload = [dict(HEADER, protected=True), *rows]
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload),
        encoding="utf-8",
    )
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        )
    )
    checks = check_dataset(config)
    assert any(c.status is PreflightStatus.FAIL and c.blocker for c in checks)


def test_identity_mismatch_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        dataset=DatasetSection(
            dataset_id="other-dataset",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        )
    )
    checks = check_dataset(config)
    assert any(c.status is PreflightStatus.FAIL and c.blocker for c in checks)


# ---------------------------------------------------------------- data contract


def test_data_contract_mock_skips() -> None:
    check = check_data_contract(make_config())
    assert check.status is PreflightStatus.SKIP


def test_data_contract_unknown_adapter_skips() -> None:
    check = check_data_contract(
        make_config(model=ModelSection(model_id="m", adapter_type="nope"))
    )
    assert check.status is PreflightStatus.SKIP


def test_data_contract_packed_only_adapter_blocks(tmp_path: Path) -> None:
    from clouda_training.adapters.capabilities import ModelCapabilities
    from clouda_training.adapters.descriptor import ModelAdapterDescriptor
    from clouda_training.adapters.registry import get_default_registry
    from clouda_training.preflight.checks_system import ensure_adapters_registered

    ensure_adapters_registered()
    registry = get_default_registry()
    if not registry.is_registered("packed_only_test"):
        registry.register(
            ModelAdapterDescriptor(
                adapter_type="packed_only_test",
                adapter_version="1.0.0",
                model_family="test",
                task_family="ocr",
                capabilities=ModelCapabilities(),
                supported_data_modes=("packed_jsonl",),
                checkpoint_compatibility_id="test@1",
                upstream_repository="test/repo",
                upstream_revision="abc",
            ),
            lambda **kw: None,
        )
    config = make_config(
        model=ModelSection(model_id="m", adapter_type="packed_only_test")
    )
    check = check_data_contract(config)
    assert check.status is PreflightStatus.FAIL and check.blocker


# ---------------------------------------------------------------- resume checks


def test_resume_skip_when_fresh() -> None:
    check = check_resume(make_config())
    assert check.status is PreflightStatus.SKIP


def test_resume_missing_checkpoint_blocks(tmp_path: Path) -> None:
    config = make_config(
        checkpoint=CheckpointSection(resume_from=str(tmp_path / "missing_ckpt"))
    )
    check = check_resume(config)
    assert check.status is PreflightStatus.FAIL and check.blocker


def _write_ckpt(tmp_path: Path, config: ExperimentConfig, **overrides: object) -> Path:
    ckpt = tmp_path / "step-00000003"
    ckpt.mkdir(parents=True, exist_ok=True)
    state = ckpt / "state.json"
    state.write_text(json.dumps({"step": 3, "epoch": 3.0}), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "step": 3,
        "epoch": 3.0,
        "timestamp": "now",
        "metrics": {"loss": 0.5},
        "config_hash": config.hash,
        "run_id": "run-1",
        "integrity_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
        "experiment_name": config.experiment.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.dataset.dataset_id,
        "dataset_version": config.dataset.dataset_version,
    }
    payload.update(overrides)  # type: ignore[arg-type]
    (ckpt / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return ckpt


def test_resume_compatible_checkpoint_passes(tmp_path: Path) -> None:
    config = make_config()
    ckpt = _write_ckpt(tmp_path, config)
    config = make_config(
        checkpoint=CheckpointSection(resume_from=str(ckpt)),
    )
    check = check_resume(config)
    assert check.status is PreflightStatus.PASS, check.detail


def test_resume_experiment_mismatch_blocks(tmp_path: Path) -> None:
    config = make_config()
    ckpt = _write_ckpt(tmp_path, config)
    other = make_config(
        experiment=ExperimentSection(name="different_experiment"),
        checkpoint=CheckpointSection(resume_from=str(ckpt)),
    )
    check = check_resume(other)
    assert check.status is PreflightStatus.FAIL and check.blocker


def test_resume_model_family_mismatch_blocks(tmp_path: Path) -> None:
    config = make_config()
    ckpt = _write_ckpt(tmp_path, config)
    other = make_config(
        model=ModelSection(model_id="other/model"),
        checkpoint=CheckpointSection(resume_from=str(ckpt)),
    )
    check = check_resume(other)
    assert check.status is PreflightStatus.FAIL and check.blocker


def test_resume_cross_adapter_rejected(tmp_path: Path) -> None:
    from clouda_training.preflight.checks_system import ensure_adapters_registered

    ensure_adapters_registered()
    config = make_config()
    ckpt = _write_ckpt(
        tmp_path,
        config,
        adapter_identity={"adapter_id": "hunyuanocr15_sft"},
    )
    other = make_config(
        model=ModelSection(model_id="mock/ocr", adapter_type="qwen_vl_sft"),
        checkpoint=CheckpointSection(resume_from=str(ckpt)),
    )
    check = check_resume(other)
    assert check.status is PreflightStatus.FAIL and check.blocker
    assert "cross-adapter" in check.detail


def test_resume_corrupt_torch_state_blocks(tmp_path: Path) -> None:
    config = make_config()
    ckpt = _write_ckpt(tmp_path, config, torch_state_file="state.bin")
    (ckpt / "state.bin").write_bytes(b"tampered")
    other = make_config(checkpoint=CheckpointSection(resume_from=str(ckpt)))
    check = check_resume(other)
    assert check.status is PreflightStatus.FAIL and check.blocker


# ---------------------------------------------------------------- orchestrator


def test_orchestrator_ready_for_synthetic_mock(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        ),
        runtime=RuntimeSection(output_root=tmp_path / "out"),
    )
    report = run_preflight(config, dataset_row_count=2)
    # mock adapter + everything valid; UNAVAILABLE hooks are warnings only
    assert report.final_status() in (
        PreflightFinalStatus.READY,
        PreflightFinalStatus.READY_WITH_WARNINGS,
    )
    payload = report.to_dict()
    assert payload["final_status"] in {"READY", "READY_WITH_WARNINGS"}
    assert payload["training_plan"] is not None
    assert payload["training_plan"]["effective_batch_size"] >= 1
    # machine-readable: JSON-serializable
    json.dumps(payload)


def test_orchestrator_not_ready_never_runs_training(tmp_path: Path) -> None:
    # Missing manifest + bad config -> NOT_READY; must not create output dirs.
    config = make_config(
        dataset=DatasetSection(
            dataset_id="x",
            dataset_version="v1",
            manifest_path=tmp_path / "missing.jsonl",
        ),
        training=dataclasses.replace(TrainingSection(), batch_size=0),
        runtime=RuntimeSection(output_root=tmp_path / "out"),
    )
    report = run_preflight(config, dataset_row_count=None)
    assert report.final_status() is PreflightFinalStatus.NOT_READY
    assert report.blockers, "blockers must be explicit"
    # preflight must not start training or create run structure (only the
    # documented output_root writability probe, which the runtime would
    # create anyway — no checkpoints/metrics/experiment dirs allowed)
    assert not (tmp_path / "out" / "checkpoints").exists()
    assert not (tmp_path / "out" / "logs").exists()


def test_orchestrator_unavailable_hooks_warn_not_block(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        ),
        runtime=RuntimeSection(output_root=tmp_path / "out"),
    )
    report = run_preflight(config, dataset_row_count=2)
    unavailable = [
        c for c in report.all_checks() if c.status is PreflightStatus.UNAVAILABLE
    ]
    assert len(unavailable) == 3  # quality gate, data loader, env doctor
    # and they are warnings, not blockers
    assert report.final_status() is not PreflightFinalStatus.NOT_READY


# ---------------------------------------------------------------- E2E: READY -> run


def test_ready_preflight_then_tiny_run(tmp_path: Path) -> None:
    """Synthetic READY preflight corresponds to an actually runnable tiny run."""
    torch = pytest.importorskip("torch")
    from clouda_training.experiments.metrics import MetricLogger
    from clouda_training.runtime.torch_backend import TorchTrainerBackend
    from tests.hunyuan.mock_hunyuan import MockHunyuanProcessor
    from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter

    manifest = write_manifest(tmp_path, clean_rows())
    config = make_config(
        model=ModelSection(model_id="mock-hunyuan", adapter_type="torch"),
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        ),
        training=TrainingSection(seed=7, max_steps=2, batch_size=2, learning_rate=0.01),
        runtime=RuntimeSection(output_root=tmp_path / "out"),
    )
    report = run_preflight(config, dataset_row_count=2, write_probe=False)
    assert report.final_status() is not PreflightFinalStatus.NOT_READY, [
        b.reason for b in report.blockers
    ]

    # tiny real run through TorchTrainerBackend with the mock processor
    adapter = HunyuanOCR15SFTAdapter(
        local_model_path="unused-mock", processor=MockHunyuanProcessor()
    )
    from tests.hunyuan.mock_hunyuan import MockHunyuanVLForConditionalGeneration

    model = MockHunyuanVLForConditionalGeneration()
    adapter.model = model
    backend = TorchTrainerBackend.__new__(TorchTrainerBackend)
    backend.torch = torch
    backend.metrics = MetricLogger(tmp_path / "metrics.jsonl", "probe")
    backend.config = config
    backend.adapter = adapter
    backend.model = model
    backend.device = torch.device("cpu")
    backend.fail_at_step = None
    backend.interrupt_at_step = None
    backend._build_optimizer_and_scheduler()
    params_before = [p.detach().clone() for p in model.parameters()]
    result = backend.train(start_step=0)
    assert result.final_step == 2
    assert any(
        not torch.equal(a, b)
        for a, b in zip(params_before, [p for p in model.parameters()])
    )
