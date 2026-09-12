from __future__ import annotations

from pathlib import Path

import pytest

from .test_catalog import _catalog


def _service(tmp_path: Path):
    from clouda_lab.dashboard.training import TrainingService

    catalog = _catalog(tmp_path)
    return TrainingService(catalog.settings, catalog)


def test_registered_adapter_capabilities_are_honest_and_path_free(tmp_path: Path):
    service = _service(tmp_path)

    models = {item["adapter_id"]: item for item in service.list_models()}
    assert {"hunyuanocr15_sft", "qwen_vl_sft"} <= set(models)
    hunyuan = models["hunyuanocr15_sft"]
    assert hunyuan["code_integration"] == "PASS"
    assert hunyuan["upstream_repository"] == "Tencent-Hunyuan/HunyuanOCR"
    assert hunyuan["weights"] == "NOT INSTALLED"
    assert hunyuan["processor"] == "NOT INSTALLED"
    assert hunyuan["gpu_validation"] == "DEFERRED"
    assert hunyuan["real_training"] == "NOT VALIDATED"
    assert hunyuan["capabilities"]["supports_resume"] is True
    assert str(tmp_path) not in repr(models)


def test_plan_identity_is_deterministic_and_config_is_inspectable(tmp_path: Path):
    service = _service(tmp_path)
    payload = {
        "experiment_name": "safe-plan",
        "adapter_type": "hunyuanocr15_sft",
        "model_id": "Tencent-Hunyuan/HunyuanOCR-1.5",
        "dataset_id": "safe-set",
        "precision": "bf16",
        "seed": 42,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "epochs": 1,
        "max_steps": 4,
        "learning_rate": 0.00005,
        "checkpoint_frequency": 2,
    }

    first = service.create_plan(payload)
    second = service.create_plan(payload)

    assert first["plan_id"] == second["plan_id"]
    assert first["config_id"] == second["config_id"]
    assert first["dataset_id"] == "safe-set@v1"
    assert first["expected_runtime_backend"] == "cuda"
    assert first["config"]["runtime"]["offline"] is True
    assert first["config"]["runtime"]["dry_run"] is True
    assert str(tmp_path) not in repr(first)
    assert service.get_plan(first["plan_id"])["plan_id"] == first["plan_id"]
    assert len(service.list_plans()) == 1


def test_preflight_preserves_canonical_sections_and_no_gpu_failure(tmp_path: Path):
    service = _service(tmp_path)
    plan = service.create_plan(
        {
            "experiment_name": "gpu-inspection",
            "adapter_type": "qwen_vl_sft",
            "model_id": "Qwen/Qwen3-VL",
            "dataset_id": "safe-set",
            "precision": "bf16",
            "max_steps": 2,
        }
    )

    report = service.run_preflight(plan["plan_id"], write_probe=False)

    assert report["final_status"] in {"READY", "READY_WITH_WARNINGS", "NOT_READY"}
    assert {section["name"] for section in report["sections"]} >= {
        "Configuration",
        "Model adapter",
        "Device & precision",
        "Dataset",
        "Checkpoint",
    }
    assert any(
        check["name"] == "device" and check["status"] in {"FAIL", "WARN", "SKIP"}
        for section in report["sections"]
        for check in section["checks"]
    )
    assert str(tmp_path) not in repr(report)


def test_plan_rejects_unknown_adapter_and_protected_dataset(tmp_path: Path):
    service = _service(tmp_path)

    for payload in (
        {"adapter_type": "unknown", "dataset_id": "safe-set"},
        {"adapter_type": "hunyuanocr15_sft", "dataset_id": "holdout-set"},
    ):
        try:
            service.create_plan(payload)
        except (KeyError, ValueError, PermissionError):
            pass
        else:
            raise AssertionError(f"unsafe plan accepted: {payload}")


def test_runs_checkpoints_and_resume_validation_use_canonical_framework(
    tmp_path: Path,
):
    from clouda_training.experiments.config import (
        CheckpointSection,
        DatasetSection,
        ExperimentConfig,
        ExperimentSection,
        ModelSection,
        RuntimeSection,
        TrainingSection,
    )
    from clouda_training.experiments.runs import run_experiment

    service = _service(tmp_path)
    internal = service.catalog._record("safe-set")
    config = ExperimentConfig(
        experiment=ExperimentSection(name="dashboard-dry-run"),
        model=ModelSection(model_id="mock/clouda", adapter_type="mock"),
        dataset=DatasetSection(
            dataset_id="safe-set",
            dataset_version="v1",
            manifest_path=internal["manifest"],
            split="train",
        ),
        training=TrainingSection(max_steps=2, batch_size=1),
        checkpoint=CheckpointSection(save_steps=1, save_total_limit=2),
        runtime=RuntimeSection(
            device="cpu",
            output_root=service.settings.runs_root,
            dry_run=True,
            offline=True,
        ),
    )
    with pytest.raises(RuntimeError):
        run_experiment(config, fail_at_step=2)

    runs = service.list_runs()
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    detail = service.get_run(run_id)
    assert detail["run"]["status"] == "FAILED"
    assert str(tmp_path) not in repr(detail)
    checkpoints = service.list_checkpoints(run_id)
    assert len(checkpoints) == 1
    assert all(item["integrity"] == "PASS" for item in checkpoints)
    resume = service.resume_check(run_id)
    assert resume["compatible"] is True
    assert resume["available"] is False
    assert resume["status"] == "BLOCKED"
    assert resume["execution"] == "DISABLED"
    assert resume["configuration_match"] == "PASS"
    assert resume["dataset_match"] == "PASS"
    assert resume["model_match"] == "PASS"
