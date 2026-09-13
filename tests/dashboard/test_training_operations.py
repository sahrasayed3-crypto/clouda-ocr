from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_catalog import _catalog


def _wait(tasks, task_id: str):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = tasks.get_task(task_id)
        if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return task
        time.sleep(0.01)
    raise AssertionError("training task did not finish")


def _service(tmp_path: Path):
    from clouda_lab.dashboard.tasks import OperationTaskService
    from clouda_lab.dashboard.training import TrainingService

    catalog = _catalog(tmp_path)
    tasks = OperationTaskService(
        catalog.settings.tasks_root,
        browser_roots=(catalog.settings.repo_root,),
        max_workers=1,
    )
    return TrainingService(catalog.settings, catalog, tasks=tasks), tasks


def test_training_start_fails_closed_when_preflight_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, tasks = _service(tmp_path)
    monkeypatch.setattr(service, "get_plan", lambda plan_id: {"plan_id": plan_id})
    monkeypatch.setattr(
        service,
        "_config",
        lambda plan_id: SimpleNamespace(runtime=SimpleNamespace(dry_run=False)),
    )
    monkeypatch.setattr(
        service,
        "run_preflight",
        lambda plan_id, write_probe=True: {
            "final_status": "NOT_READY",
            "blockers": [{"reason": "No CUDA device"}],
        },
    )

    with pytest.raises(PermissionError, match="preflight"):
        service.start_training("plan-safe")
    tasks.shutdown()


def test_ready_training_delegates_once_to_canonical_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, tasks = _service(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(service, "get_plan", lambda plan_id: {"plan_id": plan_id})
    monkeypatch.setattr(
        service,
        "_config",
        lambda plan_id: SimpleNamespace(runtime=SimpleNamespace(dry_run=False)),
    )
    monkeypatch.setattr(
        service,
        "run_preflight",
        lambda plan_id, write_probe=True: {
            "final_status": "READY",
            "blockers": [],
        },
    )

    def fake_start(_self, config_path):
        calls.append(Path(config_path).name)
        return {"run_id": "run-real", "status": "COMPLETED"}

    monkeypatch.setattr(
        "clouda_lab.training_orchestrator.TrainingOrchestrator.start", fake_start
    )
    task = service.start_training("plan-safe")
    finished = _wait(tasks, task["task_id"])
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert calls == ["plan-safe.config.yaml"]
    assert finished["result"]["run_id"] == "run-real"


def test_stop_capability_is_truthful_until_runtime_supports_cooperation(tmp_path: Path):
    service, tasks = _service(tmp_path)
    capability = service.stop_capability()
    tasks.shutdown()

    assert capability["available"] is False
    assert "cooperative" in capability["reason"].lower()


def test_verified_managed_assets_produce_non_dry_server_owned_plan(tmp_path: Path):
    from clouda_lab.dashboard.training import TrainingService

    catalog = _catalog(tmp_path)
    asset_root = catalog.settings.models_root / "hunyuan-local"
    asset_root.mkdir(parents=True)

    class Models:
        @staticmethod
        def list_models():
            return [
                {
                    "training_adapter": "hunyuanocr15_sft",
                    "assets": {"status": "VERIFIED", "asset_id": "hunyuan-local"},
                }
            ]

    service = TrainingService(catalog.settings, catalog, models=Models())
    plan = service.create_plan(
        {
            "adapter_type": "hunyuanocr15_sft",
            "dataset_id": "safe-set",
            "max_steps": 2,
        }
    )

    assert plan["config"]["runtime"]["dry_run"] is False
    assert plan["config"]["model"]["model_id"] == "data/models/hunyuan-local"
    assert plan["model_id"] == "Tencent-Hunyuan/HunyuanOCR"
    assert str(tmp_path) not in repr(plan)


def test_training_start_requires_expiring_single_use_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, tasks = _service(tmp_path)
    monkeypatch.setattr(
        service,
        "get_plan",
        lambda plan_id: {
            "plan_id": plan_id,
            "config_id": "config-safe",
            "model_id": "published-model",
            "dataset_id": "safe-set@v1",
            "output_path": "runs",
            "config": {
                "experiment": {"name": "safe-experiment"},
                "training": {"max_steps": 2},
            },
            "plan": {"hardware": {"gpu_count": 1}},
        },
    )
    monkeypatch.setattr(
        service,
        "run_preflight",
        lambda plan_id, write_probe=False: {"final_status": "READY"},
    )
    monkeypatch.setattr(
        service,
        "_config",
        lambda plan_id: SimpleNamespace(runtime=SimpleNamespace(dry_run=False)),
    )
    started: list[str] = []

    def fake_start(plan_id: str):
        started.append(plan_id)
        return {"task_id": "task-real"}

    monkeypatch.setattr(
        service,
        "start_training",
        fake_start,
    )

    plan = service.create_start_plan("plan-safe")
    with pytest.raises(PermissionError, match="confirmation"):
        service.confirm_start(plan["plan_id"], "wrong")
    assert service.confirm_start(plan["plan_id"], plan["confirmation_token"]) == {
        "task_id": "task-real"
    }
    with pytest.raises(PermissionError, match="already used"):
        service.confirm_start(plan["plan_id"], plan["confirmation_token"])
    tasks.shutdown()
    assert started == ["plan-safe"]
