from __future__ import annotations

import time
from pathlib import Path

import pytest


def _wait(service, task_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return task
        time.sleep(0.01)
    raise AssertionError("operation did not finish")


def test_model_catalog_contains_complete_published_inventory_and_adapter_split():
    from clouda_lab.dashboard.models import ModelCatalogService
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(Path.cwd())
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = ModelCatalogService(settings, tasks)
    models = service.list_models()
    tasks.shutdown()

    assert [item["name"] for item in models] == [
        "HunyuanOCR-1.5",
        "MBZUAI/AIN-7B",
        "Qari OCR 0.4.0",
        "Qwen3-VL-4B-Instruct",
        "DeepSeek-OCR-2",
        "Arabic Nougat Large",
        "dots.mocr",
        "PaddleOCR-VL-1.6",
    ]
    trainable = {item["name"] for item in models if item["training_supported"]}
    assert trainable == {"HunyuanOCR-1.5", "Qwen3-VL-4B-Instruct"}
    hunyuan = next(item for item in models if item["name"] == "HunyuanOCR-1.5")
    assert hunyuan["training_adapter"] == "hunyuanocr15_sft"
    assert hunyuan["benchmark"]["status"] == "COMPLETE"
    assert hunyuan["download"]["available"] is False
    assert "approved model download manifest" in hunyuan["download"]["reason"]
    dots = next(item for item in models if item["name"] == "dots.mocr")
    assert dots["benchmark"]["rankable"] is False
    assert dots["training_supported"] is False


def test_model_assets_are_configured_only_by_managed_id_and_verified_canonically(
    tmp_path: Path,
):
    from clouda_lab.dashboard.models import ModelCatalogService
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    source_benchmarks = Path.cwd() / "benchmarks" / "ocr_arabic"
    target_benchmarks = settings.benchmarks_root / "ocr_arabic"
    target_benchmarks.mkdir(parents=True)
    for name in ("models.csv", "results.csv"):
        (target_benchmarks / name).write_bytes((source_benchmarks / name).read_bytes())
    asset_root = settings.models_root / "qwen-local"
    asset_root.mkdir(parents=True)
    (asset_root / "config.json").write_text("{}", encoding="utf-8")
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = ModelCatalogService(settings, tasks)

    configured = service.configure_assets("qwen3-vl-4b-instruct", "qwen-local")
    assert configured["asset_status"] == "CONFIGURED_UNVERIFIED"
    assert configured["asset_id"] == "qwen-local"
    assert str(tmp_path) not in repr(configured)
    with pytest.raises(ValueError, match="canonical identifier"):
        service.configure_assets("qwen3-vl-4b-instruct", "../private")

    task = service.verify_assets("qwen3-vl-4b-instruct")
    finished = _wait(tasks, task["task_id"])
    detail = service.get_model("qwen3-vl-4b-instruct")
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert finished["result"]["asset_check"]["status"] == "PASS"
    assert detail["assets"]["status"] == "VERIFIED"
    assert detail["assets"]["image_processor_status"] == "NOT_SEPARATELY_VERIFIED"
    assert detail["assets"]["text_encoder_status"] == "NOT_SEPARATELY_VERIFIED"


def test_dependency_remediation_is_display_only_and_uses_fixed_project_group():
    from clouda_lab.dashboard.models import ModelCatalogService
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(Path.cwd())
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    qwen = ModelCatalogService(settings, tasks).get_model("qwen3-vl-4b-instruct")
    tasks.shutdown()

    remediation = qwen["dependencies"]["remediation"]
    assert remediation["installation_endpoint"] is False
    assert remediation["optional_group"] == "training-torch"
    assert remediation["command"] == "python -m pip install -e .[training-torch]"
    assert remediation["complete_for_adapter"] is False
    assert "transformers" in " ".join(remediation["uncovered_requirements"])


def test_model_removal_is_confirmed_and_recoverable(tmp_path: Path):
    from clouda_lab.dashboard.models import ModelCatalogService
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    source_benchmarks = Path.cwd() / "benchmarks" / "ocr_arabic"
    target_benchmarks = settings.benchmarks_root / "ocr_arabic"
    target_benchmarks.mkdir(parents=True)
    for name in ("models.csv", "results.csv"):
        (target_benchmarks / name).write_bytes((source_benchmarks / name).read_bytes())
    assets = settings.models_root / "qwen-local"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text("{}", encoding="utf-8")
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = ModelCatalogService(settings, tasks)
    service.configure_assets("qwen3-vl-4b-instruct", "qwen-local")

    plan = service.create_removal_plan("qwen3-vl-4b-instruct")
    assert plan["bytes"] == 2
    with pytest.raises(PermissionError, match="confirmation"):
        service.remove_assets(plan["plan_id"], "wrong")
    task = service.remove_assets(plan["plan_id"], plan["confirmation_token"])
    finished = _wait(tasks, task["task_id"])
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert not assets.exists()
    assert any(settings.trash_root.rglob("config.json"))
