from __future__ import annotations

import shutil
from pathlib import Path


def _service(tmp_path: Path):
    from clouda_lab.dashboard.benchmarks import BenchmarkWorkspaceService
    from clouda_lab.dashboard.models import ModelCatalogService
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    source = Path.cwd() / "benchmarks" / "ocr_arabic"
    target = settings.benchmarks_root / "ocr_arabic"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    models = ModelCatalogService(settings, tasks)
    return BenchmarkWorkspaceService(settings, models), tasks


def test_inventory_and_results_use_published_actual_metadata(tmp_path: Path):
    service, tasks = _service(tmp_path)
    inventory = service.inventory()
    results = service.results()
    tasks.shutdown()

    assert len(inventory["models"]) == 8
    assert inventory["benchmark"]["benchmark_id"] == "clouda-ocr-arabic-177-v1"
    assert inventory["dataset"]["purpose"] == "PROTECTED_EVALUATION"
    assert inventory["dataset"]["training_allowed"] is False
    assert [item["name"] for item in results["ranked"][:2]] == [
        "HunyuanOCR-1.5",
        "MBZUAI/AIN-7B",
    ]
    assert {item["name"] for item in results["unranked"]} == {
        "dots.mocr",
        "PaddleOCR-VL-1.6",
    }


def test_plan_identity_is_deterministic_and_execution_is_truthfully_blocked(
    tmp_path: Path,
):
    service, tasks = _service(tmp_path)
    selected = ["qwen3-vl-4b-instruct", "hunyuanocr-1-5"]
    first = service.create_plan({"model_ids": selected})
    second = service.create_plan({"model_ids": list(reversed(selected))})
    tasks.shutdown()

    assert first["plan_id"] == second["plan_id"]
    assert first["execution"]["available"] is False
    assert "runner" in first["execution"]["reason"].lower()
    assert first["dataset"]["training_allowed"] is False
    assert service.get_plan(first["plan_id"])["plan_id"] == first["plan_id"]


def test_pairwise_comparison_never_ranks_incomplete_runs(tmp_path: Path):
    service, tasks = _service(tmp_path)
    comparison = service.compare("hunyuanocr-1-5", "dots-mocr")
    tasks.shutdown()

    assert comparison["left"]["status"] == "COMPLETE"
    assert comparison["right"]["status"] == "PARTIAL"
    assert comparison["winner"] is None
    assert comparison["reason"] == "Both runs must be complete and rankable"
