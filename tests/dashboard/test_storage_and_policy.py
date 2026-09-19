from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_storage_reports_managed_categories_and_free_space(tmp_path: Path):
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.storage import StorageService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    (settings.dataset_downloads_root / "sample").mkdir(parents=True)
    (settings.dataset_downloads_root / "sample" / "page.bin").write_bytes(b"123")
    (settings.models_root / "model").mkdir(parents=True)
    (settings.models_root / "model" / "config.json").write_bytes(b"{}")
    (settings.runs_root / "run-a").mkdir(parents=True)
    (settings.runs_root / "run-a" / "metrics.jsonl").write_bytes(b"1234")

    status = StorageService(settings).status()

    categories = {item["id"]: item for item in status["categories"]}
    assert categories["dataset_downloads"]["bytes"] == 3
    assert categories["models"]["bytes"] == 2
    assert categories["runs"]["bytes"] >= 4
    assert categories["dataset_downloads"]["location"] == "data/downloads"
    assert status["disk"]["free_bytes"] > 0
    assert status["disk"]["status"] in {"AVAILABLE", "LOW", "CRITICAL"}
    assert str(tmp_path) not in repr(status)


def test_storage_refuses_symlinks_in_managed_roots(tmp_path: Path):
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.storage import StorageService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    settings.models_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, settings.models_root / "escape", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(PermissionError, match="symbolic links"):
        StorageService(settings).status()


def test_network_policy_is_explicit_and_task_counts_are_real(tmp_path: Path):
    from clouda_lab.dashboard.settings import LabSettings
    from clouda_lab.dashboard.storage import StorageService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = LabSettings.from_repo(tmp_path / "repo")
    settings.repo_root.mkdir()
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    queued = tasks.enqueue("DATASET_DOWNLOAD", "source-a", lambda context: {"ok": True})
    tasks.shutdown()

    status = StorageService(settings, tasks=tasks).status()

    assert status["tasks"]["total"] == 1
    assert status["tasks"]["downloads"] == 1
    assert status["network_policy"] == {
        "default": "OFFLINE",
        "automatic_downloads": False,
        "external_providers": False,
        "explicit_dataset_sample_downloads": True,
        "confirmation_required": True,
    }
    assert queued["kind"] == "DATASET_DOWNLOAD"
