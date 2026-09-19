from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image

from clouda_data.datasets.downloader import DownloadResult, DownloadedFile


def _wait(service, task_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return task
        time.sleep(0.01)
    raise AssertionError("operation did not finish")


def _settings(tmp_path: Path):
    from clouda_lab.dashboard.settings import LabSettings

    repo = tmp_path / "repo"
    repo.mkdir()
    return LabSettings.from_repo(repo)


def _registry(path: Path) -> Path:
    payload = {
        "sources": [
            {
                "source_id": "allowed_source",
                "name": "Allowed Source",
                "classification": "approved_with_conditions",
                "license": "Apache-2.0",
                "license_verified": True,
                "commercial_use_status": "allowed",
                "redistribution_status": "allowed",
                "attribution_requirements": "preserve notice",
                "sample_size_bytes": 16,
                "dataset_size_bytes": 1024,
                "download_method": "https",
                "requires_authentication": False,
                "requires_form": False,
                "requires_account": False,
                "sample_assets": [
                    {
                        "filename": "sample.txt",
                        "url": "https://datasets.example.invalid/sample.txt",
                        "size_bytes": 16,
                    }
                ],
                "metadata": {},
                "official_url": "https://datasets.example.invalid",
                "risk_level": "low",
                "verification_date": "2026-09-13",
                "license_notes": ["Attribution is required."],
            },
            {
                "source_id": "blocked_source",
                "name": "Blocked Source",
                "classification": "unclear_license",
                "license": "not stated",
                "license_verified": False,
                "commercial_use_status": "unknown",
                "redistribution_status": "unknown",
                "attribution_requirements": "unknown",
                "sample_size_bytes": 8,
                "download_method": "https",
                "sample_assets": [],
                "metadata": {},
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_source_catalog_has_complete_inventory_and_separate_license_states(
    tmp_path: Path,
):
    from clouda_lab.dashboard.datasets import DatasetOperationsService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = _settings(tmp_path)
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = DatasetOperationsService(settings, tasks)
    sources = service.list_sources()
    tasks.shutdown()

    assert len(sources) == 13
    rasam = next(item for item in sources if item["source_id"] == "rasam_dataset")
    assert rasam["license_state"] == "APPROVED_WITH_CONDITIONS"
    assert rasam["training_allowed"] is True
    assert rasam["benchmark_allowed"] is False
    assert rasam["benchmark_reason"] == "evaluation permission is not recorded"
    assert rasam["download"]["available"] is True
    unclear = next(item for item in sources if item["source_id"] == "pats_a01")
    assert unclear["license_state"] == "UNCLEAR_LICENSE"
    assert unclear["training_allowed"] is False
    assert unclear["download"]["available"] is False


def test_download_requires_server_plan_exact_confirmation_and_is_single_use(
    tmp_path: Path,
):
    from clouda_lab.dashboard.datasets import DatasetOperationsService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = _settings(tmp_path)
    registry = _registry(settings.repo_root / "registry.json")
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )

    def fake_download(source_id: str, **kwargs):
        destination = settings.dataset_downloads_root / source_id / "sample.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"canonical sample")
        kwargs["progress_callback"](16, 16)
        return DownloadResult(
            source_id,
            True,
            False,
            [
                DownloadedFile(
                    url="https://datasets.example.invalid/sample.txt",
                    path=str(destination),
                    size_bytes=16,
                    checksum_sha256="a" * 64,
                )
            ],
        )

    service = DatasetOperationsService(
        settings, tasks, registry_path=registry, download_fn=fake_download
    )
    plan = service.create_download_plan("allowed_source")
    assert plan["estimated_bytes"] == 16
    assert plan["destination"] == "data/downloads/allowed_source"
    assert plan["expected_files"] == ["sample.txt"]
    assert plan["license"] == "Apache-2.0"
    assert plan["confirmation_token"]
    persisted_plan = next(settings.confirmations_root.glob("*.json")).read_text(
        encoding="utf-8"
    )
    assert plan["confirmation_token"] not in persisted_plan
    assert "confirmation_digest" in persisted_plan
    assert "confirmation_token" not in persisted_plan

    with pytest.raises(PermissionError, match="confirmation"):
        service.start_download(plan["plan_id"], "wrong")
    task = service.start_download(plan["plan_id"], plan["confirmation_token"])
    finished = _wait(tasks, task["task_id"])
    with pytest.raises(PermissionError, match="already used"):
        service.start_download(plan["plan_id"], plan["confirmation_token"])
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert finished["result"]["source_id"] == "allowed_source"
    assert finished["completed_bytes"] == 16


def test_download_plan_fails_closed_for_unclear_license(tmp_path: Path):
    from clouda_lab.dashboard.datasets import DatasetOperationsService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = _settings(tmp_path)
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = DatasetOperationsService(
        settings, tasks, registry_path=_registry(settings.repo_root / "registry.json")
    )

    with pytest.raises(PermissionError, match="license"):
        service.create_download_plan("blocked_source")
    tasks.shutdown()


def test_managed_import_validates_and_registers_through_canonical_ingestion(
    tmp_path: Path,
):
    from clouda_lab.dashboard.datasets import DatasetOperationsService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = _settings(tmp_path)
    source = settings.dataset_imports_root / "local_arabic"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(source / "page.png")
    (source / "gt.txt").write_text("نص عربي", encoding="utf-8")
    (source / "source_manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc_001",
                        "source_path": "page.png",
                        "source_type": "image",
                        "language": "ar",
                        "source_license": "operator-reviewed",
                    }
                ],
                "pages": [
                    {
                        "document_id": "doc_001",
                        "page_id": "doc_001_p001",
                        "page_number": 1,
                        "source_path": "page.png",
                        "source_type": "image",
                        "language": "ar",
                        "ground_truth_path": "gt.txt",
                        "source_license": "operator-reviewed",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = DatasetOperationsService(
        settings, tasks, registry_path=_registry(settings.repo_root / "registry.json")
    )

    assert service.list_imports() == [
        {"import_id": "local_arabic", "manifest_present": True}
    ]
    validation = service.validate_import("local_arabic")
    assert validation["ok"] is True
    task = service.register_import("local_arabic")
    finished = _wait(tasks, task["task_id"])
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert (settings.repo_root / "data/manifests/page_manifest.json").is_file()
    assert (source / "page.png").is_file()
    with pytest.raises(ValueError, match="canonical identifier"):
        service.validate_import("../outside")


def test_dataset_removal_moves_only_managed_copy_to_recoverable_trash(tmp_path: Path):
    from clouda_lab.dashboard.datasets import DatasetOperationsService
    from clouda_lab.dashboard.tasks import OperationTaskService

    settings = _settings(tmp_path)
    managed = settings.dataset_downloads_root / "allowed_source"
    managed.mkdir(parents=True)
    (managed / "data.bin").write_bytes(b"1234")
    manifest = settings.download_manifests_root / "allowed_source.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    tasks = OperationTaskService(
        settings.tasks_root, browser_roots=(settings.repo_root,), max_workers=1
    )
    service = DatasetOperationsService(
        settings, tasks, registry_path=_registry(settings.repo_root / "registry.json")
    )

    plan = service.create_removal_plan("allowed_source")
    assert plan["bytes"] == 4
    task = service.remove_download(plan["plan_id"], plan["confirmation_token"])
    finished = _wait(tasks, task["task_id"])
    tasks.shutdown()

    assert finished["status"] == "COMPLETED"
    assert not managed.exists()
    assert not manifest.exists()
    assert any(settings.trash_root.rglob("data.bin"))
