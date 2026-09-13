from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest


def _wait(service, task_id: str, statuses: set[str], timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if task["status"] in statuses:
            return task
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach {statuses}")


def test_task_records_progress_result_and_survives_service_restart(tmp_path: Path):
    from clouda_lab.dashboard.tasks import OperationTaskService

    root = tmp_path / "tasks"
    service = OperationTaskService(root, browser_roots=(tmp_path,), max_workers=1)

    def worker(context):
        context.update(
            phase="VERIFYING",
            progress=0.5,
            completed_bytes=5,
            total_bytes=10,
            detail="checking",
        )
        return {"artifact": str(tmp_path / "managed" / "result.json")}

    task = service.enqueue("DATASET_VERIFY", "rasam_dataset", worker)
    completed = _wait(service, task["task_id"], {"COMPLETED"})
    service.shutdown()

    assert completed["phase"] == "COMPLETED"
    assert completed["progress"] == 1.0
    assert completed["completed_bytes"] == 5
    assert completed["total_bytes"] == 10
    assert completed["result"] == {"artifact": "managed/result.json"}

    reopened = OperationTaskService(root, browser_roots=(tmp_path,), max_workers=1)
    assert reopened.get_task(task["task_id"])["status"] == "COMPLETED"
    reopened.shutdown()


def test_task_cancellation_is_cooperative_and_persisted(tmp_path: Path):
    from clouda_lab.dashboard.tasks import OperationCancelled, OperationTaskService

    entered = threading.Event()
    service = OperationTaskService(
        tmp_path / "tasks", browser_roots=(tmp_path,), max_workers=1
    )

    def worker(context):
        entered.set()
        while True:
            context.raise_if_cancelled()
            time.sleep(0.01)

    task = service.enqueue("DATASET_DOWNLOAD", "rasam_dataset", worker)
    assert entered.wait(timeout=2)
    cancellation = service.cancel(task["task_id"])
    cancelled = _wait(service, task["task_id"], {"CANCELLED"})
    service.shutdown()

    assert cancellation["cancel_requested"] is True
    assert cancelled["status"] == "CANCELLED"
    assert cancelled["error"]["code"] == OperationCancelled.code


def test_task_errors_are_sanitized_and_unknown_kinds_are_rejected(tmp_path: Path):
    from clouda_lab.dashboard.tasks import OperationTaskService

    service = OperationTaskService(
        tmp_path / "tasks", browser_roots=(tmp_path,), max_workers=1
    )

    def worker(_context):
        raise RuntimeError(
            f"token=super-secret failed at {tmp_path / 'private' / 'file.bin'}"
        )

    task = service.enqueue("MODEL_VERIFY", "hunyuanocr15", worker)
    failed = _wait(service, task["task_id"], {"FAILED"})

    with pytest.raises(ValueError, match="unsupported operation kind"):
        service.enqueue("RUN_BROWSER_COMMAND", "target", worker)
    service.shutdown()

    assert "super-secret" not in repr(failed)
    assert str(tmp_path) not in repr(failed)
    assert failed["error"]["code"] == "RuntimeError"
    assert failed["error"]["message"].startswith("token=[REDACTED]")


def test_unfinished_task_is_failed_closed_after_restart(tmp_path: Path):
    from clouda_lab.dashboard.tasks import OperationTaskService
    from clouda_training.experiments.io import atomic_write_json

    root = tmp_path / "tasks"
    root.mkdir()
    atomic_write_json(
        root / "unfinished.json",
        {
            "schema_version": "clouda.lab.task.v1",
            "task_id": "unfinished",
            "kind": "DATASET_DOWNLOAD",
            "target_id": "rasam_dataset",
            "status": "RUNNING",
            "phase": "DOWNLOADING",
            "progress": 0.25,
            "cancel_requested": False,
        },
    )

    service = OperationTaskService(root, browser_roots=(tmp_path,), max_workers=1)
    recovered = service.get_task("unfinished")
    service.shutdown()

    assert recovered["status"] == "FAILED"
    assert recovered["error"]["code"] == "service_restarted"
    assert recovered["progress"] == 0.25
