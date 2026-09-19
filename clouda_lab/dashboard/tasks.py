from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from clouda_training.experiments.io import atomic_write_json, read_json

from .security import browser_safe, safe_identifier

TASK_SCHEMA_VERSION = "clouda.lab.task.v1"
ALLOWED_OPERATION_KINDS = frozenset(
    {
        "DATASET_DOWNLOAD",
        "DATASET_VERIFY",
        "DATASET_IMPORT_VALIDATE",
        "DATASET_IMPORT_REGISTER",
        "DATASET_QUALITY",
        "DATASET_DEDUP",
        "DATASET_DERIVE",
        "DATASET_REMOVE",
        "MODEL_VERIFY",
        "MODEL_REMOVE",
        "TRAINING_START",
        "TRAINING_RESUME",
        "HARDWARE_VALIDATE",
    }
)


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OperationCancelled(RuntimeError):
    code = "operation_cancelled"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskContext:
    def __init__(self, service: OperationTaskService, task_id: str) -> None:
        self._service = service
        self.task_id = task_id

    def cancelled(self) -> bool:
        return bool(self._service._read(self.task_id).get("cancel_requested"))

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise OperationCancelled("Operation cancelled by the user")

    def update(
        self,
        *,
        phase: str | None = None,
        progress: float | None = None,
        completed_bytes: int | None = None,
        total_bytes: int | None = None,
        speed_bytes_per_second: float | None = None,
        checksum_status: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        if phase is not None:
            changes["phase"] = str(phase)
        if progress is not None:
            changes["progress"] = max(0.0, min(float(progress), 1.0))
        if completed_bytes is not None:
            changes["completed_bytes"] = max(0, int(completed_bytes))
        if total_bytes is not None:
            changes["total_bytes"] = max(0, int(total_bytes))
        if speed_bytes_per_second is not None:
            changes["speed_bytes_per_second"] = max(0.0, float(speed_bytes_per_second))
        if checksum_status is not None:
            changes["checksum_status"] = str(checksum_status)
        if detail is not None:
            changes["detail"] = str(detail)
        return self._service._update(self.task_id, **changes)


TaskWorker = Callable[[TaskContext], dict[str, Any] | None]


class OperationTaskService:
    """Persistent, closed-enum task execution for Clouda Lab operations."""

    def __init__(
        self,
        root: str | Path,
        *,
        browser_roots: tuple[Path, ...],
        max_workers: int = 2,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.browser_roots = tuple(Path(item).resolve() for item in browser_roots)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(max_workers), 4)),
            thread_name_prefix="clouda-lab-operation",
        )
        self._futures: dict[str, Future[Any]] = {}
        self._recover_unfinished()

    def _path(self, task_id: str) -> Path:
        return self.root / f"{safe_identifier(task_id)}.json"

    def _read(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(task_id)
            if not path.is_file():
                raise KeyError(f"Unknown task: {task_id}")
            return read_json(path)

    def _write(self, payload: dict[str, Any]) -> None:
        # Task records are a browser-facing audit surface, not an execution
        # descriptor. Persist the same redacted representation we return so a
        # worker exception can never leave secrets or private paths on disk.
        safe_payload = browser_safe(payload, self.browser_roots)
        atomic_write_json(self._path(str(safe_payload["task_id"])), safe_payload)

    def _update(self, task_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            payload = self._read(task_id)
            payload.update(changes)
            payload["updated_at"] = _utc_now()
            self._write(payload)
            return browser_safe(payload, self.browser_roots)

    def _recover_unfinished(self) -> None:
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    payload = read_json(path)
                except Exception:
                    continue
                if payload.get("status") not in {
                    TaskStatus.QUEUED.value,
                    TaskStatus.RUNNING.value,
                }:
                    continue
                payload.update(
                    {
                        "status": TaskStatus.FAILED.value,
                        "phase": TaskStatus.FAILED.value,
                        "completed_at": _utc_now(),
                        "updated_at": _utc_now(),
                        "error": {
                            "code": "service_restarted",
                            "message": "The local service restarted before completion",
                        },
                    }
                )
                self._write(payload)

    def enqueue(
        self,
        kind: str,
        target_id: str,
        worker: TaskWorker,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in ALLOWED_OPERATION_KINDS:
            raise ValueError(f"unsupported operation kind: {kind}")
        safe_identifier(target_id)
        task_id = f"task-{uuid.uuid4().hex}"
        now = _utc_now()
        payload = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "kind": kind,
            "target_id": target_id,
            "status": TaskStatus.QUEUED.value,
            "phase": "PENDING",
            "progress": 0.0,
            "completed_bytes": 0,
            "total_bytes": None,
            "speed_bytes_per_second": None,
            "checksum_status": "NOT_STARTED",
            "detail": None,
            "metadata": dict(metadata or {}),
            "result": None,
            "error": None,
            "cancel_requested": False,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
        }
        with self._lock:
            self._write(payload)
            self._futures[task_id] = self._executor.submit(self._run, task_id, worker)
        return browser_safe(payload, self.browser_roots)

    def _run(self, task_id: str, worker: TaskWorker) -> None:
        context = TaskContext(self, task_id)
        try:
            context.raise_if_cancelled()
            self._update(
                task_id,
                status=TaskStatus.RUNNING.value,
                phase=TaskStatus.RUNNING.value,
                started_at=_utc_now(),
            )
            result = worker(context) or {}
            context.raise_if_cancelled()
            self._update(
                task_id,
                status=TaskStatus.COMPLETED.value,
                phase=TaskStatus.COMPLETED.value,
                progress=1.0,
                result=result,
                completed_at=_utc_now(),
            )
        except OperationCancelled as exc:
            self._update(
                task_id,
                status=TaskStatus.CANCELLED.value,
                phase=TaskStatus.CANCELLED.value,
                completed_at=_utc_now(),
                error={"code": exc.code, "message": str(exc)},
            )
        except Exception as exc:
            self._update(
                task_id,
                status=TaskStatus.FAILED.value,
                phase=TaskStatus.FAILED.value,
                completed_at=_utc_now(),
                error={"code": type(exc).__name__, "message": str(exc)},
            )

    def get_task(self, task_id: str) -> dict[str, Any]:
        return browser_safe(self._read(task_id), self.browser_roots)

    def list_tasks(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                payload = read_json(path)
            except Exception:
                continue
            if payload.get("schema_version") != TASK_SCHEMA_VERSION:
                continue
            if kind is not None and payload.get("kind") != kind:
                continue
            records.append(browser_safe(payload, self.browser_roots))
        return sorted(
            records, key=lambda item: str(item.get("created_at", "")), reverse=True
        )

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._read(task_id)
            if payload.get("status") in {
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
            }:
                raise ValueError("completed task cannot be cancelled")
            payload["cancel_requested"] = True
            payload["updated_at"] = _utc_now()
            future = self._futures.get(task_id)
            if future is not None and future.cancel():
                payload.update(
                    {
                        "status": TaskStatus.CANCELLED.value,
                        "phase": TaskStatus.CANCELLED.value,
                        "completed_at": _utc_now(),
                        "error": {
                            "code": OperationCancelled.code,
                            "message": "Operation cancelled before it started",
                        },
                    }
                )
            self._write(payload)
            return browser_safe(payload, self.browser_roots)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = [
    "ALLOWED_OPERATION_KINDS",
    "OperationCancelled",
    "OperationTaskService",
    "TaskContext",
    "TaskStatus",
]
