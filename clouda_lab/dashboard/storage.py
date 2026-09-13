from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .security import browser_safe
from .settings import LabSettings

if TYPE_CHECKING:
    from .tasks import OperationTaskService


def _bounded_tree_bytes(root: Path, managed_root: Path) -> int:
    if not root.exists():
        return 0
    resolved = root.resolve()
    resolved.relative_to(managed_root.resolve())
    if root.is_symlink():
        raise PermissionError("managed storage refuses symbolic links")
    total = 0
    for item in root.rglob("*"):
        if item.is_symlink():
            raise PermissionError("managed storage refuses symbolic links")
        item.resolve().relative_to(resolved)
        if item.is_file():
            total += item.stat().st_size
    return total


class StorageService:
    """Read-only capacity accounting over server-owned Clouda roots."""

    def __init__(
        self,
        settings: LabSettings,
        *,
        tasks: OperationTaskService | None = None,
    ) -> None:
        self.settings = settings
        self.tasks = tasks

    def _categories(self) -> list[dict[str, Any]]:
        roots = (
            (
                "dataset_downloads",
                "Dataset downloads",
                self.settings.dataset_downloads_root,
            ),
            ("dataset_imports", "Dataset imports", self.settings.dataset_imports_root),
            ("models", "Model assets", self.settings.models_root),
            ("results", "Results Store", self.settings.results_root),
            ("runs", "Runs and Lab state", self.settings.runs_root),
            ("benchmarks", "Benchmark metadata", self.settings.benchmarks_root),
        )
        return [
            {
                "id": category_id,
                "label": label,
                "bytes": _bounded_tree_bytes(root, self.settings.repo_root),
                "location": root.relative_to(self.settings.repo_root).as_posix(),
                "present": root.exists(),
            }
            for category_id, label, root in roots
        ]

    def _task_state(self) -> dict[str, int]:
        records = self.tasks.list_tasks() if self.tasks is not None else []
        pending = [
            item for item in records if item.get("status") in {"QUEUED", "RUNNING"}
        ]
        downloads = [item for item in records if item.get("kind") == "DATASET_DOWNLOAD"]
        return {
            "total": len(records),
            "pending": len(pending),
            "downloads": len(downloads),
            "pending_downloads": sum(item in pending for item in downloads),
        }

    def status(self) -> dict[str, Any]:
        self.settings.repo_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.settings.repo_root)
        ratio = usage.free / usage.total if usage.total else 0.0
        disk_status = "AVAILABLE"
        if ratio < 0.05:
            disk_status = "CRITICAL"
        elif ratio < 0.10:
            disk_status = "LOW"
        payload = {
            "schema_version": "clouda.lab.storage.v1",
            "categories": self._categories(),
            "disk": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_ratio": ratio,
                "status": disk_status,
            },
            "tasks": self._task_state(),
            "network_policy": {
                "default": "OFFLINE",
                "automatic_downloads": False,
                "external_providers": False,
                "explicit_dataset_sample_downloads": True,
                "confirmation_required": True,
            },
        }
        return browser_safe(payload, (self.settings.repo_root,))


__all__ = ["StorageService"]
