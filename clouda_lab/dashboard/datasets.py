from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from clouda_data.datasets.downloader import (
    DEFAULT_SAMPLE_LIMIT,
    DownloadCancelled,
    DownloadResult,
    disk_free_bytes,
    download_dataset_sample,
    result_to_dict,
    verify_download,
)
from clouda_data.datasets.registry import (
    assert_source_download_allowed,
    estimate_source_download,
    get_source,
    list_sources,
    verify_license,
)
from clouda_data.ingestion.workflow import (
    ingest_source_manifest,
    plan_to_dict,
    validate_source_manifest_file,
)
from clouda_data.locations import default_foundation_registry_path

from .confirmations import ConfirmationStore
from .security import browser_safe, safe_identifier
from .settings import LabSettings
from .tasks import OperationCancelled, OperationTaskService, TaskContext

DownloadFunction = Callable[..., DownloadResult]

_LICENSE_STATES = {
    "approved": "APPROVED",
    "approved_with_conditions": "APPROVED_WITH_CONDITIONS",
    "research_only": "RESEARCH_ONLY",
    "unclear_license": "UNCLEAR_LICENSE",
    "rejected": "BLOCKED",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError("managed dataset operations refuse symbolic links")
        if path.is_file():
            total += path.stat().st_size
    return total


class DatasetOperationsService:
    """Operational facade over canonical dataset acquisition and ingestion."""

    def __init__(
        self,
        settings: LabSettings,
        tasks: OperationTaskService,
        *,
        registry_path: str | Path | None = None,
        download_fn: DownloadFunction = download_dataset_sample,
    ) -> None:
        self.settings = settings
        self.tasks = tasks
        self.registry_path = Path(
            registry_path or default_foundation_registry_path()
        ).resolve()
        self.download_fn = download_fn
        self.confirmations = ConfirmationStore(
            settings.confirmations_root,
            browser_roots=(settings.repo_root,),
        )

    def _source(self, source_id: str) -> dict[str, Any]:
        safe_identifier(source_id)
        return get_source(source_id, self.registry_path)

    def list_sources(self) -> list[dict[str, Any]]:
        return [
            self._source_state(source) for source in list_sources(self.registry_path)
        ]

    def source_detail(self, source_id: str) -> dict[str, Any]:
        return self._source_state(self._source(source_id))

    def _source_state(self, source: dict[str, Any]) -> dict[str, Any]:
        source_id = str(source["source_id"])
        license_result = verify_license(source)
        local_root = self.settings.dataset_downloads_root / source_id
        manifest = self.settings.download_manifests_root / f"{source_id}.json"
        download_available = bool(
            license_result["sample_download_allowed"]
            and source.get("sample_assets")
            and not source.get("requires_authentication")
            and not source.get("requires_form")
            and not source.get("requires_account")
        )
        reasons = list(license_result.get("reasons", []))
        if not source.get("sample_assets"):
            reasons.append("no canonical sample asset is registered")
        if source.get("requires_authentication") or source.get("requires_account"):
            reasons.append("authentication or an account is required")
        if source.get("requires_form"):
            reasons.append("terms or a form must be completed outside Clouda Lab")
        state = {
            "source_id": source_id,
            "name": source.get("name"),
            "classification": source.get("classification"),
            "license": source.get("license"),
            "license_verified": license_result["license_verified"],
            "license_state": _LICENSE_STATES.get(
                str(source.get("classification")), "MANUAL_REVIEW_REQUIRED"
            ),
            "license_reasons": reasons,
            "attribution_requirements": source.get("attribution_requirements"),
            "commercial_restrictions": source.get("commercial_use_status"),
            "redistribution": source.get("redistribution_status"),
            "training_allowed": license_result["commercial_use_allowed"],
            "benchmark_allowed": False,
            "benchmark_reason": "evaluation permission is not recorded",
            "estimated_sample_bytes": source.get("sample_size_bytes"),
            "estimated_dataset_bytes": source.get("dataset_size_bytes"),
            "provider": source.get("download_method"),
            "authentication_required": bool(
                source.get("requires_authentication") or source.get("requires_account")
            ),
            "local_state": "DOWNLOADED" if local_root.is_dir() else "NOT_DOWNLOADED",
            "download_manifest_present": manifest.is_file(),
            "download": {
                "available": download_available,
                "scope": "sample_only" if download_available else None,
                "reason": None if download_available else "; ".join(reasons),
            },
        }
        return browser_safe(state, (self.settings.repo_root,))

    def create_download_plan(self, source_id: str) -> dict[str, Any]:
        source = self._source(source_id)
        estimate = estimate_source_download(source, sample_only=True)
        estimated = int(estimate.get("estimated_bytes") or 0)
        max_bytes = max(estimated, DEFAULT_SAMPLE_LIMIT if not estimated else estimated)
        assert_source_download_allowed(source, full_dataset=False, max_bytes=max_bytes)
        if not source.get("sample_assets"):
            raise PermissionError("download blocked: no canonical sample assets")
        destination = self.settings.dataset_downloads_root / source_id
        details = {
            "item": source.get("name"),
            "item_type": "dataset",
            "provider": source.get("download_method"),
            "license": source.get("license"),
            "classification": source.get("classification"),
            "estimated_bytes": estimated or None,
            "max_bytes": max_bytes,
            "expected_files": [
                str(asset.get("filename") or asset.get("path") or "registered asset")
                for asset in source.get("sample_assets", [])
            ],
            "destination": destination.relative_to(self.settings.repo_root).as_posix(),
            "disk_free_bytes": disk_free_bytes(self.settings.repo_root),
            "authentication_required": bool(
                source.get("requires_authentication") or source.get("requires_account")
            ),
            "usage_restrictions": list(source.get("license_notes", [])),
        }
        if details["disk_free_bytes"] < max_bytes:
            raise OSError("insufficient free disk space for canonical download limit")
        return self.confirmations.issue("DATASET_DOWNLOAD", source_id, details)

    def start_download(self, plan_id: str, confirmation: str) -> dict[str, Any]:
        plan = self.confirmations.consume(
            plan_id, confirmation, kind="DATASET_DOWNLOAD"
        )
        source_id = str(plan["target_id"])

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(
                phase="CHECKING",
                total_bytes=plan.get("estimated_bytes"),
                detail="Checking canonical source and destination",
            )

            def progress(completed: int, total: int | None) -> None:
                context.raise_if_cancelled()
                context.update(
                    phase="DOWNLOADING",
                    completed_bytes=completed,
                    total_bytes=total,
                    progress=(completed / total if total else 0.0),
                )

            try:
                result = self.download_fn(
                    source_id,
                    project_root=self.settings.repo_root,
                    registry_path=self.registry_path,
                    max_bytes=int(plan["max_bytes"]),
                    progress_callback=progress,
                    cancellation_check=context.cancelled,
                )
            except DownloadCancelled as exc:
                raise OperationCancelled(str(exc)) from exc
            context.update(phase="VERIFYING", detail="Verifying downloaded files")
            if not result.ok:
                raise RuntimeError(
                    "; ".join(result.issues) or "dataset download failed"
                )
            return result_to_dict(result)

        return self.tasks.enqueue(
            "DATASET_DOWNLOAD",
            source_id,
            worker,
            metadata={
                "item": plan.get("item"),
                "item_type": "dataset",
                "provider": plan.get("provider"),
                "destination": plan.get("destination"),
            },
        )

    def verify_source(self, source_id: str) -> dict[str, Any]:
        self._source(source_id)

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(phase="VERIFYING")
            result = verify_download(source_id, project_root=self.settings.repo_root)
            if not result.ok:
                raise RuntimeError("; ".join(result.issues))
            return result_to_dict(result)

        return self.tasks.enqueue("DATASET_VERIFY", source_id, worker)

    def list_imports(self) -> list[dict[str, Any]]:
        root = self.settings.dataset_imports_root
        if not root.is_dir():
            return []
        records = []
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_dir():
                continue
            try:
                safe_identifier(path.name)
            except ValueError:
                continue
            records.append(
                {
                    "import_id": path.name,
                    "manifest_present": (path / "source_manifest.json").is_file(),
                }
            )
        return records

    def _import_manifest(self, import_id: str) -> Path:
        safe_identifier(import_id)
        root = self.settings.dataset_imports_root.resolve()
        candidate = (root / import_id / "source_manifest.json").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError("import escaped the managed root") from exc
        if not candidate.is_file():
            raise FileNotFoundError("managed import source_manifest.json is missing")
        return candidate

    def validate_import(self, import_id: str) -> dict[str, Any]:
        plan = validate_source_manifest_file(
            self._import_manifest(import_id), self.settings.repo_root
        )
        return browser_safe(plan_to_dict(plan), (self.settings.repo_root,))

    def register_import(self, import_id: str) -> dict[str, Any]:
        manifest = self._import_manifest(import_id)

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(phase="VERIFYING")
            for relative in (
                "data/raw/documents",
                "data/raw/pages",
                "data/raw/ground_truth",
                "data/raw/layout_annotations",
                "data/manifests",
                "outputs/reports",
            ):
                (self.settings.repo_root / relative).mkdir(parents=True, exist_ok=True)
            validation = ingest_source_manifest(
                manifest, self.settings.repo_root, dry_run=True
            )
            if not validation.ok:
                raise ValueError("managed import failed canonical validation")
            context.raise_if_cancelled()
            context.update(phase="REGISTERING")
            result = ingest_source_manifest(
                manifest, self.settings.repo_root, dry_run=False
            )
            return plan_to_dict(result)

        return self.tasks.enqueue("DATASET_IMPORT_REGISTER", import_id, worker)

    def create_removal_plan(self, source_id: str) -> dict[str, Any]:
        self._source(source_id)
        root = self.settings.dataset_downloads_root / source_id
        if not root.is_dir():
            raise FileNotFoundError("managed dataset copy is not present")
        return self.confirmations.issue(
            "DATASET_REMOVE",
            source_id,
            {
                "item": source_id,
                "item_type": "dataset",
                "bytes": _tree_bytes(root),
                "destination": root.relative_to(self.settings.repo_root).as_posix(),
                "recoverable": True,
            },
        )

    def remove_download(self, plan_id: str, confirmation: str) -> dict[str, Any]:
        plan = self.confirmations.consume(plan_id, confirmation, kind="DATASET_REMOVE")
        source_id = str(plan["target_id"])
        root = self.settings.dataset_downloads_root / source_id
        manifest = self.settings.download_manifests_root / f"{source_id}.json"

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(phase="REMOVING")
            destination = (
                self.settings.trash_root
                / f"dataset-{source_id}-{_utc_stamp()}-{uuid.uuid4().hex[:8]}"
            )
            destination.mkdir(parents=True, exist_ok=False)
            if root.is_dir():
                shutil.move(str(root), destination / "files")
            if manifest.is_file():
                shutil.move(str(manifest), destination / "download-manifest.json")
            return {
                "source_id": source_id,
                "recoverable_trash": str(destination),
                "bytes": plan.get("bytes", 0),
            }

        return self.tasks.enqueue("DATASET_REMOVE", source_id, worker)


__all__ = ["DatasetOperationsService"]
