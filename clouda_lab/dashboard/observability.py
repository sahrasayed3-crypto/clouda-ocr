from __future__ import annotations

import csv
import json
from itertools import islice
from typing import Any, Mapping

from clouda_contracts.checksums import sha256_file
from clouda_data.results.service import ResultsService

from .catalog import DatasetCatalog
from .security import browser_safe, safe_identifier
from .settings import LabSettings
from .storage import StorageService
from .training import TrainingService


class ObservabilityService:
    def __init__(
        self,
        settings: LabSettings,
        catalog: DatasetCatalog,
        training: TrainingService,
        storage: StorageService | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.training = training
        self.storage = storage or StorageService(settings)
        self._doctor_snapshot: dict[str, Any] | None = None
        self._hardware_snapshot: dict[str, Any] | None = None

    def _results_service(self) -> ResultsService:
        return ResultsService(self.settings.results_root, read_only=True)

    def results(self, filters: Mapping[str, str | None]) -> dict[str, Any]:
        if not self.settings.results_root.is_dir():
            return {"datasets": [], "models": [], "runs": []}
        service = self._results_service()
        model_id = filters.get("model") or filters.get("model_id")
        dataset_id = filters.get("dataset") or filters.get("dataset_id")
        status = filters.get("status")
        runs = service.list_runs(
            model_id=str(model_id) if model_id else None,
            dataset_id=str(dataset_id) if dataset_id else None,
            status=str(status) if status else None,
        )
        experiment = str(filters.get("experiment") or "").strip().lower()
        benchmark = str(filters.get("benchmark") or "").strip().lower()
        date = str(filters.get("date") or "").strip()
        if experiment:
            runs = [
                run
                for run in runs
                if experiment
                in str(
                    (run.get("metadata") or {}).get("experiment_id")
                    or (run.get("metadata") or {}).get("experiment_name")
                    or run.get("config_hash")
                    or ""
                ).lower()
            ]
        if benchmark:
            runs = [
                run
                for run in runs
                if benchmark
                in str((run.get("metadata") or {}).get("benchmark_id") or "").lower()
            ]
        if date:
            runs = [
                run
                for run in runs
                if date
                in str(
                    run.get("started_at")
                    or run.get("ended_at")
                    or (run.get("metadata") or {}).get("created_at")
                    or ""
                )
            ]
        limit = max(1, min(int(filters.get("limit") or 200), 500))
        runs = runs[:limit]
        return browser_safe(
            {
                "datasets": service.list_datasets(),
                "models": service.list_models(),
                "runs": runs,
            },
            (self.settings.repo_root,),
        )

    def result_detail(self, run_id: str, *, metric_limit: int = 200) -> dict[str, Any]:
        safe_identifier(run_id)
        service = self._results_service()
        run = service.get_run(run_id)
        try:
            summary: dict[str, Any] = service.store.load_summary(run_id)
        except KeyError:
            summary = {"status": "NOT AVAILABLE"}
        bounded_metric_limit = max(1, min(metric_limit, 500))
        metric_records = list(
            islice(service.store.iter_metrics(run_id), bounded_metric_limit + 1)
        )
        metrics = [metric.to_dict() for metric in metric_records[:bounded_metric_limit]]
        integrity = service.verify(run_id)
        integrity["valid"] = bool(integrity.get("ok"))
        return browser_safe(
            {
                "run": run,
                "summary": summary,
                "metrics": metrics,
                "metrics_truncated": len(metric_records) > bounded_metric_limit,
                "integrity": integrity,
            },
            (self.settings.repo_root,),
        )

    def benchmarks(self, filters: Mapping[str, str | None]) -> dict[str, Any]:
        root = self.settings.benchmarks_root / "ocr_arabic"
        release_path = root / "release.json"
        results_path = root / "results.csv"
        manifest: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        if release_path.is_file():
            try:
                manifest = json.loads(release_path.read_text(encoding="utf-8"))
                manifest["artifact_sha256"] = sha256_file(release_path)
            except (OSError, json.JSONDecodeError):
                manifest = {
                    "status": "FAILED",
                    "reason": "Unreadable benchmark release",
                }
        if results_path.is_file():
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    row: dict[str, Any] = dict(raw)
                    row["status"] = str(row.get("status") or "NOT RUN").upper()
                    row["rankable"] = (
                        str(row.get("rankable", "false")).lower() == "true"
                    )
                    if row["status"] != "COMPLETE":
                        row["rankable"] = False
                    model_filter = filters.get("model")
                    status_filter = filters.get("status")
                    if (
                        model_filter
                        and str(model_filter).lower()
                        not in str(row.get("model", "")).lower()
                    ):
                        continue
                    if status_filter and row["status"] != str(status_filter).upper():
                        continue
                    rows.append(row)
                    limit = max(1, min(int(filters.get("limit") or 200), 500))
                    if len(rows) >= limit:
                        break
        return browser_safe(
            {
                "manifest": manifest,
                "results": rows,
                "automatic_execution": False,
                "source": "immutable local benchmark metadata",
            },
            (self.settings.repo_root,),
        )

    def latest_doctor(self) -> dict[str, Any]:
        return self._doctor_snapshot or {"status": "NOT RUN", "report": None}

    def run_doctor(self, *, deep: bool = False) -> dict[str, Any]:
        from clouda_data.doctor import collect_report

        report = collect_report(deep=deep).to_dict()
        self._doctor_snapshot = browser_safe(report, (self.settings.repo_root,))
        self._hardware_snapshot = None
        return self._doctor_snapshot

    def hardware(self) -> dict[str, Any]:
        if self._hardware_snapshot is not None:
            return self._hardware_snapshot
        report = self._doctor_snapshot or {}
        sections = {
            section.get("id"): section for section in report.get("sections", [])
        }
        gpu_section = sections.get("gpu", {"checks": []})
        if not gpu_section.get("checks"):
            from clouda_data.doctor.training import check_gpu

            gpu_section = check_gpu().to_dict()
        cuda = next(
            (
                check
                for check in gpu_section.get("checks", [])
                if check.get("id") == "gpu.cuda"
            ),
            None,
        )
        details = dict(cuda.get("details", {})) if cuda else {}
        available = bool(details.get("cuda_available", False))
        runtime = report.get("runtime", {})
        storage = sections.get("storage", {})
        self._hardware_snapshot = browser_safe(
            {
                "cpu": {
                    "machine": runtime.get("machine"),
                    "platform": runtime.get("platform"),
                    "status": "AVAILABLE" if runtime else "DEFERRED",
                },
                "storage": storage,
                "gpu": {
                    "available": available,
                    "status": "AVAILABLE" if available else "UNAVAILABLE",
                    "cuda_version": details.get("cuda_build"),
                    "torch_cuda_available": available,
                    "devices": details.get("devices", []),
                    "bf16_supported": details.get("bf16_supported"),
                    "reason": details.get("reason") or (cuda or {}).get("message"),
                },
                "logical_validation_separate": True,
                "source": "Clouda Doctor GPU capability check",
            },
            (self.settings.repo_root,),
        )
        return self._hardware_snapshot

    def overview(self) -> dict[str, Any]:
        datasets = self.catalog.list_datasets()
        runs = self.training.list_runs()
        models = self.training.list_models()
        statuses: dict[str, int] = {}
        for run in runs:
            status = str(run.get("status", "UNKNOWN"))
            statuses[status] = statuses.get(status, 0) + 1
        hardware = self.hardware()
        benchmark = self.benchmarks({})
        doctor = self.latest_doctor()
        storage = self.storage.status()
        return {
            "schema_version": "clouda.lab.overview.v1",
            "repository": (
                "AVAILABLE"
                if self.settings.repo_root.is_dir()
                and (self.settings.repo_root / "pyproject.toml").is_file()
                else "UNAVAILABLE"
            ),
            "datasets": len(datasets),
            "derived_datasets": sum(
                bool(item.get("parent_datasets")) for item in datasets
            ),
            "experiments": len(
                {
                    run.get("experiment_name")
                    for run in runs
                    if run.get("experiment_name")
                }
            ),
            "runs": statuses,
            "available_model_adapters": sum(bool(item["available"]) for item in models),
            "gpu": hardware["gpu"]["status"],
            "training_readiness": (
                "READY"
                if hardware["gpu"]["available"]
                and any(item["available"] for item in models)
                and any(
                    item.get("safety", {}).get("training_allowed") for item in datasets
                )
                else "DEFERRED"
            ),
            "doctor": doctor.get("overall_status", doctor.get("status", "NOT RUN")),
            "benchmark": {
                "manifest": benchmark.get("manifest"),
                "results": len(benchmark.get("results", [])),
            },
            "offline": "ACTIVE",
            "network_policy": storage["network_policy"],
            "storage": {
                "status": storage["disk"]["status"],
                "free_bytes": storage["disk"]["free_bytes"],
                "pending_tasks": storage["tasks"]["pending"],
                "pending_downloads": storage["tasks"]["pending_downloads"],
            },
        }


__all__ = ["ObservabilityService"]
