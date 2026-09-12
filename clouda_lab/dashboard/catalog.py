from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from clouda_contracts.protection import normalize_marker, record_is_protected
from clouda_data.pretraining.manifest import iter_manifest
from clouda_data.quality.manifest_adapter import manifest_sha256
from clouda_data.results.service import ResultsService
from clouda_data.training_data.input_contract import validate_canonical_manifest
from clouda_training.experiments.config import load_experiment_config

from .security import (
    browser_safe,
    safe_identifier,
    safe_output_label,
    safe_relative_label,
    sanitize_payload,
)
from .settings import LabSettings


class DatasetCatalog:
    """Bounded adapter over canonical training configs and manifests."""

    def __init__(self, settings: LabSettings) -> None:
        self.settings = settings
        self._detail_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}

    def _configs(self) -> Iterator[Path]:
        root = self.settings.repo_root / "configs" / "training"
        if not root.is_dir():
            return
        for pattern in ("*.json", "*.yaml", "*.yml"):
            yield from sorted(root.glob(pattern))

    def list_sources(self) -> list[dict[str, Any]]:
        from clouda_data.datasets.registry import list_sources, verify_license

        records = []
        for source in list_sources():
            license_state = verify_license(source)
            records.append(
                {
                    "source_id": source.get("source_id"),
                    "name": source.get("name"),
                    "classification": source.get("classification"),
                    "license": source.get("license"),
                    "license_verified": license_state["license_verified"],
                    "commercial_use_allowed": license_state["commercial_use_allowed"],
                    "download_enabled": False,
                    "download_reason": "Downloads are disabled in Clouda Lab",
                }
            )
        return sanitize_payload(records)

    def _records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in self._configs():
            try:
                config = load_experiment_config(path)
            except Exception:
                continue
            manifest = config.dataset.manifest_path.resolve(strict=False)
            if not manifest.is_file():
                continue
            dataset_id = str(config.dataset.dataset_id)
            try:
                safe_identifier(dataset_id)
            except ValueError:
                continue
            records.setdefault(
                dataset_id,
                {
                    "dataset_id": dataset_id,
                    "version": str(config.dataset.dataset_version),
                    "manifest": manifest,
                    "split": str(config.dataset.split),
                    "config": path,
                },
            )
        return records

    def _record(self, dataset_id: str) -> dict[str, Any]:
        safe_identifier(dataset_id)
        try:
            return self._records()[dataset_id]
        except KeyError as exc:
            raise KeyError(f"unknown local dataset: {dataset_id}") from exc

    def _results_records(self) -> dict[str, dict[str, Any]]:
        if not self.settings.results_root.is_dir():
            return {}
        service = ResultsService(self.settings.results_root, read_only=True)
        records: dict[str, dict[str, Any]] = {}
        for payload in service.list_datasets():
            dataset_id = str(payload.get("dataset_id") or "")
            try:
                safe_identifier(dataset_id)
            except ValueError:
                continue
            prior = records.get(dataset_id)
            if prior is None or str(payload.get("version", "1")) > str(
                prior.get("version", "1")
            ):
                records[dataset_id] = payload
        return records

    def list_datasets(self) -> list[dict[str, Any]]:
        manifests = self._records()
        datasets = [self._detail(record) for _, record in sorted(manifests.items())]
        datasets.extend(
            self._results_detail(record)
            for dataset_id, record in sorted(self._results_records().items())
            if dataset_id not in manifests
        )
        return datasets

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        safe_identifier(dataset_id)
        manifest_record = self._records().get(dataset_id)
        if manifest_record is not None:
            return self._detail(manifest_record)
        result_record = self._results_records().get(dataset_id)
        if result_record is not None:
            return self._results_detail(result_record)
        raise KeyError(f"unknown local dataset: {dataset_id}")

    def _results_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload["dataset_id"])
        version = str(payload.get("version", "1"))
        metadata = payload.get("metadata") or {}
        protected = record_is_protected(payload) or record_is_protected(metadata)
        return browser_safe(
            {
                "dataset_id": dataset_id,
                "identity": f"{dataset_id}@{version}",
                "version": version,
                "name": payload.get("name"),
                "description": payload.get("description"),
                "source_type": "results_store",
                "source_lineage": metadata.get("source_id"),
                "row_count": metadata.get("row_count"),
                "sample_count": metadata.get("sample_count"),
                "page_count": metadata.get("page_count"),
                "artifact_root": None,
                "manifest_hash": payload.get("manifest_sha256"),
                "created_at": payload.get("registered_at"),
                "updated_at": payload.get("registered_at"),
                "safety": {
                    "protected_holdout": protected,
                    "evaluation_only": True,
                    "training_allowed": False,
                },
                "quality": {"status": "NOT AVAILABLE", "verdict": None},
                "dedup_status": metadata.get("dedup_status", "NOT AVAILABLE"),
                "integrity": {
                    "manifest_readable": False,
                    "reason": "No configured local canonical manifest",
                },
                "loader": {
                    "compatible": False,
                    "reason": "Results Store metadata is not a training manifest",
                },
                "lineage": [f"{dataset_id}@{version}", "Results Store"],
                "parent_datasets": metadata.get("parent_datasets", []),
            },
            (self.settings.repo_root,),
        )

    def _detail(self, record: dict[str, Any]) -> dict[str, Any]:
        manifest = Path(record["manifest"])
        stat = manifest.stat()
        cache_key = str(record["dataset_id"])
        cached = self._detail_cache.get(cache_key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        stream = iter_manifest(manifest)
        try:
            header = next(stream)
        except StopIteration:
            header = {}
        protected = record_is_protected(header)
        roles = {
            normalize_marker(str(header.get("dataset_role", header.get("role", ""))))
        }
        row_count = 0
        for row in stream:
            row_count += 1
            protected = protected or record_is_protected(row)
            roles.add(
                normalize_marker(str(row.get("dataset_role", row.get("role", ""))))
            )
        evaluation_only = "evaluation_only" in roles
        compatible = False
        compatibility_reason = "NOT VALIDATED"
        try:
            identity = validate_canonical_manifest(
                manifest,
                dataset_id=record["dataset_id"],
                dataset_version=record["version"],
                split=record["split"],
            )
            compatible = True
            compatibility_reason = (
                identity.system if hasattr(identity, "system") else "CANONICAL"
            )
        except Exception as exc:
            compatibility_reason = f"{type(exc).__name__}: {sanitize_payload(str(exc))}"
        lineage = [
            str(value)
            for key in (
                "derived_from",
                "source_dataset_version",
                "parent_dataset",
                "quality_run_id",
            )
            if (value := header.get(key))
        ]
        lineage.append(f"{record['dataset_id']}@{record['version']}")
        path_label = safe_relative_label(manifest.parent, (self.settings.repo_root,))
        detail = browser_safe(
            {
                "dataset_id": record["dataset_id"],
                "identity": f"{record['dataset_id']}@{record['version']}",
                "version": record["version"],
                "source_type": "canonical_manifest",
                "source_lineage": header.get("source_id") or header.get("derived_from"),
                "row_count": row_count,
                "sample_count": row_count,
                "page_count": header.get("page_count"),
                "artifact_root": path_label,
                "manifest_hash": manifest_sha256(manifest),
                "manifest_identity": header.get("manifest_sha256"),
                "created_at": datetime.fromtimestamp(
                    stat.st_ctime, timezone.utc
                ).isoformat(),
                "updated_at": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "safety": {
                    "protected_holdout": protected,
                    "evaluation_only": evaluation_only,
                    "training_allowed": not protected
                    and not evaluation_only
                    and compatible,
                },
                "quality": self.quality_summary(record["dataset_id"]),
                "dedup_status": header.get("dedup_status", "NOT AVAILABLE"),
                "integrity": {
                    "manifest_readable": True,
                    "declared_row_count": header.get("_row_count"),
                    "actual_row_count": row_count,
                    "row_count_matches": header.get("_row_count") == row_count,
                },
                "loader": {
                    "compatible": compatible,
                    "reason": compatibility_reason,
                },
                "lineage": lineage,
                "parent_datasets": header.get("parent_datasets", []),
            },
            (self.settings.repo_root,),
        )
        self._detail_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, detail)
        return detail

    def preview(self, dataset_id: str, limit: int = 10) -> dict[str, Any]:
        record = self._record(dataset_id)
        bounded = max(1, min(int(limit), self.settings.preview_limit))
        stream = iter_manifest(record["manifest"])
        header = next(stream, None) or {}
        protected_manifest = record_is_protected(header)
        records: list[dict[str, Any]] = []
        has_more = False
        for row in stream:
            if len(records) >= bounded:
                has_more = True
                break
            records.append(self._safe_row(row, protected_context=protected_manifest))
        return {
            "dataset_id": dataset_id,
            "limit": bounded,
            "returned": len(records),
            "truncated": has_more,
            "records": records,
        }

    def _safe_row(
        self, row: dict[str, Any], *, protected_context: bool = False
    ) -> dict[str, Any]:
        if protected_context or record_is_protected(row):
            return sanitize_payload(
                {
                    key: row.get(key)
                    for key in (
                        "sample_id",
                        "source_id",
                        "target_split",
                        "split",
                        "dataset_role",
                        "role",
                    )
                    if row.get(key) is not None
                }
                | {"protected": True, "content_withheld": True}
            )
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if (
                "path" in key.lower()
                and isinstance(value, str)
                and Path(value).is_absolute()
            ):
                cleaned[key] = (
                    safe_relative_label(value, (self.settings.repo_root,))
                    or "[PRIVATE PATH]"
                )
            else:
                cleaned[key] = value
        return browser_safe(cleaned, (self.settings.repo_root,))

    def quality_summary(self, dataset_id: str) -> dict[str, Any]:
        safe_identifier(dataset_id)
        report = self._quality_report_path(dataset_id)
        if not report.is_file():
            return {"status": "NOT RUN", "verdict": None}
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "FAIL",
                "verdict": None,
                "reason": "Unreadable quality report",
            }
        return browser_safe(
            {
                "status": "AVAILABLE",
                "verdict": payload.get("verdict"),
                "run_id": payload.get("run_id"),
                "severity_counts": payload.get("severity_counts", {}),
                "rejected_samples": len(payload.get("exclusions", [])),
                "duplicate_clusters": (payload.get("clusters") or {}).get("count", 0),
                "report": payload,
            },
            (self.settings.repo_root,),
        )

    def _quality_report_path(self, dataset_id: str) -> Path:
        safe_identifier(dataset_id)
        digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:32]
        return self.settings.quality_root / f"dataset-{digest}.report.json"

    def run_quality(
        self, dataset_id: str, *, max_samples: int = 0, duplicates_only: bool = False
    ) -> dict[str, Any]:
        from clouda_data.pretraining.hashing import atomic_write_text
        from clouda_data.quality.config import QualityGateConfig
        from clouda_data.quality.gate import run_quality_gate
        from clouda_data.quality.policy import quarantine_sample_ids
        from clouda_data.quality.report import build_report_payload

        record = self._record(dataset_id)
        bounded = max(0, min(int(max_samples), self.settings.quality_scan_limit))
        scan = run_quality_gate(
            str(record["manifest"]),
            QualityGateConfig(),
            max_samples=bounded,
            no_near_duplicates=False,
        )
        payload = build_report_payload(
            scan.result,
            protected_ids=frozenset(
                quarantine_sample_ids(scan.samples, scan.exclusions)
            ),
        )
        payload.update(
            {
                "dataset_id": dataset_id,
                "dataset_version": str(record["version"]),
                "verdict": scan.result.verdict.value,
                "manifest_sha256": scan.manifest_sha256,
                "quarantine_count": len(scan.quarantine_ids),
                "duplicates_only": bool(duplicates_only),
            }
        )
        self.settings.quality_root.mkdir(parents=True, exist_ok=True)
        target = self._quality_report_path(dataset_id)
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
        self._detail_cache.pop(dataset_id, None)
        return self.quality_summary(dataset_id)

    def derive(self, dataset_id: str, output_label: str) -> dict[str, Any]:
        from clouda_data.quality.config import QualityGateConfig
        from clouda_data.quality.derived import write_clean_manifest
        from clouda_data.quality.gate import run_quality_gate

        safe_output_label(output_label)
        record = self._record(dataset_id)
        scan = run_quality_gate(str(record["manifest"]), QualityGateConfig())
        self.settings.quality_root.mkdir(parents=True, exist_ok=True)
        output = self.settings.quality_root / f"{output_label}.manifest.jsonl"
        result = write_clean_manifest(
            Path(record["manifest"]),
            scan.samples,
            scan.exclusions,
            scan.quarantine_ids,
            scan.run,
            output,
        )
        return browser_safe(
            {
                "dataset_id": dataset_id,
                "derived_dataset": result,
                "artifact": safe_relative_label(output, (self.settings.repo_root,)),
            },
            (self.settings.repo_root,),
        )


__all__ = ["DatasetCatalog"]
