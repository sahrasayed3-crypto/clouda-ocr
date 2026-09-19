from __future__ import annotations

import csv
import hashlib
import json
from typing import Any, Mapping

from clouda_contracts.checksums import sha256_file
from clouda_training.experiments.io import atomic_write_json, read_json

from .models import ModelCatalogService
from .security import browser_safe, safe_identifier
from .settings import LabSettings


class BenchmarkWorkspaceService:
    """Planning and comparison over immutable published benchmark evidence."""

    def __init__(self, settings: LabSettings, models: ModelCatalogService) -> None:
        self.settings = settings
        self.models = models
        self.root = settings.benchmarks_root / "ocr_arabic"

    def _release(self) -> dict[str, Any]:
        path = self.root / "release.json"
        payload = read_json(path)
        payload["artifact_sha256"] = sha256_file(path)
        return payload

    def _evaluation_permissions(self) -> dict[str, Any]:
        path = self.root / "source_manifest.csv"
        rows: list[dict[str, str]] = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            "source_count": len(rows),
            "evaluation_allowed": bool(rows)
            and all(row.get("evaluation_permission") == "ALLOWED" for row in rows),
            "evidence_ids": sorted(
                {str(row.get("evidence_id")) for row in rows if row.get("evidence_id")}
            ),
        }

    def inventory(self) -> dict[str, Any]:
        release = self._release()
        permissions = self._evaluation_permissions()
        return browser_safe(
            {
                "schema_version": "clouda.lab.benchmark-workspace.v1",
                "benchmark": release,
                "dataset": {
                    "purpose": "PROTECTED_EVALUATION",
                    "training_allowed": False,
                    "manifest_sha256": release.get("manifest_sha256"),
                    **permissions,
                },
                "models": self.models.list_models(),
                "execution": {
                    "available": False,
                    "reason": (
                        "No canonical local runner is registered for the eight "
                        "published OCR model integrations"
                    ),
                },
            },
            (self.settings.repo_root,),
        )

    def results(self) -> dict[str, Any]:
        models = self.models.list_models()
        records = [
            {
                "catalog_id": item["catalog_id"],
                "name": item["name"],
                **dict(item["benchmark"]),
            }
            for item in models
        ]
        ranked = sorted(
            (
                item
                for item in records
                if item["status"] == "COMPLETE"
                and item["rankable"]
                and item["normalized_arabic_cer"] is not None
            ),
            key=lambda item: float(item["normalized_arabic_cer"]),
        )
        unranked = [item for item in records if item not in ranked]
        return {
            "schema_version": "clouda.lab.benchmark-results.v1",
            "primary_metric": "normalized_arabic_cer",
            "direction": "lower_is_better",
            "ranked": ranked,
            "unranked": unranked,
            "source": "published actual benchmark metadata",
        }

    def create_plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_ids = payload.get("model_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("model_ids must be a non-empty list")
        if len(raw_ids) > 8:
            raise ValueError("at most eight published models may be selected")
        selected_ids = sorted({safe_identifier(str(item)) for item in raw_ids})
        inventory = {item["catalog_id"]: item for item in self.models.list_models()}
        missing = [item for item in selected_ids if item not in inventory]
        if missing:
            raise KeyError(f"unknown published model: {missing[0]}")
        release = self._release()
        identity = {
            "schema_version": "clouda.lab.benchmark-plan.v1",
            "benchmark_id": release["benchmark_id"],
            "benchmark_manifest_sha256": release["manifest_sha256"],
            "model_ids": selected_ids,
            "metrics": ["cer", "wer", "normalized_arabic_cer"],
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        plan_id = (
            "benchmark-plan-"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        )
        record = {
            **identity,
            "plan_id": plan_id,
            "dataset": {
                "purpose": "PROTECTED_EVALUATION",
                "training_allowed": False,
            },
            "models": [
                {
                    "catalog_id": item,
                    "name": inventory[item]["name"],
                    "eligible": False,
                    "reason": "No canonical published-model benchmark runner is registered",
                }
                for item in selected_ids
            ],
            "execution": {
                "available": False,
                "reason": (
                    "No canonical published-model benchmark runner is registered; "
                    "existing results remain inspectable"
                ),
            },
        }
        self.settings.benchmark_plans_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.settings.benchmark_plans_root / f"{plan_id}.json", record
        )
        return browser_safe(record, (self.settings.repo_root,))

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        safe_identifier(plan_id)
        path = self.settings.benchmark_plans_root / f"{plan_id}.json"
        if not path.is_file():
            raise KeyError(f"unknown benchmark plan: {plan_id}")
        return browser_safe(read_json(path), (self.settings.repo_root,))

    def compare(self, left_id: str, right_id: str) -> dict[str, Any]:
        safe_identifier(left_id)
        safe_identifier(right_id)
        result_set = self.results()
        records = {
            item["catalog_id"]: item
            for item in [*result_set["ranked"], *result_set["unranked"]]
        }
        if left_id not in records or right_id not in records:
            raise KeyError("unknown published model comparison target")
        left, right = records[left_id], records[right_id]
        comparable = all(
            item["status"] == "COMPLETE" and item["rankable"] for item in (left, right)
        )
        winner = None
        reason = "Both runs must be complete and rankable"
        if comparable:
            metric = "normalized_arabic_cer"
            winner = min((left, right), key=lambda item: float(item[metric]))[
                "catalog_id"
            ]
            reason = "Lower published normalized Arabic CER"
        return {
            "left": left,
            "right": right,
            "winner": winner,
            "reason": reason,
            "actual_results_only": True,
        }


__all__ = ["BenchmarkWorkspaceService"]
