from __future__ import annotations

import csv
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from clouda_training.adapters.registry import get_default_registry
from clouda_training.experiments.io import atomic_write_json, read_json
from clouda_training.preflight.checks_system import (
    check_local_model,
)

from .confirmations import ConfirmationStore
from .security import browser_safe, safe_identifier
from .settings import LabSettings
from .tasks import OperationTaskService, TaskContext

MODEL_STATE_SCHEMA = "clouda.lab.model-assets.v1"


def _catalog_id(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return safe_identifier(value)


def _match_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError("managed model operations refuse symbolic links")
        if path.is_file():
            total += path.stat().st_size
    return total


class ModelCatalogService:
    """Join published benchmark evidence with canonical training descriptors."""

    def __init__(self, settings: LabSettings, tasks: OperationTaskService) -> None:
        self.settings = settings
        self.tasks = tasks
        self.models_csv = settings.benchmarks_root / "ocr_arabic" / "models.csv"
        self.results_csv = settings.benchmarks_root / "ocr_arabic" / "results.csv"
        self.state_file = settings.model_state_root / "assets.json"
        self.confirmations = ConfirmationStore(
            settings.confirmations_root,
            browser_roots=(settings.repo_root,),
        )

    @staticmethod
    def _registry():
        from clouda_training.hunyuan.registration import register_hunyuan_adapters
        from clouda_training.qwen.registration import register_qwen_adapters

        register_hunyuan_adapters()
        register_qwen_adapters()
        return get_default_registry()

    def _published_models(self) -> list[dict[str, str]]:
        if not self.models_csv.is_file():
            return []
        with self.models_csv.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _results(self) -> dict[str, dict[str, str]]:
        if not self.results_csv.is_file():
            return {}
        with self.results_csv.open(encoding="utf-8-sig", newline="") as handle:
            return {str(row["model"]): row for row in csv.DictReader(handle)}

    def _adapter_map(self) -> dict[str, Any]:
        registry = self._registry()
        return {
            adapter_id: registry.get(adapter_id)
            for adapter_id in registry.list_adapters()
        }

    def _state(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {"schema_version": MODEL_STATE_SCHEMA, "models": {}}
        payload = read_json(self.state_file)
        if payload.get("schema_version") != MODEL_STATE_SCHEMA:
            raise ValueError("Unsupported Clouda Lab model asset state")
        return payload

    def _save_state(self, payload: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_file, payload)

    @staticmethod
    def _descriptor_for(record: dict[str, str], adapters: dict[str, Any]):
        candidates = {
            _match_key(record.get("model", "")),
            _match_key(record.get("model_identifier", "")),
        }
        for adapter_id, descriptor in adapters.items():
            values = {
                _match_key(descriptor.model_family),
                _match_key(descriptor.upstream_repository),
                _match_key(descriptor.upstream_repository.split("/")[-1]),
            }
            if any(
                candidate and value and (candidate in value or value in candidate)
                for candidate in candidates
                for value in values
            ):
                return adapter_id, descriptor
        return None, None

    def list_models(self) -> list[dict[str, Any]]:
        results = self._results()
        adapters = self._adapter_map()
        state = self._state().get("models", {})
        models: list[dict[str, Any]] = []
        for record in self._published_models():
            name = str(record["model"])
            catalog_id = _catalog_id(name)
            adapter_id, descriptor = self._descriptor_for(record, adapters)
            result = results.get(name, {})
            configured = dict(state.get(catalog_id, {}))
            asset_id = configured.get("asset_id")
            asset_root = self.settings.models_root / str(asset_id or "")
            asset_present = bool(asset_id and asset_root.is_dir())
            configured_status = str(configured.get("status", "CONFIGURED_UNVERIFIED"))
            asset_status = (
                configured_status
                if asset_present
                else ("MISSING" if asset_id else "NOT_CONFIGURED")
            )
            if descriptor is not None:
                dependency = {
                    "name": "dependencies",
                    "status": "UNCHECKED",
                    "detail": (
                        "Dependency imports are checked by the canonical training "
                        "preflight when an operation is planned"
                    ),
                    "blocker": False,
                }
                requirements = list(descriptor.required_optional_dependencies)
                capabilities = descriptor.capabilities.summary()
            else:
                dependency = {
                    "name": "dependencies",
                    "status": "NOT_APPLICABLE",
                    "detail": "No local executable adapter is registered",
                    "blocker": False,
                }
                requirements = []
                capabilities = {}
            models.append(
                {
                    "catalog_id": catalog_id,
                    "name": name,
                    "model_identifier": record.get("model_identifier"),
                    "license_status": record.get("model_license_status"),
                    "training_label_permission": record.get(
                        "training_label_permission"
                    ),
                    "benchmark_supported": True,
                    "local_inference_supported": False,
                    "training_supported": bool(
                        descriptor
                        and (
                            descriptor.capabilities.supports_full_finetune
                            or descriptor.capabilities.supports_selective_finetune
                        )
                    ),
                    "training_adapter": adapter_id,
                    "precision": (
                        list(descriptor.supported_precision) if descriptor else []
                    ),
                    "devices": list(descriptor.supported_devices) if descriptor else [],
                    "gpu_required": bool(
                        descriptor and "cuda" in descriptor.supported_devices
                    ),
                    "capabilities": capabilities,
                    "benchmark": {
                        "status": result.get(
                            "status", record.get("evaluation_status", "NOT_RECORDED")
                        ),
                        "pages": int(result.get("pages") or 0),
                        "cer": float(result["cer"]) if result.get("cer") else None,
                        "wer": float(result["wer"]) if result.get("wer") else None,
                        "normalized_arabic_cer": (
                            float(result["normalized_arabic_cer"])
                            if result.get("normalized_arabic_cer")
                            else None
                        ),
                        "seconds_per_page": (
                            float(result["seconds_per_page"])
                            if result.get("seconds_per_page")
                            else None
                        ),
                        "hardware": result.get("gpu"),
                        "rankable": _bool(result.get("rankable")),
                    },
                    "assets": {
                        "status": asset_status,
                        "asset_id": asset_id,
                        "location": (
                            asset_root.relative_to(self.settings.repo_root).as_posix()
                            if asset_present
                            else None
                        ),
                        "image_processor_status": "NOT_SEPARATELY_VERIFIED",
                        "text_encoder_status": "NOT_SEPARATELY_VERIFIED",
                        "last_validation": configured.get("last_validation"),
                    },
                    "dependencies": {
                        "check": dependency,
                        "requirements": requirements,
                        "remediation": {
                            "installation_endpoint": False,
                            "optional_group": (
                                "training-torch" if descriptor else None
                            ),
                            "command": (
                                "python -m pip install -e .[training-torch]"
                                if descriptor
                                else None
                            ),
                            "complete_for_adapter": False,
                            "uncovered_requirements": [
                                item
                                for item in requirements
                                if not item.startswith("torch")
                            ],
                        },
                    },
                    "download": {
                        "available": False,
                        "reason": (
                            "No approved model download manifest with license, size, "
                            "expected files, and checksums is registered"
                        ),
                    },
                    "notes": record.get("notes"),
                }
            )
        return browser_safe(models, (self.settings.repo_root,))

    def get_model(self, catalog_id: str) -> dict[str, Any]:
        safe_identifier(catalog_id)
        for model in self.list_models():
            if model["catalog_id"] == catalog_id:
                return model
        raise KeyError(f"Unknown model: {catalog_id}")

    def configure_assets(self, catalog_id: str, asset_id: str) -> dict[str, Any]:
        self.get_model(catalog_id)
        safe_identifier(asset_id)
        root = (self.settings.models_root / asset_id).resolve()
        managed_root = self.settings.models_root.resolve()
        try:
            root.relative_to(managed_root)
        except ValueError as exc:
            raise PermissionError("model asset directory escaped managed root") from exc
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError("managed model asset directory does not exist")
        state = self._state()
        state.setdefault("models", {})[catalog_id] = {
            "asset_id": asset_id,
            "status": "CONFIGURED_UNVERIFIED",
            "configured_at": datetime.now(timezone.utc).isoformat(),
            "last_validation": None,
        }
        self._save_state(state)
        return browser_safe(
            {
                "catalog_id": catalog_id,
                "asset_id": asset_id,
                "asset_status": "CONFIGURED_UNVERIFIED",
                "location": root,
            },
            (self.settings.repo_root,),
        )

    def _configured(self, catalog_id: str) -> tuple[dict[str, Any], Path]:
        model = self.get_model(catalog_id)
        configured = self._state().get("models", {}).get(catalog_id)
        if not configured:
            raise ValueError("model assets are not configured")
        root = (self.settings.models_root / str(configured["asset_id"])).resolve()
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("configured model assets are missing")
        return model, root

    def verify_assets(self, catalog_id: str) -> dict[str, Any]:
        model, root = self._configured(catalog_id)
        adapter_id = model.get("training_adapter")
        if not adapter_id:
            raise ValueError("no canonical local model verifier is registered")

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(phase="VERIFYING")
            config = SimpleNamespace(
                model=SimpleNamespace(adapter_type=adapter_id, model_id=str(root))
            )
            asset_check = check_local_model(config).to_dict()
            state = self._state()
            configured = state.setdefault("models", {})[catalog_id]
            now = datetime.now(timezone.utc).isoformat()
            configured["status"] = (
                "VERIFIED" if asset_check["status"] == "PASS" else "INVALID"
            )
            configured["last_validation"] = now
            configured["asset_check"] = asset_check
            self._save_state(state)
            if asset_check["status"] != "PASS":
                raise ValueError(str(asset_check["detail"]))
            return {
                "catalog_id": catalog_id,
                "asset_check": asset_check,
                "validated_at": now,
            }

        return self.tasks.enqueue("MODEL_VERIFY", catalog_id, worker)

    def create_removal_plan(self, catalog_id: str) -> dict[str, Any]:
        _model, root = self._configured(catalog_id)
        return self.confirmations.issue(
            "MODEL_REMOVE",
            catalog_id,
            {
                "item": catalog_id,
                "item_type": "model",
                "asset_id": root.name,
                "bytes": _tree_bytes(root),
                "destination": root.relative_to(self.settings.repo_root).as_posix(),
                "recoverable": True,
            },
        )

    def remove_assets(self, plan_id: str, confirmation: str) -> dict[str, Any]:
        plan = self.confirmations.consume(plan_id, confirmation, kind="MODEL_REMOVE")
        catalog_id = str(plan["target_id"])
        _model, root = self._configured(catalog_id)

        def worker(context: TaskContext) -> dict[str, Any]:
            context.update(phase="REMOVING")
            destination = (
                self.settings.trash_root / f"model-{catalog_id}-{uuid.uuid4().hex[:12]}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root), destination)
            state = self._state()
            state.setdefault("models", {}).pop(catalog_id, None)
            self._save_state(state)
            return {
                "catalog_id": catalog_id,
                "recoverable_trash": str(destination),
                "bytes": plan.get("bytes", 0),
            }

        return self.tasks.enqueue("MODEL_REMOVE", catalog_id, worker)


__all__ = ["ModelCatalogService"]
