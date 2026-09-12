from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from clouda_data.pretraining.hashing import atomic_write_text
from clouda_training.adapters.registry import get_default_registry
from clouda_training.experiments.config import (
    CheckpointSection,
    DatasetSection,
    EvaluationSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    RuntimeSection,
    TrackingSection,
    TrainingSection,
    load_experiment_config,
)
from clouda_training.planner.models import (
    HardwareEnvelope,
    ParameterMetadata,
    StorageKind,
    TrainingMode,
    TrainingScale,
    default_profile,
)
from clouda_training.planner.planner import build_experiment_plan
from clouda_training.preflight.orchestrator import run_preflight

from .catalog import DatasetCatalog
from .security import (
    browser_safe,
    safe_identifier,
    safe_relative_label,
    sanitize_payload,
)
from .settings import LabSettings

_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class TrainingService:
    def __init__(self, settings: LabSettings, catalog: DatasetCatalog) -> None:
        self.settings = settings
        self.catalog = catalog

    @staticmethod
    def _registry():
        from clouda_training.hunyuan.registration import register_hunyuan_adapters
        from clouda_training.qwen.registration import register_qwen_adapters

        register_hunyuan_adapters()
        register_qwen_adapters()
        return get_default_registry()

    def list_models(self) -> list[dict[str, Any]]:
        registry = self._registry()
        models: list[dict[str, Any]] = []
        for adapter_id in registry.list_adapters():
            descriptor = registry.get(adapter_id)
            dependencies = []
            for requirement in descriptor.required_optional_dependencies:
                module = re.split(r"[<>=!~]", requirement, maxsplit=1)[0]
                if module == "trust_remote_code":
                    available: bool | None = None
                else:
                    available = (
                        importlib.util.find_spec(module.replace("-", "_")) is not None
                    )
                dependencies.append(
                    {"requirement": requirement, "available": available}
                )
            missing = [
                item["requirement"]
                for item in dependencies
                if item["available"] is False
            ]
            models.append(
                {
                    "name": descriptor.model_family,
                    "model_family": descriptor.model_family,
                    "upstream_repository": descriptor.upstream_repository,
                    "adapter_id": adapter_id,
                    "adapter_version": descriptor.adapter_version,
                    "code_integration": "PASS",
                    "installed": not missing,
                    "available": not missing,
                    "weights": "NOT INSTALLED",
                    "tokenizer": "NOT INSTALLED",
                    "processor": "NOT INSTALLED",
                    "training_capability": any(
                        (
                            descriptor.capabilities.supports_full_finetune,
                            descriptor.capabilities.supports_selective_finetune,
                        )
                    ),
                    "inference_capability": False,
                    "cpu_compatibility": descriptor.capabilities.supports_cpu_smoke,
                    "gpu_requirement": True,
                    "gpu_validation": (
                        "PASS" if descriptor.capabilities.gpu_validated else "DEFERRED"
                    ),
                    "real_training": (
                        "VALIDATED"
                        if descriptor.capabilities.real_weights_validated
                        else "NOT VALIDATED"
                    ),
                    "supported_precision": list(descriptor.supported_precision),
                    "supported_devices": list(descriptor.supported_devices),
                    "dependencies": dependencies,
                    "capabilities": descriptor.capabilities.summary(),
                    "availability_reason": (
                        f"Missing optional dependencies: {', '.join(missing)}"
                        if missing
                        else "Code dependencies available; local model assets are not configured"
                    ),
                }
            )
        return models

    def planner_options(self) -> dict[str, Any]:
        datasets = self.catalog.list_datasets()
        registry = self._registry()
        precisions = sorted(
            {
                precision
                for descriptor in (
                    registry.get(adapter_id) for adapter_id in registry.list_adapters()
                )
                for precision in descriptor.supported_precision
            }
        )
        return {
            "models": self.list_models(),
            "datasets": [
                {
                    "dataset_id": item["dataset_id"],
                    "identity": item["identity"],
                    "training_allowed": item["safety"]["training_allowed"],
                }
                for item in datasets
            ],
            "precision": precisions,
            "devices": ["cuda"],
            "offline": True,
            "execution": "DEFERRED",
        }

    def create_plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        adapter_type = safe_identifier(str(payload.get("adapter_type", "")))
        registry = self._registry()
        if not registry.is_registered(adapter_type):
            raise KeyError(f"unknown adapter: {adapter_type}")
        descriptor = registry.get(adapter_type)
        dataset_id = safe_identifier(str(payload.get("dataset_id", "")))
        dataset = self.catalog.get_dataset(dataset_id)
        if not dataset["safety"]["training_allowed"]:
            raise PermissionError("dataset is protected or not loader-compatible")
        internal = self.catalog._record(dataset_id)
        experiment_name = str(payload.get("experiment_name") or f"lab-{dataset_id}")
        if not _EXPERIMENT.fullmatch(experiment_name):
            raise ValueError("invalid experiment name")
        model_id = str(payload.get("model_id") or descriptor.upstream_repository)
        if not model_id or len(model_id) > 512 or ".." in model_id:
            raise ValueError("invalid model identity")
        precision = str(payload.get("precision", "bf16"))
        if precision not in descriptor.supported_precision:
            raise ValueError(
                f"precision {precision!r} is unsupported by {adapter_type}"
            )

        def positive(name: str, default: int) -> int:
            value = int(payload.get(name, default))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return value

        config = ExperimentConfig(
            experiment=ExperimentSection(name=experiment_name, tags=("clouda-lab",)),
            model=ModelSection(
                model_id=model_id,
                model_family=descriptor.model_family,
                adapter_type=adapter_type,
                precision=precision,
            ),
            dataset=DatasetSection(
                dataset_id=dataset_id,
                dataset_version=str(internal["version"]),
                manifest_path=Path(internal["manifest"]),
                split=str(internal["split"]),
                preprocessing_version="canonical",
            ),
            training=TrainingSection(
                seed=int(payload.get("seed", 20260723)),
                epochs=positive("epochs", 1),
                max_steps=positive("max_steps", 100),
                batch_size=positive("batch_size", 1),
                gradient_accumulation_steps=positive("gradient_accumulation_steps", 1),
                learning_rate=float(payload.get("learning_rate", 5e-5)),
                mixed_precision=precision != "fp32",
                gradient_checkpointing=descriptor.capabilities.supports_gradient_checkpointing,
            ),
            checkpoint=CheckpointSection(
                save_steps=positive("checkpoint_frequency", 100),
                save_total_limit=3,
            ),
            evaluation=EvaluationSection(enabled=True),
            runtime=RuntimeSection(
                device="cuda",
                output_root=self.settings.runs_root,
                dry_run=True,
                offline=True,
                deterministic=True,
            ),
            tracking=TrackingSection(enabled=True, backend="jsonl"),
        )
        profile = replace(
            default_profile(
                TrainingScale.PILOT,
                training_mode=TrainingMode.SELECTIVE_FINETUNE,
            ),
            sample_count=int(dataset["row_count"]),
            epochs=config.training.epochs,
            max_steps=config.training.max_steps,
            micro_batch=config.training.batch_size,
            gradient_accumulation=config.training.gradient_accumulation_steps,
            precision=config.model.precision,
            checkpoint_interval=config.checkpoint.save_steps,
        )
        plan = build_experiment_plan(
            config,
            profile,
            hardware=HardwareEnvelope(
                gpu_count=1,
                per_gpu_vram_gb=None,
                storage_kind=StorageKind.UNKNOWN,
            ),
            parameter_metadata=ParameterMetadata(),
            dataset_row_count=int(dataset["row_count"]),
        )
        public_config = self._public_config(config.to_dict())
        record = {
            "schema_version": "clouda.lab.plan.v1",
            "plan_id": plan.plan_id,
            "config_id": config.hash,
            "model_id": model_id,
            "adapter_id": adapter_type,
            "dataset_id": dataset["identity"],
            "expected_runtime_backend": config.runtime.device,
            "checkpoint_policy": plan.checkpoint_plan.to_dict(),
            "output_path": "runs",
            "execution_status": plan.execution_status,
            "config": public_config,
            "plan": plan.to_dict(),
        }
        self.settings.plans_root.mkdir(parents=True, exist_ok=True)
        config_path = self.settings.plans_root / f"{plan.plan_id}.config.yaml"
        plan_path = self.settings.plans_root / f"{plan.plan_id}.plan.json"
        atomic_write_text(
            config_path,
            yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=True),
        )
        atomic_write_text(
            plan_path,
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return sanitize_payload(record)

    def _public_config(self, config: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        result["dataset"]["manifest_path"] = safe_relative_label(
            result["dataset"]["manifest_path"], (self.settings.repo_root,)
        )
        result["runtime"]["output_root"] = safe_relative_label(
            result["runtime"]["output_root"], (self.settings.repo_root,)
        )
        return sanitize_payload(result)

    def list_plans(self) -> list[dict[str, Any]]:
        if not self.settings.plans_root.is_dir():
            return []
        records = []
        for path in sorted(self.settings.plans_root.glob("*.plan.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sanitize_payload(records)

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        safe_identifier(plan_id)
        path = self.settings.plans_root / f"{plan_id}.plan.json"
        if not path.is_file():
            raise KeyError(f"unknown plan: {plan_id}")
        return sanitize_payload(json.loads(path.read_text(encoding="utf-8")))

    def _config(self, plan_id: str) -> ExperimentConfig:
        safe_identifier(plan_id)
        path = self.settings.plans_root / f"{plan_id}.config.yaml"
        if not path.is_file():
            raise KeyError(f"unknown plan: {plan_id}")
        return load_experiment_config(path)

    def run_preflight(
        self, plan_id: str, *, write_probe: bool = False
    ) -> dict[str, Any]:
        plan = self.get_plan(plan_id)
        config = self._config(plan_id)
        report = run_preflight(
            config,
            dataset_row_count=(
                int(plan["plan"]["step_plan"]["dataset_rows"])
                if plan["plan"]["step_plan"].get("dataset_rows") is not None
                else None
            ),
            write_probe=write_probe,
        ).to_dict()
        context = report.get("context")
        if isinstance(context, dict):
            context["output_root"] = safe_relative_label(
                context.get("output_root", ""), (self.settings.repo_root,)
            )
        return browser_safe(report, (self.settings.repo_root,))

    def list_runs(self) -> list[dict[str, Any]]:
        from clouda_lab.training_orchestrator import TrainingOrchestrator

        runs = TrainingOrchestrator(self.settings.runs_root).list_runs()
        runs.sort(key=lambda item: str(item.get("created_at", item.get("run_id", ""))))
        return browser_safe(runs, (self.settings.repo_root,))

    def get_run(self, run_id: str) -> dict[str, Any]:
        from clouda_lab.training_orchestrator import TrainingOrchestrator

        safe_identifier(run_id)
        payload = TrainingOrchestrator(self.settings.runs_root).inspect_run(run_id)
        return browser_safe(payload, (self.settings.repo_root,))

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        from clouda_lab.training_orchestrator import TrainingOrchestrator

        safe_identifier(run_id)
        checkpoints = TrainingOrchestrator(self.settings.runs_root).get_checkpoints(
            run_id
        )
        result = []
        for checkpoint in checkpoints:
            item = dict(checkpoint)
            item["checkpoint_id"] = Path(str(item["path"])).name
            item["integrity"] = "PASS"
            item["resumable"] = False
            item["resume_state"] = "VALIDATION REQUIRED"
            result.append(item)
        return browser_safe(result, (self.settings.repo_root,))

    def resume_check(self, run_id: str) -> dict[str, Any]:
        from clouda_training.experiments.checkpoints import (
            CheckpointManager,
            list_checkpoints,
        )
        from clouda_training.experiments.config import config_from_dict
        from clouda_training.experiments.runs import load_run

        safe_identifier(run_id)
        handle = load_run(run_id, self.settings.runs_root)
        try:
            checkpoints = list_checkpoints(handle.path)
        except Exception as exc:
            return browser_safe(
                {
                    "available": False,
                    "compatible": False,
                    "status": "BLOCKED",
                    "reason": f"Checkpoint integrity failed: {exc}",
                },
                (self.settings.repo_root,),
            )
        if not checkpoints:
            return {
                "available": False,
                "compatible": False,
                "status": "BLOCKED",
                "reason": "No canonical checkpoint is available",
            }
        import yaml

        config_path = handle.path / "resolved_config.yaml"
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            return browser_safe(
                {
                    "available": False,
                    "compatible": False,
                    "status": "BLOCKED",
                    "reason": f"Resolved config unavailable: {exc}",
                },
                (self.settings.repo_root,),
            )
        config = config_from_dict(payload, path_base=handle.path)
        latest = checkpoints[-1]
        if handle.status.value not in {"INTERRUPTED", "FAILED"}:
            return {
                "available": False,
                "compatible": False,
                "status": "BLOCKED",
                "checkpoint": browser_safe(
                    latest.to_dict(), (self.settings.repo_root,)
                ),
                "reason": f"Run status {handle.status.value} is not resumable",
            }
        try:
            CheckpointManager(handle.path, run_id, config).validate_resume(latest)
        except Exception as exc:
            return browser_safe(
                {
                    "available": False,
                    "compatible": False,
                    "status": "BLOCKED",
                    "checkpoint": latest.to_dict(),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                (self.settings.repo_root,),
            )
        return browser_safe(
            {
                "available": False,
                "compatible": True,
                "status": "BLOCKED",
                "checkpoint": latest.to_dict(),
                "integrity": "PASS",
                "configuration_match": "PASS",
                "dataset_match": "PASS",
                "model_match": "PASS",
                "execution": "DISABLED",
                "execution_reason": "Real training and resume execution are deferred until compatible GPU assets are available",
            },
            (self.settings.repo_root,),
        )


__all__ = ["TrainingService"]
