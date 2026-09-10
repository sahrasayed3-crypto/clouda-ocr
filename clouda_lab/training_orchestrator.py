"""Training Orchestrator — thin facade over the Training Experiment Framework.

The framework (``clouda_training.experiments``) owns experiment configs,
runs, metrics, checkpoints, resume, dry-run/mock lifecycle, and the run
registry. This facade exposes high-level operations for the future UI/API
and **adds nothing parallel**: every operation delegates.

Real training stays fail-closed: the framework only permits mock/dry_run
adapters, and this facade never constructs real trainer adapters.

Phase 15 adds the safe selection→experiment pipeline:
``create_training_experiment_from_selection`` validates the selection
(holdout guard), writes the derived manifest, records provenance, and
resolves an experiment config the framework accepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from clouda_contracts.checksums import sha256_file
from clouda_data.training_data.loader import (
    StreamingTrainingDataLoader,
    loader_config_hash,
)
from clouda_data.training_data.models import (
    BatchConfig,
    ShuffleConfig,
    ShuffleMode,
    TrainingDataConfig,
    ValidationMode,
)
from clouda_data.training_data.sharding import build_shards
from clouda_training.experiments import (
    ConfigError,
    ExperimentRegistry,
    compare_runs,
    list_checkpoints,
    list_runs,
    load_experiment_config,
    load_run,
    resume_run,
    run_experiment,
)

from .dataset_selection import (
    SelectionCriteria,
    SelectionResult,
    select_samples,
    validate_derived_manifest_for_training,
    write_selection_manifest,
)
from .holdout_guard import assert_selection_safe
from .selection_history import SelectionHistory

TRAINING_PURPOSE = "training_subset"


class TrainingOrchestrator:
    """High-level, UI/API-facing operations over the experiment framework."""

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.registry = ExperimentRegistry(self.runs_root)

    # -- inspection (pure reads) -------------------------------------------

    def list_runs(
        self, experiment: str | None = None, **kwargs
    ) -> list[dict[str, Any]]:
        if experiment:
            return [
                r.to_dict()
                for r in self.registry.runs(**kwargs)
                if r.path.parent.name == experiment
            ]
        return [r.to_dict() for r in list_runs(self.runs_root, **kwargs)]

    def inspect_run(self, run_id: str) -> dict[str, Any]:
        handle = load_run(run_id, self.runs_root)
        return {
            "run": handle.to_dict(),
            "summary": handle.summary(),
            "metrics": list(handle.metrics()),
        }

    def run_status(self, run_id: str) -> str:
        return load_run(run_id, self.runs_root).status.value

    def get_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return list(load_run(run_id, self.runs_root).metrics())

    def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        handle = load_run(run_id, self.runs_root)
        return [c.to_dict() for c in list_checkpoints(handle.path)]

    def compare(self, left_run_id: str, right_run_id: str) -> dict[str, Any]:
        left = load_run(left_run_id, self.runs_root)
        right = load_run(right_run_id, self.runs_root)
        return compare_runs(left, right)

    # -- lifecycle (mock/dry-run only) --------------------------------------

    def start_dry_run(
        self,
        config_path: str | Path,
        *,
        training_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a validated experiment config in dry-run (mock) mode."""
        config = load_experiment_config(Path(config_path))
        if not config.runtime.dry_run:
            raise ConfigError(
                "Orchestrator only permits dry_run configs; real training "
                "adapters remain disabled"
            )
        loader = (
            self.create_training_data_loader(training_data)
            if training_data is not None
            else None
        )
        handle = run_experiment(config, data_loader=loader)
        return handle.to_dict()

    def resume(
        self,
        run_id: str,
        *,
        training_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume passthrough — the framework enforces its own integrity checks."""
        loader = (
            self.create_training_data_loader(training_data)
            if training_data is not None
            else None
        )
        handle = resume_run(run_id, self.runs_root, data_loader=loader)
        return handle.to_dict()

    def create_training_data_loader(
        self, descriptor: Mapping[str, Any]
    ) -> StreamingTrainingDataLoader:
        """Construct the canonical runtime loader from prepared local paths."""
        if descriptor.get("schema_version") != "clouda.training_data.prepared.v1":
            raise ValueError("Unsupported prepared training-data descriptor")
        manifest_path = Path(str(descriptor["manifest_path"]))
        shard_index_path = Path(str(descriptor["shard_index_path"]))
        if sha256_file(manifest_path) != descriptor.get("manifest_sha256"):
            raise ValueError("Prepared training-data manifest hash mismatch")
        if sha256_file(shard_index_path) != descriptor.get("shard_index_sha256"):
            raise ValueError("Prepared training-data shard-index hash mismatch")
        config = TrainingDataConfig(
            dataset_id=str(descriptor["dataset_id"]),
            dataset_version=str(descriptor["dataset_version"]),
            global_seed=int(descriptor["global_seed"]),
            shuffle=ShuffleConfig(mode=ShuffleMode.NONE),
            batch=BatchConfig(batch_size=int(descriptor["batch_size"])),
            validation_mode=ValidationMode.NONE,
        )
        loader = StreamingTrainingDataLoader(
            shard_index_path=shard_index_path,
            loader_config=config,
            manifest_path=manifest_path,
            dataset_root=Path(str(descriptor["dataset_root"])),
        )
        if loader.config_hash != descriptor.get("loader_config_hash"):
            raise ValueError("Prepared training-data loader config hash mismatch")
        return loader

    def validate_experiment(self, config_path: str | Path) -> dict[str, Any]:
        config = load_experiment_config(Path(config_path))
        return {
            "valid": True,
            "experiment": config.experiment.name,
            "dataset_id": config.dataset.dataset_id,
            "dataset_version": config.dataset.dataset_version,
            "split": config.dataset.split,
            "manifest": str(config.dataset.manifest_path),
            "dry_run": config.runtime.dry_run,
            "adapter": config.model.adapter_type,
        }

    # -- Phase 15: selection → experiment pipeline ---------------------------

    def create_training_experiment_from_selection(
        self,
        *,
        manifest_path: str,
        criteria: SelectionCriteria,
        seed: int = 20260723,
        output_dir: str | Path,
        experiment_name: str,
        scores: dict[str, float] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        dry_run: bool = True,
        history: SelectionHistory | None = None,
    ) -> dict[str, Any]:
        """Selection → derived manifest → experiment config → optional dry-run.

        Rejects protected data before anything is written. The derived
        manifest embeds full selection provenance in its header.
        """
        selection: SelectionResult = select_samples(
            manifest_path, criteria, seed=seed, scores=scores
        )
        return self.create_training_experiment_from_result(
            selection=selection,
            seed=seed,
            output_dir=output_dir,
            experiment_name=experiment_name,
            config_overrides=config_overrides,
            dry_run=dry_run,
            history=history,
        )

    def create_training_experiment_from_result(
        self,
        *,
        selection: SelectionResult,
        output_dir: str | Path,
        experiment_name: str,
        seed: int | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        dry_run: bool = True,
        history: SelectionHistory | None = None,
    ) -> dict[str, Any]:
        """Create an experiment directly from an existing canonical selection."""
        if seed is not None and seed != selection.seed:
            raise ValueError("Experiment seed must match the selection seed")
        resolved_seed = selection.seed
        if not selection.sample_ids:
            raise ValueError(
                "Selection produced no rows; refusing to create an experiment"
            )
        # Fail closed: verify every selected row against the holdout guard.
        assert_selection_safe(list(selection.rows))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        derived_manifest = out_dir / f"{selection.selection_id}.manifest.jsonl"
        derived_dataset_id = f"lab_selection_{selection.selection_id}"
        derived_dataset_version = selection.selection_id
        header = write_selection_manifest(
            selection,
            str(derived_manifest),
            dataset_id=derived_dataset_id,
            dataset_version=derived_dataset_version,
        )
        validation = validate_derived_manifest_for_training(str(derived_manifest))

        if history is not None:
            history.record(
                purpose=TRAINING_PURPOSE,
                selection_id=selection.selection_id,
                sample_ids=list(selection.sample_ids),
                source_manifest=selection.source_manifest,
                source_manifest_sha256=selection.source_manifest_sha256,
                metadata={"derived_manifest": str(derived_manifest)},
            )

        config = _build_experiment_config(
            experiment_name=experiment_name,
            dataset_id=derived_dataset_id,
            dataset_version=derived_dataset_version,
            manifest_path=derived_manifest,
            split=str(selection.criteria.get("split") or "train"),
            seed=resolved_seed,
            dry_run=dry_run,
            output_root=self.runs_root.resolve(),
            overrides=config_overrides,
        )
        batch_size = int(config["training"].get("batch_size", 8))
        loader_config = TrainingDataConfig(
            dataset_id=derived_dataset_id,
            dataset_version=derived_dataset_version,
            global_seed=resolved_seed,
            shuffle=ShuffleConfig(mode=ShuffleMode.NONE),
            batch=BatchConfig(batch_size=batch_size),
            validation_mode=ValidationMode.NONE,
        )
        training_data_dir = out_dir / f"{selection.selection_id}.training-data"
        shard_index = build_shards(
            derived_manifest,
            training_data_dir,
            loader_config.shard,
            dataset_id=derived_dataset_id,
            dataset_version=derived_dataset_version,
        )
        source_path = Path(selection.source_manifest)
        dataset_root = (
            source_path.parent.resolve()
            if source_path.is_file()
            else derived_manifest.parent.resolve()
        )
        training_data = {
            "schema_version": "clouda.training_data.prepared.v1",
            "dataset_id": derived_dataset_id,
            "dataset_version": derived_dataset_version,
            "manifest_path": str(derived_manifest.resolve()),
            "manifest_sha256": sha256_file(derived_manifest),
            "shard_index_path": str((training_data_dir / "shard_index.json").resolve()),
            "shard_index_sha256": sha256_file(training_data_dir / "shard_index.json"),
            "shard_ids": [entry.shard_id for entry in shard_index.shards],
            "dataset_root": str(dataset_root),
            "global_seed": resolved_seed,
            "batch_size": batch_size,
            "loader_config_hash": loader_config_hash(loader_config),
        }
        config_path = out_dir / f"{selection.selection_id}.experiment.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        return {
            "selection": selection.to_dict(),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "manifest_header": header,
            "validation": validation,
            "experiment_config": config,
            "experiment_config_path": str(config_path),
            "training_data": training_data,
            "dry_run_requested": dry_run,
        }


def _build_experiment_config(
    *,
    experiment_name: str,
    dataset_id: str,
    dataset_version: str,
    manifest_path: Path,
    split: str,
    seed: int,
    dry_run: bool,
    output_root: Path,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Framework-shaped config dict (matches ExperimentConfig sections)."""
    config: dict[str, Any] = {
        "experiment": {
            "name": experiment_name,
            "description": "Clouda Lab selection-driven experiment",
            "tags": ["clouda-lab", "selection-driven"],
        },
        "model": {
            "model_id": "clouda-lab/mock",
            "adapter_type": "mock",
        },
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "manifest_path": str(manifest_path),
            "split": split,
        },
        "training": {
            "seed": seed,
            "epochs": 1,
            "max_steps": 4,
        },
        "checkpoint": {
            "save_strategy": "steps",
            "save_steps": 2,
            "save_total_limit": 2,
            "keep_best": True,
            "metric_for_best": "loss",
            "greater_is_better": False,
        },
        "evaluation": {
            "enabled": True,
            "eval_split": "validation",
            "eval_steps": 2,
            "metrics": ["cer", "wer"],
        },
        "tracking": {
            "enabled": True,
            "backend": "jsonl",
            "log_steps": 1,
        },
        "runtime": {
            "device": "cpu",
            "output_root": str(output_root),
            "dry_run": bool(dry_run),
            "offline": True,
            "deterministic": True,
        },
    }
    if overrides:
        for dotted, value in overrides.items():
            section, _, field = dotted.partition(".")
            if section not in config or not field:
                raise ConfigError(f"Invalid override key: {dotted}")
            config[section][field] = value
    return config


__all__ = [
    "TRAINING_PURPOSE",
    "TrainingOrchestrator",
]
