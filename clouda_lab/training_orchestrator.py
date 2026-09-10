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
from typing import Any, Mapping, Sequence

import yaml

from clouda_contracts.checksums import sha256_file
from clouda_training.experiments import (
    ConfigError,
    ExperimentConfig,
    ExperimentRegistry,
    RunHandle,
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

    def start_dry_run(self, config_path: str | Path) -> dict[str, Any]:
        """Run a validated experiment config in dry-run (mock) mode."""
        config = load_experiment_config(Path(config_path))
        if not config.runtime.dry_run:
            raise ConfigError(
                "Orchestrator only permits dry_run configs; real training "
                "adapters remain disabled"
            )
        handle = run_experiment(config)
        return handle.to_dict()

    def resume(self, run_id: str) -> dict[str, Any]:
        """Resume passthrough — the framework enforces its own integrity checks."""
        handle = resume_run(run_id, self.runs_root)
        return handle.to_dict()

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
        if not selection.sample_ids:
            raise ValueError("Selection produced no rows; refusing to create an experiment")
        # Fail closed: verify every selected row against the holdout guard.
        assert_selection_safe(list(selection.rows))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        derived_manifest = out_dir / f"{selection.selection_id}.manifest.jsonl"
        header = write_selection_manifest(selection, str(derived_manifest))
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
            dataset_id=str(header.get("dataset_id", f"lab_selection_{selection.selection_id}")),
            dataset_version=selection.selection_id,
            manifest_path=derived_manifest,
            split=str(
                (criteria.split or "train")
            ),
            seed=seed,
            dry_run=dry_run,
            output_root=self.runs_root.resolve(),
            overrides=config_overrides,
        )
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
