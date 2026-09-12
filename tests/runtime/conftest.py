"""Shared fixtures for runtime tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def torch_manifest(tmp_path: Path) -> Path:
    rows = [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": 1,
            "dataset_role": "training",
        },
        {
            "sample_id": "sample-1",
            "target_split": "train",
            "source_id": "synthetic-test",
            "source_path": "page.png",
            "source_license": "Apache-2.0",
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture()
def torch_config(torch_manifest: Path, tmp_path: Path):
    from clouda_training.experiments import load_experiment_config

    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": {"name": "torch_runtime_probe", "tags": ["test"]},
        "model": {
            "model_id": "synthetic/linear",
            "revision": "probe-v1",
            "model_family": "synthetic",
            "adapter_type": "torch",
        },
        "dataset": {
            "dataset_id": "synthetic-test",
            "dataset_version": "v1",
            "manifest_path": str(torch_manifest),
            "split": "train",
        },
        "training": {
            "seed": 17,
            "max_steps": 12,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.05,
            "scheduler": "linear",
            "max_grad_norm": 1.0,
        },
        "checkpoint": {
            "save_strategy": "steps",
            "save_steps": 4,
            "save_total_limit": 3,
        },
        "evaluation": {"enabled": False},
        "runtime": {
            "device": "cpu",
            "output_root": str(tmp_path / "runs"),
            "dry_run": False,
            "offline": True,
            "deterministic": True,
        },
        "tracking": {"enabled": True, "backend": "jsonl", "log_steps": 1},
    }
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_experiment_config(config_path)
