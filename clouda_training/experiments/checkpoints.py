from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clouda_contracts.checksums import sha256_file

from .config import ExperimentConfig
from .io import atomic_write_json, read_json
from .metrics import utc_now


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    metadata_path: Path
    step: int
    epoch: float
    timestamp: str
    metrics: dict[str, float]
    config_hash: str
    run_id: str
    integrity_sha256: str
    is_best: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["metadata_path"] = str(self.metadata_path)
        return payload


def _load_checkpoint(path: Path) -> CheckpointInfo:
    metadata_path = path / "metadata.json"
    payload = read_json(metadata_path)
    required = {
        "step",
        "epoch",
        "timestamp",
        "metrics",
        "config_hash",
        "run_id",
        "integrity_sha256",
        "experiment_name",
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_version",
    }
    if required - set(payload):
        raise ValueError(f"Corrupt checkpoint metadata: {metadata_path}")
    state = path / "state.json"
    if not state.is_file() or sha256_file(state) != payload["integrity_sha256"]:
        raise ValueError(f"Checkpoint integrity validation failed: {path}")
    return CheckpointInfo(
        path=path,
        metadata_path=metadata_path,
        step=int(payload["step"]),
        epoch=float(payload["epoch"]),
        timestamp=str(payload["timestamp"]),
        metrics={str(k): float(v) for k, v in payload["metrics"].items()},
        config_hash=str(payload["config_hash"]),
        run_id=str(payload["run_id"]),
        integrity_sha256=str(payload["integrity_sha256"]),
        is_best=bool(payload.get("is_best", False)),
    )


def list_checkpoints(run_path: str | Path) -> list[CheckpointInfo]:
    root = Path(run_path) / "checkpoints"
    if not root.is_dir():
        return []
    return [
        _load_checkpoint(path) for path in sorted(root.glob("step-*")) if path.is_dir()
    ]


class CheckpointManager:
    def __init__(self, run_path: Path, run_id: str, config: ExperimentConfig) -> None:
        self.root = run_path / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = config

    def save(
        self, *, step: int, epoch: float, metrics: dict[str, float]
    ) -> CheckpointInfo:
        final = self.root / f"step-{step:08d}"
        if final.exists():
            return _load_checkpoint(final)
        staging = self.root / f".step-{step:08d}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        atomic_write_json(staging / "state.json", {"step": step, "epoch": epoch})
        digest = sha256_file(staging / "state.json")
        payload = {
            "schema_version": 1,
            "step": step,
            "epoch": epoch,
            "timestamp": utc_now(),
            "metrics": metrics,
            "config_hash": self.config.hash,
            "run_id": self.run_id,
            "integrity_sha256": digest,
            "experiment_name": self.config.experiment.name,
            "model_id": self.config.model.model_id,
            "model_revision": self.config.model.revision,
            "dataset_id": self.config.dataset.dataset_id,
            "dataset_version": self.config.dataset.dataset_version,
            "is_best": False,
        }
        atomic_write_json(staging / "metadata.json", payload)
        staging.rename(final)
        self._mark_best_and_retain()
        return _load_checkpoint(final)

    def _mark_best_and_retain(self) -> None:
        items = list_checkpoints(self.root.parent)
        metric = self.config.checkpoint.metric_for_best
        eligible = [item for item in items if metric in item.metrics]
        best_path: Path | None = None
        if eligible and self.config.checkpoint.keep_best:

            def metric_value(item: CheckpointInfo) -> float:
                return item.metrics[metric]

            best_path = (
                max(eligible, key=metric_value).path
                if self.config.checkpoint.greater_is_better
                else min(eligible, key=metric_value).path
            )
        for item in items:
            payload = read_json(item.metadata_path)
            wanted = item.path == best_path
            if bool(payload.get("is_best")) != wanted:
                payload["is_best"] = wanted
                atomic_write_json(item.metadata_path, payload)
        limit = self.config.checkpoint.save_total_limit
        keep: set[Path] = set()
        if best_path is not None:
            keep.add(best_path)
        remaining = limit - len(keep)
        if remaining:
            newest = [item.path for item in reversed(items) if item.path not in keep]
            keep.update(newest[:remaining])
        for item in items:
            if item.path not in keep:
                shutil.rmtree(item.path)

    def latest(self) -> CheckpointInfo | None:
        items = list_checkpoints(self.root.parent)
        return items[-1] if items else None

    def validate_resume(self, checkpoint: CheckpointInfo) -> None:
        payload = read_json(checkpoint.metadata_path)
        expected = {
            "experiment_name": self.config.experiment.name,
            "model_id": self.config.model.model_id,
            "model_revision": self.config.model.revision,
            "dataset_id": self.config.dataset.dataset_id,
            "dataset_version": self.config.dataset.dataset_version,
            "config_hash": self.config.hash,
            "run_id": self.run_id,
        }
        differences = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if differences:
            raise ValueError(f"Incompatible checkpoint for resume: {differences}")
