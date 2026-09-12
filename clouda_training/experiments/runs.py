from __future__ import annotations

import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from clouda_contracts.checksums import sha256_file

from .checkpoints import CheckpointManager, list_checkpoints
from .config import ExperimentConfig, config_from_dict
from .dataset import validate_training_dataset
from .environment import apply_seed, capture_environment
from .io import atomic_write_json, read_json
from .metrics import MetricLogger, read_metrics, utc_now
from clouda_training.runtime.backend import torch_available
from clouda_training.runtime.mock_backend import MockTrainerBackend


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    path: Path
    status: RunStatus

    def metrics(self) -> list[dict[str, Any]]:
        return list(read_metrics(self.path / "metrics.jsonl"))

    def summary(self) -> dict[str, Any]:
        return read_json(self.path / "summary.json")

    def to_dict(self) -> dict[str, Any]:
        payload = read_json(self.path / "metadata.json")
        payload.update(read_json(self.path / "status.json"))
        if (self.path / "summary.json").is_file():
            payload["summary"] = self.summary()
        payload["path"] = str(self.path)
        return payload


def _git_metadata() -> tuple[str | None, bool | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _new_run_id(name: str, config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}__{timestamp}__{config_hash[:8]}-{uuid.uuid4().hex[:6]}"


def _status(path: Path, status: RunStatus, **details: Any) -> None:
    prior = read_json(path / "status.json") if (path / "status.json").is_file() else {}
    now = utc_now()
    history = list(prior.get("history", []))
    history.append({"status": status.value, "timestamp": now})
    atomic_write_json(
        path / "status.json",
        {
            "schema_version": 1,
            "run_id": path.name,
            "status": status.value,
            "created_at": prior.get("created_at", utc_now()),
            "updated_at": now,
            "history": history,
            **details,
        },
    )
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        metadata["status"] = status.value
        if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}:
            metadata["end_timestamp"] = details.get("end_timestamp", now)
        atomic_write_json(metadata_path, metadata)


def _summarize(
    run_path: Path, *, resumed_from_step: int, duration: float
) -> dict[str, Any]:
    records = list(read_metrics(run_path / "metrics.jsonl"))
    final: dict[str, float] = {}
    best: dict[str, float] = {}
    for record in records:
        name, value = str(record["metric_name"]), float(record["value"])
        final[name] = value
        best[name] = (
            max(best.get(name, value), value)
            if name in {"samples_per_second", "tokens_per_second"}
            else min(best.get(name, value), value)
        )
    checkpoints = list_checkpoints(run_path)
    best_checkpoint = next(
        (str(item.path) for item in checkpoints if item.is_best), None
    )
    return {
        "schema_version": 1,
        "run_id": run_path.name,
        "status": RunStatus.COMPLETED.value,
        "duration_seconds": duration,
        "resumed_from_step": resumed_from_step,
        "final_metrics": final,
        "best_metrics": best,
        "best_checkpoint": best_checkpoint,
        "checkpoint_count": len(checkpoints),
    }


_INTEGRITY_FILES = (
    "resolved_config.yaml",
    "metadata.json",
    "status.json",
    "metrics.jsonl",
    "summary.json",
    "environment.json",
)


def _write_integrity(run_path: Path) -> None:
    atomic_write_json(
        run_path / "artifacts" / "integrity.json",
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "files": {
                name: sha256_file(run_path / name)
                for name in _INTEGRITY_FILES
                if (run_path / name).is_file()
            },
        },
    )


def verify_run_integrity(run_path: str | Path) -> dict[str, Any]:
    path = Path(run_path)
    manifest = read_json(path / "artifacts" / "integrity.json")
    missing: list[str] = []
    mismatches: list[str] = []
    for name, expected in manifest.get("files", {}).items():
        artifact = path / name
        if not artifact.is_file():
            missing.append(name)
        elif sha256_file(artifact) != expected:
            mismatches.append(name)
    return {
        "valid": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
    }


def _execute(
    run_path: Path,
    config: ExperimentConfig,
    *,
    start_step: int,
    fail_at_step: int | None,
    interrupt_at_step: int | None,
    data_loader: Any | None = None,
) -> RunHandle:
    _status(run_path, RunStatus.RUNNING, start_step=start_step)
    seed_details = apply_seed(
        config.training.seed, deterministic=config.runtime.deterministic
    )
    metadata = read_json(run_path / "metadata.json")
    metadata["determinism"] = seed_details
    atomic_write_json(run_path / "metadata.json", metadata)
    manager = CheckpointManager(run_path, run_path.name, config)
    metric_logger = MetricLogger(run_path / "metrics.jsonl", run_path.name)
    trainer: Any
    if config.model.adapter_type == "torch":
        if not torch_available():
            raise ImportError(
                "adapter_type='torch' requires PyTorch. Install with: "
                "pip install clouda-pdf[training-torch]  (or: pip install torch)"
            )
        from clouda_training.runtime.torch_backend import TorchTrainerBackend

        trainer = TorchTrainerBackend(
            config,
            metric_logger,
            manager,
            fail_at_step=fail_at_step,
            interrupt_at_step=interrupt_at_step,
        )
    else:
        trainer = MockTrainerBackend(
            config,
            metric_logger,
            manager,
            fail_at_step=fail_at_step,
            interrupt_at_step=interrupt_at_step,
            data_loader=data_loader,
        )
    try:
        result = trainer.train(start_step=start_step)
        summary = _summarize(
            run_path,
            resumed_from_step=result.resumed_from_step,
            duration=result.duration_seconds,
        )
        atomic_write_json(run_path / "summary.json", summary)
        _status(
            run_path,
            RunStatus.COMPLETED,
            end_timestamp=utc_now(),
            final_step=result.final_step,
        )
        _write_integrity(run_path)
        return RunHandle(run_path.name, run_path, RunStatus.COMPLETED)
    except KeyboardInterrupt:
        _status(
            run_path,
            RunStatus.INTERRUPTED,
            end_timestamp=utc_now(),
            last_completed_step=(
                start_step
                if not list(read_metrics(run_path / "metrics.jsonl"))
                else max(
                    int(row["step"]) for row in read_metrics(run_path / "metrics.jsonl")
                )
            ),
        )
        raise
    except Exception as exc:
        _status(
            run_path,
            RunStatus.FAILED,
            end_timestamp=utc_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def run_experiment(
    config: ExperimentConfig,
    *,
    fail_at_step: int | None = None,
    interrupt_at_step: int | None = None,
    data_loader: Any | None = None,
) -> RunHandle:
    identity = validate_training_dataset(config.dataset)
    if config.model.adapter_type == "torch":
        if not torch_available():
            raise RuntimeError(
                "Real torch training requested but PyTorch is not installed"
            )
    elif not config.runtime.dry_run or config.model.adapter_type not in {
        "mock",
        "dry_run",
    }:
        raise RuntimeError("Real training adapters are not enabled; use mock/dry_run")
    experiment_root = config.runtime.output_root / config.experiment.name
    experiment_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = _new_run_id(config.experiment.name, config.hash)
        run_path = experiment_root / run_id
        try:
            run_path.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Unable to allocate a unique run directory")
    for directory in ("checkpoints", "logs", "artifacts"):
        (run_path / directory).mkdir()
    commit, dirty = _git_metadata()
    started = utc_now()
    metadata = {
        "schema_version": 1,
        "experiment_name": config.experiment.name,
        "run_id": run_id,
        "config_hash": config.hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "start_timestamp": started,
        "end_timestamp": None,
        "status": RunStatus.CREATED.value,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "random_seed": config.training.seed,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.dataset.dataset_id,
        "dataset_version": config.dataset.dataset_version,
        "dataset_manifest_path": str(config.dataset.manifest_path),
        "dataset_manifest_hash": identity.manifest_hash,
        "dataset_split": config.dataset.split,
        "dataset_rows": identity.row_count,
        "source_provenance": {
            "source_ids": list(identity.source_ids),
            "licenses": list(identity.source_licenses),
        },
        "preprocessing_version": config.dataset.preprocessing_version,
        "code_version": commit,
        "command_line": sys.argv,
        "resume_source": config.checkpoint.resume_from,
        "tags": list(config.experiment.tags),
    }
    if data_loader is not None:
        metadata["training_data"] = _training_data_identity(data_loader)
    _status(run_path, RunStatus.CREATED, start_timestamp=started)
    atomic_write_json(run_path / "metadata.json", metadata)
    try:
        (run_path / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        atomic_write_json(run_path / "environment.json", capture_environment())
    except Exception as exc:
        _status(
            run_path,
            RunStatus.FAILED,
            end_timestamp=utc_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    return _execute(
        run_path,
        config,
        start_step=0,
        fail_at_step=fail_at_step,
        interrupt_at_step=interrupt_at_step,
        data_loader=data_loader,
    )


def load_run(run_id: str, runs_root: str | Path) -> RunHandle:
    matches = list(Path(runs_root).glob(f"*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Run id not found or ambiguous: {run_id}")
    status = RunStatus(read_json(matches[0] / "status.json")["status"])
    return RunHandle(run_id, matches[0], status)


def resume_run(
    run_id: str, runs_root: str | Path, *, data_loader: Any | None = None
) -> RunHandle:
    run = load_run(run_id, runs_root)
    payload = yaml.safe_load(
        (run.path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    config = config_from_dict(payload, path_base=run.path)
    validate_training_dataset(config.dataset)
    manager = CheckpointManager(run.path, run_id, config)
    checkpoint = manager.latest()
    if checkpoint is None:
        raise ValueError("Run has no checkpoint to resume")
    manager.validate_resume(checkpoint)
    if run.status not in {RunStatus.INTERRUPTED, RunStatus.FAILED}:
        raise ValueError(f"Run status {run.status.value} is not resumable")
    metadata = read_json(run.path / "metadata.json")
    recorded_training_data = metadata.get("training_data")
    if recorded_training_data is not None:
        if data_loader is None:
            raise ValueError(
                "Training-data checkpoint requires a compatible data loader to resume"
            )
        current_training_data = _training_data_identity(data_loader)
        if current_training_data != recorded_training_data:
            raise ValueError(
                "Incompatible training-data identity for resume: "
                f"{current_training_data!r}"
            )
        from clouda_data.training_data.checkpoint_bridge import LoaderCheckpointHook

        LoaderCheckpointHook(data_loader).restore_from_checkpoint(checkpoint.path)
    metadata["resume_source"] = str(checkpoint.path)
    atomic_write_json(run.path / "metadata.json", metadata)
    return _execute(
        run.path,
        config,
        start_step=checkpoint.step,
        fail_at_step=None,
        interrupt_at_step=None,
        data_loader=data_loader,
    )


def _training_data_identity(data_loader: Any) -> dict[str, Any]:
    identity = data_loader.open()
    return {
        "schema_version": "clouda.training_data.run_lineage.v1",
        "dataset_id": identity.dataset_id,
        "dataset_version": identity.dataset_version,
        "manifest_sha256": identity.manifest_sha256,
        "loader_config_hash": data_loader.config_hash,
        "shard_index_sha256": sha256_file(data_loader.index_path),
        "shard_ids": [entry.shard_id for entry in data_loader.index.shards],
    }


def list_runs(
    runs_root: str | Path,
    *,
    status: RunStatus | None = None,
    tags: set[str] | None = None,
) -> list[RunHandle]:
    results: list[RunHandle] = []
    root = Path(runs_root)
    if not root.is_dir():
        return results
    for path in root.glob("*/*"):
        if (
            not (path / "metadata.json").is_file()
            or not (path / "status.json").is_file()
        ):
            continue
        handle = RunHandle(
            path.name, path, RunStatus(read_json(path / "status.json")["status"])
        )
        metadata = read_json(path / "metadata.json")
        if status is not None and handle.status is not status:
            continue
        if tags and not tags.issubset(set(metadata.get("tags", []))):
            continue
        results.append(handle)
    return sorted(
        results,
        key=lambda item: read_json(item.path / "metadata.json").get(
            "start_timestamp", ""
        ),
        reverse=True,
    )
