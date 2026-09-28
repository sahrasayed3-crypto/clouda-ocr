"""Hardening tests for the real torch training backend.

Covers the failure modes the canonical runtime must surface explicitly:
non-finite loss, CUDA out-of-memory, unvalidated mixed-precision requests,
and unsafe resume offsets. All run on CPU with tiny synthetic components.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clouda_training.experiments.config import ExperimentConfig  # noqa: E402
from clouda_training.experiments.metrics import MetricLogger  # noqa: E402
from clouda_training.runtime.adapter import (  # noqa: E402
    SyntheticLinearAdapter,
    build_synthetic_dataset_pairs,
)
from clouda_training.runtime.torch_backend import TorchTrainerBackend  # noqa: E402


class _Mgr:
    def save(self, **kw):  # noqa: ANN003
        return None

    def latest(self):
        return None


def _backend(config: ExperimentConfig, adapter, tmp_path: Path) -> TorchTrainerBackend:
    metrics = MetricLogger(tmp_path / "metrics.jsonl", "probe")
    return TorchTrainerBackend(config, metrics, _Mgr(), adapter=adapter)


def test_non_finite_loss_aborts_run(torch_config, tmp_path: Path) -> None:
    class NaNAtStep2(SyntheticLinearAdapter):
        def forward_loss(self, model, batch):  # noqa: ANN001
            loss = super().forward_loss(model, batch)
            if self._step == 2:
                return loss * float("nan")
            return loss

        def make_batch(self, step, batch_size, seed):  # noqa: ANN001
            self._step = step
            return super().make_batch(step, batch_size, seed)

    backend = _backend(torch_config, NaNAtStep2(), tmp_path)
    with pytest.raises(RuntimeError, match="[Nn]on-finite.*step 2"):
        backend.train(start_step=0)


def test_infinite_grad_norm_aborts_run(torch_config, tmp_path: Path) -> None:
    class InfGradAdapter(SyntheticLinearAdapter):
        def forward_loss(self, model, batch):  # noqa: ANN001
            loss = super().forward_loss(model, batch)
            # huge-but-finite loss produces an inf grad norm after backward
            return loss * 1e300

    config = dataclasses.replace(
        torch_config,
        training=dataclasses.replace(torch_config.training, max_grad_norm=1.0),
    )
    backend = _backend(config, InfGradAdapter(), tmp_path)
    with pytest.raises(RuntimeError, match="[Nn]on-finite"):
        backend.train(start_step=0)


def test_cuda_oom_reports_actionable_error(torch_config, tmp_path: Path) -> None:
    class OOMAtStep2(SyntheticLinearAdapter):
        def forward_loss(self, model, batch):  # noqa: ANN001
            if self._step == 2:
                raise torch.cuda.OutOfMemoryError(
                    "CUDA out of memory. Tried to allocate 2.00 GiB"
                )
            return super().forward_loss(model, batch)

        def make_batch(self, step, batch_size, seed):  # noqa: ANN001
            self._step = step
            return super().make_batch(step, batch_size, seed)

    backend = _backend(torch_config, OOMAtStep2(), tmp_path)
    with pytest.raises(RuntimeError, match="[Oo]ut of memory.*batch_size"):
        backend.train(start_step=0)


def test_mixed_precision_requires_explicit_support(torch_config) -> None:
    config = dataclasses.replace(
        torch_config,
        training=dataclasses.replace(torch_config.training, mixed_precision=True),
    )
    with pytest.raises(RuntimeError, match="mixed_precision"):
        _backend(config, SyntheticLinearAdapter(), Path("_unused"))


def test_resume_rejects_start_step_older_than_checkpoint(
    torch_config, tmp_path: Path
) -> None:
    """start_step must never rewind behind the restored checkpoint state."""
    from clouda_training.experiments.checkpoints import CheckpointManager

    run_path = tmp_path / "run"
    run_path.mkdir()
    manager = CheckpointManager(run_path, "probe", torch_config)
    metrics = MetricLogger(tmp_path / "metrics.jsonl", "probe")
    backend = TorchTrainerBackend(torch_config, metrics, manager)
    info = manager.save(step=4, epoch=4.0, metrics={"loss": 0.5})
    backend._save_checkpoint(step=4, metrics={"loss": 0.5})
    assert info.path.is_dir()

    with pytest.raises(RuntimeError, match="[Cc]heckpoint.*step"):
        backend.train(start_step=2)


def test_full_synthetic_run_matches_dataset_contract(
    torch_config, tmp_path: Path
) -> None:
    """The reference trainer learns the fixed synthetic mapping end to end."""
    config = dataclasses.replace(
        torch_config,
        training=dataclasses.replace(
            torch_config.training, max_steps=300, learning_rate=0.05
        ),
    )
    adapter = SyntheticLinearAdapter()
    backend = _backend(config, adapter, tmp_path)
    backend.train(start_step=0)
    pairs = build_synthetic_dataset_pairs(count=64, seed=torch_config.training.seed)
    inputs = torch.tensor([row[0] for row in pairs], dtype=torch.float64)
    targets = torch.tensor([[row[1]] for row in pairs], dtype=torch.float64)
    with torch.no_grad():
        mse = float(torch.nn.functional.mse_loss(backend.model(inputs), targets))
    assert mse < 0.05, f"synthetic trainer failed to learn; final val MSE={mse:.4f}"


def test_checkpoint_metrics_json_is_serializable(torch_config, tmp_path: Path) -> None:
    """Checkpoint payloads must remain strict JSON (no tensor leakage)."""
    from clouda_training.experiments.checkpoints import CheckpointManager

    run_path = tmp_path / "run"
    run_path.mkdir()
    manager = CheckpointManager(run_path, "probe", torch_config)
    metrics = MetricLogger(tmp_path / "metrics.jsonl", "probe")
    backend = TorchTrainerBackend(torch_config, metrics, manager)
    backend._save_checkpoint(step=4, metrics={"loss": 0.25})
    payload = json.loads(
        (run_path / "checkpoints" / "step-00000004" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["metrics"] == {"loss": 0.25}
