"""Periodic evaluation hooks in the real torch runtime.

The adapter may expose an optional ``evaluate(model) -> dict[str, float]``
hook. When present (and evaluation is enabled), the runtime must run it on
its cadence, persist the metrics with the configured eval split, and feed
best-checkpoint selection. Adapters without the hook must be unaffected.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clouda_training.experiments.checkpoints import (  # noqa: E402
    CheckpointManager,
    list_checkpoints,
)
from clouda_training.experiments.metrics import MetricLogger  # noqa: E402
from clouda_training.runtime.adapter import (  # noqa: E402
    SyntheticLinearAdapter,
    build_synthetic_dataset_pairs,
)
from clouda_training.runtime.torch_backend import TorchTrainerBackend  # noqa: E402


class EvaluatingLinearAdapter(SyntheticLinearAdapter):
    """Reference adapter with a deterministic validation hook."""

    input_features = 3

    def build_model(self, config):  # noqa: ANN001
        self._seed = config.training.seed
        return super().build_model(config)

    def evaluate(self, model):  # noqa: ANN001
        pairs = build_synthetic_dataset_pairs(count=64, seed=self._seed + 1)
        inputs = torch.tensor([row[0] for row in pairs], dtype=torch.float64)
        targets = torch.tensor([[row[1]] for row in pairs], dtype=torch.float64)
        with torch.no_grad():
            value = float(torch.nn.functional.mse_loss(model(inputs), targets))
        return {"validation_loss": value}


def _config(torch_config, tmp_path: Path):
    return dataclasses.replace(
        torch_config,
        evaluation=replace(
            torch_config.evaluation,
            enabled=True,
            eval_steps=2,
            eval_split="validation",
            metrics=("cer", "wer"),
        ),
        checkpoint=replace(
            torch_config.checkpoint,
            save_steps=2,
            metric_for_best="validation_loss",
            greater_is_better=False,
            keep_best=True,
        ),
    )


def _run(config, tmp_path: Path, adapter) -> tuple[TorchTrainerBackend, list[dict]]:
    run_path = tmp_path / "run"
    run_path.mkdir()
    manager = CheckpointManager(run_path, "probe", config)
    metrics = MetricLogger(tmp_path / "metrics.jsonl", "probe")
    backend = TorchTrainerBackend(config, metrics, manager, adapter=adapter)
    backend.train(start_step=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return backend, rows


def test_evaluation_hook_logs_on_cadence(torch_config, tmp_path: Path) -> None:
    config = _config(torch_config, tmp_path)
    _, rows = _run(config, tmp_path, EvaluatingLinearAdapter())
    eval_rows = [
        row
        for row in rows
        if row["metric_name"] == "validation_loss" and row["split"] == "validation"
    ]
    assert eval_rows, "evaluation hook never produced metrics"
    steps = sorted({row["step"] for row in eval_rows})
    assert steps and all(step % config.evaluation.eval_steps == 0 for step in steps)


def test_evaluation_metrics_drive_best_checkpoint(torch_config, tmp_path: Path) -> None:
    config = _config(torch_config, tmp_path)
    _run(config, tmp_path, EvaluatingLinearAdapter())
    checkpoints = list_checkpoints(tmp_path / "run")
    evaluated = [item for item in checkpoints if "validation_loss" in item.metrics]
    assert evaluated, "checkpoint metadata is missing evaluation metrics"
    assert any(item.is_best for item in evaluated), "no best checkpoint selected"


def test_adapter_without_evaluate_hook_is_unaffected(
    torch_config, tmp_path: Path
) -> None:
    class NoEvalAdapter(SyntheticLinearAdapter):
        # Deliberate protocol probe: falsy, non-callable attribute -> backend
        # treats the hook as absent. None overrides the inherited method only
        # at the adapter-contract boundary, not in the real adapter base.
        evaluate = None  # type: ignore[assignment]

    config = _config(torch_config, tmp_path)
    _, rows = _run(config, tmp_path, NoEvalAdapter())
    assert not [
        row for row in rows if row["metric_name"] == "validation_loss"
    ], "adapter without evaluate() must not produce evaluation metrics"
