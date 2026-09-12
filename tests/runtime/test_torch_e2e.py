"""End-to-end tests for the real torch training backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from clouda_training.runtime.adapter import SyntheticLinearAdapter  # noqa: E402
from clouda_training.runtime.backend import torch_available  # noqa: E402
from clouda_training.runtime.torch_backend import TorchTrainerBackend  # noqa: E402


def test_torch_backend_reports_capability() -> None:
    assert torch_available() is True


def test_real_optimization_learns_synthetic_task(torch_config) -> None:
    """Genuine optimization: parameters change and loss decreases measurably."""
    adapter = SyntheticLinearAdapter()

    class _Mgr:
        save = lambda self, **kw: None  # noqa: E731
        latest = lambda self: None  # noqa: E731

    from clouda_training.experiments.metrics import MetricLogger
    from pathlib import Path
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    metrics = MetricLogger(tmp / "metrics.jsonl", "probe")
    backend = TorchTrainerBackend(torch_config, metrics, _Mgr(), adapter=adapter)
    model = backend.model  # the model the backend actually optimizes

    # loss before training (fresh model)
    inputs, targets = adapter.make_batch(
        step=1, batch_size=8, seed=torch_config.training.seed
    )
    loss_before = float(adapter.forward_loss(model, (inputs, targets)).detach())

    params_before = [p.detach().clone() for p in adapter.trainable_parameters(model)]
    backend.train(start_step=0)
    params_after = adapter.trainable_parameters(model)

    # parameters genuinely changed
    changed = any(not torch.equal(a, b) for a, b in zip(params_before, params_after))
    assert changed, "optimizer steps did not modify parameters"

    # loss after training on the same distribution is measurably lower
    losses = [
        row["value"]
        for row in map(
            json.loads, (tmp / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if row["metric_name"] == "loss"
    ]
    assert losses, "no loss metrics recorded"
    assert (
        losses[-1] < loss_before * 0.9
    ), f"expected learning; loss_before={loss_before:.4f} final={losses[-1]:.4f}"


import json  # noqa: E402


def test_gradient_accumulation_matches_reference(torch_config) -> None:
    """Grad accumulation approximates the full-batch gradient."""
    adapter = SyntheticLinearAdapter()
    import tempfile
    from pathlib import Path
    from clouda_training.experiments.metrics import MetricLogger

    tmp = Path(tempfile.mkdtemp())
    metrics = MetricLogger(tmp / "metrics.jsonl", "probe")

    class _Mgr:
        save = lambda self, **kw: None  # noqa: E731
        latest = lambda self: None  # noqa: E731

    backend = TorchTrainerBackend(torch_config, metrics, _Mgr(), adapter=adapter)
    model = backend.model

    # two micro-batches of 4 vs one batch of 8, same data
    micro_grads = None
    for micro in range(2):
        model.zero_grad(set_to_none=True)
        inputs, targets = backend._step_batch(step=1, accumulation_index=micro)
        loss = adapter.forward_loss(model, (inputs, targets))
        (loss / 2).backward()
        grads = [
            p.grad.detach().clone() if p.grad is not None else None
            for p in model.parameters()
        ]
        if micro_grads is None:
            micro_grads = grads
        else:
            micro_grads = [
                a + b if a is not None and b is not None else None
                for a, b in zip(micro_grads, grads)
            ]

    model.zero_grad(set_to_none=True)
    inputs_a, targets_a = backend._step_batch(step=1, accumulation_index=0)
    inputs_b, targets_b = backend._step_batch(step=1, accumulation_index=1)
    full_inputs = torch.cat([inputs_a, inputs_b])
    full_targets = torch.cat([targets_a, targets_b])
    loss = adapter.forward_loss(model, (full_inputs, full_targets))
    loss.backward()
    full_grads = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    ]

    assert micro_grads is not None
    for micro_grad, full in zip(micro_grads, full_grads):
        if micro_grad is None or full is None:
            continue
        assert torch.allclose(
            micro_grad, full, atol=1e-10
        ), "accumulated gradients diverge from full-batch reference"


def test_gradient_clipping_respects_max_norm(torch_config) -> None:
    from dataclasses import replace

    torch_config = replace(
        torch_config, training=replace(torch_config.training, max_grad_norm=0.01)
    )
    adapter = SyntheticLinearAdapter()

    import tempfile
    from pathlib import Path
    from clouda_training.experiments.metrics import MetricLogger

    tmp = Path(tempfile.mkdtemp())
    metrics = MetricLogger(tmp / "metrics.jsonl", "probe")

    class _Mgr:
        save = lambda self, **kw: None  # noqa: E731
        latest = lambda self: None  # noqa: E731

    backend = TorchTrainerBackend(torch_config, metrics, _Mgr(), adapter=adapter)
    model = backend.model
    inputs, targets = adapter.make_batch(step=1, batch_size=8, seed=1)
    loss = adapter.forward_loss(model, (inputs, targets))
    loss.backward()
    norm_before = torch.norm(
        torch.cat(
            [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
        )
    )
    assert float(norm_before) > 0.01
    clipped = float(
        torch.nn.utils.clip_grad_norm_(
            adapter.trainable_parameters(model), torch_config.training.max_grad_norm
        )
    )
    norm_after = torch.norm(
        torch.cat(
            [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
        )
    )
    assert torch.isclose(
        norm_after,
        torch.tensor(torch_config.training.max_grad_norm, dtype=torch.float64),
        rtol=1e-4,
    )
    assert clipped >= float(norm_after)  # scale factor was applied


def test_scheduler_changes_learning_rate(torch_config) -> None:
    from dataclasses import replace

    torch_config = replace(
        torch_config,
        training=replace(torch_config.training, warmup_steps=4, scheduler="linear"),
    )
    adapter = SyntheticLinearAdapter()
    import tempfile
    from pathlib import Path
    from clouda_training.experiments.metrics import MetricLogger

    tmp = Path(tempfile.mkdtemp())
    metrics = MetricLogger(tmp / "metrics.jsonl", "probe")

    class _Mgr:
        save = lambda self, **kw: None  # noqa: E731
        latest = lambda self: None  # noqa: E731

    backend = TorchTrainerBackend(torch_config, metrics, _Mgr(), adapter=adapter)
    # lr values are logged during train; lrs list below comes from metrics
    backend.train(start_step=0)
    rows = [
        json.loads(line)
        for line in (tmp / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    lrs = [row["value"] for row in rows if row["metric_name"] == "learning_rate"]
    assert len(lrs) >= 2
    assert lrs[-1] > lrs[0]  # warmup ramps up
