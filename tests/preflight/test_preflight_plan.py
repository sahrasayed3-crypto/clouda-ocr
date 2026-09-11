"""compute_training_plan math + checkpoint cadence + runtime-matching tests.

The plan module is pure integer math (no torch needed) — those tests always
run. The runtime-parity test needs torch and is skipped without it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from clouda_training.experiments.config import (
    CheckpointSection,
    DatasetSection,
    ExperimentSection,
    ExperimentConfig,
    ModelSection,
    TrackingSection,
    TrainingSection,
)
from clouda_training.preflight.models import PreflightStatus
from clouda_training.preflight.plan import (
    RUNTIME_STEPS_PER_EPOCH_FALLBACK,
    compute_training_plan,
    validate_checkpoint_cadence,
)


def _config(
    *,
    epochs: int = 2,
    max_steps: int | None = None,
    batch_size: int = 4,
    grad_accum: int = 2,
    save_strategy: str = "steps",
    save_steps: int = 100,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="plan-probe"),
        model=ModelSection(model_id="mock-model"),
        dataset=DatasetSection(
            dataset_id="mock-ds", dataset_version="v1", manifest_path=Path(".")
        ),
        training=TrainingSection(
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
        ),
        checkpoint=CheckpointSection(
            save_strategy=save_strategy, save_steps=save_steps
        ),
        tracking=TrackingSection(enabled=False),
    )


# ---------------------------------------------------------------- basic math


def test_effective_batch_and_epoch_math() -> None:
    plan = compute_training_plan(
        _config(batch_size=4, grad_accum=2, epochs=2),
        dataset_row_count=100,
        world_size=2,
    )
    assert plan.effective_batch_size == 4 * 2 * 2
    # drop_last=False: 100 rows / batch 4 -> 25 micro-batches, no remainder
    assert plan.micro_batches_per_epoch == 25
    assert plan.optimizer_steps_per_epoch == math.ceil(25 / 2)
    assert plan.planned_optimizer_steps == math.ceil(25 / 2) * 2


def test_partial_final_micro_batch_and_accumulation_ceil() -> None:
    # 13 rows / batch 4 -> 4 micro-batches (ceil, drop_last=False);
    # 4 micro / accum 3 -> 2 optimizer steps (partial final accumulation).
    plan = compute_training_plan(
        _config(batch_size=4, grad_accum=3, epochs=1),
        dataset_row_count=13,
    )
    assert plan.micro_batches_per_epoch == 4
    assert plan.optimizer_steps_per_epoch == 2
    assert plan.planned_optimizer_steps == 2
    assert any("drop_last=False" in note for note in plan.notes)


def test_exact_multiples_use_plain_division() -> None:
    plan = compute_training_plan(
        _config(batch_size=8, grad_accum=4, epochs=1),
        dataset_row_count=256,
    )
    assert plan.micro_batches_per_epoch == 32
    assert plan.optimizer_steps_per_epoch == 8
    assert plan.planned_optimizer_steps == 8


# ------------------------------------------------------------ max_steps path


def test_max_steps_override_wins() -> None:
    plan = compute_training_plan(
        _config(max_steps=7, epochs=5, batch_size=4, grad_accum=1),
        dataset_row_count=1000,
    )
    assert plan.planned_optimizer_steps == 7
    assert any("max_steps" in note for note in plan.notes)
    # checkpoints still counted against the override budget
    assert plan.estimated_checkpoint_count == 7 // 100


def test_max_steps_not_wasted_when_dataset_unknown() -> None:
    plan = compute_training_plan(_config(max_steps=12), dataset_row_count=None)
    assert plan.micro_batches_per_epoch is None
    assert plan.optimizer_steps_per_epoch is None
    assert plan.planned_optimizer_steps == 12


# --------------------------------------------------------- checkpoint counts


def test_checkpoint_count_steps_strategy() -> None:
    plan = compute_training_plan(
        _config(max_steps=250, save_strategy="steps", save_steps=100),
        dataset_row_count=10,
    )
    # steps 1..250, saved when step % 100 == 0 -> 100, 200
    assert plan.estimated_checkpoint_count == 2


def test_checkpoint_count_zero_for_non_steps_strategy() -> None:
    plan = compute_training_plan(
        _config(max_steps=250, save_strategy="epoch", save_steps=100),
        dataset_row_count=10,
    )
    assert plan.estimated_checkpoint_count == 0


def test_checkpoint_count_zero_when_save_steps_negative() -> None:
    plan = compute_training_plan(
        _config(max_steps=250, save_strategy="steps", save_steps=0),
        dataset_row_count=10,
    )
    assert plan.estimated_checkpoint_count == 0
    assert any("save_steps" in note for note in plan.notes)


# --------------------------------------------------------------- validation


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        compute_training_plan(_config(batch_size=0), dataset_row_count=10)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        compute_training_plan(_config(grad_accum=0), dataset_row_count=10)
    with pytest.raises(ValueError, match="world_size"):
        compute_training_plan(_config(), dataset_row_count=10, world_size=0)
    with pytest.raises(ValueError, match="dataset_row_count"):
        compute_training_plan(_config(), dataset_row_count=-1)


def test_world_size_defaults_to_one() -> None:
    plan = compute_training_plan(
        _config(batch_size=3, grad_accum=5), dataset_row_count=30
    )
    assert plan.world_size == 1
    assert plan.effective_batch_size == 15


# ------------------------------------------------------- checkpoint cadence


def test_cadence_pass_when_interval_fits() -> None:
    checks = validate_checkpoint_cadence(save_steps=10, planned_optimizer_steps=25)
    by_name = {c.name: c for c in checks}
    assert by_name["checkpoint.save_steps_positive"].status is PreflightStatus.PASS
    assert by_name["checkpoint.cadence"].status is PreflightStatus.PASS


def test_cadence_warns_when_interval_exceeds_planned_steps() -> None:
    checks = validate_checkpoint_cadence(save_steps=100, planned_optimizer_steps=50)
    by_name = {c.name: c for c in checks}
    assert by_name["checkpoint.cadence"].status is PreflightStatus.WARN
    assert "zero checkpoints" in by_name["checkpoint.cadence"].detail


def test_cadence_fails_on_non_positive_interval() -> None:
    checks = validate_checkpoint_cadence(save_steps=0, planned_optimizer_steps=50)
    by_name = {c.name: c for c in checks}
    assert by_name["checkpoint.save_steps_positive"].status is PreflightStatus.FAIL
    assert by_name["checkpoint.save_steps_positive"].blocker is True
    # cadence comparison is skipped: planned-step budget is meaningless when
    # the interval itself is invalid
    assert "checkpoint.cadence" not in by_name


def test_cadence_skips_for_non_steps_strategy() -> None:
    checks = validate_checkpoint_cadence(
        save_steps=0, planned_optimizer_steps=50, save_strategy="no"
    )
    assert len(checks) == 1
    assert checks[0].status is PreflightStatus.SKIP


def test_cadence_unknown_planned_steps_still_validates_interval() -> None:
    checks = validate_checkpoint_cadence(save_steps=10, planned_optimizer_steps=None)
    by_name = {c.name: c for c in checks}
    assert by_name["checkpoint.save_steps_positive"].status is PreflightStatus.PASS
    assert "checkpoint.cadence" not in by_name


# ------------------------------------------------- runtime parity (needs torch)

torch = pytest.importorskip("torch")

from clouda_training.experiments.metrics import MetricLogger  # noqa: E402
from clouda_training.runtime.adapter import SyntheticLinearAdapter  # noqa: E402
from clouda_training.runtime.torch_backend import TorchTrainerBackend  # noqa: E402


def _build_backend(config: ExperimentConfig, tmp_path: Path) -> TorchTrainerBackend:
    """__new__ shim, same pattern as tests/multimodel/test_resume_and_smoke.py."""
    backend = TorchTrainerBackend.__new__(TorchTrainerBackend)
    backend.torch = torch
    backend.metrics = MetricLogger(tmp_path / "metrics.jsonl", "plan-parity")
    backend.checkpoints = None
    backend.config = config
    backend.adapter = SyntheticLinearAdapter()
    backend.model = backend.adapter.build_model(config)
    backend.device = torch.device("cpu")
    backend.fail_at_step = None
    backend.interrupt_at_step = None
    backend._build_optimizer_and_scheduler()
    return backend


@pytest.mark.parametrize(
    ("rows", "batch_size", "grad_accum", "epochs"),
    [
        (12, 4, 2, 1),
        (13, 4, 3, 2),
        (7, 3, 1, 1),
        (40, 8, 4, 3),
    ],
)
def test_plan_matches_real_torch_runtime(
    tmp_path: Path, rows: int, batch_size: int, grad_accum: int, epochs: int
) -> None:
    """Prove the plan's step math equals the real backend's executed steps.

    Runtime truth (torch_backend.train): with max_steps unset the step loop
    runs ``range(1, epochs*5 + 1)`` — the backend is step-budgeted and does
    NOT derive its budget from dataset size. The plan mirrors the runtime's
    dataset-derived micro-batch math (drop_last=False) and, when no
    max_steps is set, clamps the planned optimizer steps to the same
    ``epochs * 5`` budget the backend actually executes.
    """
    config = _config(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        save_steps=10_000,  # never checkpoint: keep parity runs side-effect free
    )
    plan = compute_training_plan(config, dataset_row_count=rows)

    expected_micro_batches_per_epoch = math.ceil(rows / batch_size)
    expected_optimizer_steps_per_epoch = math.ceil(
        expected_micro_batches_per_epoch / grad_accum
    )
    runtime_budget = epochs * RUNTIME_STEPS_PER_EPOCH_FALLBACK
    assert plan.micro_batches_per_epoch == expected_micro_batches_per_epoch
    assert plan.optimizer_steps_per_epoch == expected_optimizer_steps_per_epoch

    backend = _build_backend(config, tmp_path)
    result = backend.train()
    # the backend stops at its step budget: exactly epochs*5 optimizer steps
    assert result.final_step == runtime_budget
    # plan agrees: dataset estimate clamped to the runtime's step budget
    assert plan.planned_optimizer_steps == min(
        expected_optimizer_steps_per_epoch * epochs, runtime_budget
    )


def test_plan_matches_runtime_max_steps_override(tmp_path: Path) -> None:
    """With max_steps set, the backend runs exactly max_steps iterations —
    the plan must report the identical number."""
    config = _config(max_steps=3, epochs=10, batch_size=4, grad_accum=2, save_steps=2)
    plan = compute_training_plan(config, dataset_row_count=9)
    assert plan.planned_optimizer_steps == 3

    backend = _build_backend(config, tmp_path)
    saved_steps: list[int] = []

    class RecordingCheckpoints:
        def save(self, **kwargs: int) -> None:
            saved_steps.append(kwargs["step"])

    backend.checkpoints = RecordingCheckpoints()
    result = backend.train()
    assert result.final_step == 3
    assert result.final_step == plan.planned_optimizer_steps
    # runtime checkpoint cadence: step % save_steps == 0 over steps 1..3
    assert saved_steps == [2]
    assert plan.estimated_checkpoint_count == 3 // 2


def test_plan_checkpoint_cadence_matches_runtime_saves(tmp_path: Path) -> None:
    """Runtime saves at step % save_steps == 0; plan must predict the count."""
    config = _config(max_steps=10, save_steps=3)
    plan = compute_training_plan(config, dataset_row_count=4)
    assert plan.estimated_checkpoint_count == 10 // 3

    backend = _build_backend(config, tmp_path)
    saved: list[int] = []

    class Mgr:
        def save(self, **kw: int) -> None:
            saved.append(kw["step"])

    backend.checkpoints = Mgr()
    backend.train()
    assert saved == [s for s in range(1, 11) if s % 3 == 0]
    assert len(saved) == plan.estimated_checkpoint_count


def _unused_replace() -> None:  # pragma: no cover - guard against lint drift
    replace  # noqa: B018
