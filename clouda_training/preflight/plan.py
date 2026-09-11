"""Pure-integer training-plan math mirroring the real runtime semantics.

This module performs NO I/O and imports neither torch nor transformers.
Every formula below mirrors what ``clouda_training.runtime.torch_backend.
TorchTrainerBackend.train`` actually does:

* the step loop is ``for step in range(start_step + 1, maximum + 1)`` with
  ``maximum = max_steps or epochs * 5`` — one optimizer step per iteration;
* each optimizer step consumes ``gradient_accumulation_steps`` micro-batches
  (``for micro in range(accumulation)``), each of ``batch_size`` samples
  (``adapter.make_batch(step, batch_size, seed)``);
* the backend is single-process, so the plan's ``world_size`` factor is a
  forward-looking scaling estimate (1 = exact runtime match);
* a step-based checkpoint is written whenever
  ``save_strategy == "steps" and step % save_steps == 0``, so within
  ``planned`` steps the checkpoint count is ``planned // save_steps``.

The dataset row math assumes ``drop_last=False``: the runtime consumes every
sample it is given (synthetic streaming adapter), so the trailing partial
batch is kept and rounded with ceiling division.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from clouda_training.preflight.models import (
    PreflightCheck,
    PreflightStatus,
    TrainingPlanSummary,
)

if TYPE_CHECKING:
    from clouda_training.experiments.config import ExperimentConfig

__all__ = [
    "RUNTIME_STEPS_PER_EPOCH_FALLBACK",
    "compute_training_plan",
    "validate_checkpoint_cadence",
]

# torch_backend.train(): maximum = max_steps or epochs * 5. The real backend
# is step-budgeted, not epoch-driven, so its no-max_steps budget is a fixed
# 5 optimizer steps per epoch regardless of dataset size. The plan reports
# the dataset-derived estimate per spec and notes this runtime fallback.
RUNTIME_STEPS_PER_EPOCH_FALLBACK = 5


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division (no floats, exact for large ints)."""
    return -(-numerator // denominator)


def compute_training_plan(
    config: ExperimentConfig,
    *,
    dataset_row_count: int | None,
    world_size: int = 1,
) -> TrainingPlanSummary:
    """Compute the planned training shape with pure integer math.

    Formulas (mirroring ``torch_backend.train()``):

    * ``effective_batch_size = batch_size * gradient_accumulation_steps *
      world_size`` — samples consumed per optimizer step.
    * ``micro_batches_per_epoch = ceil(dataset_row_count / batch_size)``
      (``drop_last=False``: the trailing partial batch is kept, because the
      runtime consumes all samples).
    * ``optimizer_steps_per_epoch = ceil(micro_batches_per_epoch /
      gradient_accumulation_steps)`` — the final partial accumulation group
      still triggers one ``optimizer.step()`` in the runtime.
    * ``planned_optimizer_steps = max_steps`` when set (override wins,
      matching ``maximum = max_steps or ...``); otherwise
      ``optimizer_steps_per_epoch * epochs``.
    * ``estimated_checkpoint_count = planned_optimizer_steps // save_steps``
      when ``save_strategy == "steps"`` and ``save_steps > 0`` (steps run
      from 1..planned, saved when ``step % save_steps == 0``); 0 otherwise.

    Raises ``ValueError`` for non-positive ``batch_size`` /
    ``gradient_accumulation_steps`` / ``world_size`` or negative
    ``dataset_row_count`` — the same misconfigurations the config checks
    reject, enforced here so the plan can never compute garbage.
    """
    training = config.training
    checkpoint = config.checkpoint

    if training.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {training.batch_size!r}")
    if training.gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be > 0, got "
            f"{training.gradient_accumulation_steps!r}"
        )
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size!r}")
    if dataset_row_count is not None and dataset_row_count < 0:
        raise ValueError(f"dataset_row_count must be >= 0, got {dataset_row_count!r}")

    notes: list[str] = [
        "drop_last=False assumption: runtime consumes all samples; "
        "trailing partial batch is kept (ceiling division)",
        "effective_batch_size = batch_size * gradient_accumulation_steps "
        "* world_size",
    ]

    effective_batch_size = (
        training.batch_size * training.gradient_accumulation_steps * world_size
    )

    micro_batches_per_epoch: int | None = None
    optimizer_steps_per_epoch: int | None = None
    if dataset_row_count is not None:
        micro_batches_per_epoch = _ceil_div(dataset_row_count, training.batch_size)
        optimizer_steps_per_epoch = _ceil_div(
            micro_batches_per_epoch, training.gradient_accumulation_steps
        )

    max_steps = training.max_steps
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0 when set, got {max_steps!r}")
        planned_optimizer_steps = max_steps
        notes.append("max_steps override wins over the epoch-based estimate")
    else:
        planned_optimizer_steps = (optimizer_steps_per_epoch or 0) * training.epochs
        notes.append(
            "no max_steps: planned steps use the dataset-derived epoch "
            "estimate; the real torch backend budgets "
            f"epochs * {RUNTIME_STEPS_PER_EPOCH_FALLBACK} optimizer steps "
            "(torch_backend.train step loop) — keep both in mind"
        )

    if checkpoint.save_strategy == "steps":
        if checkpoint.save_steps > 0:
            estimated_checkpoint_count = (
                planned_optimizer_steps // checkpoint.save_steps
            )
        else:
            estimated_checkpoint_count = 0
            notes.append(
                "save_steps <= 0: no step-based checkpoints planned; "
                "validate_checkpoint_cadence reports FAIL"
            )
    else:
        estimated_checkpoint_count = 0
        notes.append(
            f"save_strategy={checkpoint.save_strategy!r}: no step-based "
            "checkpoints planned"
        )

    return TrainingPlanSummary(
        effective_batch_size=effective_batch_size,
        micro_batches_per_epoch=micro_batches_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        planned_optimizer_steps=planned_optimizer_steps,
        estimated_checkpoint_count=estimated_checkpoint_count,
        world_size=world_size,
        notes=tuple(notes),
    )


def validate_checkpoint_cadence(
    *,
    save_steps: int,
    planned_optimizer_steps: int | None,
    save_strategy: str = "steps",
) -> tuple[PreflightCheck, ...]:
    """Validate the checkpoint interval against the planned run length.

    Returns machine-readable :class:`PreflightCheck` items:

    * ``checkpoint.save_steps_positive`` — ``FAIL`` blocker when
      ``save_steps <= 0`` under a step-based strategy (the runtime would
      never save, or divide by zero in ``step % save_steps``); ``PASS``
      otherwise. Non-step strategies yield ``SKIP`` (nothing to validate).
    * ``checkpoint.cadence`` — ``WARN`` when the interval exceeds the
      planned optimizer steps (zero checkpoints would be written for the
      whole run); ``PASS`` when at least one checkpoint is expected.
    """
    checks: list[PreflightCheck] = []

    if save_strategy != "steps":
        return (
            PreflightCheck(
                name="checkpoint.cadence",
                status=PreflightStatus.SKIP,
                detail=(
                    f"save_strategy={save_strategy!r}: no step-based "
                    "checkpoint cadence to validate"
                ),
            ),
        )

    if save_steps <= 0:
        checks.append(
            PreflightCheck(
                name="checkpoint.save_steps_positive",
                status=PreflightStatus.FAIL,
                detail=(
                    f"save_steps must be > 0 under save_strategy='steps', "
                    f"got {save_steps!r}; no checkpoints would ever be "
                    "written"
                ),
                blocker=True,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="checkpoint.save_steps_positive",
                status=PreflightStatus.PASS,
                detail=f"save_steps={save_steps}",
            )
        )

    if save_steps > 0 and planned_optimizer_steps is not None:
        if save_steps > planned_optimizer_steps:
            checks.append(
                PreflightCheck(
                    name="checkpoint.cadence",
                    status=PreflightStatus.WARN,
                    detail=(
                        f"save_steps={save_steps} exceeds the "
                        f"{planned_optimizer_steps} planned optimizer steps; "
                        "zero checkpoints would be written this run"
                    ),
                )
            )
        else:
            expected = planned_optimizer_steps // save_steps
            checks.append(
                PreflightCheck(
                    name="checkpoint.cadence",
                    status=PreflightStatus.PASS,
                    detail=(
                        f"~{expected} checkpoint(s) at every "
                        f"{save_steps} of {planned_optimizer_steps} steps"
                    ),
                )
            )

    return tuple(checks)
