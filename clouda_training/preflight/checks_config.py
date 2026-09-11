"""Configuration validation checks for the training preflight validator.

``validate_config`` turns an :class:`ExperimentConfig` into a list of
:class:`PreflightCheck`. Every invalid field yields exactly one ``FAIL``
check (with ``blocker=True`` — a misconfigured run must never start) and
an actionable detail string; every valid field yields exactly one ``PASS``
check, so the list length is constant for a given config shape.

These checks are pure: no I/O except the ``resume_from`` filesystem
existence probe (see ``checkpoint.resume_from`` below). Torch and
transformers are never imported here.
"""

from __future__ import annotations

import math
from pathlib import Path

from clouda_training.experiments.config import ExperimentConfig
from clouda_training.preflight.models import PreflightCheck, PreflightStatus

# Schedulers actually supported by TorchTrainerBackend._build_optimizer_and_scheduler().
_SUPPORTED_SCHEDULERS = ("none", "linear", "cosine")
# Devices the runtime can target (cpu always; cuda via torch.cuda).
_SUPPORTED_DEVICES = ("cpu", "cuda")
# Checkpoint strategies the preflight admits for a training run.
_SUPPORTED_SAVE_STRATEGIES = ("none", "steps")

__all__ = ["validate_config"]


def _check(name: str, ok: bool, detail_pass: str, detail_fail: str) -> PreflightCheck:
    if ok:
        return PreflightCheck(
            name=name, status=PreflightStatus.PASS, detail=detail_pass, blocker=True
        )
    return PreflightCheck(
        name=name, status=PreflightStatus.FAIL, detail=detail_fail, blocker=True
    )


def _number_check(
    name: str,
    value: object,
    *,
    minimum: float,
    inclusive: bool,
    label: str,
    requirement: str,
) -> PreflightCheck:
    """Validate a numeric field: must be a real, finite number >=/> minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        got = type(value).__name__
        return _check(
            name,
            False,
            "",
            f"{label} must be a number, got {got}: {value!r}. Set {label} to {requirement}.",
        )
    if not math.isfinite(value):
        return _check(
            name,
            False,
            "",
            f"{label} must be a finite number, got {value!r}. Set {label} to {requirement}.",
        )
    ok = value >= minimum if inclusive else value > minimum
    return _check(
        name,
        ok,
        f"{label} = {value}",
        f"{label} = {value!r} but it must be {requirement}.",
    )


def validate_config(config: ExperimentConfig) -> list[PreflightCheck]:
    """Validate an experiment configuration; return one check per rule.

    Every check is a blocker: a ``FAIL`` here means the configuration is
    invalid and training must not start.
    """
    checks: list[PreflightCheck] = []
    training = config.training
    checkpoint = config.checkpoint
    runtime = config.runtime
    tracking = config.tracking

    checks.append(
        _number_check(
            "training.batch_size",
            training.batch_size,
            minimum=1,
            inclusive=True,
            label="training.batch_size",
            requirement="a positive integer (>= 1)",
        )
    )
    checks.append(
        _number_check(
            "training.gradient_accumulation_steps",
            training.gradient_accumulation_steps,
            minimum=1,
            inclusive=True,
            label="training.gradient_accumulation_steps",
            requirement="a positive integer (>= 1)",
        )
    )
    checks.append(
        _number_check(
            "training.learning_rate",
            training.learning_rate,
            minimum=0,
            inclusive=False,
            label="training.learning_rate",
            requirement="a positive float (e.g. 5e-5)",
        )
    )

    # At least one finite training plan: epochs >= 1 OR max_steps > 0.
    epochs_ok = (
        isinstance(training.epochs, int)
        and not isinstance(training.epochs, bool)
        and training.epochs >= 1
    )
    max_steps = training.max_steps
    max_steps_ok = (
        isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0
    )
    checks.append(
        _check(
            "training.epochs_or_max_steps",
            epochs_ok or max_steps_ok,
            f"training plan: epochs={training.epochs}, max_steps={max_steps}",
            "no finite training plan: set training.epochs >= 1 or "
            "training.max_steps > 0 (at least one must define a bounded run)",
        )
    )

    # max_steps, WHEN SET, must be positive: the runtime honors it as an
    # exact step budget, so max_steps<=0 would silently run ZERO optimizer
    # steps and report a successful no-op training run (false-READY guard).
    if max_steps is not None and not max_steps_ok:
        checks.append(
            _check(
                "training.max_steps_positive",
                False,
                f"training.max_steps={max_steps}",
                (
                    f"training.max_steps must be > 0 when set "
                    f"(got {max_steps!r}); set it to null/0 to use the "
                    "epochs-based budget instead"
                ),
            )
        )

    checks.append(
        _number_check(
            "training.weight_decay",
            training.weight_decay,
            minimum=0,
            inclusive=True,
            label="training.weight_decay",
            requirement="a non-negative float (>= 0)",
        )
    )
    checks.append(
        _number_check(
            "training.warmup_steps",
            training.warmup_steps,
            minimum=0,
            inclusive=True,
            label="training.warmup_steps",
            requirement="a non-negative integer (>= 0)",
        )
    )

    scheduler = training.scheduler
    checks.append(
        _check(
            "training.scheduler",
            scheduler in _SUPPORTED_SCHEDULERS,
            f"training.scheduler = {scheduler!r}",
            f"training.scheduler = {scheduler!r} but the runtime only supports "
            f"{list(_SUPPORTED_SCHEDULERS)}",
        )
    )

    checks.append(
        _number_check(
            "training.max_grad_norm",
            training.max_grad_norm,
            minimum=0,
            inclusive=True,
            label="training.max_grad_norm",
            requirement="a non-negative float (>= 0)",
        )
    )

    checks.append(
        _check(
            "training.seed",
            training.seed is not None,
            f"training.seed = {training.seed}",
            "training.seed is not set; set it to an integer for reproducible runs",
        )
    )

    device = runtime.device
    checks.append(
        _check(
            "runtime.device",
            device in _SUPPORTED_DEVICES,
            f"runtime.device = {device!r}",
            f"runtime.device = {device!r} but only {list(_SUPPORTED_DEVICES)} "
            "are supported",
        )
    )

    precision = config.model.precision
    checks.append(
        _check(
            "model.precision",
            isinstance(precision, str) and precision.strip() != "",
            f"model.precision = {precision!r}",
            "model.precision must be a non-empty string (e.g. 'float32')",
        )
    )

    # mixed_precision is a boolean flag on TrainingSection; the runtime treats
    # it as an on/off switch, so only True/False are valid values.
    mixed = training.mixed_precision
    checks.append(
        _check(
            "training.mixed_precision",
            isinstance(mixed, bool),
            f"training.mixed_precision = {mixed}",
            f"training.mixed_precision = {mixed!r} but it must be a boolean "
            "(True or False)",
        )
    )

    if tracking.enabled:
        checks.append(
            _number_check(
                "tracking.log_steps",
                tracking.log_steps,
                minimum=1,
                inclusive=True,
                label="tracking.log_steps",
                requirement="a positive integer (>= 1) because tracking is enabled",
            )
        )
    else:
        checks.append(
            _check(
                "tracking.log_steps",
                True,
                "tracking is disabled; log_steps is not validated",
                "",
            )
        )

    save_strategy = checkpoint.save_strategy
    strategy_ok = save_strategy in _SUPPORTED_SAVE_STRATEGIES
    cadence_ok = save_strategy != "steps" or (
        isinstance(checkpoint.save_steps, int)
        and not isinstance(checkpoint.save_steps, bool)
        and checkpoint.save_steps > 0
    )
    if strategy_ok and cadence_ok:
        detail = f"checkpoint.save_strategy = {save_strategy!r}, save_steps = {checkpoint.save_steps}"
    elif not strategy_ok:
        detail = (
            f"checkpoint.save_strategy = {save_strategy!r} but only "
            f"{list(_SUPPORTED_SAVE_STRATEGIES)} are supported by the preflight"
        )
    else:
        detail = (
            f"checkpoint.save_steps = {checkpoint.save_steps!r} but it must be a "
            "positive integer (>= 1) when save_strategy is 'steps'"
        )
    checks.append(
        _check(
            "checkpoint.save_strategy",
            strategy_ok and cadence_ok,
            detail if strategy_ok and cadence_ok else "",
            detail,
        )
    )

    resume_from = checkpoint.resume_from
    if resume_from is None or (
        isinstance(resume_from, str) and not resume_from.strip()
    ):
        checks.append(
            _check(
                "checkpoint.resume_from",
                True,
                "no resume_from configured (fresh run)",
                "",
            )
        )
    else:
        resume_path = Path(str(resume_from))
        checks.append(
            _check(
                "checkpoint.resume_from",
                resume_path.exists(),
                f"checkpoint.resume_from = {resume_from!r} (path exists)",
                f"checkpoint.resume_from = {resume_from!r} but the path does not "
                "exist; fix the path or clear resume_from to start fresh",
            )
        )

    output_root = runtime.output_root
    output_str = str(output_root).strip() if output_root is not None else ""
    checks.append(
        _check(
            "runtime.output_root",
            output_str not in ("", "."),
            f"runtime.output_root = {output_root}",
            "runtime.output_root is empty; set it to a writable directory path",
        )
    )

    return checks
