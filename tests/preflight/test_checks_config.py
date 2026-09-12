"""Tests for clouda_training.preflight.checks_config.validate_config."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from clouda_training.experiments.config import (
    CheckpointSection,
    DatasetSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    RuntimeSection,
    TrackingSection,
    TrainingSection,
)
from clouda_training.preflight.checks_config import validate_config
from clouda_training.preflight.models import PreflightStatus


def make_config(**sections: object) -> ExperimentConfig:
    """Build a valid baseline config, overriding whole sections by name."""
    overrides = {
        "experiment": ExperimentSection(name="preflight_probe"),
        "model": ModelSection(model_id="mock/ocr"),
        "dataset": DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=Path("manifest.jsonl"),
        ),
        "training": TrainingSection(),
        "checkpoint": CheckpointSection(),
        "runtime": RuntimeSection(),
        "tracking": TrackingSection(),
    }
    overrides.update(sections)  # type: ignore[arg-type]
    return ExperimentConfig(**overrides)  # type: ignore[arg-type]


def test_valid_config_passes_all_checks() -> None:
    checks = validate_config(make_config())
    assert checks, "expected at least one check"
    for check in checks:
        assert check.status is PreflightStatus.PASS, f"{check.name}: {check.detail}"
        assert check.blocker is True


def test_every_expected_check_is_present() -> None:
    names = {c.name for c in validate_config(make_config())}
    expected = {
        "training.batch_size",
        "training.gradient_accumulation_steps",
        "training.learning_rate",
        "training.epochs_or_max_steps",
        "training.weight_decay",
        "training.warmup_steps",
        "training.scheduler",
        "training.max_grad_norm",
        "training.seed",
        "runtime.device",
        "model.precision",
        "training.mixed_precision",
        "tracking.log_steps",
        "checkpoint.save_strategy",
        "checkpoint.resume_from",
        "runtime.output_root",
    }
    assert names == expected


def test_check_list_is_stable_across_calls() -> None:
    config = make_config()
    first = [c.to_dict() for c in validate_config(config)]
    second = [c.to_dict() for c in validate_config(config)]
    assert first == second


def test_batch_size_zero_fails() -> None:
    config = make_config(training=dataclasses.replace(TrainingSection(), batch_size=0))
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.batch_size"
    assert fails[0].blocker is True
    assert "positive" in fails[0].detail


def test_batch_size_negative_fails() -> None:
    config = make_config(training=dataclasses.replace(TrainingSection(), batch_size=-2))
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.batch_size"


def test_gradient_accumulation_steps_zero_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), gradient_accumulation_steps=0)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.gradient_accumulation_steps"


def test_learning_rate_zero_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), learning_rate=0.0)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.learning_rate"


def test_learning_rate_negative_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), learning_rate=-1e-4)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.learning_rate"


def test_no_finite_plan_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), epochs=0, max_steps=None)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.epochs_or_max_steps"
    assert "epochs" in fails[0].detail or "max_steps" in fails[0].detail


def test_negative_epochs_with_positive_max_steps_passes_plan() -> None:
    """max_steps > 0 alone satisfies the finite-plan rule even with epochs < 1."""
    config = make_config(
        training=dataclasses.replace(TrainingSection(), epochs=0, max_steps=10)
    )
    plan = [
        c for c in validate_config(config) if c.name == "training.epochs_or_max_steps"
    ]
    assert len(plan) == 1
    assert plan[0].status is PreflightStatus.PASS


def test_epochs_one_with_max_steps_none_passes_plan() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), epochs=1, max_steps=None)
    )
    plan = [
        c for c in validate_config(config) if c.name == "training.epochs_or_max_steps"
    ]
    assert len(plan) == 1
    assert plan[0].status is PreflightStatus.PASS


def test_weight_decay_negative_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), weight_decay=-0.1)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.weight_decay"


def test_weight_decay_zero_passes() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), weight_decay=0.0)
    )
    check = next(
        c for c in validate_config(config) if c.name == "training.weight_decay"
    )
    assert check.status is PreflightStatus.PASS


def test_warmup_steps_negative_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), warmup_steps=-1)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.warmup_steps"


def test_scheduler_unknown_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), scheduler="polynomial")
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.scheduler"
    assert "polynomial" in fails[0].detail


def test_scheduler_each_supported_value_passes() -> None:
    for scheduler in ("none", "linear", "cosine"):
        config = make_config(
            training=dataclasses.replace(TrainingSection(), scheduler=scheduler)
        )
        check = next(
            c for c in validate_config(config) if c.name == "training.scheduler"
        )
        assert check.status is PreflightStatus.PASS, scheduler


def test_max_grad_norm_negative_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), max_grad_norm=-0.5)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.max_grad_norm"


def test_seed_none_fails() -> None:
    config = make_config(training=dataclasses.replace(TrainingSection(), seed=None))  # type: ignore[arg-type]
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.seed"


def test_device_unknown_fails() -> None:
    config = make_config(runtime=dataclasses.replace(RuntimeSection(), device="tpu"))
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "runtime.device"
    assert "tpu" in fails[0].detail


def test_device_cpu_and_cuda_pass() -> None:
    for device in ("cpu", "cuda"):
        config = make_config(
            runtime=dataclasses.replace(RuntimeSection(), device=device)
        )
        check = next(c for c in validate_config(config) if c.name == "runtime.device")
        assert check.status is PreflightStatus.PASS, device


def test_precision_empty_fails() -> None:
    config = make_config(
        model=dataclasses.replace(ModelSection(model_id="mock/ocr"), precision="")
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "model.precision"


def test_precision_whitespace_fails() -> None:
    config = make_config(
        model=dataclasses.replace(ModelSection(model_id="mock/ocr"), precision="   ")
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "model.precision"


def test_mixed_precision_non_bool_fails() -> None:
    config = make_config(
        training=dataclasses.replace(TrainingSection(), mixed_precision="yes")  # type: ignore[arg-type]
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "training.mixed_precision"


def test_mixed_precision_true_and_false_pass() -> None:
    for value in (True, False):
        config = make_config(
            training=dataclasses.replace(TrainingSection(), mixed_precision=value)
        )
        check = next(
            c for c in validate_config(config) if c.name == "training.mixed_precision"
        )
        assert check.status is PreflightStatus.PASS, value


def test_tracking_disabled_skips_log_steps_validation() -> None:
    config = make_config(
        tracking=dataclasses.replace(TrackingSection(), enabled=False, log_steps=0)
    )
    check = next(c for c in validate_config(config) if c.name == "tracking.log_steps")
    assert check.status is PreflightStatus.PASS


def test_tracking_enabled_log_steps_zero_fails() -> None:
    config = make_config(
        tracking=dataclasses.replace(TrackingSection(), enabled=True, log_steps=0)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "tracking.log_steps"


def test_tracking_enabled_log_steps_negative_fails() -> None:
    config = make_config(
        tracking=dataclasses.replace(TrackingSection(), enabled=True, log_steps=-3)
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "tracking.log_steps"


def test_checkpoint_save_strategy_unknown_fails() -> None:
    config = make_config(
        checkpoint=dataclasses.replace(CheckpointSection(), save_strategy="epoch")
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "checkpoint.save_strategy"


def test_checkpoint_save_steps_zero_with_steps_strategy_fails() -> None:
    config = make_config(
        checkpoint=dataclasses.replace(
            CheckpointSection(), save_strategy="steps", save_steps=0
        )
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "checkpoint.save_strategy"


def test_checkpoint_save_strategy_none_skips_save_steps_validation() -> None:
    config = make_config(
        checkpoint=dataclasses.replace(
            CheckpointSection(), save_strategy="none", save_steps=0
        )
    )
    check = next(
        c for c in validate_config(config) if c.name == "checkpoint.save_strategy"
    )
    assert check.status is PreflightStatus.PASS


def test_resume_from_missing_path_fails(tmp_path: Path) -> None:
    config = make_config(
        checkpoint=dataclasses.replace(
            CheckpointSection(), resume_from=str(tmp_path / "does_not_exist")
        )
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "checkpoint.resume_from"
    assert "does not exist" in fails[0].detail
    assert fails[0].blocker is True


def test_resume_from_existing_path_passes(tmp_path: Path) -> None:
    existing = tmp_path / "checkpoint_dir"
    existing.mkdir()
    config = make_config(
        checkpoint=dataclasses.replace(CheckpointSection(), resume_from=str(existing))
    )
    check = next(
        c for c in validate_config(config) if c.name == "checkpoint.resume_from"
    )
    assert check.status is PreflightStatus.PASS


def test_resume_from_none_passes() -> None:
    config = make_config(
        checkpoint=dataclasses.replace(CheckpointSection(), resume_from=None)
    )
    check = next(
        c for c in validate_config(config) if c.name == "checkpoint.resume_from"
    )
    assert check.status is PreflightStatus.PASS
    assert "fresh" in check.detail


def test_output_root_empty_fails() -> None:
    config = make_config(
        runtime=dataclasses.replace(RuntimeSection(), output_root=Path(""))
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "runtime.output_root"


def test_output_root_whitespace_only_fails() -> None:
    config = make_config(
        runtime=dataclasses.replace(RuntimeSection(), output_root=Path("   "))
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert len(fails) == 1
    assert fails[0].name == "runtime.output_root"


def test_output_root_valid_path_passes(tmp_path: Path) -> None:
    config = make_config(
        runtime=dataclasses.replace(RuntimeSection(), output_root=tmp_path / "runs")
    )
    check = next(c for c in validate_config(config) if c.name == "runtime.output_root")
    assert check.status is PreflightStatus.PASS


def test_multiple_invalid_fields_each_produce_own_fail() -> None:
    config = make_config(
        training=dataclasses.replace(
            TrainingSection(), batch_size=0, learning_rate=0.0
        ),
        runtime=dataclasses.replace(RuntimeSection(), device="tpu"),
    )
    fails = [c for c in validate_config(config) if c.status is PreflightStatus.FAIL]
    assert {c.name for c in fails} == {
        "training.batch_size",
        "training.learning_rate",
        "runtime.device",
    }
    assert len(fails) == 3
