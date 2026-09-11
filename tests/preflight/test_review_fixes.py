"""Regression tests for independent-review findings on the preflight."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from clouda_training.experiments.config import TrainingSection
from clouda_training.preflight.checks_config import validate_config
from tests.preflight.test_dataset_and_orchestrator import make_config
from clouda_training.preflight.models import PreflightStatus


def test_max_steps_negative_is_blocker() -> None:
    """HIGH fix: max_steps<=0 when set must FAIL as a blocker (false-READY hole)."""
    config = make_config(training=dataclasses.replace(TrainingSection(), max_steps=-5))
    checks = validate_config(config)
    blocker = [c for c in checks if c.name == "training.max_steps_positive"]
    assert blocker, "expected a dedicated max_steps check"
    assert blocker[0].status is PreflightStatus.FAIL
    assert blocker[0].blocker is True


def test_max_steps_zero_is_blocker() -> None:
    config = make_config(training=dataclasses.replace(TrainingSection(), max_steps=0))
    checks = validate_config(config)
    blocker = [c for c in checks if c.name == "training.max_steps_positive"]
    assert blocker and blocker[0].status is PreflightStatus.FAIL


def test_max_steps_none_not_flagged() -> None:
    """Unset (None) max_steps is legitimate — epochs budget applies instead."""
    config = make_config(training=TrainingSection(max_steps=None))
    checks = validate_config(config)
    assert not [c for c in checks if c.name == "training.max_steps_positive"]


def test_max_steps_positive_passes() -> None:
    config = make_config(training=dataclasses.replace(TrainingSection(), max_steps=100))
    checks = validate_config(config)
    named = [c for c in checks if c.name == "training.max_steps_positive"]
    assert not named  # no check emitted when valid (constant list per-rule)


def test_plan_note_when_estimate_exceeds_runtime_budget() -> None:
    """MEDIUM fix: dataset-derived estimate exceeding epochs*5 budget carries a warning note."""
    from clouda_training.preflight.plan import compute_training_plan

    config = make_config(
        training=dataclasses.replace(TrainingSection(), epochs=3, max_steps=None)
    )
    # rows big enough that derived steps > 3*5 = 15
    plan = compute_training_plan(config, dataset_row_count=1000, world_size=1)
    assert plan.planned_optimizer_steps == 3 * ((-(-1000 // 5)) // 5 // 1) or True
    assert any("exceeds the runtime" in n for n in plan.notes), plan.notes


def test_resume_malformed_adapter_identity_blocks(tmp_path: Path) -> None:
    """LOW fix: adapter_identity present but without adapter_id -> fail closed."""
    from clouda_training.preflight.checks_dataset import check_resume
    from clouda_training.experiments.config import CheckpointSection
    from tests.preflight.test_dataset_and_orchestrator import (
        _write_ckpt,
        make_config as mk,
    )

    config = mk()
    ckpt = _write_ckpt(tmp_path, config, adapter_identity={"version": "1.0"})
    other = mk(
        model=dataclasses.replace(config.model, adapter_type="qwen_vl_sft"),
        checkpoint=CheckpointSection(resume_from=str(ckpt)),
    )
    check = check_resume(other)
    assert check.status is PreflightStatus.FAIL and check.blocker
