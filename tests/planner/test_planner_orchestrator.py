"""Planner orchestrator tests: plan building, matrix, identity, preflight flow."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

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
from clouda_training.planner.models import (
    EstimateSource,
    ExperimentProfile,
    HardwareEnvelope,
    HardwareFit,
    ParameterMetadata,
    StorageKind,
    TrainingMode,
    TrainingScale,
    default_profile,
)
from clouda_training.planner.planner import (
    build_experiment_matrix,
    build_experiment_plan,
    generate_training_config,
    render_plan_report,
)


def make_manifest(tmp_path: Path, row_count: int = 5000) -> Path:
    manifest = tmp_path / "m.jsonl"
    header = (
        '{"_schema_version":"clouda.pretraining.manifest.v1",'
        f'"_row_count":{row_count},"dataset_id":"synthetic-ar",'
        '"dataset_version":"v1"}\n'
    )
    manifest.write_text(header, encoding="utf-8")
    return manifest


@pytest.fixture()
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="qwen_pilot"),
        model=ModelSection(model_id="Qwen/Qwen3-VL-4B", adapter_type="qwen_vl_sft"),
        dataset=DatasetSection(
            dataset_id="synthetic-ar",
            dataset_version="v1",
            manifest_path=make_manifest(tmp_path),
            split="train",
        ),
        training=TrainingSection(
            seed=7, epochs=2, batch_size=1, gradient_accumulation_steps=8
        ),
        checkpoint=CheckpointSection(
            save_strategy="steps", save_steps=100, save_total_limit=5
        ),
        runtime=RuntimeSection(output_root=tmp_path / "out"),
        tracking=TrackingSection(enabled=False),
    )


HARDWARE = HardwareEnvelope(
    gpu_count=1, per_gpu_vram_gb=48, storage_kind=StorageKind.NVME, free_space_gb=500
)
PARAMS = ParameterMetadata(
    parameter_count=4_000_000_000, trainable_parameter_count=4_000_000_000
)


def pilot_profile() -> ExperimentProfile:
    profile = default_profile(
        TrainingScale.PILOT, training_mode=TrainingMode.SELECTIVE_FINETUNE
    )
    return dataclasses.replace(
        profile,
        sample_count=5000,
        epochs=2,
        micro_batch=1,
        gradient_accumulation=8,
        checkpoint_interval=100,
    )


def test_plan_deterministic_identity(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    plan2 = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert plan.plan_id == plan2.plan_id


def test_plan_id_changes_with_hardware(base_config: ExperimentConfig) -> None:
    plan_a = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    plan_b = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HardwareEnvelope(gpu_count=1, per_gpu_vram_gb=96),
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert plan_a.plan_id != plan_b.plan_id


def test_plan_id_json_has_no_absolute_paths(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    payload = json.dumps(plan.to_dict(), ensure_ascii=False)
    assert str(base_config.dataset.manifest_path) not in payload
    assert str(base_config.runtime.output_root) not in payload


def test_step_math_matches_preflight(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    # 5000 rows / batch 1 = 5000 micro; /accum 8 = 625 steps/epoch; x2 epochs
    assert plan.step_plan.effective_batch_size == 8
    assert plan.step_plan.micro_batches_per_epoch == 5000
    assert plan.step_plan.optimizer_steps_per_epoch == 625
    assert plan.step_plan.planned_optimizer_steps == 1250
    # checkpoint: 1250 // 100
    assert plan.checkpoint_plan.expected_checkpoint_count == 12


def test_vram_components_labeled_honestly(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert plan.resources is not None
    memory = plan.resources.memory
    # 4B params bf16 = 8 GB weights (DERIVED)
    assert memory.weights_bytes.source is EstimateSource.DERIVED
    assert memory.weights_bytes.value is not None
    assert abs(memory.weights_bytes.value / 2**30 - 7.45) < 0.1
    # AdamW state HEURISTIC
    assert memory.optimizer_states_bytes.source is EstimateSource.HEURISTIC
    # activations honest UNKNOWN
    assert memory.activations_bytes.source is EstimateSource.UNKNOWN
    assert memory.activations_bytes.value is None
    # fit verdict conservative
    assert plan.resources.fit in (HardwareFit.MAY_FIT, HardwareFit.LIKELY_TOO_LARGE)


def test_assumptions_ledger_populated(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    topics = {a.topic for a in plan.assumptions}
    assert "parameter_count" in topics
    assert "activation_memory" in topics
    assert "gpu_profile" in topics
    assert "world_size" in topics
    assert "precision" in topics


def test_matrix_small_deterministic(base_config: ExperimentConfig) -> None:
    matrix = build_experiment_matrix(
        base_config,
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert 3 <= len(matrix) <= 6
    scales = [p.profile.scale for p in matrix]
    assert scales[0] is TrainingScale.SMOKE
    assert TrainingScale.FULL in scales
    matrix2 = build_experiment_matrix(
        base_config,
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert [p.plan_id for p in matrix] == [p.plan_id for p in matrix2]


def test_recommendation_conservative(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert "RECOMMENDED FOR FIRST VALIDATION" in plan.recommendation
    assert "BEST" not in plan.recommendation.upper().replace("BEST-VALIDATED", "")


def test_execution_deferred_without_gpu(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert "DEFERRED" in plan.execution_status


def test_runtime_unknown_without_measurement(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    assert plan.runtime.estimated_runtime_seconds.value is None


def test_cost_unknown_without_runtime(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HardwareEnvelope(
            gpu_count=1,
            per_gpu_vram_gb=48,
            measured_throughput_seconds_per_step=None,
        ),
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
        cost_per_gpu_hour=2.0,
    )
    # runtime UNKNOWN -> cost must not be invented
    assert plan.cost is None or plan.cost.estimated_cost is None


def test_human_report_contains_required_sections(
    base_config: ExperimentConfig,
) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    report = render_plan_report(plan, model_label="Qwen3-VL-4B")
    for section in (
        "CLOUDA TRAINING EXPERIMENT PLAN",
        "Qwen3-VL-4B",
        "Effective batch: 8",
        "Total optimizer steps: 1250",
        "Activations: UNKNOWN",
        "MAY_FIT",
        "DEFERRED",
        "ASSUMPTIONS",
    ):
        assert section in report, section


def test_json_output_serializable(base_config: ExperimentConfig) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    payload = plan.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert plan.plan_id in text
    assert "training_mode" in text or "profile" in text


# ------------------------------------------------- preflight integration


def test_generated_config_flows_into_preflight(
    base_config: ExperimentConfig, tmp_path: Path
) -> None:
    """Planner -> generated config -> Preflight (never bypassed)."""
    from clouda_training.preflight.orchestrator import run_preflight
    from clouda_training.preflight.models import PreflightFinalStatus

    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    generated = generate_training_config(plan, base_config=base_config)
    # structural hand-off: the generated config is a real ExperimentConfig
    # the preflight orchestrator accepts without modification.
    report = run_preflight(generated, dataset_row_count=5000, write_probe=False)
    assert isinstance(report.final_status(), PreflightFinalStatus)
    # On this torch-less machine the qwen adapter deps are missing, so the
    # honest outcome is NOT_READY — the important assertion is that preflight
    # ran and produced explicit blockers (never a bypass).
    assert report.final_status() is PreflightFinalStatus.NOT_READY
    assert report.blockers


def test_generated_config_carries_profile_math(
    base_config: ExperimentConfig,
) -> None:
    plan = build_experiment_plan(
        base_config,
        pilot_profile(),
        hardware=HARDWARE,
        parameter_metadata=PARAMS,
        dataset_row_count=5000,
    )
    generated = generate_training_config(plan, base_config=base_config)
    assert generated.training.batch_size == 1
    assert generated.training.gradient_accumulation_steps == 8
    assert generated.training.epochs == 2
    assert generated.checkpoint.save_steps == 100
