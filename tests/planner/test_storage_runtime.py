"""Tests for planner.storage_runtime: provenance-gated estimates.

Every test pins the core rule: no number without provenance, and UNKNOWN
stays UNKNOWN (no value) until real inputs exist.
"""

from __future__ import annotations

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
    HardwareEnvelope,
    ParameterMetadata,
    PlanningWarning,
    StorageKind,
)
from clouda_training.planner.storage_runtime import (
    RUNTIME_UNKNOWN_MESSAGE,
    build_checkpoint_plan,
    classify_storage_io,
    estimate_checkpoint_footprint,
    estimate_cost,
    estimate_dataset_bytes,
    estimate_runtime,
    plan_storage_runtime,
)


def make_config(tmp_path: Path, **overrides: object) -> ExperimentConfig:
    sections = {
        "experiment": ExperimentSection(name="planner_storage"),
        "model": ModelSection(model_id="mock/ocr"),
        "dataset": DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=tmp_path / "manifest.jsonl",
        ),
        "training": TrainingSection(max_steps=10, batch_size=2),
        "checkpoint": CheckpointSection(save_steps=4, save_total_limit=2),
        "runtime": RuntimeSection(output_root=tmp_path / "out"),
        "tracking": TrackingSection(),
    }
    sections.update(overrides)  # type: ignore[arg-type]
    return ExperimentConfig(**sections)  # type: ignore[arg-type]


def write_manifest(tmp_path: Path) -> Path:
    header = {
        "_schema_version": "clouda.pretraining.manifest.v1",
        "_row_count": 2,
        "dataset_id": "synthetic-test",
        "dataset_version": "v1",
    }
    rows = [
        {
            "sample_id": "ar-1",
            "target_split": "train",
            "source_id": "synthetic-test",
            "image_path": "p/1.png",
            "ground_truth": "نص",
            "source_license": "Apache-2.0",
        },
        {
            "sample_id": "ar-2",
            "target_split": "train",
            "source_id": "synthetic-test",
            "image_path": "p/2.png",
            "ground_truth": "نص ثانٍ",
            "source_license": "Apache-2.0",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [header, *rows]),
        encoding="utf-8",
    )
    return manifest


KNOWN_PARAMS = ParameterMetadata(
    parameter_count=1_000,
    trainable_parameter_count=1_000,
    declared_model_size_bytes=4_000,  # fp32: 1000 params x 4 bytes
)
UNKNOWN_PARAMS = ParameterMetadata()


# ------------------------------------------------------------ dataset bytes


def test_dataset_bytes_measured_from_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    est = estimate_dataset_bytes(manifest)
    assert est.value == manifest.stat().st_size
    assert est.source is EstimateSource.MEASURED
    assert est.confidence.value == "HIGH"


def test_dataset_bytes_unknown_without_manifest(tmp_path: Path) -> None:
    est = estimate_dataset_bytes(tmp_path / "missing.jsonl")
    assert est.value is None
    assert est.source is EstimateSource.UNKNOWN
    est_none = estimate_dataset_bytes(None)
    assert est_none.value is None
    assert est_none.source is EstimateSource.UNKNOWN


# ------------------------------------------------------- checkpoint footprint


def test_checkpoint_footprint_known_inputs_derived() -> None:
    # model 4000 B + optimizer 1000 x 8 = 8000 B -> 12000 B per checkpoint;
    # retained x2 -> 24000 B total.
    est = estimate_checkpoint_footprint(
        parameter_metadata=KNOWN_PARAMS, retained_count=2
    )
    assert est.value == 24_000
    assert est.source is EstimateSource.DERIVED


def test_checkpoint_footprint_unknown_without_real_model() -> None:
    est = estimate_checkpoint_footprint(
        parameter_metadata=UNKNOWN_PARAMS, retained_count=2
    )
    assert est.value is None
    assert est.source is EstimateSource.UNKNOWN
    assert "UNKNOWN UNTIL REAL MODEL IS AVAILABLE" in (est.note or "")


def test_checkpoint_footprint_missing_parameter_count_is_unknown() -> None:
    partial = ParameterMetadata(declared_model_size_bytes=4_000)
    est = estimate_checkpoint_footprint(parameter_metadata=partial, retained_count=1)
    assert est.value is None
    assert est.source is EstimateSource.UNKNOWN


# --------------------------------------------------------------- I/O class


def test_hdd_io_warning_is_qualitative() -> None:
    kind, warnings = classify_storage_io(HardwareEnvelope(storage_kind=StorageKind.HDD))
    assert kind is StorageKind.HDD
    messages = " ".join(w.message for w in warnings)
    assert "HDD" in messages and "bottleneck" in messages
    # no invented throughput numbers anywhere
    assert "MB/s" not in messages and "GB/s" not in messages


def test_nvme_io_no_bottleneck_warning() -> None:
    kind, warnings = classify_storage_io(
        HardwareEnvelope(storage_kind=StorageKind.NVME)
    )
    assert kind is StorageKind.NVME
    assert all("bottleneck" not in w.message.lower() for w in warnings)


def test_measured_throughput_stored_as_measured_only() -> None:
    _kind, warnings = classify_storage_io(
        HardwareEnvelope(
            storage_kind=StorageKind.SSD,
            measured_throughput_seconds_per_step=0.5,
        )
    )
    throughputs = [w for w in warnings if w.code == "io.throughput"]
    assert len(throughputs) == 1
    assert "MEASURED" in throughputs[0].message
    assert "not extrapolated" in throughputs[0].message


# ----------------------------------------------------------------- runtime


def test_runtime_unknown_without_measurement() -> None:
    est = estimate_runtime(HardwareEnvelope(), planned_optimizer_steps=100)
    assert est.estimated_runtime_seconds.value is None
    assert est.estimated_runtime_seconds.source is EstimateSource.UNKNOWN
    assert RUNTIME_UNKNOWN_MESSAGE in (est.estimated_runtime_seconds.note or "")
    assert est.seconds_per_step.value is None
    assert est.seconds_per_step.source is EstimateSource.UNKNOWN


def test_runtime_computed_with_measured_input() -> None:
    est = estimate_runtime(
        HardwareEnvelope(measured_throughput_seconds_per_step=0.25),
        planned_optimizer_steps=40,
    )
    assert est.estimated_runtime_seconds.value == pytest.approx(10.0)
    assert est.estimated_runtime_seconds.source is EstimateSource.DERIVED
    assert est.seconds_per_step.source is EstimateSource.MEASURED


def test_runtime_no_assumed_seconds_per_step() -> None:
    # Fail-closed: without a MEASURED value the runtime is UNKNOWN even
    # when the planned step count is known — no seconds/step is invented.
    est = estimate_runtime(HardwareEnvelope(), planned_optimizer_steps=40)
    assert est.estimated_runtime_seconds.value is None
    assert est.seconds_per_step.value is None


# -------------------------------------------------------------------- cost


def test_cost_gated_on_runtime_estimation() -> None:
    unknown_runtime = estimate_runtime(HardwareEnvelope(), planned_optimizer_steps=100)
    assert estimate_cost(unknown_runtime, cost_per_gpu_hour=2.0, gpu_count=1) is None


def test_cost_gated_on_supplied_rate() -> None:
    runtime = estimate_runtime(
        HardwareEnvelope(measured_throughput_seconds_per_step=1.0),
        planned_optimizer_steps=100,
    )
    assert estimate_cost(runtime, cost_per_gpu_hour=None, gpu_count=1) is None


def test_cost_estimated_when_runtime_and_rate_present() -> None:
    runtime = estimate_runtime(
        HardwareEnvelope(measured_throughput_seconds_per_step=36.0),
        planned_optimizer_steps=100,
    )
    cost = estimate_cost(runtime, cost_per_gpu_hour=3.0, gpu_count=1)
    assert cost is not None
    # 100 x 36s = 3600s = 1 GPU-hour x 3.0
    assert cost.estimated_cost == pytest.approx(3.0)
    assert cost.label == "ESTIMATED"
    assert cost.source is EstimateSource.DERIVED


# ------------------------------------------------------- checkpoint plan


def test_checkpoint_plan_uses_preflight_math(tmp_path: Path) -> None:
    config = make_config(tmp_path)  # max_steps=10, save_steps=4
    plan, planned_steps = build_checkpoint_plan(config, dataset_row_count=8)
    assert planned_steps == 10
    assert plan.save_steps == 4
    assert plan.expected_checkpoint_count == 10 // 4  # preflight: planned//steps
    assert plan.save_total_limit == 2
    assert plan.estimated_retained_count == 2  # min(expected, limit)


def test_checkpoint_plan_no_limit_retains_all(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        checkpoint=CheckpointSection(save_steps=4, save_total_limit=0),
    )
    plan, _ = build_checkpoint_plan(config, dataset_row_count=8)
    # save_total_limit=0 means "no retention cap" for planning purposes
    assert plan.estimated_retained_count == plan.expected_checkpoint_count


def test_checkpoint_plan_matches_compute_training_plan(tmp_path: Path) -> None:
    """Direct equivalence with the preflight math (no fork)."""
    from clouda_training.preflight.plan import compute_training_plan

    config = make_config(tmp_path, training=TrainingSection(max_steps=23))
    plan, planned_steps = build_checkpoint_plan(config, dataset_row_count=100)
    summary = compute_training_plan(config, dataset_row_count=100)
    assert planned_steps == summary.planned_optimizer_steps == 23
    assert plan.expected_checkpoint_count == summary.estimated_checkpoint_count


# --------------------------------------------------------- top-level plan


def test_plan_storage_runtime_known_inputs(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    config = make_config(
        tmp_path,
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
        ),
    )
    result = plan_storage_runtime(
        config,
        hardware=HardwareEnvelope(
            storage_kind=StorageKind.SSD,
            measured_throughput_seconds_per_step=0.5,
        ),
        parameter_metadata=KNOWN_PARAMS,
        dataset_row_count=8,
        cost_per_gpu_hour=2.0,
    )
    storage = result["storage"]
    assert storage.known_minimum_bytes == manifest.stat().st_size + 24_000
    assert storage.recommended_safety_reserve_bytes == int(
        (manifest.stat().st_size + 24_000) * 0.5
    )
    assert result["checkpoint_total_bytes"].value == 12_000 * 2  # retained x2
    runtime = result["runtime"]
    assert runtime.estimated_runtime_seconds.value == pytest.approx(10 * 0.5)
    cost = result["cost"]
    assert cost is not None
    assert cost.label == "ESTIMATED"
    assert cost.estimated_cost == pytest.approx(10 * 0.5 / 3600 * 2.0)


def test_plan_storage_runtime_unknown_inputs_stay_unknown(tmp_path: Path) -> None:
    config = make_config(tmp_path)  # manifest missing
    result = plan_storage_runtime(
        config,
        hardware=HardwareEnvelope(storage_kind=StorageKind.HDD),
        parameter_metadata=UNKNOWN_PARAMS,
        dataset_row_count=8,
        cost_per_gpu_hour=2.0,
    )
    storage = result["storage"]
    assert result["dataset_bytes"].value is None
    assert result["checkpoint_total_bytes"].value is None
    assert storage.known_minimum_bytes == 0
    assert storage.known_minimum_source is EstimateSource.UNKNOWN
    contributors = " ".join(storage.unknown_contributors)
    assert "dataset" in contributors and "checkpoint" in contributors
    runtime = result["runtime"]
    assert runtime.estimated_runtime_seconds.value is None
    assert RUNTIME_UNKNOWN_MESSAGE in (runtime.estimated_runtime_seconds.note or "")
    assert result["cost"] is None
    codes = {w.code for w in result["warnings"]}
    assert "runtime.estimate" in codes
    assert any(
        w.code == "io.storage_class" and "HDD" in w.message for w in result["warnings"]
    )
    # every warning is a typed PlanningWarning and JSON-serializable
    assert all(isinstance(w, PlanningWarning) for w in result["warnings"])
    snapshot = {
        "storage": storage.to_dict(),
        "runtime": runtime.to_dict(),
        "checkpoint_plan": result["checkpoint_plan"].to_dict(),
        "warnings": [w.to_dict() for w in result["warnings"]],
    }
    assert json.dumps(snapshot)


def test_plan_storage_runtime_step_count_single_source(tmp_path: Path) -> None:
    """Runtime uses the SAME planned steps as the preflight math."""
    config = make_config(tmp_path)
    result = plan_storage_runtime(
        config,
        hardware=HardwareEnvelope(measured_throughput_seconds_per_step=2.0),
        parameter_metadata=KNOWN_PARAMS,
        dataset_row_count=8,
    )
    assert result["planned_optimizer_steps"] == 10
    assert result["runtime"].planned_optimizer_steps == 10
    assert result["runtime"].estimated_runtime_seconds.value == pytest.approx(20.0)
