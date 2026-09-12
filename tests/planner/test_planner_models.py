"""Tests for clouda_training.planner.models (typed planning domain model)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from clouda_training.planner import (
    BYTES_PER_GIB,
    DEFAULT_PROFILES,
    Estimate,
    EstimateConfidence,
    EstimateSource,
    ExperimentPlan,
    ExperimentProfile,
    HardwareEnvelope,
    HardwareFit,
    ParameterMetadata,
    PlanningAssumption,
    PlanningIdentity,
    PlanningWarning,
    StorageEstimate,
    StorageKind,
    TrainingMode,
    TrainingScale,
    default_profile,
    worst_confidence,
)
from clouda_training.planner.models import (
    CheckpointPlan,
    CostEstimate,
    MemoryComponentEstimates,
    ResourceEstimate,
    RuntimeEstimate,
    StepPlan,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _identity(
    profile: ExperimentProfile | None = None,
    hardware: HardwareEnvelope | None = None,
) -> PlanningIdentity:
    return PlanningIdentity(
        adapter_type="hunyuanocr15_sft",
        adapter_identity={
            "adapter_type": "hunyuanocr15_sft",
            "upstream_repository": "Tencent-Hunyuan/HunyuanOCR",
            "upstream_revision": "c55965d3da1e",
        },
        dataset_id="arabic-ocr-sft",
        dataset_version="v1",
        profile=(
            profile if profile is not None else default_profile(TrainingScale.SMOKE)
        ),
        hardware=hardware,
    )


def _minimal_plan() -> ExperimentPlan:
    profile = default_profile(TrainingScale.PILOT)
    hardware = HardwareEnvelope(
        gpu_count=1,
        per_gpu_vram_gb=24.0,
        system_ram_gb=64.0,
        storage_kind=StorageKind.NVME,
        free_space_gb=500.0,
    )
    storage = StorageEstimate(
        known_minimum_bytes=10 * BYTES_PER_GIB,
        known_minimum_source=EstimateSource.DECLARED,
        known_minimum_confidence=EstimateConfidence.MEDIUM,
        unknown_contributors=("checkpoint export size",),
        recommended_safety_reserve_bytes=BYTES_PER_GIB,
    )
    return ExperimentPlan(
        plan_id=_identity(profile, hardware).plan_id(),
        identity=_identity(profile, hardware),
        profile=profile,
        hardware=hardware,
        step_plan=StepPlan(
            effective_batch_size=8,
            micro_batches_per_epoch=63,
            optimizer_steps_per_epoch=16,
            planned_optimizer_steps=200,
            source=EstimateSource.DERIVED,
            confidence=EstimateConfidence.MEDIUM,
        ),
        checkpoint_plan=CheckpointPlan(
            save_steps=50,
            expected_checkpoint_count=4,
            save_total_limit=3,
            estimated_retained_count=3,
        ),
        storage=storage,
        runtime=RuntimeEstimate(
            seconds_per_step=Estimate(
                value=2.5,
                unit="s/step",
                source=EstimateSource.MEASURED,
                confidence=EstimateConfidence.HIGH,
            ),
            planned_optimizer_steps=200,
            estimated_runtime_seconds=Estimate(
                value=500.0,
                unit="s",
                source=EstimateSource.DERIVED,
                confidence=EstimateConfidence.HIGH,
            ),
        ),
        cost=CostEstimate(
            currency="USD",
            cost_per_gpu_hour=2.0,
            gpus=1,
            estimated_runtime_seconds=500.0,
            estimated_cost=round(500.0 / 3600.0 * 2.0 * 1, 2),
        ),
        assumptions=(
            PlanningAssumption(
                topic="activations",
                detail="no activation profile supplied",
                source=EstimateSource.UNKNOWN,
                confidence=EstimateConfidence.UNKNOWN,
            ),
        ),
        warnings=(
            PlanningWarning(
                code="ACTIVATIONS_UNKNOWN",
                message="activation memory unknown until a smoke test runs",
            ),
        ),
        recommendation="PILOT conservative — RECOMMENDED FOR FIRST VALIDATION",
        confidence=EstimateConfidence.MEDIUM,
    )


# --------------------------------------------------------------------------
# enums + defaults
# --------------------------------------------------------------------------


class TestEnumsAndDefaults:
    def test_estimate_source_members(self) -> None:
        assert {s.value for s in EstimateSource} == {
            "MEASURED",
            "DECLARED",
            "DERIVED",
            "HEURISTIC",
            "UNKNOWN",
        }

    def test_estimate_confidence_members(self) -> None:
        assert {c.value for c in EstimateConfidence} == {
            "HIGH",
            "MEDIUM",
            "LOW",
            "UNKNOWN",
        }

    def test_training_scale_members(self) -> None:
        assert {s.value for s in TrainingScale} == {
            "SMOKE",
            "PILOT",
            "MEDIUM",
            "FULL",
            "CUSTOM",
        }

    def test_hardware_fit_members(self) -> None:
        assert {f.value for f in HardwareFit} == {
            "LIKELY_FITS",
            "MAY_FIT",
            "LIKELY_TOO_LARGE",
            "UNKNOWN",
        }

    def test_str_enum_serializes_as_plain_string(self) -> None:
        assert json.dumps(TrainingScale.SMOKE.value) == '"SMOKE"'
        assert f"{EstimateSource.MEASURED}" == "EstimateSource.MEASURED"

    def test_default_profiles_cover_non_custom_scales(self) -> None:
        assert set(DEFAULT_PROFILES) == {
            TrainingScale.SMOKE,
            TrainingScale.PILOT,
            TrainingScale.MEDIUM,
            TrainingScale.FULL,
        }

    def test_smoke_defaults_are_tiny_and_very_low_steps(self) -> None:
        smoke = DEFAULT_PROFILES[TrainingScale.SMOKE]
        pilot = DEFAULT_PROFILES[TrainingScale.PILOT]
        medium = DEFAULT_PROFILES[TrainingScale.MEDIUM]
        full = DEFAULT_PROFILES[TrainingScale.FULL]
        # SMOKE: tiny samples, very low steps.
        assert (
            smoke.sample_count
            < pilot.sample_count
            < medium.sample_count
            < full.sample_count
        )
        assert smoke.max_steps is not None and smoke.max_steps <= 10
        # PILOT small but enough for signal; MEDIUM intermediate; FULL dataset-driven.
        assert pilot.max_steps is not None and 0 < pilot.max_steps < medium.max_steps  # type: ignore[operator]
        assert full.max_steps is None  # dataset-driven steps
        for scale in (TrainingScale.SMOKE, TrainingScale.PILOT, TrainingScale.MEDIUM):
            assert DEFAULT_PROFILES[scale].epochs >= 1

    def test_full_scale_defaults_use_full_dataset_intent(self) -> None:
        full = DEFAULT_PROFILES[TrainingScale.FULL]
        assert full.sample_count == max(
            p.sample_count for p in DEFAULT_PROFILES.values()
        )

    def test_default_profile_all_fields_configurable(self) -> None:
        custom = default_profile(
            TrainingScale.SMOKE, training_mode=TrainingMode.FULL_FINETUNE
        )
        assert custom.training_mode is TrainingMode.FULL_FINETUNE
        override = dataclasses.replace(custom, sample_count=7, epochs=9, micro_batch=3)
        assert (override.sample_count, override.epochs, override.micro_batch) == (
            7,
            9,
            3,
        )

    def test_effective_batch_size(self) -> None:
        profile = dataclasses.replace(
            default_profile(TrainingScale.MEDIUM),
            micro_batch=4,
            gradient_accumulation=8,
        )
        assert profile.effective_batch_size() == 32


# --------------------------------------------------------------------------
# provenance propagation
# --------------------------------------------------------------------------


class TestProvenance:
    def test_every_estimate_carries_source_and_confidence(self) -> None:
        est = Estimate(value=1.5, unit="GB")
        # Defaults are fail-closed: UNKNOWN/UNKNOWN, never a fake value.
        assert est.source is EstimateSource.UNKNOWN
        assert est.confidence is EstimateConfidence.UNKNOWN

    def test_measured_runtime_propagates_into_plan_dict(self) -> None:
        plan = _minimal_plan()
        d = plan.to_dict()
        assert d["runtime"]["seconds_per_step"]["source"] == "MEASURED"
        assert d["runtime"]["seconds_per_step"]["confidence"] == "HIGH"
        assert d["runtime"]["estimated_runtime_seconds"]["source"] == "DERIVED"

    def test_unknown_runtime_is_none_valued(self) -> None:
        runtime = RuntimeEstimate()
        assert runtime.seconds_per_step.value is None
        assert runtime.seconds_per_step.source is EstimateSource.UNKNOWN
        assert runtime.estimated_runtime_seconds.value is None
        d = runtime.to_dict()
        assert d["seconds_per_step"]["value"] is None

    def test_worst_confidence_ordering(self) -> None:
        assert worst_confidence([]) is EstimateConfidence.UNKNOWN
        assert (
            worst_confidence(
                [
                    EstimateConfidence.HIGH,
                    EstimateConfidence.LOW,
                    EstimateConfidence.MEDIUM,
                ]
            )
            is EstimateConfidence.LOW
        )
        assert (
            worst_confidence([EstimateConfidence.HIGH, EstimateConfidence.MEDIUM])
            is EstimateConfidence.MEDIUM
        )
        assert (
            worst_confidence([EstimateConfidence.HIGH, EstimateConfidence.UNKNOWN])
            is EstimateConfidence.UNKNOWN
        )

    def test_parameter_metadata_declared_source(self) -> None:
        empty = ParameterMetadata()
        assert empty.source() is EstimateSource.UNKNOWN
        declared = ParameterMetadata(parameter_count=1_000_000)
        assert declared.source() is EstimateSource.DECLARED
        declared2 = ParameterMetadata(declared_model_size_bytes=123)
        assert declared2.source() is EstimateSource.DECLARED

    def test_memory_component_estimates_provenance(self) -> None:
        mem = MemoryComponentEstimates(
            weights_bytes=Estimate(
                value=8_000_000_000,
                unit="B",
                source=EstimateSource.DECLARED,
                confidence=EstimateConfidence.MEDIUM,
            ),
            optimizer_states_bytes=Estimate(
                value=16_000_000_000,
                unit="B",
                source=EstimateSource.HEURISTIC,
                confidence=EstimateConfidence.LOW,
            ),
        )
        comps = mem.iter_components()
        assert comps["weights_bytes"].source is EstimateSource.DECLARED
        assert comps["optimizer_states_bytes"].source is EstimateSource.HEURISTIC
        assert comps["activations_bytes"].source is EstimateSource.UNKNOWN
        assert mem.known_lower_bound_bytes() == 8_000_000_000 + 16_000_000_000
        assert not mem.all_known()
        assert mem.worst_confidence() is EstimateConfidence.UNKNOWN

    def test_resource_estimate_fit_default_is_unknown(self) -> None:
        res = ResourceEstimate()
        assert res.fit is HardwareFit.UNKNOWN
        assert res.lower_bound_source is EstimateSource.UNKNOWN


# --------------------------------------------------------------------------
# to_dict JSON serializability
# --------------------------------------------------------------------------


class TestToDict:
    def test_plan_to_dict_is_json_serializable(self) -> None:
        plan = _minimal_plan()
        blob = json.dumps(plan.to_dict(), sort_keys=True)
        assert isinstance(blob, str)
        round_trip = json.loads(blob)
        assert round_trip["plan_id"] == plan.plan_id
        assert round_trip["confidence"] == "MEDIUM"

    def test_plan_to_dict_enums_are_plain_strings(self) -> None:
        d = _minimal_plan().to_dict()
        assert d["profile"]["scale"] == "PILOT"
        assert d["profile"]["training_mode"] == "LORA_PEFT"
        assert d["hardware"]["storage_kind"] == "NVMe"
        assert d["storage"]["known_minimum_source"] == "DECLARED"
        assert d["cost"]["label"] == "ESTIMATED"
        assert d["assumptions"][0]["source"] == "UNKNOWN"

    def test_all_model_to_dicts_are_json_serializable(self) -> None:
        models: list[Any] = [
            Estimate(1, "B"),
            ExperimentProfile(scale=TrainingScale.SMOKE, sample_count=8),
            HardwareEnvelope(),
            ParameterMetadata(parameter_count=10),
            StepPlan(None, None, None, None),
            CheckpointPlan(),
            StorageEstimate(known_minimum_bytes=0),
            RuntimeEstimate(),
            CostEstimate("USD", 1.0, 1, 3600.0, 1.0),
            PlanningAssumption("t", "d"),
            PlanningWarning("C", "m"),
            MemoryComponentEstimates(),
            ResourceEstimate(),
            PlanningIdentity(dataset_id="d"),
            _minimal_plan(),
        ]
        for model in models:
            assert isinstance(json.dumps(model.to_dict()), str), type(model)

    def test_estimate_unknown_helper(self) -> None:
        est = Estimate.unknown("s/step", note="no measurement")
        assert est.value is None
        assert est.to_dict() == {
            "value": None,
            "unit": "s/step",
            "source": "UNKNOWN",
            "confidence": "UNKNOWN",
            "note": "no measurement",
        }


# --------------------------------------------------------------------------
# frozen immutability
# --------------------------------------------------------------------------


class TestFrozen:
    @pytest.mark.parametrize(
        "model",
        [
            Estimate(1, "B"),
            ExperimentProfile(scale=TrainingScale.SMOKE, sample_count=8),
            HardwareEnvelope(),
            ParameterMetadata(),
            PlanningAssumption("t", "d"),
            PlanningWarning("C", "m"),
            PlanningIdentity(dataset_id="d"),
            MemoryComponentEstimates(),
            ResourceEstimate(),
            StepPlan(None, None, None, None),
            CheckpointPlan(),
            StorageEstimate(known_minimum_bytes=0),
            RuntimeEstimate(),
            CostEstimate("USD", 1.0, 1, 3600.0, 1.0),
        ],
        ids=lambda m: type(m).__name__,
    )
    def test_dataclass_is_frozen(self, model: Any) -> None:
        assert model.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
        field_name = dataclasses.fields(model)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(model, field_name, None)

    def test_plan_models_are_frozen(self) -> None:
        plan = _minimal_plan()
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.plan_id = "nope"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.profile = dataclasses.replace(plan.profile, sample_count=1)  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.runtime.seconds_per_step = Estimate.unknown("s/step")  # type: ignore[misc]

    def test_estimate_value_frozen(self) -> None:
        est = Estimate(value=1.0, unit="s")
        with pytest.raises(dataclasses.FrozenInstanceError):
            est.value = 999.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# portable identity / plan_id (no absolute paths)
# --------------------------------------------------------------------------


class TestPortableIdentity:
    def test_plan_id_deterministic_across_instances(self) -> None:
        first = _identity()
        second = _identity()
        assert first.plan_id() == second.plan_id()
        assert len(first.plan_id()) == 16

    def test_plan_id_changes_when_inputs_change(self) -> None:
        base = _identity()
        other_scale = _identity(profile=default_profile(TrainingScale.FULL))
        other_hw = _identity(
            hardware=HardwareEnvelope(gpu_count=2, per_gpu_vram_gb=24.0)
        )
        assert base.plan_id() != other_scale.plan_id()
        assert base.plan_id() != other_hw.plan_id()

    def test_no_absolute_path_fields_in_portable_inputs(self) -> None:
        portable = _identity().portable_dict()

        def walk(value: Any, key_hint: str = "") -> None:
            if isinstance(value, dict):
                for key, sub in value.items():
                    assert "path" not in key.lower(), f"path field leaked: {key}"
                    assert "root" not in key.lower(), f"root field leaked: {key}"
                    assert "dir" not in key.lower(), f"dir field leaked: {key}"
                    walk(sub, key)
            elif isinstance(value, str):
                assert not value.startswith(
                    ("/", "\\", "C:", "F:")
                ), f"absolute path leaked under {key_hint!r}: {value!r}"

        walk(portable)

    def test_no_path_fields_on_input_models(self) -> None:
        for model in (
            ExperimentProfile,
            HardwareEnvelope,
            ParameterMetadata,
            PlanningIdentity,
        ):
            for f in dataclasses.fields(model):
                assert "path" not in f.name.lower(), (model.__name__, f.name)
                assert f.name != "output_root"

    def test_plan_id_stable_under_dict_key_order(self) -> None:
        a = PlanningIdentity(
            adapter_identity={"a": 1, "b": 2},
            dataset_id="x",
            profile=default_profile(TrainingScale.SMOKE),
        )
        b = PlanningIdentity(
            adapter_identity={"b": 2, "a": 1},
            dataset_id="x",
            profile=default_profile(TrainingScale.SMOKE),
        )
        assert a.plan_id() == b.plan_id()

    def test_plan_id_differs_with_policy_version(self) -> None:
        base = PlanningIdentity(dataset_id="d")
        bumped = dataclasses.replace(base, planning_policy_version="2")
        assert base.plan_id() != bumped.plan_id()


# --------------------------------------------------------------------------
# model sanity details
# --------------------------------------------------------------------------


class TestModelDetails:
    def test_storage_estimate_dict_shapes(self) -> None:
        storage = StorageEstimate(
            known_minimum_bytes=5,
            estimated_working_range_bytes=(10, 20),
            unknown_contributors=("a", "b"),
        )
        d = storage.to_dict()
        assert d["estimated_working_range_bytes"] == [10, 20]  # tuple -> list
        assert d["unknown_contributors"] == ["a", "b"]

    def test_bytes_per_gib(self) -> None:
        assert BYTES_PER_GIB == 1024**3

    def test_step_plan_notes_round_trip(self) -> None:
        plan = StepPlan(
            effective_batch_size=8,
            micro_batches_per_epoch=1,
            optimizer_steps_per_epoch=1,
            planned_optimizer_steps=10,
            notes=("a", "b"),
        )
        assert plan.to_dict()["notes"] == ["a", "b"]

    def test_experiment_plan_confidence_aggregates_children(self) -> None:
        plan = _minimal_plan()
        # cost LOW, step MEDIUM, storage MEDIUM, runtime HIGH -> overall LOW
        assert plan.overall_confidence() is EstimateConfidence.LOW
        degraded = dataclasses.replace(
            plan,
            runtime=RuntimeEstimate(planned_optimizer_steps=10),
            cost=None,
            resources=None,
        )
        assert degraded.overall_confidence() is EstimateConfidence.UNKNOWN
