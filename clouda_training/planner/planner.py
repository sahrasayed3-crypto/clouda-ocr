"""Experiment planner orchestrator: build full ExperimentPlans.

Assembles step math (via the canonical preflight plan module), VRAM
estimates (memory.py), storage/runtime/cost (storage_runtime.py), the
assumptions ledger, deterministic plan identity, the small candidate matrix,
and the conservative first-run recommendation.

Boundaries respected:
- Planner designs experiments; the Training Preflight Validator verifies
  concrete configs; the runtime executes. The planner never runs training.
- No forks of preflight math: step planning calls
  ``clouda_training.preflight.plan.compute_training_plan``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from clouda_training.experiments.config import (
    CheckpointSection,
    DatasetSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    RuntimeSection,
    TrainingSection,
)
from clouda_training.planner.models import (
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
    ResourceEstimate,
    StepPlan,
    TrainingMode,
    TrainingScale,
    default_profile,
)
from clouda_training.preflight.plan import compute_training_plan

PLANNING_POLICY_VERSION = "1.0.0"

_MATRIX_SCALES: tuple[TrainingScale, ...] = (
    TrainingScale.SMOKE,
    TrainingScale.PILOT,
    TrainingScale.PILOT,  # conservative vs higher-batch variant below
    TrainingScale.MEDIUM,
    TrainingScale.FULL,
)


def build_experiment_plan(
    config: ExperimentConfig,
    profile: ExperimentProfile,
    *,
    hardware: HardwareEnvelope | None = None,
    parameter_metadata: ParameterMetadata | None = None,
    measured_activations_bytes: int | None = None,
    cost_per_gpu_hour: float | None = None,
    dataset_row_count: int | None = None,
    optimizer_state_multiplier: float | None = None,
) -> ExperimentPlan:
    """Build a full plan for one experiment profile.

    Read-only: no training, no downloads, no side effects.
    """
    from clouda_training.planner.memory import estimate_vram
    from clouda_training.planner.storage_runtime import (
        build_checkpoint_plan,
        plan_storage_runtime,
    )

    hardware = hardware or HardwareEnvelope()
    parameter_metadata = parameter_metadata or ParameterMetadata()

    # -- training mode validation against adapter capabilities -------------
    mode_warnings = _validate_training_mode(config, profile)

    # -- step plan: reuse canonical preflight math (no fork) ---------------
    row_count = (
        dataset_row_count
        if dataset_row_count is not None
        else _row_count_from_manifest(config)
    )
    preflight_summary = compute_training_plan(
        config, dataset_row_count=row_count, world_size=profile.world_size
    )
    step_plan = StepPlan(
        effective_batch_size=preflight_summary.effective_batch_size,
        micro_batches_per_epoch=preflight_summary.micro_batches_per_epoch,
        optimizer_steps_per_epoch=preflight_summary.optimizer_steps_per_epoch,
        planned_optimizer_steps=preflight_summary.planned_optimizer_steps,
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.HIGH,
        notes=preflight_summary.notes,
    )

    # -- checkpoint plan (delegates to storage_runtime, which itself calls
    #    the preflight math) ------------------------------------------------
    checkpoint_plan, planned_steps = build_checkpoint_plan(
        config, dataset_row_count=row_count, world_size=profile.world_size
    )

    # -- VRAM estimate ------------------------------------------------------
    vram = estimate_vram(
        parameter_metadata,
        hardware=hardware,
        precision=profile.precision,
        training_mode=profile.training_mode,
        measured_activations_bytes=measured_activations_bytes,
        optimizer_state_multiplier=optimizer_state_multiplier or 2.0,
    )
    resources: ResourceEstimate = vram["resources"]

    # -- storage / runtime / cost ------------------------------------------
    storage_runtime = plan_storage_runtime(
        config,
        hardware=hardware,
        parameter_metadata=parameter_metadata,
        dataset_row_count=row_count,
        cost_per_gpu_hour=cost_per_gpu_hour,
    )
    storage = storage_runtime["storage"]
    runtime = storage_runtime["runtime"]
    cost = storage_runtime["cost"]

    # -- assumptions ledger --------------------------------------------------
    assumptions: list[PlanningAssumption] = []
    components = resources.memory

    def _assume(
        topic: str,
        detail: str,
        source: EstimateSource,
        confidence: EstimateConfidence = EstimateConfidence.UNKNOWN,
    ) -> None:
        assumptions.append(
            PlanningAssumption(
                topic=topic, detail=detail, source=source, confidence=confidence
            )
        )

    if parameter_metadata.parameter_count is not None:
        _assume(
            "parameter_count",
            f"supplied manually: {parameter_metadata.parameter_count:,} "
            "(weights not installed)",
            EstimateSource.DECLARED,
            EstimateConfidence.HIGH,
        )
    else:
        _assume(
            "parameter_count",
            "unknown — VRAM lower bound unavailable",
            EstimateSource.UNKNOWN,
        )
    if components.activations_bytes.source is EstimateSource.UNKNOWN:
        _assume(
            "activation_memory",
            "unknown (model/input/hardware dependent)",
            EstimateSource.UNKNOWN,
        )
    _assume(
        "gpu_profile",
        f"no real {config.model.adapter_type} GPU profile exists in Clouda — "
        "memory components are theoretical bounds, not measurements",
        EstimateSource.UNKNOWN,
    )
    _assume(
        "world_size",
        f"assumed {profile.world_size} (planning value)",
        EstimateSource.DECLARED,
        EstimateConfidence.HIGH,
    )
    if profile.precision in ("bf16", "fp16"):
        _assume(
            "precision",
            f"{profile.precision} planned but not hardware validated "
            "(precision widths are storage math, not measured VRAM)",
            EstimateSource.DECLARED,
            EstimateConfidence.MEDIUM,
        )
    for warning in mode_warnings:
        _assume("training_mode", warning.message, EstimateSource.DECLARED)

    # VRAM-fit warnings join the plan's warning list.
    warnings: tuple[PlanningWarning, ...] = tuple(mode_warnings) + tuple(
        vram["warnings"]
    )

    # -- identity -----------------------------------------------------------
    identity = PlanningIdentity(
        adapter_type=config.model.adapter_type,
        adapter_identity=_adapter_identity(config),
        dataset_id=config.dataset.dataset_id,
        dataset_version=config.dataset.dataset_version,
        profile=profile,
        hardware=hardware,
    )
    plan_id = _plan_id(identity)

    execution_status = (
        "DEFERRED — GPU NOT AVAILABLE"
        if resources.fit
        in (HardwareFit.LIKELY_TOO_LARGE, HardwareFit.MAY_FIT, HardwareFit.UNKNOWN)
        else "PLAN READY"
    )

    return ExperimentPlan(
        plan_id=plan_id,
        identity=identity,
        profile=profile,
        hardware=hardware,
        step_plan=step_plan,
        checkpoint_plan=checkpoint_plan,
        storage=storage,
        runtime=runtime,
        cost=cost,
        resources=resources,
        assumptions=tuple(assumptions),
        warnings=warnings,
        recommendation=_recommend_first_validation(resources, profile),
        execution_status=execution_status,
        confidence=_overall_confidence(resources, step_plan),
    )


def _validate_training_mode(
    config: ExperimentConfig, profile: ExperimentProfile
) -> tuple[PlanningWarning, ...]:
    """Reject training modes the adapter does not declare support for."""
    from clouda_training.preflight.checks_system import (
        ensure_adapters_registered,
        resolve_descriptor,
    )

    adapter_type = config.model.adapter_type
    if (
        adapter_type in {"mock", "torch"}
        or profile.training_mode is not TrainingMode.LORA_PEFT
    ):
        return ()
    ensure_adapters_registered()
    try:
        descriptor = resolve_descriptor(adapter_type)
        caps = descriptor.capabilities
    except Exception:  # noqa: BLE001 — unknown adapters reported by preflight
        return ()
    if not getattr(caps, "supports_lora", False):
        return (
            PlanningWarning(
                code="LORA_NOT_SUPPORTED",
                message=(
                    f"adapter {adapter_type!r} does not declare LoRA/PEFT "
                    "support in Clouda — the full/selective path is the "
                    "validated planning target"
                ),
            ),
        )
    return ()


def _adapter_identity(config: ExperimentConfig) -> dict[str, Any] | None:
    if config.model.adapter_type in {"mock", "torch"}:
        return None
    try:
        from clouda_training.preflight.checks_system import resolve_descriptor

        return dict(resolve_descriptor(config.model.adapter_type).identity_dict())
    except Exception:  # noqa: BLE001 — identity is informational
        return None


def _row_count_from_manifest(config: ExperimentConfig) -> int | None:
    """Cheap row count from the manifest header when available (metadata-first)."""
    manifest = Path(config.dataset.manifest_path)
    if not manifest.is_file():
        return None
    try:
        first = manifest.read_text(encoding="utf-8").splitlines()[0]
        header = json.loads(first)
        count = header.get("_row_count")
        return int(count) if count else None
    except (OSError, json.JSONDecodeError, IndexError, ValueError):
        return None


def _plan_id(identity: PlanningIdentity) -> str:
    payload = json.dumps(identity.portable_dict(), sort_keys=True, ensure_ascii=False)
    return "plan-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _recommend_first_validation(
    resources: ResourceEstimate, profile: ExperimentProfile
) -> str:
    """Conservative recommendation — never claims optimality."""
    if resources.fit is HardwareFit.LIKELY_TOO_LARGE:
        return (
            "RECOMMENDED FOR FIRST VALIDATION: reduce the profile (SMOKE/PILOT, "
            "SELECTIVE_FINETUNE, or LoRA when supported) — current estimate "
            "exceeds the target VRAM lower bound"
        )
    return (
        "RECOMMENDED FOR FIRST VALIDATION: PILOT conservative — no real GPU "
        "measurement exists and these weights have never been trained in "
        "Clouda; use the smallest plan that validates end-to-end behavior"
    )


def _overall_confidence(
    resources: ResourceEstimate, step_plan: StepPlan
) -> EstimateConfidence:
    """Categorical confidence: worst across blocks (never a fake score)."""
    return min(step_plan.confidence, resources.lower_bound_confidence)


# ----------------------------------------------------------------- matrix


def build_experiment_matrix(
    config: ExperimentConfig,
    *,
    hardware: HardwareEnvelope | None = None,
    parameter_metadata: ParameterMetadata | None = None,
    dataset_row_count: int | None = None,
    max_candidates: int = 5,
) -> list[ExperimentPlan]:
    """Small deterministic candidate matrix (default 5, never combinatorial).

    A — Smoke, B — Pilot conservative, C — Pilot higher batch,
    D — Medium, E — Full planned.
    """
    hardware = hardware or HardwareEnvelope()
    parameter_metadata = parameter_metadata or ParameterMetadata()
    rows = dataset_row_count or _row_count_from_manifest(config)

    candidates: list[ExperimentProfile] = [
        default_profile(TrainingScale.SMOKE),
        default_profile(TrainingScale.PILOT),
        dataclasses.replace(
            default_profile(TrainingScale.PILOT), micro_batch=2, gradient_accumulation=4
        ),
        default_profile(TrainingScale.MEDIUM),
        default_profile(TrainingScale.FULL),
    ][:max_candidates]
    if rows is not None:
        # Bound every candidate's sample_count to the actual dataset size.
        candidates = [
            dataclasses.replace(p, sample_count=min(p.sample_count, rows))
            for p in candidates
        ]

    plans: list[ExperimentPlan] = []
    for profile in candidates:
        profile_config = _apply_profile(config, profile, rows)
        plans.append(
            build_experiment_plan(
                profile_config,
                profile,
                hardware=hardware,
                parameter_metadata=parameter_metadata,
                dataset_row_count=rows,
            )
        )
    return plans


def _apply_profile(
    config: ExperimentConfig, profile: ExperimentProfile, rows: int | None
) -> ExperimentConfig:
    """Apply the profile's training knobs onto a copy of the base config."""
    sample_limit = (
        profile.sample_count
        if rows is None
        else min(profile.sample_count, rows) or None
    )
    return ExperimentConfig(
        experiment=ExperimentSection(
            name=f"{config.experiment.name}-{profile.scale.value.lower()}"
        ),
        model=config.model,
        dataset=DatasetSection(
            dataset_id=config.dataset.dataset_id,
            dataset_version=config.dataset.dataset_version,
            manifest_path=config.dataset.manifest_path,
            split=config.dataset.split,
            sample_limit=sample_limit,
        ),
        training=TrainingSection(
            seed=config.training.seed,
            epochs=profile.epochs,
            max_steps=profile.max_steps,
            batch_size=profile.micro_batch,
            gradient_accumulation_steps=profile.gradient_accumulation,
            learning_rate=config.training.learning_rate,
            scheduler=config.training.scheduler,
        ),
        checkpoint=CheckpointSection(
            save_strategy=config.checkpoint.save_strategy,
            save_steps=(
                profile.checkpoint_interval
                if profile.checkpoint_interval
                else config.checkpoint.save_steps
            ),
            save_total_limit=config.checkpoint.save_total_limit,
        ),
        runtime=config.runtime,
        tracking=config.tracking,
    )


# ------------------------------------------------------- config generation


def generate_training_config(
    plan: ExperimentPlan,
    *,
    base_config: ExperimentConfig,
    model_path: str | None = None,
) -> ExperimentConfig:
    """Emit the canonical training config a preflight can consume.

    Flow: Planner -> generated config -> Preflight -> READY/NOT_READY.
    The planner never bypasses preflight.
    """
    profile = plan.profile
    return ExperimentConfig(
        experiment=ExperimentSection(
            name=f"{base_config.experiment.name}-{plan.profile.scale.value.lower()}"
        ),
        model=ModelSection(
            model_id=base_config.model.model_id,
            revision=base_config.model.revision,
            model_family=base_config.model.model_family,
            adapter_type=base_config.model.adapter_type,
            trust_remote_code=base_config.model.trust_remote_code,
            precision=profile.precision,
        ),
        dataset=base_config.dataset,
        training=TrainingSection(
            seed=base_config.training.seed,
            epochs=profile.epochs,
            max_steps=profile.max_steps,
            batch_size=profile.micro_batch,
            gradient_accumulation_steps=profile.gradient_accumulation,
            learning_rate=base_config.training.learning_rate,
            scheduler=base_config.training.scheduler,
        ),
        checkpoint=CheckpointSection(
            save_strategy=base_config.checkpoint.save_strategy,
            save_steps=profile.checkpoint_interval or base_config.checkpoint.save_steps,
            save_total_limit=base_config.checkpoint.save_total_limit,
            resume_from=base_config.checkpoint.resume_from,
        ),
        runtime=RuntimeSection(
            device=base_config.runtime.device,
            output_root=base_config.runtime.output_root,
            dry_run=base_config.runtime.dry_run,
        ),
        tracking=base_config.tracking,
    )


# ------------------------------------------------------------- reporting


def render_plan_report(plan: ExperimentPlan, *, model_label: str) -> str:
    """Human-readable report (example format from the planning spec)."""
    from clouda_training.planner.memory import bytes_to_gib

    if plan.resources is None:
        return (
            f"CLOUDA TRAINING EXPERIMENT PLAN\n\nModel:\n{model_label}\n"
            "\nResource estimate: UNAVAILABLE (no parameter metadata)\n"
            f"\nSTATUS: {plan.execution_status}"
        )

    lines: list[str] = ["CLOUDA TRAINING EXPERIMENT PLAN", ""]
    lines.append(f"Model:\n{model_label}")
    lines.append(f"\nProfile:\n{plan.profile.scale.value}")
    lines.append(
        f"\nDataset:\n{plan.profile.sample_count:,} samples"
        if plan.profile.sample_count
        else "\nDataset:\n(configured manifest; row count not counted here)"
    )
    sp = plan.step_plan
    lines.append("\nTraining math:")
    lines.append(f"  Effective batch: {sp.effective_batch_size}")
    lines.append(f"  Micro batches/epoch: {sp.micro_batches_per_epoch}")
    lines.append(f"  Optimizer steps/epoch: {sp.optimizer_steps_per_epoch}")
    lines.append(f"  Total optimizer steps: {sp.planned_optimizer_steps}")
    cp = plan.checkpoint_plan
    lines.append("\nCheckpoints:")
    lines.append(
        f"  Every {cp.save_steps} steps -> expected {cp.expected_checkpoint_count}"
    )
    if cp.save_total_limit:
        lines.append(f"  Retention: keep last {cp.save_total_limit}")
    r = plan.resources
    lines.append("\nResource estimate (per GPU):")
    weights_gib = bytes_to_gib(r.memory.weights_bytes.value)
    lines.append(
        f"  Weights lower bound: " f"{weights_gib:.1f} GiB"
        if weights_gib is not None
        else "  Weights lower bound: UNKNOWN"
    )
    lines.append(f"    ({r.memory.weights_bytes.source.value})")
    opt_gib = bytes_to_gib(r.memory.optimizer_states_bytes.value)
    lines.append(
        "  Optimizer state: "
        + (f"{opt_gib:.1f} GiB" if opt_gib is not None else "UNKNOWN")
        + f" ({r.memory.optimizer_states_bytes.source.value})"
    )
    lines.append(
        f"  Activations: {r.memory.activations_bytes.source.value}"
        if r.memory.activations_bytes.value is None
        else f"  Activations: {bytes_to_gib(r.memory.activations_bytes.value):.1f} GiB (MEASURED)"
    )
    lines.append(f"  Estimated VRAM status: {r.fit.value}")
    lines.append(f"  Confidence: {plan.confidence}")
    s = plan.storage
    lines.append("\nStorage:")
    min_gib = bytes_to_gib(s.known_minimum_bytes)
    lines.append(
        "  Known minimum: "
        + (f"{min_gib:.2f} GiB" if min_gib is not None else "UNKNOWN")
    )
    for contributor in s.unknown_contributors:
        lines.append(f"  Unknown: {contributor}")
    lines.append(f"\nSTATUS: {plan.execution_status}")
    lines.append(f"\nNext action:\n{plan.recommendation}")
    if plan.assumptions:
        lines.append("\nASSUMPTIONS")
        for assumption in plan.assumptions:
            lines.append(
                f"- [{assumption.source.value}/{assumption.confidence.value}] "
                f"{assumption.topic}: {assumption.detail}"
            )
    return "\n".join(lines)


__all__ = [
    "PLANNING_POLICY_VERSION",
    "build_experiment_matrix",
    "build_experiment_plan",
    "generate_training_config",
    "render_plan_report",
]
