"""Storage, I/O, runtime, and cost planning for the experiment planner.

Provenance rules (enforced here, not negotiable):

* Dataset bytes are the size of the manifest file when it exists
  (MEASURED). Sample payload bytes (images/audio) are never guessed —
  they are listed as unknown contributors.
* Checkpoint footprint = (declared model bytes + derived optimizer-state
  bytes) x retained count, only when the declared inputs are known;
  otherwise the estimate is literally
  ``CHECKPOINT SIZE: UNKNOWN UNTIL REAL MODEL IS AVAILABLE``.
* I/O classification is qualitative (NVMe/SSD/HDD/NETWORK/UNKNOWN) with
  warning text only. No MB/s numbers are ever invented; an
  operator-supplied seconds-per-step is stored with source MEASURED.
* Runtime = planned_steps x seconds_per_step ONLY when seconds_per_step
  has source MEASURED (operator hardware smoke test); otherwise
  ``TRAINING TIME: UNKNOWN UNTIL HARDWARE SMOKE TEST``.
* Cost is computed only when runtime is estimated AND cost_per_gpu_hour
  is supplied; always labeled ESTIMATED. No cloud pricing is fetched.

Checkpoint cadence/count math is NOT forked: it is taken from
``clouda_training.preflight.plan.compute_training_plan``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from clouda_training.planner.models import (
    BYTES_PER_GIB,
    CheckpointPlan,
    CostEstimate,
    Estimate,
    EstimateConfidence,
    EstimateSource,
    HardwareEnvelope,
    ParameterMetadata,
    PlanningWarning,
    RuntimeEstimate,
    StorageEstimate,
    StorageKind,
)

if TYPE_CHECKING:
    from clouda_training.experiments.config import ExperimentConfig

__all__ = [
    "CHECKPOINT_UNKNOWN_MESSAGE",
    "RUNTIME_UNKNOWN_MESSAGE",
    "OPTIMIZER_STATE_BYTES_PER_PARAMETER_DEFAULT",
    "SAFETY_RESERVE_FRACTION",
    "estimate_dataset_bytes",
    "estimate_checkpoint_footprint",
    "classify_storage_io",
    "estimate_runtime",
    "estimate_cost",
    "build_checkpoint_plan",
    "plan_storage_runtime",
]

CHECKPOINT_UNKNOWN_MESSAGE = "CHECKPOINT SIZE: UNKNOWN UNTIL REAL MODEL IS AVAILABLE"
RUNTIME_UNKNOWN_MESSAGE = "TRAINING TIME: UNKNOWN UNTIL HARDWARE SMOKE TEST"

#: AdamW fp32 first + second moment = 8 bytes/parameter. Documented
#: heuristic (the same constant memory.py uses for its default
#: optimizer-state multiplier); configurable by the caller.
OPTIMIZER_STATE_BYTES_PER_PARAMETER_DEFAULT = 8.0

#: Recommended free-space safety reserve as a fraction of the known
#: storage minimum. Documented heuristic, labeled HEURISTIC — never
#: presented as a measured fact.
SAFETY_RESERVE_FRACTION = 0.5


# --------------------------------------------------------------------- storage


def estimate_dataset_bytes(manifest_path: Path | None) -> Estimate:
    """Dataset storage estimate from the manifest file when it exists.

    The manifest is the only artifact whose size is actually measurable at
    planning time: sample payloads live outside it (paths are stored
    relative to each source root), so their bytes stay UNKNOWN and are
    surfaced by the caller via ``unknown_contributors``.
    """
    if manifest_path is None:
        return Estimate.unknown("B", note="no manifest path configured")
    path = Path(manifest_path)
    if not path.is_file():
        return Estimate.unknown("B", note=f"manifest file not found: {path.name}")
    size = path.stat().st_size
    return Estimate(
        value=size,
        unit="B",
        source=EstimateSource.MEASURED,
        confidence=EstimateConfidence.HIGH,
        note=(
            "size of the manifest file only (header + rows); sample "
            "payload bytes (images) are NOT included"
        ),
    )


def estimate_checkpoint_footprint(
    *,
    parameter_metadata: ParameterMetadata,
    retained_count: int | None,
    optimizer_state_bytes_per_parameter: float = (
        OPTIMIZER_STATE_BYTES_PER_PARAMETER_DEFAULT
    ),
) -> Estimate:
    """Per-checkpoint footprint: (model bytes + optimizer state) per copy.

    Requires ``declared_model_size_bytes`` AND ``parameter_count`` (to
    derive optimizer state). Missing either -> UNKNOWN with the exact
    brief-mandated message. The optimizer-state term is the documented
    AdamW heuristic (fp32 moments), source HEURISTIC.
    """
    declared = parameter_metadata.declared_model_size_bytes
    params = parameter_metadata.parameter_count
    if declared is None or params is None or params <= 0:
        return Estimate.unknown(
            "B",
            note=(
                f"{CHECKPOINT_UNKNOWN_MESSAGE} (needs declared_model_size_bytes "
                f"and parameter_count; got declared_model_size_bytes={declared!r}, "
                f"parameter_count={params!r})"
            ),
        )
    optimizer_bytes = params * optimizer_state_bytes_per_parameter
    per_checkpoint = declared + optimizer_bytes
    if retained_count is not None:
        if retained_count <= 0:
            return Estimate(
                value=0,
                unit="B",
                source=EstimateSource.DERIVED,
                confidence=EstimateConfidence.MEDIUM,
                note="retained_count <= 0: no checkpoints retained on disk",
            )
        total = per_checkpoint * retained_count
        note = (
            f"({declared} declared model bytes + {optimizer_bytes:.0f} derived "
            f"optimizer-state bytes) x {retained_count} retained"
        )
    else:
        total = per_checkpoint
        note = (
            f"per checkpoint: {declared} declared model bytes + "
            f"{optimizer_bytes:.0f} derived optimizer-state bytes; "
            "retention count unknown"
        )
    return Estimate(
        value=total,
        unit="B",
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.MEDIUM,
        note=note,
    )


def _safety_reserve(known_minimum: int | None) -> Estimate | None:
    if known_minimum is None or known_minimum <= 0:
        return None
    reserve = int(known_minimum * SAFETY_RESERVE_FRACTION)
    return Estimate(
        value=reserve,
        unit="B",
        source=EstimateSource.HEURISTIC,
        confidence=EstimateConfidence.LOW,
        note=(
            f"{int(SAFETY_RESERVE_FRACTION * 100)}% of the known minimum "
            "(documented heuristic — real overhead is unknown until a "
            "real run writes to disk)"
        ),
    )


# ------------------------------------------------------------------ I/O class


def classify_storage_io(
    hardware: HardwareEnvelope,
) -> tuple[StorageKind, tuple[PlanningWarning, ...]]:
    """Qualitative I/O classification and warnings. No throughput numbers.

    The only numeric throughput ever recorded here is the operator's own
    measurement carried on the hardware envelope (source MEASURED).
    """
    kind = hardware.storage_kind
    warnings: list[PlanningWarning] = []
    if kind is StorageKind.HDD:
        warnings.append(
            PlanningWarning(
                code="io.storage_class",
                message=(
                    "HDD storage may bottleneck high-throughput training "
                    "I/O (checkpoint writes and sample reads); prefer "
                    "SSD/NVMe for the working set"
                ),
            )
        )
    elif kind is StorageKind.NETWORK:
        warnings.append(
            PlanningWarning(
                code="io.storage_class",
                message=(
                    "NETWORK storage adds latency/jitter to checkpoint "
                    "writes and sample reads; validate with a real "
                    "throughput measurement before long runs"
                ),
            )
        )
    elif kind is StorageKind.NVME:
        warnings.append(
            PlanningWarning(
                code="io.storage_class",
                message=(
                    "NVMe storage: suitable for high-throughput training "
                    "I/O in the planned workloads (qualitative, not "
                    "measured)"
                ),
            )
        )
    elif kind is StorageKind.UNKNOWN:
        warnings.append(
            PlanningWarning(
                code="io.storage_class",
                message=(
                    "storage class UNKNOWN: I/O suitability cannot be "
                    "assessed; declare storage_kind in the hardware "
                    "envelope"
                ),
            )
        )
    if hardware.measured_throughput_seconds_per_step is not None:
        warnings.append(
            PlanningWarning(
                code="io.throughput",
                message=(
                    "operator MEASURED throughput recorded: "
                    f"{hardware.measured_throughput_seconds_per_step} "
                    "seconds/step (stored as MEASURED, not extrapolated)"
                ),
            )
        )
    return kind, tuple(warnings)


# -------------------------------------------------------------------- runtime


def estimate_runtime(
    hardware: HardwareEnvelope,
    *,
    planned_optimizer_steps: int | None,
) -> RuntimeEstimate:
    """Source-gated runtime estimate.

    ``planned_steps x seconds_per_step`` is computed ONLY when a real
    seconds-per-step measurement exists (source MEASURED — the operator's
    hardware smoke test). Anything else stays UNKNOWN with the
    brief-mandated message; no seconds-per-step value is ever assumed.
    """
    measured = hardware.measured_throughput_seconds_per_step
    if measured is None or measured <= 0:
        return RuntimeEstimate(
            seconds_per_step=Estimate.unknown(
                "s/step", note="no hardware smoke-test measurement supplied"
            ),
            planned_optimizer_steps=planned_optimizer_steps,
            estimated_runtime_seconds=Estimate.unknown(
                "s", note=RUNTIME_UNKNOWN_MESSAGE
            ),
        )
    seconds_per_step = Estimate(
        value=measured,
        unit="s/step",
        source=EstimateSource.MEASURED,
        confidence=EstimateConfidence.HIGH,
        note="operator hardware smoke-test measurement",
    )
    if planned_optimizer_steps is None or planned_optimizer_steps <= 0:
        return RuntimeEstimate(
            seconds_per_step=seconds_per_step,
            planned_optimizer_steps=planned_optimizer_steps,
            estimated_runtime_seconds=Estimate.unknown(
                "s",
                note="planned optimizer steps unknown; cannot multiply",
            ),
        )
    total = planned_optimizer_steps * measured
    return RuntimeEstimate(
        seconds_per_step=seconds_per_step,
        planned_optimizer_steps=planned_optimizer_steps,
        estimated_runtime_seconds=Estimate(
            value=total,
            unit="s",
            source=EstimateSource.DERIVED,
            confidence=EstimateConfidence.MEDIUM,
            note=(
                f"{planned_optimizer_steps} planned optimizer steps x "
                f"{measured} measured seconds/step"
            ),
        ),
    )


# ----------------------------------------------------------------------- cost


def estimate_cost(
    runtime: RuntimeEstimate,
    *,
    cost_per_gpu_hour: float | None,
    gpu_count: int,
) -> CostEstimate | None:
    """ESTIMATED cost, only when runtime is estimated AND a rate is given.

    Returns ``None`` when gated out. No cloud pricing is fetched, ever.
    """
    if cost_per_gpu_hour is None or cost_per_gpu_hour <= 0:
        return None
    total = runtime.estimated_runtime_seconds
    if total.value is None or total.source is not EstimateSource.DERIVED:
        return None
    total_seconds = float(total.value)
    gpu_hours = total_seconds * gpu_count / 3600.0
    cost = gpu_hours * cost_per_gpu_hour
    return CostEstimate(
        currency="USD",
        cost_per_gpu_hour=cost_per_gpu_hour,
        gpus=gpu_count,
        estimated_runtime_seconds=total_seconds,
        estimated_cost=cost,
        label="ESTIMATED",
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.LOW,
    )


# ----------------------------------------------------------------- checkpoints


def build_checkpoint_plan(
    config: ExperimentConfig,
    *,
    dataset_row_count: int | None,
    world_size: int = 1,
) -> tuple[CheckpointPlan, int | None]:
    """Checkpoint cadence/count via the preflight plan math (no fork).

    Calls ``clouda_training.preflight.plan.compute_training_plan`` and
    mirrors its ``planned // save_steps`` semantics exactly. Returns
    ``(plan, planned_optimizer_steps)`` so runtime/cost reuse the same
    step count.
    """
    from clouda_training.preflight.plan import compute_training_plan

    summary = compute_training_plan(
        config,
        dataset_row_count=dataset_row_count,
        world_size=world_size,
    )
    checkpoint = config.checkpoint
    expected: int | None = None
    retained: int | None = None
    if checkpoint.save_strategy == "steps" and checkpoint.save_steps > 0:
        expected = summary.estimated_checkpoint_count
        limit = checkpoint.save_total_limit
        if limit is not None and limit > 0 and expected is not None:
            retained = min(expected, limit)
        elif expected is not None:
            retained = expected
    plan = CheckpointPlan(
        save_strategy=checkpoint.save_strategy,
        save_steps=checkpoint.save_steps,
        expected_checkpoint_count=expected,
        save_total_limit=checkpoint.save_total_limit,
        estimated_retained_count=retained,
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.MEDIUM,
    )
    return plan, summary.planned_optimizer_steps


# ------------------------------------------------------------------- top level


def plan_storage_runtime(
    config: ExperimentConfig,
    *,
    hardware: HardwareEnvelope,
    parameter_metadata: ParameterMetadata,
    dataset_row_count: int | None = None,
    cost_per_gpu_hour: float | None = None,
) -> dict[str, Any]:
    """Assemble storage + I/O + runtime + cost + checkpoint planning.

    Returns a dict with ``storage``, ``runtime``, ``cost`` (or None),
    ``checkpoint_plan``, ``planned_optimizer_steps``, and ``warnings`` —
    consumable by ``planner.build_experiment_plan``.
    """
    warnings: list[PlanningWarning] = []

    # Checkpoint cadence/count FIRST: runtime needs the planned step count
    # from the same preflight math (single source of truth, no fork).
    checkpoint_plan, planned_steps = build_checkpoint_plan(
        config,
        dataset_row_count=dataset_row_count,
        world_size=hardware.gpu_count if hardware.gpu_count > 1 else 1,
    )

    # -- storage --------------------------------------------------------
    dataset_bytes = estimate_dataset_bytes(config.dataset.manifest_path)
    if dataset_bytes.value is None:
        warnings.append(
            PlanningWarning(
                code="storage.dataset_bytes",
                message=(
                    "dataset byte size UNKNOWN: manifest file missing; "
                    "sample payload bytes are never guessed"
                ),
            )
        )
    retained = checkpoint_plan.estimated_retained_count
    checkpoint_bytes = estimate_checkpoint_footprint(
        parameter_metadata=parameter_metadata,
        retained_count=retained,
    )
    per_checkpoint = estimate_checkpoint_footprint(
        parameter_metadata=parameter_metadata,
        retained_count=1,
    )

    unknown_contributors: list[str] = []
    known_parts: list[int] = []
    if dataset_bytes.value is None:
        unknown_contributors.append("dataset payload bytes (manifest missing)")
    else:
        known_parts.append(int(dataset_bytes.value))
    if checkpoint_bytes.value is None:  # UNKNOWN sentinel
        unknown_contributors.append(
            "checkpoint footprint (needs declared model bytes + "
            "parameter count from the real model)"
        )
    else:
        known_parts.append(int(checkpoint_bytes.value))
    unknown_contributors.append(
        "optimizer/workspace scratch files and framework logging output"
    )
    known_minimum = sum(known_parts) if known_parts else None
    reserve = _safety_reserve(known_minimum)

    storage_notes: list[str] = [
        "known_minimum covers only measurable artifacts (manifest file, "
        "retained checkpoints when declared inputs exist)",
    ]
    if reserve is not None:
        storage_notes.append(f"safety reserve: {reserve.note}")
    else:
        storage_notes.append("safety reserve: no known minimum, reserve unset")

    if known_minimum is not None:
        reserve_bytes = 0
        if reserve is not None and reserve.value is not None:
            reserve_bytes = int(reserve.value)
        storage = StorageEstimate(
            known_minimum_bytes=known_minimum,
            known_minimum_source=EstimateSource.DERIVED,
            known_minimum_confidence=EstimateConfidence.MEDIUM,
            unknown_contributors=tuple(unknown_contributors),
            recommended_safety_reserve_bytes=reserve_bytes,
            reserve_source=EstimateSource.HEURISTIC,
            reserve_confidence=EstimateConfidence.LOW,
        )
    else:
        storage = StorageEstimate(
            known_minimum_bytes=0,
            known_minimum_source=EstimateSource.UNKNOWN,
            known_minimum_confidence=EstimateConfidence.UNKNOWN,
            unknown_contributors=tuple(unknown_contributors),
            recommended_safety_reserve_bytes=0,
            reserve_source=EstimateSource.HEURISTIC,
            reserve_confidence=EstimateConfidence.LOW,
        )

    # -- I/O ------------------------------------------------------------
    _kind, io_warnings = classify_storage_io(hardware)
    warnings.extend(io_warnings)

    # -- runtime + cost ---------------------------------------------------
    runtime = estimate_runtime(hardware, planned_optimizer_steps=planned_steps)
    if runtime.estimated_runtime_seconds.value is None:
        warnings.append(
            PlanningWarning(code="runtime.estimate", message=RUNTIME_UNKNOWN_MESSAGE)
        )
    cost = estimate_cost(
        runtime,
        cost_per_gpu_hour=cost_per_gpu_hour,
        gpu_count=hardware.gpu_count,
    )
    if cost is None and cost_per_gpu_hour is not None:
        warnings.append(
            PlanningWarning(
                code="cost.estimate",
                message=(
                    "cost not estimated: runtime is "
                    f"{RUNTIME_UNKNOWN_MESSAGE}; run the hardware smoke "
                    "test first"
                ),
            )
        )

    return {
        "storage": storage,
        "runtime": runtime,
        "cost": cost,
        "checkpoint_plan": checkpoint_plan,
        "planned_optimizer_steps": planned_steps,
        "dataset_bytes": dataset_bytes,
        "checkpoint_bytes_per_checkpoint": per_checkpoint,
        "checkpoint_total_bytes": checkpoint_bytes,
        "warnings": tuple(warnings),
        "constants": {"BYTES_PER_GIB": BYTES_PER_GIB},
    }
