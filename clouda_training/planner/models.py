"""Typed planning domain model for the training experiment planner.

This module is the *vocabulary* of the planner package: every quantity a
plan carries is either a plain value or an :class:`Estimate` stamped with
an :class:`EstimateSource` (where the number came from) and an
:class:`EstimateConfidence` (how trustworthy it is).  Provenance is not
optional — "never present guesses as facts" is enforced by the type system:

* :class:`EstimateSource` — ``MEASURED`` (observed on real hardware),
  ``DECLARED`` (operator/spec supplied), ``DERIVED`` (computed from other
  proven numbers), ``HEURISTIC`` (documented rule of thumb), ``UNKNOWN``.
* :class:`EstimateConfidence` — ``HIGH``/``MEDIUM``/``LOW``/``UNKNOWN``.

Fail-closed semantics
---------------------
* Anything not backed by real evidence is ``UNKNOWN``/``UNKNOWN`` — never a
  plausible-looking fabricated number.
* All models are frozen dataclasses (immutability is asserted in tests).
* Every model exposes ``to_dict()`` returning a JSON-serializable snapshot
  (enums -> ``.value``, tuples -> lists, ints/floats/strs/bools as-is).
* No torch/transformers imports; stdlib only; no I/O.
* Portable identity: nothing that enters a ``plan_id`` hash may contain a
  filesystem path.  :class:`HardwareEnvelope` and :class:`ExperimentProfile`
  carry no path fields at all; :class:`PlanningIdentity` exposes the exact
  portable input set used for the deterministic ``plan_id``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    "BYTES_PER_GIB",
    "CHECKPOINT_INTERVAL_DEFAULTS",
    "DEFAULT_PROFILES",
    "EPOCHS_DEFAULTS",
    "GRADIENT_ACCUMULATION_DEFAULTS",
    "MAX_STEPS_DEFAULTS",
    "MICRO_BATCH_DEFAULTS",
    "PLANNING_POLICY_VERSION",
    "PRECISION_DEFAULTS",
    "SAMPLE_COUNT_DEFAULTS",
    "WORLD_SIZE_DEFAULTS",
    "CheckpointPlan",
    "CostEstimate",
    "Estimate",
    "EstimateConfidence",
    "EstimateSource",
    "ExperimentPlan",
    "ExperimentProfile",
    "HardwareEnvelope",
    "HardwareFit",
    "MemoryComponentEstimates",
    "ParameterMetadata",
    "PlanningAssumption",
    "PlanningIdentity",
    "PlanningWarning",
    "ResourceEstimate",
    "RuntimeEstimate",
    "StepPlan",
    "StorageEstimate",
    "StorageKind",
    "TrainingMode",
    "TrainingScale",
    "default_profile",
    "worst_confidence",
]

# Exact conversions used by the planner (1 GiB = 2**30 bytes).
BYTES_PER_GIB = 1 << 30

# Bumped when any planner input semantics change; plan_id hashes include it.
PLANNING_POLICY_VERSION = "1"


# --------------------------------------------------------------------------
# Provenance enums
# --------------------------------------------------------------------------


class EstimateSource(str, Enum):
    """Where an estimated quantity came from."""

    MEASURED = "MEASURED"
    DECLARED = "DECLARED"
    DERIVED = "DERIVED"
    HEURISTIC = "HEURISTIC"
    UNKNOWN = "UNKNOWN"


class EstimateConfidence(str, Enum):
    """How trustworthy an estimated quantity is."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


# Worst-first ordering; index() is the confidence rank.
_CONFIDENCE_ORDER: tuple[EstimateConfidence, ...] = (
    EstimateConfidence.UNKNOWN,
    EstimateConfidence.LOW,
    EstimateConfidence.MEDIUM,
    EstimateConfidence.HIGH,
)


def worst_confidence(
    confidences: Sequence[EstimateConfidence],
) -> EstimateConfidence:
    """Worst (least confident) entry; ``UNKNOWN`` when the list is empty."""
    if not confidences:
        return EstimateConfidence.UNKNOWN
    return min(confidences, key=_CONFIDENCE_ORDER.index)


@dataclass(frozen=True)
class Estimate:
    """A numeric quantity with mandatory provenance."""

    value: float | int | None
    unit: str
    source: EstimateSource = EstimateSource.UNKNOWN
    confidence: EstimateConfidence = EstimateConfidence.UNKNOWN
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value,
            "confidence": self.confidence.value,
            "note": self.note,
        }

    @classmethod
    def unknown(cls, unit: str, note: str = "") -> "Estimate":
        """Explicit 'we do not know' estimate (never a fabricated value)."""
        return cls(
            value=None,
            unit=unit,
            source=EstimateSource.UNKNOWN,
            confidence=EstimateConfidence.UNKNOWN,
            note=note,
        )


# --------------------------------------------------------------------------
# Scale / mode enums
# --------------------------------------------------------------------------


class TrainingScale(str, Enum):
    """Coarse experiment scale tiers."""

    SMOKE = "SMOKE"
    PILOT = "PILOT"
    MEDIUM = "MEDIUM"
    FULL = "FULL"
    CUSTOM = "CUSTOM"


class TrainingMode(str, Enum):
    """What fraction of the model is trained."""

    FULL_FINETUNE = "FULL_FINETUNE"
    SELECTIVE_FINETUNE = "SELECTIVE_FINETUNE"
    LORA_PEFT = "LORA_PEFT"


class StorageKind(str, Enum):
    """Qualitative storage class for I/O bottleneck reasoning."""

    NVME = "NVMe"
    SSD = "SSD"
    HDD = "HDD"
    NETWORK = "NETWORK"
    UNKNOWN = "UNKNOWN"


class HardwareFit(str, Enum):
    """Fail-closed memory-fit classification.

    ``LIKELY_FITS`` requires measured evidence for a matching config;
    anything less (unknown/heuristic activations) is at best ``MAY_FIT``
    and must be labelled as requiring a real smoke test.
    """

    LIKELY_FITS = "LIKELY_FITS"
    MAY_FIT = "MAY_FIT"
    LIKELY_TOO_LARGE = "LIKELY_TOO_LARGE"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# Per-scale defaults (documented heuristics, all overridable)
# --------------------------------------------------------------------------
# SMOKE: tiny sample count / very low step count, just proves the loop runs.
# PILOT: small subset large enough to produce a training signal.
# MEDIUM: intermediate scale.
# FULL: the complete dataset; steps come from the dataset, so only epochs
# and accumulation get defaults (max_steps stays None -> dataset-driven).

# --------------------------------------------------------------------------
# Input models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentProfile:
    """What we intend to train: scale + training shape.

    Every field is configurable; the documented per-scale defaults live in
    ``DEFAULT_PROFILES`` / ``default_profile()``.  ``scale=CUSTOM`` means the
    operator set the numbers explicitly and defaults must not be applied.

    No filesystem paths here — profiles are part of the portable
    ``plan_id`` input set.
    """

    scale: TrainingScale
    sample_count: int
    epochs: int = 1
    max_steps: int | None = None
    micro_batch: int = 1
    gradient_accumulation: int = 1
    world_size: int = 1
    precision: str = "bf16"
    training_mode: TrainingMode = TrainingMode.LORA_PEFT
    checkpoint_interval: int = 50

    def effective_batch_size(self) -> int:
        """Samples consumed per optimizer step (planning-only metadata)."""
        return self.micro_batch * self.gradient_accumulation * self.world_size

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale"] = self.scale.value
        data["training_mode"] = self.training_mode.value
        data["effective_batch_size"] = self.effective_batch_size()
        return data


SAMPLE_COUNT_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 32,
    TrainingScale.PILOT: 500,
    TrainingScale.MEDIUM: 5_000,
    TrainingScale.FULL: 50_000,
}

EPOCHS_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 1,
    TrainingScale.PILOT: 2,
    TrainingScale.MEDIUM: 3,
    TrainingScale.FULL: 3,
}

MICRO_BATCH_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 1,
    TrainingScale.PILOT: 2,
    TrainingScale.MEDIUM: 4,
    TrainingScale.FULL: 4,
}

GRADIENT_ACCUMULATION_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 1,
    TrainingScale.PILOT: 4,
    TrainingScale.MEDIUM: 8,
    TrainingScale.FULL: 8,
}

WORLD_SIZE_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 1,
    TrainingScale.PILOT: 1,
    TrainingScale.MEDIUM: 1,
    TrainingScale.FULL: 1,
}

MAX_STEPS_DEFAULTS: Mapping[TrainingScale, int | None] = {
    TrainingScale.SMOKE: 10,
    TrainingScale.PILOT: 200,
    TrainingScale.MEDIUM: 2_000,
    TrainingScale.FULL: None,
}

CHECKPOINT_INTERVAL_DEFAULTS: Mapping[TrainingScale, int] = {
    TrainingScale.SMOKE: 5,
    TrainingScale.PILOT: 50,
    TrainingScale.MEDIUM: 200,
    TrainingScale.FULL: 500,
}

PRECISION_DEFAULTS: Mapping[TrainingScale, str] = {
    TrainingScale.SMOKE: "bf16",
    TrainingScale.PILOT: "bf16",
    TrainingScale.MEDIUM: "bf16",
    TrainingScale.FULL: "bf16",
}


def default_profile(
    scale: TrainingScale,
    *,
    training_mode: TrainingMode = TrainingMode.LORA_PEFT,
) -> "ExperimentProfile":
    """Build the documented per-scale default profile (all configurable)."""
    return ExperimentProfile(
        scale=scale,
        sample_count=SAMPLE_COUNT_DEFAULTS[scale],
        epochs=EPOCHS_DEFAULTS[scale],
        max_steps=MAX_STEPS_DEFAULTS[scale],
        micro_batch=MICRO_BATCH_DEFAULTS[scale],
        gradient_accumulation=GRADIENT_ACCUMULATION_DEFAULTS[scale],
        world_size=WORLD_SIZE_DEFAULTS[scale],
        precision=PRECISION_DEFAULTS[scale],
        training_mode=training_mode,
        checkpoint_interval=CHECKPOINT_INTERVAL_DEFAULTS[scale],
    )


DEFAULT_PROFILES: Mapping[TrainingScale, ExperimentProfile] = {
    scale: default_profile(scale)
    for scale in (
        TrainingScale.SMOKE,
        TrainingScale.PILOT,
        TrainingScale.MEDIUM,
        TrainingScale.FULL,
    )
}


@dataclass(frozen=True)
class HardwareEnvelope:
    """The hardware we are planning for (target machine).

    Planning-only metadata: ``gpu_count`` > 1 does NOT mean N×VRAM pools
    into one addressable pool — memory.py emits that warning, this model
    simply records the envelope.  No paths; part of the portable
    ``plan_id`` input set.
    """

    gpu_count: int = 1
    per_gpu_vram_gb: float | None = None
    system_ram_gb: float | None = None
    storage_kind: StorageKind = StorageKind.UNKNOWN
    free_space_gb: float | None = None
    measured_throughput_seconds_per_step: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["storage_kind"] = self.storage_kind.value
        return data


@dataclass(frozen=True)
class ParameterMetadata:
    """Declared model parameter facts.

    All fields optional.  When a value is supplied its provenance is
    ``DECLARED`` (it comes from the model card / adapter descriptor, not
    from a local measurement — no model download happens in planning).
    """

    parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    declared_model_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source(self) -> EstimateSource:
        """``DECLARED`` when anything is supplied, else ``UNKNOWN``."""
        if (
            self.parameter_count is not None
            or self.trainable_parameter_count is not None
            or self.declared_model_size_bytes is not None
        ):
            return EstimateSource.DECLARED
        return EstimateSource.UNKNOWN


# --------------------------------------------------------------------------
# Estimate blocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StepPlan:
    """Planned optimizer-step shape (mirrors preflight plan semantics)."""

    effective_batch_size: int | None
    micro_batches_per_epoch: int | None
    optimizer_steps_per_epoch: int | None
    planned_optimizer_steps: int | None
    source: EstimateSource = EstimateSource.DERIVED
    confidence: EstimateConfidence = EstimateConfidence.MEDIUM
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


@dataclass(frozen=True)
class CheckpointPlan:
    """Checkpoint cadence + retention footprint plan."""

    save_strategy: str = "steps"
    save_steps: int = 50
    expected_checkpoint_count: int | None = None
    save_total_limit: int | None = None
    estimated_retained_count: int | None = None
    source: EstimateSource = EstimateSource.DERIVED
    confidence: EstimateConfidence = EstimateConfidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageEstimate:
    """Disk footprint with explicit unknown contributors.

    ``known_minimum_bytes`` is what we can defend; everything we cannot
    defend is listed in ``unknown_contributors`` (never folded into a fake
    total).  ``estimated_working_range_bytes`` is a low..high heuristic
    span, or ``None`` when even the span is not defensible.
    """

    known_minimum_bytes: int
    known_minimum_source: EstimateSource = EstimateSource.UNKNOWN
    known_minimum_confidence: EstimateConfidence = EstimateConfidence.UNKNOWN
    estimated_working_range_bytes: tuple[int, int] | None = None
    working_range_source: EstimateSource = EstimateSource.UNKNOWN
    working_range_confidence: EstimateConfidence = EstimateConfidence.UNKNOWN
    unknown_contributors: tuple[str, ...] = ()
    recommended_safety_reserve_bytes: int = 0
    reserve_source: EstimateSource = EstimateSource.HEURISTIC
    reserve_confidence: EstimateConfidence = EstimateConfidence.LOW

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        rng = self.estimated_working_range_bytes
        data["estimated_working_range_bytes"] = list(rng) if rng is not None else None
        data["unknown_contributors"] = list(self.unknown_contributors)
        return data


@dataclass(frozen=True)
class RuntimeEstimate:
    """Wall-clock runtime estimate.

    ``seconds_per_step`` and ``estimated_runtime_seconds`` carry a value
    ONLY when the seconds-per-step input was MEASURED or DECLARED; the
    UNKNOWN case is represented by ``Estimate.unknown`` (value ``None``),
    never by a fabricated number.
    """

    seconds_per_step: Estimate = field(
        default_factory=lambda: Estimate.unknown("s/step")
    )
    planned_optimizer_steps: int | None = None
    estimated_runtime_seconds: Estimate = field(
        default_factory=lambda: Estimate.unknown("s")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds_per_step": self.seconds_per_step.to_dict(),
            "planned_optimizer_steps": self.planned_optimizer_steps,
            "estimated_runtime_seconds": self.estimated_runtime_seconds.to_dict(),
        }


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate, valid only when runtime was estimated AND the
    operator supplied ``cost_per_gpu_hour``.  Always labelled ESTIMATED."""

    currency: str
    cost_per_gpu_hour: float
    gpus: int
    estimated_runtime_seconds: float
    estimated_cost: float
    label: str = "ESTIMATED"
    source: EstimateSource = EstimateSource.DERIVED
    confidence: EstimateConfidence = EstimateConfidence.LOW

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningAssumption:
    """One line in the assumptions ledger.

    Every DECLARED / HEURISTIC / UNKNOWN input to the plan becomes one of
    these, so the report can show exactly which numbers are facts and
    which are bets.
    """

    topic: str
    detail: str
    source: EstimateSource = EstimateSource.UNKNOWN
    confidence: EstimateConfidence = EstimateConfidence.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        data["confidence"] = self.confidence.value
        return data


@dataclass(frozen=True)
class PlanningWarning:
    """Something the operator must know before trusting the plan."""

    code: str
    message: str
    severity: str = "WARNING"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Component estimates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryComponentEstimates:
    """Per-component VRAM breakdown (bytes) with provenance.

    The transparent VRAM framework owns the math; this model only carries
    the results.  A component with no defensible value is ``None``-valued
    with ``UNKNOWN`` provenance (fail-closed).
    """

    weights_bytes: Estimate = field(default_factory=lambda: Estimate.unknown("B"))
    gradients_bytes: Estimate = field(default_factory=lambda: Estimate.unknown("B"))
    optimizer_states_bytes: Estimate = field(
        default_factory=lambda: Estimate.unknown("B")
    )
    activations_bytes: Estimate = field(default_factory=lambda: Estimate.unknown("B"))
    temporary_buffers_bytes: Estimate = field(
        default_factory=lambda: Estimate.unknown("B")
    )
    multimodal_tensors_bytes: Estimate = field(
        default_factory=lambda: Estimate.unknown("B")
    )
    framework_overhead_bytes: Estimate = field(
        default_factory=lambda: Estimate.unknown("B")
    )

    def iter_components(self) -> dict[str, Estimate]:
        """Ordered {component_name: estimate} view."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), Estimate)
        }

    def known_lower_bound_bytes(self) -> int:
        """Sum of components that have a concrete value (defensible floor)."""
        return sum(
            int(est.value)
            for est in self.iter_components().values()
            if est.value is not None
        )

    def all_known(self) -> bool:
        """True when every component carries a concrete value."""
        return all(est.value is not None for est in self.iter_components().values())

    def worst_confidence(self) -> EstimateConfidence:
        return worst_confidence(
            [est.confidence for est in self.iter_components().values()]
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: est.to_dict() for name, est in self.iter_components().items()}


@dataclass(frozen=True)
class ResourceEstimate:
    """Aggregate hardware-resource estimate for one candidate plan."""

    memory: MemoryComponentEstimates = field(default_factory=MemoryComponentEstimates)
    lower_bound_bytes: int = 0
    lower_bound_source: EstimateSource = EstimateSource.UNKNOWN
    lower_bound_confidence: EstimateConfidence = EstimateConfidence.UNKNOWN
    fit: HardwareFit = HardwareFit.UNKNOWN
    fit_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "lower_bound_bytes": self.lower_bound_bytes,
            "lower_bound_source": self.lower_bound_source.value,
            "lower_bound_confidence": self.lower_bound_confidence.value,
            "fit": self.fit.value,
            "fit_note": self.fit_note,
        }


# --------------------------------------------------------------------------
# Portable identity (plan_id inputs)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanningIdentity:
    """The portable input set the deterministic ``plan_id`` hashes.

    HARD RULE: none of these values may contain an absolute path.  This
    model is the contract between planner.py (hasher) and the tests that
    enforce path-freeness; ``portable_dict()`` is the exact pre-hash view.
    """

    planning_policy_version: str = PLANNING_POLICY_VERSION
    adapter_type: str | None = None
    adapter_identity: Mapping[str, Any] | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    profile: ExperimentProfile | None = None
    hardware: HardwareEnvelope | None = None

    def portable_dict(self) -> dict[str, Any]:
        """JSON-serializable, path-free view hashed into the plan_id."""
        return {
            "planning_policy_version": self.planning_policy_version,
            "adapter_type": self.adapter_type,
            "adapter_identity": (
                dict(self.adapter_identity) if self.adapter_identity else None
            ),
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "profile": self.profile.to_dict() if self.profile else None,
            "hardware": self.hardware.to_dict() if self.hardware else None,
        }

    def plan_id(self) -> str:
        """Deterministic hash over the portable inputs (sha256, first 16)."""
        blob = json.dumps(self.portable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id(), "inputs": self.portable_dict()}


# --------------------------------------------------------------------------
# The full plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentPlan:
    """Complete experiment plan: everything plus its provenance."""

    plan_id: str
    identity: PlanningIdentity
    profile: ExperimentProfile
    hardware: HardwareEnvelope
    step_plan: StepPlan
    checkpoint_plan: CheckpointPlan
    storage: StorageEstimate
    runtime: RuntimeEstimate
    cost: CostEstimate | None = None
    resources: ResourceEstimate | None = None
    assumptions: tuple[PlanningAssumption, ...] = ()
    warnings: tuple[PlanningWarning, ...] = ()
    recommendation: str = ""
    execution_status: str = "PLAN READY — EXECUTION DEFERRED UNTIL GPU IS AVAILABLE"
    confidence: EstimateConfidence = EstimateConfidence.UNKNOWN

    def overall_confidence(self) -> EstimateConfidence:
        """Worst confidence across the plan's evidence-bearing blocks."""
        parts = [
            self.step_plan.confidence,
            self.checkpoint_plan.confidence,
            self.storage.known_minimum_confidence,
            self.runtime.seconds_per_step.confidence,
            self.runtime.estimated_runtime_seconds.confidence,
        ]
        if self.cost is not None:
            parts.append(self.cost.confidence)
        if self.resources is not None:
            parts.append(self.resources.lower_bound_confidence)
            parts.append(self.resources.memory.worst_confidence())
        return worst_confidence(parts)

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable snapshot."""
        return {
            "plan_id": self.plan_id,
            "identity": self.identity.to_dict(),
            "profile": self.profile.to_dict(),
            "hardware": self.hardware.to_dict(),
            "step_plan": self.step_plan.to_dict(),
            "checkpoint_plan": self.checkpoint_plan.to_dict(),
            "storage": self.storage.to_dict(),
            "runtime": self.runtime.to_dict(),
            "cost": self.cost.to_dict() if self.cost is not None else None,
            "resources": (
                self.resources.to_dict() if self.resources is not None else None
            ),
            "assumptions": [a.to_dict() for a in self.assumptions],
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendation": self.recommendation,
            "execution_status": self.execution_status,
            "confidence": self.confidence.value,
            "overall_confidence": self.overall_confidence().value,
        }
