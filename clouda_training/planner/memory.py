"""Transparent VRAM estimation for the training experiment planner.

Design contract (Wave-1 brief)
------------------------------
Safe, transparent math only.  Every component of the VRAM breakdown
carries an :class:`~clouda_training.planner.models.EstimateSource` and an
:class:`~clouda_training.planner.models.EstimateConfidence`; anything we
cannot defend is ``UNKNOWN`` with value ``None`` — never a fabricated
number dressed up as a fact.

Components (all bytes):

* ``weights_bytes`` — ``parameters x bytes_per_parameter`` using the
  documented precision width table (fp32=4, bf16=2, fp16=2).  Storage
  precision is DISTINCT from optimizer-state precision (an fp16 model
  still gets fp32 AdamW moments).
* ``gradients_bytes`` — ``trainable_parameters x gradient_width`` where
  the gradient width equals the weights storage width unless declared
  otherwise.
* ``optimizer_states_bytes`` — AdamW-style first+second fp32 moments:
  ``trainable_parameters x optimizer_state_bytes_per_parameter`` with a
  CONFIGURABLE multiplier (default 2.0 fp32 moments = 8 bytes/param).
  This is a documented HEURISTIC, never a measurement.
* ``activations_bytes`` — MEASURED (operator-supplied profile),
  HEURISTIC (adapter-verified profile), or UNKNOWN.  There is NO
  universal activation formula; nothing is ever fabricated here.
* ``temporary_buffers_bytes`` / ``multimodal_tensors_bytes`` /
  ``framework_overhead_bytes`` — UNKNOWN unless explicitly declared by
  the caller (e.g. a real smoke test).

Hardware-fit classification is fail-closed:

* lower bound > per-GPU VRAM target          -> ``LIKELY_TOO_LARGE``
* lower bound fits but activations are
  UNKNOWN/HEURISTIC (or any component unknown) -> ``MAY_FIT`` (REQUIRES
  REAL SMOKE TEST — never reported as ready)
* ``LIKELY_FITS`` only when activations are MEASURED for a matching
  configuration (real smoke-test evidence).

Multi-GPU semantics: ``gpu_count > 1`` is PLANNING-ONLY metadata.  This
framework NEVER claims N x VRAM pools into one addressable pool.  It
emits a ``PLANNING ONLY — EXECUTION BACKEND NOT YET VALIDATED`` warning
because the current runtime has no distributed support (verified: grep
of ``clouda_training/runtime/`` for ``distributed|DDP|FSDP|accelerate``
returns zero matches on this branch).

Image/sequence factors (``SequenceFactors``) are represented as
documented UNCERTAINTY (multiplicative bounds on the activation
uncertainty band), not as fake math: they widen the ``MAY_FIT`` caution,
they never produce a number.

No torch/transformers import happens at module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clouda_training.planner.models import (
    BYTES_PER_GIB,
    Estimate,
    EstimateConfidence,
    EstimateSource,
    HardwareEnvelope,
    HardwareFit,
    ParameterMetadata,
    PlanningWarning,
    TrainingMode,
)

if TYPE_CHECKING:
    from clouda_training.planner.models import MemoryComponentEstimates

__all__ = [
    "ACTIVATIONS_UNMEASURED_NOTE",
    "ADAMW_BYTES_PER_PARAM_DEFAULT",
    "BYTES_PER_GIB",
    "GRADIENT_WIDTH_TABLE",
    "PRECISION_WIDTH_TABLE",
    "SMOKE_TEST_REQUIRED_NOTE",
    "SequenceFactors",
    "bytes_to_gib",
    "estimate_component_breakdown",
    "estimate_gradients",
    "estimate_optimizer_states",
    "estimate_weights",
    "estimate_vram",
    "resolve_precision",
]


# --------------------------------------------------------------------------
# Documented width tables (the ONLY precision math in this framework)
# --------------------------------------------------------------------------
# Storage precision distinct from optimizer-state precision: bf16/fp16
# weights still get fp32 AdamW moments.  Unknown precision is fail-closed
# (UNKNOWN), never silently treated as any default width.

PRECISION_WIDTH_TABLE: dict[str, int] = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
}

#: Gradient width equals the weights storage width unless declared.
GRADIENT_WIDTH_TABLE: dict[str, int] = dict(PRECISION_WIDTH_TABLE)

#: AdamW: first + second fp32 moments = 2 x 4 = 8 bytes/parameter.
#: Documented HEURISTIC; configurable via ``optimizer_state_multiplier``.
ADAMW_BYTES_PER_PARAM_DEFAULT = 8.0

#: fp32 moment width, used when the caller supplies a moment COUNT.
FP32_MOMENT_BYTES = 4

_MULTI_GPU_WARNING_TEXT = (
    "PLANNING ONLY — EXECUTION BACKEND NOT YET VALIDATED: gpu_count>1 is "
    "planning metadata; VRAM does NOT pool across GPUs into one addressable "
    "pool, and the current runtime (clouda_training/runtime/) contains no "
    "distributed/DDP/FSDP support (verified by source search), so every GPU "
    "must hold the full per-GPU footprint independently"
)

ACTIVATIONS_UNMEASURED_NOTE = (
    "activations UNKNOWN: no measured profile supplied; a universal "
    "activation formula does not exist and none is fabricated here"
)

SMOKE_TEST_REQUIRED_NOTE = (
    "REQUIRES REAL SMOKE TEST: lower bound fits but activations/overhead "
    "are not measured for this configuration"
)


def bytes_to_gib(nbytes: int | float | None) -> float | None:
    """Exact binary-GiB conversion (1 GiB = 2**30 bytes); None-safe."""
    if nbytes is None:
        return None
    return float(nbytes) / BYTES_PER_GIB


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def resolve_precision(
    precision: str | None,
) -> tuple[int | None, EstimateSource, EstimateConfidence, str]:
    """Map a precision string to bytes/parameter via the documented table.

    Returns ``(bytes_per_parameter, source, confidence, note)``.  An
    unrecognized precision is fail-closed: ``None`` bytes with UNKNOWN
    provenance — the caller must not guess a width.
    """
    if not precision:
        return (
            None,
            EstimateSource.UNKNOWN,
            EstimateConfidence.UNKNOWN,
            "no precision declared; weights width UNKNOWN",
        )
    key = precision.strip().lower()
    width = PRECISION_WIDTH_TABLE.get(key)
    if width is None:
        return (
            None,
            EstimateSource.UNKNOWN,
            EstimateConfidence.UNKNOWN,
            f"precision {precision!r} not in documented width table "
            f"{sorted(set(PRECISION_WIDTH_TABLE.values()))}-byte tiers "
            "(fp32=4, bf16/fp16=2); width UNKNOWN",
        )
    return (
        width,
        EstimateSource.DECLARED,
        EstimateConfidence.HIGH,
        f"precision {key!r} -> {width} bytes/parameter (documented width table)",
    )


# --------------------------------------------------------------------------
# Component estimates
# --------------------------------------------------------------------------


def estimate_weights(
    parameter_metadata: ParameterMetadata,
    *,
    precision: str | None,
) -> Estimate:
    """weights = parameter_count x bytes_per_parameter (width table).

    Source is DECLARED (both inputs are declarations) and confidence HIGH
    only when BOTH are known; otherwise UNKNOWN with value None.
    """
    width, source, confidence, note = resolve_precision(precision)
    params = parameter_metadata.parameter_count
    if params is None or params <= 0 or width is None:
        missing = []
        if params is None or params <= 0:
            missing.append("parameter_count")
        if width is None:
            missing.append("bytes_per_parameter (precision)")
        return Estimate.unknown(
            "B",
            note="weights UNKNOWN: missing " + " and ".join(missing),
        )
    return Estimate(
        value=int(params) * width,
        unit="B",
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.HIGH,
        note=(
            f"{params} parameters x {width} B/param ({precision}); "
            f"parameter count {source.value.lower()}"
        ),
    )


def estimate_gradients(
    parameter_metadata: ParameterMetadata,
    *,
    precision: str | None,
    declared_gradient_width_bytes: int | None = None,
) -> Estimate:
    """gradients = trainable_parameters x gradient width.

    Gradient width = the weights storage width unless explicitly declared
    otherwise.  Trainable count falls back to the full parameter count
    only for ``FULL_FINETUNE``; otherwise missing trainable count is
    UNKNOWN (fail-closed).
    """
    width: int | None
    if declared_gradient_width_bytes is not None:
        width = declared_gradient_width_bytes
        width_source = EstimateSource.DECLARED
        width_note_text = f"declared gradient width {width} B/param"
    else:
        width, width_source, _conf, width_note_text = resolve_precision(precision)
    trainable = parameter_metadata.trainable_parameter_count
    params = parameter_metadata.parameter_count
    if width is None:
        return Estimate.unknown(
            "B", note="gradients UNKNOWN: gradient width unresolved"
        )
    if trainable is None:
        if params is not None and params > 0:
            return Estimate.unknown(
                "B",
                note=(
                    "gradients UNKNOWN: trainable_parameter_count not "
                    "declared (not assumed = full parameter count outside "
                    "FULL_FINETUNE)"
                ),
            )
        return Estimate.unknown(
            "B", note="gradients UNKNOWN: no trainable/total parameter count"
        )
    if trainable <= 0:
        return Estimate(
            value=0,
            unit="B",
            source=EstimateSource.DERIVED,
            confidence=EstimateConfidence.HIGH,
            note="trainable_parameter_count <= 0: no gradient memory",
        )
    return Estimate(
        value=int(trainable) * width,
        unit="B",
        source=EstimateSource.DERIVED,
        confidence=EstimateConfidence.HIGH,
        note=(
            f"{trainable} trainable parameters x {width} B/param; " f"{width_note_text}"
        ),
    )


def estimate_optimizer_states(
    parameter_metadata: ParameterMetadata,
    *,
    optimizer_state_multiplier: float = ADAMW_BYTES_PER_PARAM_DEFAULT
    / FP32_MOMENT_BYTES,
    declared_bytes_per_parameter: float | None = None,
) -> Estimate:
    """AdamW-style optimizer state = trainable x bytes/param multiplier.

    Default 2.0 fp32 moments = 8 bytes/parameter.  This is a documented
    HEURISTIC (configurable), never a measurement.  Master weights are
    part of what the multiplier covers when the caller widens it.
    """
    trainable = parameter_metadata.trainable_parameter_count
    params = parameter_metadata.parameter_count
    if declared_bytes_per_parameter is not None:
        per_param = float(declared_bytes_per_parameter)
        prov_source = EstimateSource.DECLARED
        prov_conf = EstimateConfidence.MEDIUM
        prov_note = f"declared {per_param} B/param optimizer state"
    else:
        per_param = optimizer_state_multiplier * FP32_MOMENT_BYTES
        prov_source = EstimateSource.HEURISTIC
        prov_conf = EstimateConfidence.LOW
        prov_note = (
            f"AdamW heuristic: {optimizer_state_multiplier:g} fp32 moment(s) "
            f"x {FP32_MOMENT_BYTES} B = {per_param:g} B/param (documented "
            "heuristic, configurable)"
        )
    if trainable is None:
        if params is not None and params > 0:
            return Estimate.unknown(
                "B",
                note=(
                    "optimizer states UNKNOWN: trainable_parameter_count "
                    "not declared (heuristic scales with trainable params)"
                ),
            )
        return Estimate.unknown(
            "B", note="optimizer states UNKNOWN: no parameter counts"
        )
    if trainable <= 0:
        return Estimate(
            value=0,
            unit="B",
            source=EstimateSource.DERIVED,
            confidence=EstimateConfidence.HIGH,
            note="trainable_parameter_count <= 0: no optimizer state",
        )
    return Estimate(
        value=int(trainable * per_param),
        unit="B",
        source=prov_source,
        confidence=prov_conf,
        note=f"{trainable} trainable parameters; {prov_note}",
    )


# --------------------------------------------------------------------------
# Activations / declared components
# --------------------------------------------------------------------------


def _activation_estimate(
    measured_activations_bytes: int | None,
    heuristic_activations_bytes: int | None,
) -> Estimate:
    """Activations: MEASURED > HEURISTIC > UNKNOWN.  Never fabricated."""
    if measured_activations_bytes is not None:
        return Estimate(
            value=int(measured_activations_bytes),
            unit="B",
            source=EstimateSource.MEASURED,
            confidence=EstimateConfidence.HIGH,
            note="operator-supplied activation profile from a real run",
        )
    if heuristic_activations_bytes is not None:
        return Estimate(
            value=int(heuristic_activations_bytes),
            unit="B",
            source=EstimateSource.HEURISTIC,
            confidence=EstimateConfidence.LOW,
            note=(
                "adapter-verified activation heuristic supplied by the "
                "caller; NOT a measurement"
            ),
        )
    return Estimate.unknown("B", note=ACTIVATIONS_UNMEASURED_NOTE)


def _optional_declared(
    declared_bytes: int | None,
    component: str,
) -> Estimate:
    """UNKNOWN unless the caller explicitly declares the component."""
    if declared_bytes is None:
        return Estimate.unknown(
            "B",
            note=f"{component} UNKNOWN unless declared; nothing fabricated",
        )
    return Estimate(
        value=int(declared_bytes),
        unit="B",
        source=EstimateSource.DECLARED,
        confidence=EstimateConfidence.MEDIUM,
        note="declared by caller (no independent verification)",
    )


# --------------------------------------------------------------------------
# Image/sequence uncertainty factors
# --------------------------------------------------------------------------


class SequenceFactors:
    """Image/sequence shape factors as DOCUMENTED UNCERTAINTY.

    These are planning inputs that widen the uncertainty band around the
    activation estimate (max_image_pixels, image_count_per_sample,
    max_tokens, pack_length).  They are NOT plugged into any formula —
    this framework does not pretend to know how vision tokens scale
    activation memory.  ``uncertainty_note`` is attached to the fit
    verdict so the operator sees exactly which knobs were not accounted
    for numerically.
    """

    __slots__ = (
        "max_image_pixels",
        "image_count_per_sample",
        "max_tokens",
        "pack_length",
    )

    def __init__(
        self,
        *,
        max_image_pixels: int | None = None,
        image_count_per_sample: int | None = None,
        max_tokens: int | None = None,
        pack_length: int | None = None,
    ) -> None:
        self.max_image_pixels = max_image_pixels
        self.image_count_per_sample = image_count_per_sample
        self.max_tokens = max_tokens
        self.pack_length = pack_length

    def uncertainty_note(self) -> str | None:
        parts: list[str] = []
        if self.max_image_pixels is not None:
            parts.append(f"max_image_pixels={self.max_image_pixels}")
        if self.image_count_per_sample is not None:
            parts.append(f"image_count_per_sample={self.image_count_per_sample}")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        if self.pack_length is not None:
            parts.append(f"pack_length={self.pack_length}")
        if not parts:
            return None
        return (
            "sequence/image factors ("
            + ", ".join(parts)
            + ") are documented UNCERTAINTY, not math: activation memory "
            "may exceed any estimate by an unquantified amount depending "
            "on the vision/token path"
        )


# --------------------------------------------------------------------------
# Top-level assembly + fit classification
# --------------------------------------------------------------------------


def estimate_component_breakdown(
    parameter_metadata: ParameterMetadata,
    *,
    precision: str | None,
    training_mode: TrainingMode = TrainingMode.FULL_FINETUNE,
    measured_activations_bytes: int | None = None,
    heuristic_activations_bytes: int | None = None,
    declared_temporary_buffers_bytes: int | None = None,
    declared_multimodal_tensors_bytes: int | None = None,
    declared_framework_overhead_bytes: int | None = None,
    optimizer_state_multiplier: float = ADAMW_BYTES_PER_PARAM_DEFAULT
    / FP32_MOMENT_BYTES,
    declared_gradient_width_bytes: int | None = None,
) -> "MemoryComponentEstimates":
    """Build the full per-component breakdown (imported lazily).

    The :class:`MemoryComponentEstimates` model is constructed here rather
    than in a signature to keep the module's public surface free of the
    dataclass import at call time; the import is from the sibling models
    module (stdlib-only chain, no torch).
    """
    from clouda_training.planner.models import MemoryComponentEstimates

    weights = estimate_weights(parameter_metadata, precision=precision)

    if training_mode is TrainingMode.FULL_FINETUNE:
        # Gradients/optimizer scale with ALL parameters: use the total
        # count explicitly (trainable == total by definition of the mode).
        grad_meta = parameter_metadata
        if grad_meta.trainable_parameter_count is None and (
            grad_meta.parameter_count is not None
        ):
            from dataclasses import replace

            grad_meta = replace(
                parameter_metadata,
                trainable_parameter_count=parameter_metadata.parameter_count,
            )
        optimizer_meta = grad_meta
    else:
        grad_meta = parameter_metadata
        optimizer_meta = parameter_metadata

    gradients = estimate_gradients(
        grad_meta,
        precision=precision,
        declared_gradient_width_bytes=declared_gradient_width_bytes,
    )
    optimizer = estimate_optimizer_states(
        optimizer_meta,
        optimizer_state_multiplier=optimizer_state_multiplier,
    )
    activations = _activation_estimate(
        measured_activations_bytes, heuristic_activations_bytes
    )
    temporary = _optional_declared(
        declared_temporary_buffers_bytes, "temporary buffers"
    )
    multimodal = _optional_declared(
        declared_multimodal_tensors_bytes, "multimodal tensors"
    )
    overhead = _optional_declared(
        declared_framework_overhead_bytes, "framework overhead"
    )
    return MemoryComponentEstimates(
        weights_bytes=weights,
        gradients_bytes=gradients,
        optimizer_states_bytes=optimizer,
        activations_bytes=activations,
        temporary_buffers_bytes=temporary,
        multimodal_tensors_bytes=multimodal,
        framework_overhead_bytes=overhead,
    )


def _has_measured_activations(components: "MemoryComponentEstimates") -> bool:
    return (
        components.activations_bytes.source is EstimateSource.MEASURED
        and components.activations_bytes.value is not None
    )


def _all_components_known(components: "MemoryComponentEstimates") -> bool:
    return all(est.value is not None for est in components.iter_components().values())


def classify_fit(
    components: "MemoryComponentEstimates",
    hardware: HardwareEnvelope,
    *,
    sequence_factors: SequenceFactors | None = None,
) -> tuple[HardwareFit, str]:
    """Fail-closed hardware-fit classification.

    * lower bound > per-GPU target                    -> ``LIKELY_TOO_LARGE``
    * lower bound fits, activations not MEASURED
      or any component unknown                        -> ``MAY_FIT`` +
      ``REQUIRES REAL SMOKE TEST`` (never "ready")
    * everything known AND activations MEASURED       -> ``LIKELY_FITS``
    * no VRAM target declared                         -> ``UNKNOWN``
    """
    target_gb = hardware.per_gpu_vram_gb
    if target_gb is None or target_gb <= 0:
        return (
            HardwareFit.UNKNOWN,
            "no per_gpu_vram_gb declared on the hardware envelope; fit "
            "cannot be assessed",
        )
    target_bytes = target_gb * BYTES_PER_GIB
    lower = components.known_lower_bound_bytes()

    if lower > target_bytes:
        over = lower - target_bytes
        return (
            HardwareFit.LIKELY_TOO_LARGE,
            (
                f"defensible lower bound {bytes_to_gib(lower):.2f} GiB "
                f"exceeds the {target_gb:g} GiB per-GPU target by "
                f"{bytes_to_gib(over):.2f} GiB — and this EXCLUDES "
                "unmeasured components (activations/overhead), so the "
                "real footprint is strictly larger"
            ),
        )

    measured = _has_measured_activations(components)
    complete = _all_components_known(components)
    seq_note = sequence_factors.uncertainty_note() if sequence_factors else None

    if measured and complete:
        note = (
            f"lower bound {bytes_to_gib(lower):.2f} GiB fits the "
            f"{target_gb:g} GiB target and activations are MEASURED for a "
            "matching configuration"
        )
        if seq_note:
            note += f"; {seq_note}"
        return HardwareFit.LIKELY_FITS, note

    note = (
        f"lower bound {bytes_to_gib(lower):.2f} GiB fits the "
        f"{target_gb:g} GiB target, but "
        + (
            "activations are not MEASURED for this configuration"
            if not measured
            else "some components remain unknown"
        )
        + f"; {SMOKE_TEST_REQUIRED_NOTE}"
    )
    if seq_note:
        note += f"; {seq_note}"
    return HardwareFit.MAY_FIT, note


def estimate_vram(
    parameter_metadata: ParameterMetadata,
    *,
    hardware: HardwareEnvelope,
    precision: str | None,
    training_mode: TrainingMode = TrainingMode.FULL_FINETUNE,
    measured_activations_bytes: int | None = None,
    heuristic_activations_bytes: int | None = None,
    declared_temporary_buffers_bytes: int | None = None,
    declared_multimodal_tensors_bytes: int | None = None,
    declared_framework_overhead_bytes: int | None = None,
    optimizer_state_multiplier: float = ADAMW_BYTES_PER_PARAM_DEFAULT
    / FP32_MOMENT_BYTES,
    declared_gradient_width_bytes: int | None = None,
    sequence_factors: SequenceFactors | None = None,
) -> dict[str, Any]:
    """Full transparent VRAM estimate + fit verdict + planning warnings.

    Returns a dict with ``components`` (MemoryComponentEstimates),
    ``lower_bound_bytes``, ``lower_bound_gib``, ``fit`` (HardwareFit),
    ``fit_note``, and ``warnings`` (tuple[PlanningWarning]) — consumable
    by ``planner.build_experiment_plan``.

    Multi-GPU: ``gpu_count > 1`` emits the PLANNING ONLY warning; the
    classification is ALWAYS against the PER-GPU target (VRAM does not
    pool — the runtime has no distributed support; verified by grep of
    ``clouda_training/runtime/`` for distributed/DDP/FSDP: zero matches).
    """
    from clouda_training.planner.models import ResourceEstimate

    components = estimate_component_breakdown(
        parameter_metadata,
        precision=precision,
        training_mode=training_mode,
        measured_activations_bytes=measured_activations_bytes,
        heuristic_activations_bytes=heuristic_activations_bytes,
        declared_temporary_buffers_bytes=declared_temporary_buffers_bytes,
        declared_multimodal_tensors_bytes=declared_multimodal_tensors_bytes,
        declared_framework_overhead_bytes=declared_framework_overhead_bytes,
        optimizer_state_multiplier=optimizer_state_multiplier,
        declared_gradient_width_bytes=declared_gradient_width_bytes,
    )
    lower = components.known_lower_bound_bytes()
    fit, fit_note = classify_fit(
        components, hardware, sequence_factors=sequence_factors
    )

    warnings: list[PlanningWarning] = []
    if hardware.gpu_count > 1:
        warnings.append(
            PlanningWarning(
                code="memory.multi_gpu_planning_only",
                message=_MULTI_GPU_WARNING_TEXT,
            )
        )
    if components.activations_bytes.value is None:
        warnings.append(
            PlanningWarning(
                code="memory.activations_unknown",
                message=ACTIVATIONS_UNMEASURED_NOTE,
            )
        )
    if fit is HardwareFit.MAY_FIT:
        warnings.append(
            PlanningWarning(
                code="memory.fit_requires_smoke_test",
                message=SMOKE_TEST_REQUIRED_NOTE,
            )
        )
    if sequence_factors is not None:
        seq_note = sequence_factors.uncertainty_note()
        if seq_note:
            warnings.append(
                PlanningWarning(
                    code="memory.sequence_factors_uncertainty",
                    message=seq_note,
                )
            )

    resources = ResourceEstimate(
        memory=components,
        lower_bound_bytes=lower,
        lower_bound_source=(
            EstimateSource.DERIVED if lower > 0 else EstimateSource.UNKNOWN
        ),
        lower_bound_confidence=components.worst_confidence(),
        fit=fit,
        fit_note=fit_note,
    )
    return {
        "components": components,
        "lower_bound_bytes": lower,
        "lower_bound_gib": bytes_to_gib(lower),
        "fit": fit,
        "fit_note": fit_note,
        "warnings": tuple(warnings),
        "resources": resources,
    }
