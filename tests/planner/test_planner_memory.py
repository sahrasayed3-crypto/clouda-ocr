"""Tests for planner.memory: transparent VRAM estimation.

Every test pins a fail-closed rule from the Wave-1 brief:

* weights/gradient/optimizer math is exact integer arithmetic over the
  documented width tables (fp32=4, bf16/fp16=2 bytes/parameter);
* the AdamW default is the documented 8 B/param heuristic;
* activations are MEASURED / HEURISTIC / UNKNOWN only — never fabricated;
* fit classification is fail-closed (LIKELY_FITS requires measured
  evidence; MAY_FIT always says REQUIRES REAL SMOKE TEST);
* multi-GPU is planning-only (no VRAM pooling, warning emitted);
* no torch import happens at module import time.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from clouda_training.planner.memory import (
    ACTIVATIONS_UNMEASURED_NOTE,
    ADAMW_BYTES_PER_PARAM_DEFAULT,
    FP32_MOMENT_BYTES,
    GRADIENT_WIDTH_TABLE,
    PRECISION_WIDTH_TABLE,
    SMOKE_TEST_REQUIRED_NOTE,
    SequenceFactors,
    bytes_to_gib,
    estimate_component_breakdown,
    estimate_gradients,
    estimate_optimizer_states,
    estimate_vram,
    estimate_weights,
    resolve_precision,
)
from clouda_training.planner.models import (
    BYTES_PER_GIB,
    EstimateSource,
    HardwareEnvelope,
    HardwareFit,
    ParameterMetadata,
    TrainingMode,
)

GB = 1_000_000_000  # decimal GB (vendor VRAM units)


def _params(n: int, trainable: int | None = None) -> ParameterMetadata:
    return ParameterMetadata(parameter_count=n, trainable_parameter_count=trainable)


def _gpu(vram_gb: float | None = 24, count: int = 1) -> HardwareEnvelope:
    return HardwareEnvelope(gpu_count=count, per_gpu_vram_gb=vram_gb)


# ------------------------------------------------------------- precision widths


def test_precision_width_table_documented_values():
    assert PRECISION_WIDTH_TABLE["fp32"] == 4
    assert PRECISION_WIDTH_TABLE["float32"] == 4
    assert PRECISION_WIDTH_TABLE["bf16"] == 2
    assert PRECISION_WIDTH_TABLE["bfloat16"] == 2
    assert PRECISION_WIDTH_TABLE["fp16"] == 2
    assert PRECISION_WIDTH_TABLE["float16"] == 2


def test_gradient_width_defaults_to_storage_width():
    assert GRADIENT_WIDTH_TABLE == PRECISION_WIDTH_TABLE


def test_resolve_precision_known_and_unknown():
    width, source, conf, _ = resolve_precision("BF16")
    assert width == 2
    assert source is EstimateSource.DECLARED
    assert conf.value == "HIGH"
    width, source, conf, note = resolve_precision("int4")
    assert width is None
    assert source is EstimateSource.UNKNOWN
    assert conf.value == "UNKNOWN"
    assert "not in documented width table" in note
    width, source, _, note = resolve_precision(None)
    assert width is None and source is EstimateSource.UNKNOWN


# ------------------------------------------------------------------ weights


def test_weights_lower_bound_math_4b_fp32_is_16gb():
    """4B parameters at fp32 (4 B/param) = exactly 16 GB of weights."""
    est = estimate_weights(_params(4_000_000_000), precision="fp32")
    assert est.value == 16_000_000_000
    assert est.source is EstimateSource.DERIVED
    assert est.confidence.value == "HIGH"


def test_weights_bf16_is_half_of_fp32():
    bf16 = estimate_weights(_params(4_000_000_000), precision="bf16")
    fp32 = estimate_weights(_params(4_000_000_000), precision="fp32")
    assert bf16.value == 8_000_000_000
    assert fp32.value == 2 * bf16.value


def test_weights_unknown_precision_is_fail_closed():
    est = estimate_weights(_params(4_000_000_000), precision="int4")
    assert est.value is None
    assert est.source is EstimateSource.UNKNOWN


def test_weights_missing_param_count_is_unknown():
    est = estimate_weights(ParameterMetadata(), precision="fp32")
    assert est.value is None
    assert est.source is EstimateSource.UNKNOWN


# ---------------------------------------------------------------- gradients


def test_gradients_scale_with_trainable_params_not_total():
    est = estimate_gradients(
        _params(4_000_000_000, trainable=100_000_000), precision="bf16"
    )
    assert est.value == 100_000_000 * 2


def test_gradients_full_finetune_uses_total_param_count():
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        training_mode=TrainingMode.FULL_FINETUNE,
    )
    assert comp.gradients_bytes.value == 1_000_000_000 * 2


def test_gradients_selective_without_trainable_count_is_unknown():
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        training_mode=TrainingMode.SELECTIVE_FINETUNE,
    )
    assert comp.gradients_bytes.value is None
    assert comp.gradients_bytes.source is EstimateSource.UNKNOWN


# ----------------------------------------------------------- optimizer states


def test_adamw_default_is_8_bytes_per_parameter():
    est = estimate_optimizer_states(_params(2, trainable=1_000_000_000))
    assert est.value == 8_000_000_000
    assert est.source is EstimateSource.HEURISTIC
    assert ADAMW_BYTES_PER_PARAM_DEFAULT / FP32_MOMENT_BYTES == pytest.approx(2.0)


def test_adamw_multiplier_is_configurable():
    est = estimate_optimizer_states(
        _params(2, trainable=1_000_000_000), optimizer_state_multiplier=3.0
    )
    assert est.value == 12_000_000_000
    assert est.source is EstimateSource.HEURISTIC


def test_optimizer_states_respect_declared_override():
    est = estimate_optimizer_states(
        _params(2, trainable=1_000_000_000),
        declared_bytes_per_parameter=12.0,
    )
    assert est.value == 12_000_000_000
    assert est.source is EstimateSource.DECLARED


# ------------------------------------------------------- activations + unknowns


def test_activations_unknown_is_honored_never_fabricated():
    comp = estimate_component_breakdown(_params(1_000_000_000), precision="bf16")
    assert comp.activations_bytes.value is None
    assert comp.activations_bytes.source is EstimateSource.UNKNOWN
    assert comp.temporary_buffers_bytes.value is None
    assert comp.multimodal_tensors_bytes.value is None
    assert comp.framework_overhead_bytes.value is None
    assert ACTIVATIONS_UNMEASURED_NOTE in comp.activations_bytes.note


def test_activations_measured_and_heuristic_sources():
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        measured_activations_bytes=3_000_000_000,
    )
    assert comp.activations_bytes.value == 3_000_000_000
    assert comp.activations_bytes.source is EstimateSource.MEASURED
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        heuristic_activations_bytes=3_000_000_000,
    )
    assert comp.activations_bytes.value == 3_000_000_000
    assert comp.activations_bytes.source is EstimateSource.HEURISTIC


def test_measured_takes_precedence_over_heuristic():
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        measured_activations_bytes=3_000_000_000,
        heuristic_activations_bytes=9_000_000_000,
    )
    assert comp.activations_bytes.source is EstimateSource.MEASURED
    assert comp.activations_bytes.value == 3_000_000_000


def test_overhead_only_when_declared():
    comp = estimate_component_breakdown(
        _params(1_000_000_000),
        precision="bf16",
        declared_framework_overhead_bytes=500 * BYTES_PER_GIB,
        declared_temporary_buffers_bytes=1024,
    )
    assert comp.framework_overhead_bytes.value == 500 * BYTES_PER_GIB
    assert comp.framework_overhead_bytes.source is EstimateSource.DECLARED
    assert comp.temporary_buffers_bytes.value == 1024
    assert comp.multimodal_tensors_bytes.value is None


# ------------------------------------------------------------ fit classification


def test_likely_too_large_31gb_lower_bound_vs_24gb():
    """31 GB defensible lower bound vs a 24 GB target -> LIKELY_TOO_LARGE."""
    comp = estimate_component_breakdown(
        _params(2_000_000_000, trainable=2_000_000_000),
        precision="bf16",
        declared_framework_overhead_bytes=31 * GB - 2_000_000_000 * 12,
    )
    assert comp.known_lower_bound_bytes() == 31 * GB
    result = estimate_vram(
        _params(2_000_000_000, trainable=2_000_000_000),
        hardware=_gpu(24),
        precision="bf16",
        declared_framework_overhead_bytes=31 * GB - 2_000_000_000 * 12,
    )
    assert result["fit"] is HardwareFit.LIKELY_TOO_LARGE
    assert "lower bound" in result["fit_note"]
    assert result["lower_bound_bytes"] == 31 * GB


def test_may_fit_17gb_with_unknown_activations_vs_24gb():
    """17 GB known lower bound + UNKNOWN activations vs 24 GB -> MAY_FIT."""
    params = 1_000_000_000
    overhead = 17 * GB - params * 12  # bf16 full-finetune = 12 B/param
    result = estimate_vram(
        _params(params),
        hardware=_gpu(24),
        precision="bf16",
        declared_framework_overhead_bytes=overhead,
    )
    assert result["lower_bound_bytes"] == 17 * GB
    assert result["fit"] is HardwareFit.MAY_FIT
    assert SMOKE_TEST_REQUIRED_NOTE in result["fit_note"]
    codes = {w.code for w in result["warnings"]}
    assert "memory.fit_requires_smoke_test" in codes


def test_heuristic_activations_still_may_fit_not_likely_fits():
    params = 1_000_000_000
    result = estimate_vram(
        _params(params),
        hardware=_gpu(24),
        precision="bf16",
        heuristic_activations_bytes=2 * GB,
    )
    assert result["fit"] is HardwareFit.MAY_FIT


def test_likely_fits_requires_measured_activations_and_full_known_set():
    params = 1_000_000_000
    base = dict(
        declared_temporary_buffers_bytes=1,
        declared_multimodal_tensors_bytes=1,
        declared_framework_overhead_bytes=1,
    )
    result = estimate_vram(
        _params(params),
        hardware=_gpu(24),
        precision="bf16",
        heuristic_activations_bytes=2 * GB,
        **base,
    )
    assert result["fit"] is HardwareFit.MAY_FIT
    result = estimate_vram(
        _params(params),
        hardware=_gpu(24),
        precision="bf16",
        measured_activations_bytes=2 * GB,
        **base,
    )
    assert result["fit"] is HardwareFit.LIKELY_FITS
    assert "MEASURED" in result["fit_note"]


def test_no_false_likely_fits_when_components_unknown():
    """ANY unknown component blocks LIKELY_FITS even with measured activations."""
    result = estimate_vram(
        _params(1_000_000_000),
        hardware=_gpu(24),
        precision="bf16",
        measured_activations_bytes=2 * GB,  # but overhead/temp/multimodal UNKNOWN
    )
    assert result["fit"] is HardwareFit.MAY_FIT


def test_fit_unknown_without_vram_target():
    result = estimate_vram(
        _params(1_000_000_000), hardware=_gpu(None), precision="bf16"
    )
    assert result["fit"] is HardwareFit.UNKNOWN


# ---------------------------------------------------------------- multi-GPU


def test_multi_gpu_no_pooling_warning_and_per_gpu_classification():
    """gpu_count=2: classification stays PER-GPU; PLANNING ONLY warning fires."""
    # 2B params fp32 full finetune: 16 B/param = 32 GB lower bound vs a
    # 24 GB PER-GPU target -> too large per GPU even with 2 GPUs present
    # (proves VRAM is not pooled into 48 GB).
    params = 2_000_000_000
    result = estimate_vram(
        _params(params),
        hardware=_gpu(24, count=2),
        precision="fp32",
    )
    assert result["lower_bound_bytes"] == 32_000_000_000
    assert (
        result["fit"] is HardwareFit.LIKELY_TOO_LARGE
    ), "VRAM must NOT be pooled: 24 GB lower bound vs 24 GB per-GPU target"
    codes = {w.code for w in result["warnings"]}
    assert "memory.multi_gpu_planning_only" in codes
    warning = next(
        w for w in result["warnings"] if w.code == "memory.multi_gpu_planning_only"
    )
    assert "PLANNING ONLY — EXECUTION BACKEND NOT YET VALIDATED" in (warning.message)
    assert "does NOT pool" in warning.message


def test_multi_gpu_warning_absent_for_single_gpu():
    result = estimate_vram(
        _params(1_000_000_000), hardware=_gpu(24, count=1), precision="bf16"
    )
    assert all(w.code != "memory.multi_gpu_planning_only" for w in result["warnings"])


def test_runtime_has_no_distributed_support_documented_in_warning():
    """Cite the source: grep runtime/ for distributed/DDP/FSDP -> 0 matches.

    The multi-GPU warning text asserts the runtime lacks distributed
    support; this test verifies that assertion against the actual source
    tree (offline, pure file read).
    """
    import re
    from pathlib import Path

    runtime_dir = Path(__file__).resolve().parents[2] / "clouda_training" / "runtime"
    pattern = re.compile(r"distributed|DDP|FSDP|accelerate", re.IGNORECASE)
    matches = [
        str(p.relative_to(runtime_dir.parents[1]))
        for p in runtime_dir.glob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert matches == [], (
        f"runtime gained distributed support; update the PLANNING ONLY "
        f"warning in memory.py: {matches}"
    )


# -------------------------------------------------- sequence/image uncertainty


def test_sequence_factors_are_uncertainty_not_math():
    """Factors never change any numeric value — they only widen the caution."""
    factors = SequenceFactors(
        max_image_pixels=4_000_000,
        image_count_per_sample=4,
        max_tokens=8192,
        pack_length=16384,
    )
    plain = estimate_vram(_params(1_000_000_000), hardware=_gpu(24), precision="bf16")
    with_factors = estimate_vram(
        _params(1_000_000_000),
        hardware=_gpu(24),
        precision="bf16",
        sequence_factors=factors,
    )
    assert with_factors["lower_bound_bytes"] == plain["lower_bound_bytes"]
    assert with_factors["fit"] is HardwareFit.MAY_FIT
    codes = {w.code for w in with_factors["warnings"]}
    assert "memory.sequence_factors_uncertainty" in codes
    warning = next(
        w
        for w in with_factors["warnings"]
        if w.code == "memory.sequence_factors_uncertainty"
    )
    assert "UNCERTAINTY, not math" in warning.message
    assert "max_image_pixels=4000000" in warning.message


def test_sequence_factors_none_emits_no_warning():
    result = estimate_vram(
        _params(1_000_000_000),
        hardware=_gpu(24),
        precision="bf16",
        sequence_factors=SequenceFactors(),
    )
    assert all(
        w.code != "memory.sequence_factors_uncertainty" for w in result["warnings"]
    )


# ----------------------------------------------------------------- helpers


def test_bytes_to_gib_exact_and_none_safe():
    assert bytes_to_gib(BYTES_PER_GIB) == 1.0
    assert bytes_to_gib(None) is None
    assert bytes_to_gib(16 * BYTES_PER_GIB) == 16.0


def test_resource_estimate_block_shape():
    result = estimate_vram(
        _params(1_000_000_000),
        hardware=_gpu(24),
        precision="bf16",
        measured_activations_bytes=1,
        declared_temporary_buffers_bytes=1,
        declared_multimodal_tensors_bytes=1,
        declared_framework_overhead_bytes=1,
    )
    resources = result["resources"]
    snapshot = resources.to_dict()
    assert snapshot["fit"] == HardwareFit.LIKELY_FITS.value
    assert snapshot["lower_bound_bytes"] == result["lower_bound_bytes"]
    components = resources.memory.to_dict()
    for name in (
        "weights_bytes",
        "gradients_bytes",
        "optimizer_states_bytes",
        "activations_bytes",
        "temporary_buffers_bytes",
        "multimodal_tensors_bytes",
        "framework_overhead_bytes",
    ):
        assert name in components


# ------------------------------------------------------- no torch at import


def test_no_torch_imported_at_module_import():
    """Module import in a fresh interpreter must not pull torch."""
    code = (
        "import sys;"
        "import clouda_training.planner.memory as m;"
        "assert 'torch' not in sys.modules, 'torch leaked into memory.py';"
        "print('clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
