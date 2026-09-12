"""Descriptor for the hunyuanocr15_sft adapter (honest capability claims).

Every capability flag is a VERIFIED-SUPPORT claim — see
clouda_training.adapters.capabilities. Unverified flags stay False so the
adapter fails closed instead of advertising untested behaviour.

Upstream: Tencent-Hunyuan/HunyuanOCR@c55965d3da1e (trust_remote_code path).
"""

from __future__ import annotations

from clouda_training.adapters.capabilities import ModelCapabilities
from clouda_training.adapters.descriptor import ModelAdapterDescriptor
from clouda_training.hunyuan.models import (
    HUNYUAN_COMPAT_REPOSITORY,
    HUNYUAN_COMPAT_REVISION,
)

ADAPTER_TYPE = "hunyuanocr15_sft"
ADAPTER_VERSION = "1.0.0"
MODEL_FAMILY = "hunyuanocr15"
TASK_FAMILY = "image_text_ocr_sft"

#: Deterministic checkpoint-compatibility identity: binds checkpoints to this
#: adapter code identity + verified upstream revision. Recomputed only when
#: the verified facts change (never per-process, never machine-dependent).
CHECKPOINT_COMPATIBILITY_ID = "hunyuanocr15-sft@" + HUNYUAN_COMPAT_REVISION

HUNYUAN_CAPABILITIES = ModelCapabilities(
    supports_full_finetune=True,
    supports_selective_finetune=True,
    supports_lora=False,  # upstream LoRA path exists but not validated here
    supports_gradient_checkpointing=True,
    supports_packed_sequences=True,  # official pack_data pipeline (cu_seqlens)
    supports_multimodal_batches=True,
    supports_local_only_loading=True,
    supports_bf16=True,
    supports_fp16=False,
    supports_cpu_smoke=True,  # via synthetic mock model/processor only
    supports_resume=True,
    real_weights_validated=False,  # no real HunyuanOCR weights exercised yet
    gpu_validated=False,
)

#: HunyuanOCR-1.5 requires trust_remote_code (custom model class) — honest
#: dependency declaration, heavier than the qwen native path.
HUNYUAN_REQUIRED_DEPENDENCIES = (
    "torch",
    "transformers",
    "trust_remote_code",
)

HUNYUAN_DESCRIPTOR = ModelAdapterDescriptor(
    adapter_type=ADAPTER_TYPE,
    adapter_version=ADAPTER_VERSION,
    model_family=MODEL_FAMILY,
    task_family=TASK_FAMILY,
    capabilities=HUNYUAN_CAPABILITIES,
    required_optional_dependencies=HUNYUAN_REQUIRED_DEPENDENCIES,
    supported_precision=("fp32", "bf16"),
    supported_devices=("cuda", "cpu"),
    supported_data_modes=("raw", "packed"),
    checkpoint_compatibility_id=CHECKPOINT_COMPATIBILITY_ID,
    upstream_repository=HUNYUAN_COMPAT_REPOSITORY,
    upstream_revision=HUNYUAN_COMPAT_REVISION,
    expected_model_symbols=(
        "HunYuanVLForConditionalGeneration",
        "AutoTokenizer",
        "AutoProcessor",
    ),
    expected_processor_symbols=("AutoProcessor",),
    data_contract_version="1",
)

__all__ = [
    "ADAPTER_TYPE",
    "ADAPTER_VERSION",
    "CHECKPOINT_COMPATIBILITY_ID",
    "HUNYUAN_CAPABILITIES",
    "HUNYUAN_DESCRIPTOR",
    "MODEL_FAMILY",
    "TASK_FAMILY",
]
