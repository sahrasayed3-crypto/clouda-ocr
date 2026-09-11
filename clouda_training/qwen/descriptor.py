"""Descriptor for the qwen_vl_sft adapter (honest capability claims).

Every capability flag is a VERIFIED-SUPPORT claim — see
clouda_training.adapters.capabilities. Unverified flags stay False so the
adapter fails closed instead of advertising untested behaviour.

Upstream: QwenLM/Qwen3-VL@96588727e44c78b25ba03ea03b8e12f7e64fd0da.
"""

from __future__ import annotations

from clouda_training.adapters.capabilities import ModelCapabilities
from clouda_training.adapters.descriptor import ModelAdapterDescriptor
from clouda_training.qwen.models import (
    QWEN_COMPAT_REPOSITORY,
    QWEN_COMPAT_REVISION,
    QWEN_DATA_CONTRACT_VERSION,
    QWEN_MIN_TRANSFORMERS_VERSION,
)

ADAPTER_TYPE = "qwen_vl_sft"
ADAPTER_VERSION = "1.0.0"
MODEL_FAMILY = "qwen3_vl"
TASK_FAMILY = "image_text_ocr_sft"

#: Deterministic checkpoint-compatibility identity: binds checkpoints to this
#: adapter code identity + verified upstream revision. Recomputed only when
#: the verified facts change (never per-process, never machine-dependent).
CHECKPOINT_COMPATIBILITY_ID = "qwen3vl-sft@" + QWEN_COMPAT_REVISION

QWEN_CAPABILITIES = ModelCapabilities(
    supports_full_finetune=True,
    supports_selective_finetune=True,
    supports_lora=False,  # official peft path exists upstream; not validated here
    supports_gradient_checkpointing=True,
    supports_packed_sequences=True,  # official --data_flatten/--data_packing contract
    supports_multimodal_batches=True,
    supports_local_only_loading=True,
    supports_bf16=True,
    supports_fp16=False,
    supports_cpu_smoke=True,  # via synthetic mock model/processor only
    supports_resume=True,
    real_weights_validated=False,  # no real Qwen3-VL weights exercised yet
    gpu_validated=False,
)

#: transformers >= 4.57.0 provides Qwen3VLForConditionalGeneration natively —
#: no trust_remote_code anywhere in the load path.
QWEN_REQUIRED_DEPENDENCIES = (
    "torch",
    f"transformers>={QWEN_MIN_TRANSFORMERS_VERSION}",
)

QWEN_DESCRIPTOR = ModelAdapterDescriptor(
    adapter_type=ADAPTER_TYPE,
    adapter_version=ADAPTER_VERSION,
    model_family=MODEL_FAMILY,
    task_family=TASK_FAMILY,
    capabilities=QWEN_CAPABILITIES,
    required_optional_dependencies=QWEN_REQUIRED_DEPENDENCIES,
    supported_precision=("fp32", "bf16"),
    supported_devices=("cuda", "cpu"),
    supported_data_modes=("raw", "packed"),
    checkpoint_compatibility_id=CHECKPOINT_COMPATIBILITY_ID,
    upstream_repository=QWEN_COMPAT_REPOSITORY,
    upstream_revision=QWEN_COMPAT_REVISION,
    expected_model_symbols=(
        "Qwen3VLForConditionalGeneration",
        "AutoProcessor",
        "AutoTokenizer",
    ),
    expected_processor_symbols=("Qwen3VLProcessor",),
    data_contract_version=QWEN_DATA_CONTRACT_VERSION,
)

__all__ = [
    "ADAPTER_TYPE",
    "ADAPTER_VERSION",
    "CHECKPOINT_COMPATIBILITY_ID",
    "MODEL_FAMILY",
    "QWEN_CAPABILITIES",
    "QWEN_DESCRIPTOR",
    "TASK_FAMILY",
]
