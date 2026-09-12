"""Verified upstream facts for the Qwen3-VL SFT bridge.

Pinned revision: QwenLM/Qwen3-VL@96588727e44c78b25ba03ea03b8e12f7e64fd0da
(2026-01-30, ``qwen-vl-finetune/``). Facts verified from official sources —
see UPSTREAM_COMPATIBILITY.md. Single source of truth for the adapter,
descriptor, and data adapter.
"""

from __future__ import annotations

from typing import Any

QWEN_COMPAT_REVISION = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
QWEN_COMPAT_REPOSITORY = "QwenLM/Qwen3-VL"

#: Model classes are NATIVE in transformers >= 4.57.0 — no trust_remote_code.
QWEN_MIN_TRANSFORMERS_VERSION = "4.57.0"
QWEN_MODEL_CLASS = "Qwen3VLForConditionalGeneration"
QWEN_PROCESSOR_CLASS = "Qwen3VLProcessor"

#: Upstream official data format (JSON/JSONL, media FILE PATHS; the text-level
#: ``<image>`` tag appears only in the human turn — the processor inserts the
#: model-level <|vision_start|>/<|image_pad|>/<|vision_end|> tokens).
QWEN_IMAGE_PLACEHOLDER = "<image>"
QWEN_DATA_CONTRACT_VERSION = "1"

#: Upstream selective-tuning knobs (sft_qwen3_4b.sh profile):
#: tune_mm_vision -> model.visual, tune_mm_mlp -> model.visual.merger,
#: tune_mm_llm -> model.language_model (+ tied lm_head).
QWEN_TUNABLE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "vision": ("visual",),
    "projector": ("visual.merger",),
    "llm": ("language_model", "lm_head"),
}

#: Loss masking: labels = IGNORE_INDEX except assistant (gpt) spans.
QWEN_IGNORE_INDEX = -100

QWEN_SFT_REFERENCE_PROFILE: dict[str, Any] = {
    "lr_range": [2e-7, 1e-6],
    "precision": "bf16",
    "gradient_checkpointing": True,
    "tune_mm_vision": False,
    "tune_mm_mlp": True,
    "tune_mm_llm": True,
    "model_max_length": 8192,
}

__all__ = [
    "QWEN_COMPAT_REPOSITORY",
    "QWEN_COMPAT_REVISION",
    "QWEN_DATA_CONTRACT_VERSION",
    "QWEN_IGNORE_INDEX",
    "QWEN_IMAGE_PLACEHOLDER",
    "QWEN_MIN_TRANSFORMERS_VERSION",
    "QWEN_MODEL_CLASS",
    "QWEN_PROCESSOR_CLASS",
    "QWEN_SFT_REFERENCE_PROFILE",
    "QWEN_TUNABLE_COMPONENTS",
]
