"""Qwen3-VL SFT integration bridge.

Imports cleanly without torch/transformers; heavy symbols resolve lazily.
Upstream compatibility: QwenLM/Qwen3-VL@96588727e44c78b25ba03ea03b8e12f7e64fd0da
(native in transformers >= 4.57.0 — no trust_remote_code).

Registration into the shared default registries is EXPLICIT and idempotent:
call :func:`clouda_training.qwen.registration.register_qwen_adapters` (the
multimodel test conftest does this; the lead wires it into the CLI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clouda_training.qwen.data_adapter import (
    QwenDataAdapterError,
    QwenExportConfig,
    QwenLineageReport,
    QwenTrainingDataAdapter,
)
from clouda_training.qwen.descriptor import (
    ADAPTER_TYPE,
    CHECKPOINT_COMPATIBILITY_ID,
    QWEN_CAPABILITIES,
    QWEN_DESCRIPTOR,
)
from clouda_training.qwen.models import (
    QWEN_COMPAT_REPOSITORY,
    QWEN_COMPAT_REVISION,
    QWEN_DATA_CONTRACT_VERSION,
    QWEN_IGNORE_INDEX,
    QWEN_IMAGE_PLACEHOLDER,
    QWEN_MIN_TRANSFORMERS_VERSION,
    QWEN_SFT_REFERENCE_PROFILE,
)
from clouda_training.qwen.registration import register_qwen_adapters

if TYPE_CHECKING:  # pragma: no cover
    from clouda_training.qwen.adapter import QwenVLSFTAdapter

__all__ = [
    "ADAPTER_TYPE",
    "CHECKPOINT_COMPATIBILITY_ID",
    "QWEN_CAPABILITIES",
    "QWEN_COMPAT_REPOSITORY",
    "QWEN_COMPAT_REVISION",
    "QWEN_DATA_CONTRACT_VERSION",
    "QWEN_DESCRIPTOR",
    "QWEN_IGNORE_INDEX",
    "QWEN_IMAGE_PLACEHOLDER",
    "QWEN_MIN_TRANSFORMERS_VERSION",
    "QWEN_SFT_REFERENCE_PROFILE",
    "QwenDataAdapterError",
    "QwenExportConfig",
    "QwenLineageReport",
    "QwenTrainingDataAdapter",
    "QwenVLSFTAdapter",
    "register_qwen_adapters",
]


def __getattr__(name: str) -> Any:
    if name == "QwenVLSFTAdapter":
        from clouda_training.qwen.adapter import QwenVLSFTAdapter

        return QwenVLSFTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
