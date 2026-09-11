"""Idempotent registration of the Qwen3-VL adapter + data adapter.

WHY a separate module (not package ``__init__``): the qwen package must stay
importable with zero side effects on the shared default registries; tests and
the lead's CLI wiring call :func:`register_qwen_adapters` explicitly. Both
registrations are guarded by ``is_registered`` so repeated calls (re-imports,
multiple test modules) never raise DuplicateAdapterError. No torch/transformers
import happens here — the factories are lazy.
"""

from __future__ import annotations

from typing import Any

from clouda_training.adapters.data_adapter import (
    DataAdapterRegistry,
    get_default_data_adapter_registry,
)
from clouda_training.adapters.registry import (
    ModelAdapterRegistry,
    get_default_registry,
)

from clouda_training.qwen.data_adapter import QwenTrainingDataAdapter
from clouda_training.qwen.descriptor import QWEN_DESCRIPTOR

QWEN_MODEL_ADAPTER_TYPE = "qwen_vl_sft"
QWEN_DATA_ADAPTER_TYPE = "qwen_vl_sft_data"

_registered: bool = False


def _make_qwen_adapter(**kwargs: Any) -> Any:
    """Lazy model-adapter factory: heavy imports stay inside QwenVLSFTAdapter."""
    from clouda_training.qwen.adapter import QwenVLSFTAdapter

    return QwenVLSFTAdapter(**kwargs)


def _make_qwen_data_adapter(**kwargs: Any) -> Any:
    return QwenTrainingDataAdapter(**kwargs)


def register_qwen_adapters(
    *,
    model_registry: ModelAdapterRegistry | None = None,
    data_registry: DataAdapterRegistry | None = None,
) -> None:
    """Register the qwen_vl_sft adapter(s) idempotently (default registries)."""
    global _registered
    model_reg = model_registry if model_registry is not None else get_default_registry()
    data_reg = (
        data_registry
        if data_registry is not None
        else get_default_data_adapter_registry()
    )
    if not model_reg.is_registered(QWEN_MODEL_ADAPTER_TYPE):
        model_reg.register(QWEN_DESCRIPTOR, _make_qwen_adapter)
    if not data_reg.is_registered(QWEN_DATA_ADAPTER_TYPE):
        data_reg.register(QWEN_DATA_ADAPTER_TYPE, _make_qwen_data_adapter)
    _registered = True


def qwen_adapters_registered() -> bool:
    """Whether registration has been performed in this process (test hook)."""
    return _registered


__all__ = [
    "QWEN_DATA_ADAPTER_TYPE",
    "QWEN_MODEL_ADAPTER_TYPE",
    "qwen_adapters_registered",
    "register_qwen_adapters",
]
