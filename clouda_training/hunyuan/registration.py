"""Idempotent registration of the HunyuanOCR-1.5 adapter + data adapter.

WHY a separate module (not package ``__init__``): the frozen hunyuan package
``__init__`` must not gain side effects; tests and the lead's CLI wiring call
:func:`register_hunyuan_adapters` explicitly. Both registrations are guarded
by ``is_registered`` so repeated calls never raise DuplicateAdapterError.
No torch/transformers import happens here — the factory is lazy.
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

from clouda_training.hunyuan.data_adapter import (
    HUNYUAN_DATA_ADAPTER_TYPE,
    HunyuanTrainingDataAdapter,
)
from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR

HUNYUAN_MODEL_ADAPTER_TYPE = "hunyuanocr15_sft"

_registered: bool = False


def _make_hunyuan_adapter(**kwargs: Any) -> Any:
    """Lazy model-adapter factory: heavy imports stay inside the adapter."""
    from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter

    config = kwargs.pop("config", None)
    if config is not None:
        kwargs.setdefault("local_model_path", config.model.model_id)
        kwargs.setdefault(
            "gradient_checkpointing", config.training.gradient_checkpointing
        )
        kwargs.setdefault("trust_remote_code", config.model.trust_remote_code)
    return HunyuanOCR15SFTAdapter(**kwargs)


def _make_hunyuan_data_adapter(**kwargs: Any) -> Any:
    return HunyuanTrainingDataAdapter(**kwargs)


def register_hunyuan_adapters(
    *,
    model_registry: ModelAdapterRegistry | None = None,
    data_registry: DataAdapterRegistry | None = None,
) -> None:
    """Register the hunyuanocr15_sft adapter(s) idempotently (default registries)."""
    global _registered
    model_reg = model_registry if model_registry is not None else get_default_registry()
    data_reg = (
        data_registry
        if data_registry is not None
        else get_default_data_adapter_registry()
    )
    if not model_reg.is_registered(HUNYUAN_MODEL_ADAPTER_TYPE):
        model_reg.register(HUNYUAN_DESCRIPTOR, _make_hunyuan_adapter)
    if not data_reg.is_registered(HUNYUAN_DATA_ADAPTER_TYPE):
        data_reg.register(HUNYUAN_DATA_ADAPTER_TYPE, _make_hunyuan_data_adapter)
    _registered = True


def hunyuan_adapters_registered() -> bool:
    """Whether registration has been performed in this process (test hook)."""
    return _registered


__all__ = [
    "HUNYUAN_DATA_ADAPTER_TYPE",
    "HUNYUAN_MODEL_ADAPTER_TYPE",
    "hunyuan_adapters_registered",
    "register_hunyuan_adapters",
]
