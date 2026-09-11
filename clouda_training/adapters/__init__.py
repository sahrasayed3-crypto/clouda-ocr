"""Clouda multi-model training adapter framework (Wave 1 foundation).

Public surface: capability flags (ModelCapabilities) and adapter identity
(ModelAdapterDescriptor). Heavier framework pieces (registry, data adapters)
and any torch/transformers-touching code are exported lazily via module
``__getattr__`` so ``import clouda_training.adapters`` stays cheap and works
in minimal environments without torch/transformers installed.
"""

from __future__ import annotations

from typing import Any

from clouda_training.adapters.capabilities import ModelCapabilities
from clouda_training.adapters.descriptor import (
    ModelAdapterDescriptor,
    UnsupportedCapabilityError,
)

__all__ = [
    "ModelAdapterDescriptor",
    "ModelCapabilities",
    "UnsupportedCapabilityError",
    "__getattr__",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Heavy / later-wave names resolved on first attribute access only.
    # (module path, attribute name)
    "ModelAdapterRegistry": (
        "clouda_training.adapters.registry",
        "ModelAdapterRegistry",
    ),
    "UnknownAdapterError": ("clouda_training.adapters.registry", "UnknownAdapterError"),
    "DuplicateAdapterError": (
        "clouda_training.adapters.registry",
        "DuplicateAdapterError",
    ),
    "get_default_registry": (
        "clouda_training.adapters.registry",
        "get_default_registry",
    ),
    "ModelTrainingDataAdapter": (
        "clouda_training.adapters.data_adapter",
        "ModelTrainingDataAdapter",
    ),
    "DataAdapterRegistry": (
        "clouda_training.adapters.data_adapter",
        "DataAdapterRegistry",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)
