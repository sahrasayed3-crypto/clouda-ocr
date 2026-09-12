"""Deterministic, explicitly-populated model adapter registry.

WHY explicit registration (and no plugin discovery):

* Training runs must be reproducible.  Plugin discovery makes the set of
  available adapters depend on import order and on whatever happens to be
  installed in the environment, so two machines could silently train
  different models from the same config.  Explicit registration means the
  registry state is a property of the code that was checked out.
* Fail-closed behaviour: an unregistered ``adapter_type`` is an error with
  an actionable message, never a silent fallback to some default model.

Heavy dependencies (torch, transformers) are never imported here.  The
``factory`` argument is a lazy callable; concrete adapters import their
heavyweight internals inside the factory body so that merely importing this
module stays cheap.  Concrete adapters (hunyuanocr15_sft, qwen_vl_sft)
register themselves inside their own packages in a later wave.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Tuple

from clouda_training.adapters.descriptor import ModelAdapterDescriptor

__all__ = [
    "DuplicateAdapterError",
    "ModelAdapterRegistry",
    "UnknownAdapterError",
    "AdapterFactory",
    "get_default_registry",
]

#: A lazy factory callable.  It receives the caller's keyword arguments and
#: returns a fully-constructed adapter instance.  Heavy imports belong
#: inside the factory, not at module import time.
AdapterFactory = Callable[..., Any]


class UnknownAdapterError(ValueError):
    """Raised when an ``adapter_type`` was never registered.

    Subclasses :class:`ValueError` because the caller supplied an invalid
    value.  The message always lists the currently registered adapter
    types so the fix is obvious from the traceback alone.
    """


class DuplicateAdapterError(RuntimeError):
    """Raised when registering an ``adapter_type`` that already exists.

    Subclasses :class:`RuntimeError` because this is a programming/state
    error, not bad user input.  The registry refuses to overwrite silently:
    two packages claiming the same adapter_type must be resolved explicitly
    (call :meth:`ModelAdapterRegistry.unregister` first if replacement really
    is intended).
    """


class ModelAdapterRegistry:
    """Deterministic registry mapping ``adapter_type`` -> descriptor+factory.

    Invariants:

    * Registration is explicit: only :meth:`register` populates the registry.
    * Iteration order is always sorted by ``adapter_type`` so listing,
      error messages, and downstream config generation are deterministic.
    * Registration is refused after an entry exists (no silent overwrite).
    * The registry is thread-safe; factories themselves are invoked outside
      the lock so slow construction never blocks other lookups.
    """

    def __init__(self, name: str = "model-adapters") -> None:
        self._name = name
        self._lock = threading.RLock()
        # adapter_type -> (descriptor, factory)
        self._entries: Dict[str, Tuple[ModelAdapterDescriptor, AdapterFactory]] = {}

    @property
    def name(self) -> str:
        """Human-readable registry name used in error messages."""
        return self._name

    def register(
        self, descriptor: ModelAdapterDescriptor, factory: AdapterFactory
    ) -> None:
        """Register ``descriptor`` with a lazy ``factory`` callable.

        Raises:
            TypeError: if ``descriptor`` is not a
                :class:`ModelAdapterDescriptor` or ``factory`` is not
                callable.
            DuplicateAdapterError: if ``descriptor.adapter_type`` is already
                registered.
        """
        if not isinstance(descriptor, ModelAdapterDescriptor):
            raise TypeError(
                "descriptor must be a ModelAdapterDescriptor instance, got "
                f"{type(descriptor).__name__}. Build one from "
                "clouda_training.adapters.descriptor.ModelAdapterDescriptor."
            )
        if not callable(factory):
            raise TypeError(
                "factory must be a callable that builds the adapter when "
                f"invoked; got {type(factory).__name__}. Pass a function or "
                "class, and keep heavy imports inside its body."
            )
        adapter_type = descriptor.adapter_type
        if not isinstance(adapter_type, str) or not adapter_type.strip():
            raise ValueError(
                "descriptor.adapter_type must be a non-empty string; got "
                f"{adapter_type!r}."
            )
        with self._lock:
            if adapter_type in self._entries:
                raise DuplicateAdapterError(
                    f"adapter_type {adapter_type!r} is already registered in "
                    f"registry {self._name!r} (version "
                    f"{self._entries[adapter_type][0].adapter_version!r}); "
                    "refusing to overwrite. If replacement is intended, call "
                    f"unregister({adapter_type!r}) first."
                )
            self._entries[adapter_type] = (descriptor, factory)

    def unregister(self, adapter_type: str) -> None:
        """Remove a previously registered adapter (explicit replacement path).

        Raises:
            UnknownAdapterError: if ``adapter_type`` is not registered.
        """
        with self._lock:
            if adapter_type not in self._entries:
                raise UnknownAdapterError(
                    f"cannot unregister {adapter_type!r}: it is not "
                    f"registered in registry {self._name!r}. Registered "
                    f"adapter types: {self._registered_types()}."
                )
            del self._entries[adapter_type]

    def get(self, adapter_type: str) -> ModelAdapterDescriptor:
        """Return the :class:`ModelAdapterDescriptor` for ``adapter_type``.

        Raises:
            UnknownAdapterError: if ``adapter_type`` is not registered.
        """
        with self._lock:
            try:
                return self._entries[adapter_type][0]
            except KeyError:
                raise UnknownAdapterError(
                    f"unknown adapter_type {adapter_type!r} in registry "
                    f"{self._name!r}. Registered adapter types: "
                    f"{self._registered_types()}. Import the package that "
                    "owns this adapter (registration is explicit and "
                    "happens inside the adapter's own package) or fix the "
                    "adapter_type in the config."
                ) from None

    def get_factory(self, adapter_type: str) -> AdapterFactory:
        """Return the lazy factory callable registered for ``adapter_type``.

        Raises:
            UnknownAdapterError: if ``adapter_type`` is not registered.
        """
        with self._lock:
            try:
                return self._entries[adapter_type][1]
            except KeyError:
                raise UnknownAdapterError(
                    f"unknown adapter_type {adapter_type!r} in registry "
                    f"{self._name!r}. Registered adapter types: "
                    f"{self._registered_types()}."
                ) from None

    def list_adapters(self) -> Tuple[str, ...]:
        """Return all registered ``adapter_type`` strings, sorted.

        Sorted output keeps CLI help text, config validation errors, and
        snapshot tests deterministic regardless of registration order.
        """
        with self._lock:
            return tuple(sorted(self._entries))

    def is_registered(self, adapter_type: str) -> bool:
        """Return ``True`` iff ``adapter_type`` has an entry (no side effects)."""
        with self._lock:
            return adapter_type in self._entries

    def create(self, adapter_type: str, **kwargs: Any) -> Any:
        """Build an adapter instance by invoking its lazy factory.

        The factory receives only the caller's ``**kwargs``; it is invoked
        outside the registry lock.  Exceptions raised by the factory
        propagate unchanged so the original traceback (e.g. a missing torch
        install) is preserved.

        Raises:
            UnknownAdapterError: if ``adapter_type`` is not registered.
        """
        factory = self.get_factory(adapter_type)
        return factory(**kwargs)

    def _registered_types(self) -> Tuple[str, ...]:
        """Snapshot of registered types for error messages (lock held)."""
        return tuple(sorted(self._entries))


_DEFAULT_REGISTRY: ModelAdapterRegistry | None = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def get_default_registry() -> ModelAdapterRegistry:
    """Return the process-wide default :class:`ModelAdapterRegistry`.

    WHY a singleton: the orchestrator, the CLI, and adapter packages must all
    observe the same registry state within a process without threading a
    registry instance through every call site.

    The default registry starts EMPTY: concrete adapters
    (``hunyuanocr15_sft``, ``qwen_vl_sft``) register themselves inside their
    own packages in a later wave, which keeps this module importable without
    torch/transformers and avoids import-order coupling.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        with _DEFAULT_REGISTRY_LOCK:
            if _DEFAULT_REGISTRY is None:
                _DEFAULT_REGISTRY = ModelAdapterRegistry(name="default-model-adapters")
    return _DEFAULT_REGISTRY
