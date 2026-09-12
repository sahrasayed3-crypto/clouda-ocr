"""Training-data adapter protocol and registry.

WHY a data adapter layer: model families disagree about training record
shape (image+OCR-text pairs, chat-style multimodal messages, packed
sequences, ...).  The orchestrator should never hardcode those shapes.
Instead each model family ships a :class:`ModelTrainingDataAdapter` that
turns generic manifest rows into that family's exact record format
(``export``), verifies records before a run starts (``validate``, so a bad
dataset fails fast and cheaply instead of mid-training), and advertises its
contract programmatically (``describe_contract``, so tooling and tests can
assert against it without importing the model code).

The registry here mirrors :mod:`clouda_training.adapters.registry`: fully
explicit registration, deterministic sorted iteration, no plugin discovery.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Protocol, Tuple, runtime_checkable

__all__ = [
    "DataAdapterFactory",
    "DataAdapterRegistry",
    "ModelTrainingDataAdapter",
    "UnknownDataAdapterError",
    "DuplicateDataAdapterError",
    "get_default_data_adapter_registry",
]

#: Manifest rows are family-agnostic dicts produced upstream (e.g. by the
#: dataset manifest builder).  Kept as ``Dict[str, Any]`` at this boundary:
#: enforcing a narrower shape here would couple the framework to one
#: manifest schema version.
ManifestRow = Dict[str, Any]

#: A fully exported training record, already shaped for the target family.
TrainingRecord = Dict[str, Any]

#: Result of ``validate``: a summary dict (e.g. record counts, error lists).
#: Returned as data rather than raised, so callers can report *all* problems
#: in one pass instead of dying on the first.
ValidationSummary = Dict[str, Any]


@runtime_checkable
class ModelTrainingDataAdapter(Protocol):
    """Protocol every model-family data adapter must satisfy.

    Implementations do NOT need to inherit from this class: structural
    typing (``runtime_checkable``) lets any object with the three methods
    pass an ``isinstance`` check, which keeps adapters free to subclass
    whatever base their own package prefers.
    """

    def export(
        self, manifest_rows: list[ManifestRow], config: Any
    ) -> list[TrainingRecord]:
        """Convert generic ``manifest_rows`` into family-specific records.

        Args:
            manifest_rows: family-agnostic row dicts from the dataset
                manifest.
            config: family-specific export configuration object (adapter
                packages own its type; the protocol deliberately does not
                constrain it).

        Returns:
            Records ready for the family's ``make_batch``/collate path.

        Raises:
            Exception subclasses of the implementation's choosing on
            malformed input; implementations should fail closed with
            actionable messages naming the offending row.
        """
        ...

    def validate(self, records: list[TrainingRecord]) -> ValidationSummary:
        """Check ``records`` against the family contract WITHOUT training.

        Returns a summary dict (counts, error/warning lists) rather than
        raising on the first problem, so a whole dataset can be triaged in
        one pass.  Implementations must treat an error summary as
        training-blocking in their own orchestration.
        """
        ...

    def describe_contract(self) -> Dict[str, Any]:
        """Describe the record contract this adapter produces.

        Returns a JSON-serialisable dict including at least a
        ``data_contract_version`` key and per-field descriptions, so
        downstream tooling can assert compatibility without importing the
        model package.
        """
        ...


#: A lazy factory for data adapters, mirroring the model-adapter
#: :data:`clouda_training.adapters.registry.AdapterFactory` convention:
#: heavy imports stay inside the callable.
DataAdapterFactory = Callable[..., Any]


class UnknownDataAdapterError(ValueError):
    """Raised when a ``data_adapter_type`` was never registered.

    Subclasses :class:`ValueError` (bad caller input); the message lists the
    registered data adapter types.
    """


class DuplicateDataAdapterError(RuntimeError):
    """Raised when a ``data_adapter_type`` is registered twice.

    Subclasses :class:`RuntimeError` (state/programming error); no silent
    overwrite — call ``unregister`` first if replacement is intended.
    """


class DataAdapterRegistry:
    """Deterministic registry mapping ``data_adapter_type`` -> factory.

    Mirrors :class:`clouda_training.adapters.registry.ModelAdapterRegistry`
    exactly: explicit registration only, sorted deterministic listing,
    refusal to overwrite, thread-safe, factories invoked outside the lock.

    Unlike the model-adapter registry there is no descriptor object here:
    the data adapter's own ``describe_contract()`` plays the descriptor
    role, so this registry stores ``(factory,)`` entries keyed by type.
    """

    def __init__(self, name: str = "data-adapters") -> None:
        self._name = name
        self._lock = threading.RLock()
        self._entries: Dict[str, DataAdapterFactory] = {}

    @property
    def name(self) -> str:
        """Human-readable registry name used in error messages."""
        return self._name

    def register(self, data_adapter_type: str, factory: DataAdapterFactory) -> None:
        """Register a lazy ``factory`` under ``data_adapter_type``.

        Raises:
            TypeError: if ``factory`` is not callable.
            ValueError: if ``data_adapter_type`` is empty or not a string.
            DuplicateDataAdapterError: if the type is already registered.
        """
        if not isinstance(data_adapter_type, str) or not data_adapter_type.strip():
            raise ValueError(
                "data_adapter_type must be a non-empty string; got "
                f"{data_adapter_type!r}."
            )
        if not callable(factory):
            raise TypeError(
                "factory must be a callable that builds the data adapter "
                f"when invoked; got {type(factory).__name__}. Keep heavy "
                "imports inside the factory body."
            )
        with self._lock:
            if data_adapter_type in self._entries:
                raise DuplicateDataAdapterError(
                    f"data_adapter_type {data_adapter_type!r} is already "
                    f"registered in registry {self._name!r}; refusing to "
                    "overwrite. If replacement is intended, call "
                    f"unregister({data_adapter_type!r}) first."
                )
            self._entries[data_adapter_type] = factory

    def unregister(self, data_adapter_type: str) -> None:
        """Remove a previously registered data adapter (replacement path).

        Raises:
            UnknownDataAdapterError: if the type is not registered.
        """
        with self._lock:
            if data_adapter_type not in self._entries:
                raise UnknownDataAdapterError(
                    f"cannot unregister {data_adapter_type!r}: it is not "
                    f"registered in registry {self._name!r}. Registered "
                    f"data adapter types: {self._registered_types()}."
                )
            del self._entries[data_adapter_type]

    def get(self, data_adapter_type: str) -> DataAdapterFactory:
        """Return the lazy factory registered under ``data_adapter_type``.

        Raises:
            UnknownDataAdapterError: if the type is not registered.
        """
        with self._lock:
            try:
                return self._entries[data_adapter_type]
            except KeyError:
                raise UnknownDataAdapterError(
                    f"unknown data_adapter_type {data_adapter_type!r} in "
                    f"registry {self._name!r}. Registered data adapter "
                    f"types: {self._registered_types()}. Import the package "
                    "that owns this data adapter (registration is explicit "
                    "and happens inside its own package) or fix the type "
                    "in the config."
                ) from None

    def list_adapters(self) -> Tuple[str, ...]:
        """Return all registered ``data_adapter_type`` strings, sorted."""
        with self._lock:
            return tuple(sorted(self._entries))

    def is_registered(self, data_adapter_type: str) -> bool:
        """Return ``True`` iff the type has an entry (no side effects)."""
        with self._lock:
            return data_adapter_type in self._entries

    def create(self, data_adapter_type: str, **kwargs: Any) -> Any:
        """Build a data adapter by invoking its lazy factory.

        Exceptions from the factory propagate unchanged so the original
        traceback (e.g. a missing optional dependency) is preserved.

        Raises:
            UnknownDataAdapterError: if the type is not registered.
        """
        factory = self.get(data_adapter_type)
        return factory(**kwargs)

    def _registered_types(self) -> Tuple[str, ...]:
        """Snapshot of registered types for error messages (lock held)."""
        return tuple(sorted(self._entries))


_DEFAULT_DATA_REGISTRY: DataAdapterRegistry | None = None
_DEFAULT_DATA_REGISTRY_LOCK = threading.Lock()


def get_default_data_adapter_registry() -> DataAdapterRegistry:
    """Return the process-wide default :class:`DataAdapterRegistry`.

    Starts EMPTY for the same reason as the model-adapter default registry:
    concrete data adapters use their explicit package registration functions,
    keeping this module free of heavy imports and import-order coupling.
    """
    global _DEFAULT_DATA_REGISTRY
    if _DEFAULT_DATA_REGISTRY is None:
        with _DEFAULT_DATA_REGISTRY_LOCK:
            if _DEFAULT_DATA_REGISTRY is None:
                _DEFAULT_DATA_REGISTRY = DataAdapterRegistry(
                    name="default-data-adapters"
                )
    return _DEFAULT_DATA_REGISTRY
