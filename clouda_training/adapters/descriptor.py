"""Adapter descriptors: static, reviewable identity + capability records.

A descriptor is the single source of truth the framework consults BEFORE any
heavy dependency is imported: what the adapter is, what it claims to support,
which optional packages it needs, and which upstream symbols must exist. The
registry, preflight checks, and checkpoint metadata all derive from it, so it
stays a frozen dataclass — mutating a live descriptor would let unverified
configuration masquerade as reviewed configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clouda_training.adapters.capabilities import ModelCapabilities

__all__ = ["ModelAdapterDescriptor", "UnsupportedCapabilityError"]


class UnsupportedCapabilityError(RuntimeError):
    """Raised when an operation requires a capability the adapter does not
    claim. Fail-closed on purpose: the error names the missing flag so the
    operator can fix the config instead of debugging a silent fallback."""


@dataclass(frozen=True)
class ModelAdapterDescriptor:
    """Static description of one model adapter (identity + capabilities).

    All collection fields are tuples so the descriptor stays hashable and
    safely shareable across the registry and checkpoint metadata.
    """

    adapter_type: str
    adapter_version: str
    model_family: str
    task_family: str
    capabilities: ModelCapabilities
    required_optional_dependencies: tuple[str, ...] = ()
    supported_precision: tuple[str, ...] = ()
    supported_devices: tuple[str, ...] = ()
    supported_data_modes: tuple[str, ...] = ()
    checkpoint_compatibility_id: str = ""
    upstream_repository: str = ""
    upstream_revision: str = ""
    expected_model_symbols: tuple[str, ...] = ()
    expected_processor_symbols: tuple[str, ...] = ()
    data_contract_version: str = "1"

    def require_capabilities(self, *flags: str) -> None:
        """Fail closed unless every named capability flag is True.

        ``flags`` are attribute names on :class:`ModelCapabilities`; unknown
        names raise immediately so typos cannot silently pass.
        """
        for name in flags:
            if not hasattr(self.capabilities, name):
                raise UnsupportedCapabilityError(
                    f"adapter '{self.adapter_type}': unknown capability flag "
                    f"'{name}' — check ModelCapabilities for valid names"
                )
            if not getattr(self.capabilities, name):
                raise UnsupportedCapabilityError(
                    f"adapter '{self.adapter_type}' does not declare support "
                    f"for '{name}'; capability flags are verified claims and "
                    "cannot be assumed. Request the capability be validated "
                    "before using this code path."
                )

    def identity_dict(self) -> dict[str, Any]:
        """Canonical checkpoint metadata for this adapter.

        Excluded on purpose: machine paths (local_model_path lives on the
        adapter instance) and capability claims (identity must stay stable
        even if flags are re-validated later) — checkpoints must resume
        against the same adapter/code identity, not the same opinions.
        """
        return {
            "adapter_type": self.adapter_type,
            "adapter_version": self.adapter_version,
            "model_family": self.model_family,
            "task_family": self.task_family,
            "checkpoint_compatibility_id": self.checkpoint_compatibility_id,
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
            "data_contract_version": self.data_contract_version,
            "required_optional_dependencies": list(self.required_optional_dependencies),
            "supported_precision": list(self.supported_precision),
            "supported_devices": list(self.supported_devices),
            "supported_data_modes": list(self.supported_data_modes),
        }
