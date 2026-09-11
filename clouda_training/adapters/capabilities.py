"""Honest capability flags for model adapters.

Every flag is a VERIFIED-SUPPORT claim: it asserts that the adapter has been
observed to work with this capability in the Clouda runtime, not that the
upstream model theoretically could. All flags default to ``False`` so that an
adapter which forgets to declare a capability fails closed (it is reported as
NOT supporting it) instead of accidentally advertising something untested.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelCapabilities"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Verified-support claims for one model adapter.

    Why frozen: capabilities are static, reviewed facts about an adapter —
    runtime code must not be able to flip them to make a failing path pass.
    """

    supports_full_finetune: bool = False
    supports_selective_finetune: bool = False
    supports_lora: bool = False
    supports_gradient_checkpointing: bool = False
    supports_packed_sequences: bool = False
    supports_multimodal_batches: bool = False
    supports_local_only_loading: bool = False
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_cpu_smoke: bool = False
    supports_resume: bool = False
    real_weights_validated: bool = False
    gpu_validated: bool = False

    def summary(self) -> dict[str, bool]:
        """Return the flags as a plain dict for reports and metadata."""
        return {
            "supports_full_finetune": self.supports_full_finetune,
            "supports_selective_finetune": self.supports_selective_finetune,
            "supports_lora": self.supports_lora,
            "supports_gradient_checkpointing": (self.supports_gradient_checkpointing),
            "supports_packed_sequences": self.supports_packed_sequences,
            "supports_multimodal_batches": self.supports_multimodal_batches,
            "supports_local_only_loading": self.supports_local_only_loading,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "supports_cpu_smoke": self.supports_cpu_smoke,
            "supports_resume": self.supports_resume,
            "real_weights_validated": self.real_weights_validated,
            "gpu_validated": self.gpu_validated,
        }
