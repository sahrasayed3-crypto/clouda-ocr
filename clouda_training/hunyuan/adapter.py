"""HunyuanOCR-1.5 SFT model adapter for the Clouda real-training runtime.

Implements the ModelAdapter contract from feature/real-training-runtime.
Lazy + optional: the base package imports without transformers/torch.
Local-only: model/processor load from an operator-supplied local path;
never downloads weights. Loss extraction fails explicitly when the model
output carries no loss — never fabricated.
"""

from __future__ import annotations

from typing import Any

from clouda_training.hunyuan.models import (
    HUNYUAN_COMPAT_REVISION,
    HUNYUAN_COMPAT_REPOSITORY,
)

ADAPTER_ID = "hunyuanocr15_sft"
ADAPTER_VERSION = "1.0.0"


class HunyuanAdapterError(RuntimeError):
    """Raised for adapter configuration/compatibility problems."""


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def require_transformers() -> None:
    if not transformers_available():
        raise HunyuanAdapterError(
            "transformers is required for the HunyuanOCR adapter. "
            "Install with: pip install clouda-pdf[training-hunyuan]"
        )


def model_class_available() -> bool:
    """Whether the verified upstream model class is importable."""
    if not transformers_available():
        return False
    try:
        from transformers import HunYuanVLForConditionalGeneration  # noqa: F401
    except ImportError:
        return False
    return True


class HunyuanOCR15SFTAdapter:
    """SFT adapter bridging HunyuanOCR-1.5 into the Clouda runtime.

    The generic runtime owns the loop; this adapter owns model, processor,
    model-specific batch preparation, forward/loss, and trainable components.
    """

    def __init__(
        self,
        *,
        local_model_path: str,
        tune_vision: bool = True,
        tune_projector: bool = True,
        tune_llm: bool = True,
        use_lora: bool = False,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        processor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.local_model_path = local_model_path
        self.tune_vision = tune_vision
        self.tune_projector = tune_projector
        self.tune_llm = tune_llm
        self.use_lora = use_lora
        self.gradient_checkpointing = gradient_checkpointing
        self.trust_remote_code = trust_remote_code
        self.processor = processor
        self.tokenizer = tokenizer
        self.model: Any = None

    # ------------------------------------------------------------------ identity
    def identity(self) -> dict[str, Any]:
        """Adapter metadata persisted in checkpoints (no machine-specific paths
        contaminate canonical identity; local path stays operational-only)."""
        return {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "upstream_repository": HUNYUAN_COMPAT_REPOSITORY,
            "upstream_revision": HUNYUAN_COMPAT_REVISION,
            "trainable": {
                "vision": self.tune_vision,
                "projector": self.tune_projector,
                "llm": self.tune_llm,
            },
            "lora": self.use_lora,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    # ------------------------------------------------------------------ loading
    def _require_local_path(self) -> None:
        from pathlib import Path

        path = Path(self.local_model_path)
        if not path.is_dir():
            raise HunyuanAdapterError(
                f"local HunyuanOCR model path does not exist: {path} — "
                "this adapter never downloads weights"
            )
        if not (path / "config.json").is_file():
            raise HunyuanAdapterError(
                f"missing config.json under {path} — not a local model directory"
            )

    def build_model(self, config: Any = None) -> Any:
        """Construct the model from the LOCAL path only (no network)."""
        self._require_local_path()
        require_transformers()
        import torch
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            HunYuanVLForConditionalGeneration,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        self._apply_trainable_selection()
        if self.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        return self.model

    def _apply_trainable_selection(self) -> None:
        """Freeze components per tune_* flags using safe model attributes
        (fallback: conservative name matching only when attributes absent)."""
        import torch

        model = self.model
        components: dict[str, list[Any]] = {
            "vision": [],
            "projector": [],
            "llm": [],
        }
        for name in ("vision_tower", "vision_encoder", "vision"):
            if hasattr(model, name):
                components["vision"] = [getattr(model, name)]
                break
        for name in ("mm_projector", "multi_modal_projector", "projector", "mlp"):
            if hasattr(model, name):
                components["projector"] = [getattr(model, name)]
                break
        for name in ("language_model", "llm", "text_model"):
            if hasattr(model, name):
                components["llm"] = [getattr(model, name)]
                break
        if not any(components.values()):
            # No structural attributes found: fall back to name matching.
            for name, param in model.named_parameters():
                if "vision" in name.lower():
                    components["vision"].append(param)
                elif "projector" in name.lower() or name.endswith("mlp"):
                    components["projector"].append(param)
                else:
                    components["llm"].append(param)
        flags = {
            "vision": self.tune_vision,
            "projector": self.tune_projector,
            "llm": self.tune_llm,
        }
        for component, enabled in flags.items():
            for module_or_param in components[component]:
                target = (
                    module_or_param
                    if isinstance(module_or_param, torch.nn.Parameter)
                    else getattr(
                        module_or_param, "parameters", lambda: [module_or_param]
                    )()
                )
                for param in target:
                    param.requires_grad = bool(enabled)

    def trainable_parameters(self, model: Any) -> list[Any]:
        return [p for p in model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ forward
    def forward_loss(self, model: Any, batch: Any) -> Any:
        """Forward pass returning the REAL loss from the model output.

        Supports dict-like outputs (loss key) and object outputs (.loss).
        Explicit failure when no loss exists — never fabricate one.
        """
        output = model(**batch) if isinstance(batch, dict) else model(batch)
        loss = None
        if isinstance(output, dict):
            loss = output.get("loss")
        elif hasattr(output, "loss"):
            loss = output.loss
        if loss is None:
            raise HunyuanAdapterError(
                "model output carries no loss — refusing to fabricate one; "
                "ensure labels are provided by the collator"
            )
        return loss

    def model_state(self, model: Any) -> Any:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def load_model_state(self, model: Any, state: Any) -> None:
        model.load_state_dict(state)

    # ------------------------------------------------------------------ batch
    def make_batch(self, step: int, batch_size: int, seed: int) -> Any:
        """Model-specific batch boundary.

        With a real processor this collates packed multimodal features;
        in mock/test mode a processor-backed fake supplies tensors. The
        generic runtime never sees these fields.
        """
        if self.processor is None:
            raise HunyuanAdapterError(
                "processor not loaded — call build_model() first or inject a "
                "mock processor for offline tests"
            )
        return self.processor.build_training_batch(
            step=step, batch_size=batch_size, seed=seed
        )
