"""Qwen3-VL SFT adapter for the Clouda real-training runtime.

Mirrors :mod:`clouda_training.hunyuan.adapter` (HunyuanOCR15SFTAdapter)
structure exactly. Lazy + optional: importing this module never imports
torch/transformers — heavy symbols resolve inside ``build_model``.
Local-only: model/processor load from an operator-supplied local path; the
adapter never downloads weights. Loss extraction fails explicitly when the
model output carries no loss — never fabricated.

Upstream facts (verified, pinned): QwenLM/Qwen3-VL@96588727e44c78b25ba03ea03b8e12f7e64fd0da.
Model class ``Qwen3VLForConditionalGeneration`` is native in
transformers >= 4.57.0 — NO trust_remote_code, NO remote-code execution.
"""

from __future__ import annotations

from typing import Any

from clouda_training.qwen.models import (
    QWEN_COMPAT_REPOSITORY,
    QWEN_COMPAT_REVISION,
    QWEN_MIN_TRANSFORMERS_VERSION,
)

ADAPTER_ID = "qwen_vl_sft"
ADAPTER_VERSION = "1.0.0"


class QwenAdapterError(RuntimeError):
    """Raised for Qwen adapter configuration/compatibility problems."""


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def transformers_version() -> str | None:
    if not transformers_available():
        return None
    import transformers

    return getattr(transformers, "__version__", None)


def transformers_version_ok() -> bool:
    """Whether the installed transformers is >= the verified minimum (4.57.0)."""
    version = transformers_version()
    if version is None:
        return False
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover — packaging ships with pip
        digits = "".join(ch for ch in version if ch.isdigit() or ch == ".")
        major_minor = digits.split(".")[:2]
        try:
            return tuple(int(x) for x in major_minor) >= (4, 57)
        except ValueError:
            return False

    try:
        return Version(version) >= Version(QWEN_MIN_TRANSFORMERS_VERSION)
    except InvalidVersion:  # pragma: no cover — dev/deviant version strings
        return False


def model_class_available() -> bool:
    """Whether the verified upstream model class is importable (native path)."""
    if not transformers_version_ok():
        return False
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except ImportError:
        return False
    return True


class QwenVLSFTAdapter:
    """SFT adapter bridging Qwen3-VL into the Clouda runtime.

    The generic runtime owns the loop; this adapter owns model, processor,
    model-specific batch preparation, forward/loss, and trainable components.

    Selective-tuning mapping (upstream official contract):
      * ``tune_vision``     -> ``model.visual``
      * ``tune_projector``  -> ``model.visual.merger``
      * ``tune_llm``        -> ``model.language_model`` (+ tied ``lm_head``)
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
        processor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.local_model_path = local_model_path
        self.tune_vision = tune_vision
        self.tune_projector = tune_projector
        self.tune_llm = tune_llm
        self.use_lora = use_lora
        self.gradient_checkpointing = gradient_checkpointing
        self.processor = processor
        self.tokenizer = tokenizer
        self.model: Any = None

    # ------------------------------------------------------------------ identity
    def identity(self) -> dict[str, Any]:
        """Adapter metadata persisted in checkpoints (no machine-specific paths)."""
        return {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "upstream_repository": QWEN_COMPAT_REPOSITORY,
            "upstream_revision": QWEN_COMPAT_REVISION,
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
            raise QwenAdapterError(
                f"local Qwen3-VL model path does not exist: {path} — "
                "this adapter never downloads weights"
            )
        if not (path / "config.json").is_file():
            raise QwenAdapterError(
                f"missing config.json under {path} — not a local model directory"
            )

    def build_model(self, config: Any = None) -> Any:
        """Construct the model from the LOCAL path only (no network).

        Uses the native transformers path (>= 4.57.0): no trust_remote_code,
        no remote-code execution. A separate fast=False tokenizer with
        right padding mirrors the official qwen-vl-finetune setup.
        """
        self._require_local_path()
        require_transformers()
        import torch
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen3VLForConditionalGeneration,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path,
            use_fast=False,
            padding_side="right",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        self._apply_trainable_selection()
        if self.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        return self.model

    def _apply_trainable_selection(self) -> None:
        """Freeze components per tune_* flags using the verified Qwen3-VL
        attribute layout (model.visual / model.visual.merger /
        model.language_model), falling back to name matching only when the
        attributes are absent."""
        import torch

        model = self.model
        components: dict[str, list[Any]] = {
            "vision": [],
            "projector": [],
            "llm": [],
        }
        visual = getattr(model, "visual", None)
        if visual is not None:
            components["vision"] = [visual]
        merger = getattr(visual, "merger", None)
        if merger is not None:
            components["projector"] = [merger]
        language_model = getattr(model, "language_model", None)
        if language_model is not None:
            components["llm"].append(language_model)
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            components["llm"].append(lm_head)
        if not any(components.values()):
            # No structural attributes found: fall back to name matching.
            for name, param in model.named_parameters():
                if "visual" in name.lower() and "merger" in name.lower():
                    components["projector"].append(param)
                elif "visual" in name.lower():
                    components["vision"].append(param)
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
                    [module_or_param]
                    if isinstance(module_or_param, torch.nn.Parameter)
                    else (
                        module_or_param.parameters()
                        if hasattr(module_or_param, "parameters")
                        else [module_or_param]
                    )
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
            raise QwenAdapterError(
                "model output carries no loss — refusing to fabricate one; "
                "ensure labels (-100-masked outside assistant spans) are "
                "provided by the collator"
            )
        return loss

    def model_state(self, model: Any) -> Any:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def load_model_state(self, model: Any, state: Any) -> None:
        model.load_state_dict(state)

    # ------------------------------------------------------------------ batch
    def make_batch(self, step: int, batch_size: int, seed: int) -> Any:
        """Model-specific batch boundary.

        The real processor-backed path (AutoProcessor over Qwen conversations
        records, labels masked to assistant spans) requires real weights to
        validate. In mock/test mode an injected mock processor supplies
        tensor batches; the generic runtime never sees these fields.
        """
        if self.processor is None:
            raise QwenAdapterError(
                "processor not loaded — call build_model() first or inject a "
                "mock processor for offline tests"
            )
        return self.processor.build_training_batch(
            step=step, batch_size=batch_size, seed=seed
        )


def require_transformers() -> None:
    if not transformers_available():
        raise QwenAdapterError(
            "transformers is required for the Qwen3-VL adapter "
            f"(>={QWEN_MIN_TRANSFORMERS_VERSION}). "
            "Install with: pip install transformers"
        )
    if not transformers_version_ok():
        raise QwenAdapterError(
            f"transformers >= {QWEN_MIN_TRANSFORMERS_VERSION} is required for "
            "the Qwen3-VL adapter (native Qwen3VLForConditionalGeneration, "
            f"no trust_remote_code); found {transformers_version()}"
        )
