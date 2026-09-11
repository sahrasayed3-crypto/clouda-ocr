"""HunyuanOCR-1.5 SFT integration bridge.

Imports cleanly without transformers/torch; heavy symbols resolve lazily.
Upstream compatibility: Tencent-Hunyuan/HunyuanOCR@c55965d3da1e.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clouda_training.hunyuan.compat import (
    CompatibilityMetadata,
    check_drift,
)
from clouda_training.hunyuan.exporter import export_raw_jsonl
from clouda_training.hunyuan.models import (
    ARABIC_DOCUMENT_OCR_PROMPT,
    HunyuanConversationTurn,
    HunyuanExportConfig,
    HunyuanExportReport,
    HunyuanRawSample,
    PromptProfile,
)
from clouda_training.hunyuan.packing import build_packing_plan
from clouda_training.hunyuan.preflight import run_preflight
from clouda_training.hunyuan.validators import (
    validate_packed_jsonl,
    validate_raw_jsonl,
)

if TYPE_CHECKING:  # pragma: no cover
    from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter

__all__ = [
    "ARABIC_DOCUMENT_OCR_PROMPT",
    "CompatibilityMetadata",
    "HunyuanConversationTurn",
    "HunyuanExportConfig",
    "HunyuanExportReport",
    "HunyuanOCR15SFTAdapter",
    "HunyuanRawSample",
    "PromptProfile",
    "build_packing_plan",
    "check_drift",
    "export_raw_jsonl",
    "run_preflight",
    "validate_packed_jsonl",
    "validate_raw_jsonl",
]


def __getattr__(name: str) -> Any:
    if name == "HunyuanOCR15SFTAdapter":
        from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter

        return HunyuanOCR15SFTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
