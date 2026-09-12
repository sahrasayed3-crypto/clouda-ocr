"""HunyuanOCR-1.5 SFT bridge data models.

Canonical Clouda manifests stay portable; Hunyuan-format artifacts are
generated exports with absolute paths and full lineage. Upstream schema
verified at Tencent-Hunyuan/HunyuanOCR@c55965d3da1e (see
UPSTREAM_COMPATIBILITY.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HUNYUAN_IMAGE_PLACEHOLDER = "<image>"
HUNYUAN_COMPAT_REVISION = "c55965d3da1e"
HUNYUAN_COMPAT_REPOSITORY = "Tencent-Hunyuan/HunyuanOCR"
HUNYUAN_RAW_SCHEMA_VERSION = 1
HUNYUAN_PACKED_SCHEMA_VERSION = 1
DEFAULT_PACK_LENGTH = 20480


@dataclass(frozen=True)
class PromptProfile:
    """Versioned prompt profile; identity persists in export lineage."""

    prompt_id: str
    prompt_version: int
    prompt_text: str
    language: str = "ar"
    task: str = "document_ocr_transcription"

    @property
    def identity(self) -> str:
        return f"{self.prompt_id}@v{self.prompt_version}"


# Initial Clouda Arabic document-OCR training prompt: faithful transcription.
ARABIC_DOCUMENT_OCR_PROMPT = PromptProfile(
    prompt_id="clouda_arabic_document_ocr",
    prompt_version=1,
    prompt_text=(
        "استخرج كامل محتوى الصورة النصي بدقة، واكتبه كما هو: "
        "احتفظ بالفواصل والترقيم والأرقام وفواصل الأسطر. "
        "لا تشرح ولا تلخص ولا تضف أي معلومات من عندك."
    ),
)


@dataclass(frozen=True)
class HunyuanConversationTurn:
    from_role: str  # "human" | "gpt"
    value: str


@dataclass(frozen=True)
class HunyuanRawSample:
    """One raw OCR JSONL record (upstream schema), plus export lineage."""

    image_path: list[str]
    conversations: list[HunyuanConversationTurn]
    canonical_sample_id: str  # lineage: canonical manifest row id
    prompt_identity: str

    def to_upstream_dict(self) -> dict[str, Any]:
        """Exact upstream wire format (lineage fields excluded)."""
        return {
            "image_path": list(self.image_path),
            "conversations": [
                {"from": turn.from_role, "value": turn.value}
                for turn in self.conversations
            ],
        }


@dataclass(frozen=True)
class HunyuanExportConfig:
    dataset_id: str
    dataset_version: str
    manifest_path: str
    manifest_hash: str
    split: str
    image_root: str  # directory prefix used to absolutize canonical relative paths
    prompt_profile: PromptProfile
    pack_length: int = DEFAULT_PACK_LENGTH
    exporter_version: str = "1.0.0"


@dataclass
class HunyuanExportReport:
    """Lineage + outcome of one export run."""

    config: HunyuanExportConfig
    output_path: str = ""
    output_sha256: str = ""
    exported_count: int = 0
    skipped_counts: dict[str, int] = field(default_factory=dict)
    sample_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HUNYUAN_RAW_SCHEMA_VERSION,
            "upstream_repository": HUNYUAN_COMPAT_REPOSITORY,
            "upstream_revision": HUNYUAN_COMPAT_REVISION,
            "dataset_id": self.config.dataset_id,
            "dataset_version": self.config.dataset_version,
            "manifest_path": self.config.manifest_path,
            "manifest_hash": self.config.manifest_hash,
            "split": self.config.split,
            "prompt_identity": self.config.prompt_profile.identity,
            "prompt_text": self.config.prompt_profile.prompt_text,
            "pack_length": self.config.pack_length,
            "exporter_version": self.config.exporter_version,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "exported_count": self.exported_count,
            "skipped_counts": self.skipped_counts,
            "sample_ids": self.sample_ids,
        }
