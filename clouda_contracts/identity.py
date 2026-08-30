from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    source_uri: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.document_id.strip() or self.schema_version != 1:
            raise ValueError("Invalid document identity.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentIdentity":
        return cls(
            document_id=str(value.get("document_id") or value.get("id") or ""),
            source_uri=value.get("source_uri") or value.get("source_path"),
            schema_version=int(value.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class ModelVersion:
    revision: str
    tokenizer_revision: str | None = None
    processor_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.revision.strip():
            raise ValueError("Model revision cannot be blank.")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "processor_revision": self.processor_revision,
        }


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    version: ModelVersion
    provider: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.provider.strip():
            raise ValueError("Model identity fields cannot be blank.")
        if self.schema_version != 1:
            raise ValueError("Unsupported model identity schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "provider": self.provider,
            "version": self.version.to_dict(),
        }
