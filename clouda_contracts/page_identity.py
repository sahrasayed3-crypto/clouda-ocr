from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema_versions import PAGE_IDENTITY_VERSION


@dataclass(frozen=True)
class PageIdentity:
    document_id: str
    page_number: int
    page_id: str
    source_uri: str | None = None
    schema_version: int = PAGE_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.page_id.strip():
            raise ValueError("Document and page identifiers cannot be blank.")
        if self.page_number < 1:
            raise ValueError("Page numbers are one-based.")
        if self.schema_version != PAGE_IDENTITY_VERSION:
            raise ValueError("Unsupported page identity schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_id": self.page_id,
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PageIdentity":
        return cls(
            document_id=str(value.get("document_id") or value.get("document") or ""),
            page_number=int(value.get("page_number", value.get("page_no", 0))),
            page_id=str(value.get("page_id") or value.get("id") or ""),
            source_uri=value.get("source_uri") or value.get("source_path"),
            schema_version=int(value.get("schema_version", PAGE_IDENTITY_VERSION)),
        )
