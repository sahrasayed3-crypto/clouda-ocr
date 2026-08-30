from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset_identity import DatasetIdentity
from .page_identity import PageIdentity


@dataclass(frozen=True)
class RuntimeJobReference:
    job_id: str
    queue_name: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not self.queue_name.strip():
            raise ValueError("Runtime job reference fields cannot be blank.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "queue_name": self.queue_name,
        }


@dataclass(frozen=True)
class DatasetPageReference:
    dataset: DatasetIdentity
    page: PageIdentity
    text_checksum: str
    image_checksum: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset.to_dict(),
            "page": self.page.to_dict(),
            "text_checksum": self.text_checksum,
            "image_checksum": self.image_checksum,
        }
