from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema_versions import DATASET_IDENTITY_VERSION


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    version: str
    split: str
    record_id: str
    schema_version: int = DATASET_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.dataset_id, self.version, self.split, self.record_id)
        ):
            raise ValueError("Dataset identity fields cannot be blank.")
        if self.schema_version != DATASET_IDENTITY_VERSION:
            raise ValueError("Unsupported dataset identity schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "split": self.split,
            "record_id": self.record_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetIdentity":
        return cls(
            dataset_id=str(value["dataset_id"]),
            version=str(value.get("version", value.get("dataset_version", "1"))),
            split=str(value.get("split", "unspecified")),
            record_id=str(value.get("record_id") or value.get("page_id") or ""),
            schema_version=int(value.get("schema_version", DATASET_IDENTITY_VERSION)),
        )
