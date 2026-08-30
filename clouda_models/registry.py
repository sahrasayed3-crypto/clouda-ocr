from __future__ import annotations

import json
from pathlib import Path

from .metadata import ModelMetadata


class ModelRegistry:
    def __init__(self, records: tuple[ModelMetadata, ...]) -> None:
        identifiers = [record.model_id for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate model id.")
        self._records = {record.model_id: record for record in records}

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported model registry schema.")
        return cls(tuple(ModelMetadata.from_dict(item) for item in payload["models"]))

    def get(self, model_id: str) -> ModelMetadata:
        try:
            return self._records[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {model_id}") from exc

    def deployable(self) -> tuple[ModelMetadata, ...]:
        return tuple(
            record
            for record in self._records.values()
            if record.deployment_status == "approved"
            and record.commercial_use_status == "approved"
        )

    def validate_no_enabled_placeholder(self) -> None:
        for record in self._records.values():
            unresolved = record.model_revision.casefold() in {
                "unresolved",
                "latest",
                "main",
            }
            if unresolved and record.deployment_status != "disabled":
                raise ValueError(
                    f"Model {record.model_id} has an unpinned enabled revision."
                )
