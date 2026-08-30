from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_uri: str
    base_model: str
    model_revision: str
    training_method: str
    dataset_manifest_ids: tuple[str, ...]
    experiment_id: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_uri": self.checkpoint_uri,
            "base_model": self.base_model,
            "model_revision": self.model_revision,
            "training_method": self.training_method,
            "dataset_manifest_ids": list(self.dataset_manifest_ids),
            "experiment_id": self.experiment_id,
        }
