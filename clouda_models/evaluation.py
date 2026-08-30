from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelEvaluation:
    model_id: str
    model_revision: str
    dataset_manifest_id: str
    metric_policy_id: str
    metrics: tuple[tuple[str, float], ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_manifest_id": self.dataset_manifest_id,
            "metric_policy_id": self.metric_policy_id,
            "metrics": dict(self.metrics),
        }
