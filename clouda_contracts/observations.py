from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .page_identity import PageIdentity
from .statuses import ObservationStatus


@dataclass(frozen=True)
class LayoutObservation:
    page: PageIdentity
    regions: tuple[dict[str, Any], ...]
    reading_order: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported layout observation schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "page": self.page.to_dict(),
            "regions": list(self.regions),
            "reading_order": list(self.reading_order),
        }


@dataclass(frozen=True)
class EvaluationObservation:
    page: PageIdentity
    policy_id: str
    metrics: tuple[tuple[str, float], ...]
    status: ObservationStatus = ObservationStatus.SUCCEEDED
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or self.schema_version != 1:
            raise ValueError("Invalid evaluation observation.")
        if any(value < 0 for _, value in self.metrics):
            raise ValueError("Evaluation metrics cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "page": self.page.to_dict(),
            "policy_id": self.policy_id,
            "metrics": dict(self.metrics),
            "status": self.status.value,
        }
