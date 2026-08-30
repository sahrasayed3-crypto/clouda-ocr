from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .page_identity import PageIdentity
from .schema_versions import MODEL_OBSERVATION_VERSION
from .statuses import ObservationStatus


@dataclass(frozen=True)
class ModelObservation:
    page: PageIdentity
    model_id: str
    status: ObservationStatus
    output_text: str
    quality_score: float | None = None
    elapsed_seconds: float | None = None
    schema_version: int = MODEL_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("Model id cannot be blank.")
        if self.quality_score is not None and not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between zero and one.")
        if self.schema_version != MODEL_OBSERVATION_VERSION:
            raise ValueError("Unsupported model observation schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "page": self.page.to_dict(),
            "model_id": self.model_id,
            "status": self.status.value,
            "output_text": self.output_text,
            "quality_score": self.quality_score,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelObservation":
        return cls(
            page=PageIdentity.from_dict(value["page"]),
            model_id=str(value.get("model_id") or value.get("model_used") or ""),
            status=ObservationStatus(str(value.get("status", "succeeded"))),
            output_text=str(value.get("output_text", value.get("markdown", ""))),
            quality_score=value.get("quality_score"),
            elapsed_seconds=value.get("elapsed_seconds", value.get("elapsed_time")),
            schema_version=int(value.get("schema_version", MODEL_OBSERVATION_VERSION)),
        )
