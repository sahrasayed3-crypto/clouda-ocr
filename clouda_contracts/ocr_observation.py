from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .page_identity import PageIdentity
from .schema_versions import OCR_OBSERVATION_VERSION
from .statuses import ObservationStatus


@dataclass(frozen=True)
class OCRObservation:
    page: PageIdentity
    engine_name: str
    status: ObservationStatus
    text: str = ""
    model_name: str | None = None
    confidence: float | None = None
    processing_seconds: float | None = None
    error_code: str | None = None
    schema_version: int = OCR_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not self.engine_name.strip():
            raise ValueError("Engine name cannot be blank.")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between zero and one.")
        if self.processing_seconds is not None and self.processing_seconds < 0:
            raise ValueError("Processing time cannot be negative.")
        if self.schema_version != OCR_OBSERVATION_VERSION:
            raise ValueError("Unsupported OCR observation schema version.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "page": self.page.to_dict(),
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "status": self.status.value,
            "text": self.text,
            "confidence": self.confidence,
            "processing_seconds": self.processing_seconds,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OCRObservation":
        status = str(value.get("status", "failed"))
        if status == "pending_ocr_model":
            status = "pending"
        return cls(
            page=PageIdentity.from_dict(value["page"]),
            engine_name=str(value["engine_name"]),
            model_name=value.get("model_name"),
            status=ObservationStatus(status),
            text=str(value.get("text", "")),
            confidence=value.get("confidence"),
            processing_seconds=value.get(
                "processing_seconds", value.get("processing_time")
            ),
            error_code=value.get("error_code") or value.get("error_message"),
            schema_version=int(value.get("schema_version", OCR_OBSERVATION_VERSION)),
        )
