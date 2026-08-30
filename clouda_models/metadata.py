from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    base_model_name: str
    model_revision: str
    license: str
    model_type: str
    parameter_count: int | None
    architecture: str
    active_parameter_count: int | None
    supported_image_inputs: tuple[str, ...]
    tokenizer_revision: str
    processor_revision: str
    training_method: str
    dataset_manifest_ids: tuple[str, ...]
    checkpoint_uri: str | None
    deployment_status: str
    commercial_use_status: str
    attribution_requirements: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        required = (
            self.model_id,
            self.base_model_name,
            self.model_revision,
            self.license,
            self.model_type,
            self.architecture,
            self.tokenizer_revision,
            self.processor_revision,
            self.training_method,
            self.deployment_status,
            self.commercial_use_status,
        )
        if any(not item.strip() for item in required):
            raise ValueError("Model metadata fields cannot be blank.")
        if self.architecture not in {"dense", "moe", "unknown"}:
            raise ValueError("Architecture must be dense, moe, or unknown.")
        if self.deployment_status not in {"disabled", "evaluation", "approved"}:
            raise ValueError("Invalid deployment status.")
        if self.schema_version != 1:
            raise ValueError("Unsupported model metadata schema.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_id=str(value["model_id"]),
            base_model_name=str(value["base_model_name"]),
            model_revision=str(value["model_revision"]),
            license=str(value["license"]),
            model_type=str(value["model_type"]),
            parameter_count=value.get("parameter_count"),
            architecture=str(value.get("architecture", "unknown")),
            active_parameter_count=value.get("active_parameter_count"),
            supported_image_inputs=tuple(value.get("supported_image_inputs", [])),
            tokenizer_revision=str(value["tokenizer_revision"]),
            processor_revision=str(value["processor_revision"]),
            training_method=str(value["training_method"]),
            dataset_manifest_ids=tuple(value.get("dataset_manifest_ids", [])),
            checkpoint_uri=value.get("checkpoint_uri"),
            deployment_status=str(value["deployment_status"]),
            commercial_use_status=str(value["commercial_use_status"]),
            attribution_requirements=str(value.get("attribution_requirements", "")),
            schema_version=int(value.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "supported_image_inputs": list(self.supported_image_inputs),
            "dataset_manifest_ids": list(self.dataset_manifest_ids),
        }
