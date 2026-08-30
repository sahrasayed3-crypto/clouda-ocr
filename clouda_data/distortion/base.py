from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .metadata import DistortionMetadata


@dataclass(frozen=True)
class DistortionSpec:
    name: str
    version: str = "0.1.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    severity: str = "medium"
    input_requirements: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Distortion name is required.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"{self.name}.probability must be between 0 and 1.")
        if self.severity not in {
            "none",
            "minimal",
            "light",
            "medium",
            "heavy",
            "extreme",
        }:
            raise ValueError(f"{self.name}.severity is not supported.")


class Distortion(ABC):
    spec: DistortionSpec

    def __init__(self, spec: DistortionSpec) -> None:
        spec.validate()
        self.spec = spec

    @abstractmethod
    def apply(
        self, image: Any, seed: int, context: dict[str, Any] | None = None
    ) -> tuple[Any, DistortionMetadata]:
        """Apply a distortion to an in-memory page object."""

    def validate_input(self, image: Any) -> None:
        if image is None:
            raise ValueError(f"{self.spec.name} received no image object.")


class MetadataOnlyDistortion(Distortion):
    def apply(
        self, image: Any, seed: int, context: dict[str, Any] | None = None
    ) -> tuple[Any, DistortionMetadata]:
        self.validate_input(image)
        del context
        metadata = DistortionMetadata(
            name=self.spec.name,
            version=self.spec.version,
            probability=self.spec.probability,
            severity=self.spec.severity,
            parameters=self.spec.parameters,
            random_seed=seed,
            input_requirements=self.spec.input_requirements,
            output_metadata={"mode": "metadata_only", "text_reference_changed": False},
        )
        return image, metadata
