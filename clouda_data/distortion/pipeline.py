from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import DistortionSpec, MetadataOnlyDistortion
from .metadata import DistortionMetadata
from .randomness import derive_seed
from .randomness import seeded_rng
from .registry import DistortionRegistry, default_registry


@dataclass(frozen=True)
class DistortionPipeline:
    profile_name: str
    operations: list[DistortionSpec]
    base_seed: int
    registry: DistortionRegistry = field(default_factory=default_registry)

    def validate(self) -> None:
        if len(self.operations) > 32:
            raise ValueError("Distortion chains are limited to 32 operations.")
        for operation in self.operations:
            operation.validate()

    def replay_plan(self, page_id: str) -> list[tuple[DistortionSpec, int]]:
        return [
            (
                operation,
                derive_seed(self.base_seed, page_id, operation.name, str(index)),
            )
            for index, operation in enumerate(self.operations)
        ]

    def run_metadata_only(
        self, image: Any, page_id: str
    ) -> tuple[Any, list[DistortionMetadata]]:
        self.validate()
        current = image
        metadata: list[DistortionMetadata] = []
        for operation, seed in self.replay_plan(page_id):
            current, op_metadata = MetadataOnlyDistortion(operation).apply(
                current, seed
            )
            metadata.append(op_metadata)
        return current, metadata

    def run(
        self,
        image: Any,
        page_id: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[Any, list[DistortionMetadata]]:
        """Apply a deterministic real-pixel chain.

        Probability decisions and operator seeds are derived independently so
        inserting a skipped operator cannot perturb later operator randomness.
        """
        self.validate()
        current = image.copy()
        metadata: list[DistortionMetadata] = []
        for index, (operation, seed) in enumerate(self.replay_plan(page_id)):
            decision = seeded_rng(self.base_seed, page_id, operation.name, str(index))
            if decision.random() > operation.probability:
                continue
            current, op_metadata = self.registry.create(operation).apply(
                current, seed, context
            )
            metadata.append(op_metadata)
        if self.operations and not metadata:
            operation, seed = self.replay_plan(page_id)[0]
            current, op_metadata = self.registry.create(operation).apply(
                current, seed, context
            )
            metadata.append(op_metadata)
        return current, metadata
