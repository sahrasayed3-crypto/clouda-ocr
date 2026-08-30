from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    plan_id: str
    task: str
    dataset_ids: tuple[str, ...]
    page_limit: int | None
    seed: int
    split_ratios: tuple[float, float, float]
    base_model: str
    training_method: str
    augmentation_profiles: tuple[str, ...] = ()
    curriculum: tuple[str, ...] = ("clean", "light", "mixed")
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.plan_id or not self.task or not self.base_model:
            raise ValueError("Training configuration identifiers cannot be blank.")
        if not self.dataset_ids:
            raise ValueError("At least one dataset is required.")
        if self.page_limit is not None and self.page_limit < 1:
            raise ValueError("Page limit must be positive.")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to one.")
        if any(ratio < 0 for ratio in self.split_ratios):
            raise ValueError("Split ratios cannot be negative.")
        if self.schema_version != 1:
            raise ValueError("Unsupported training config schema.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        return cls(
            plan_id=str(value["plan_id"]),
            task=str(value["task"]),
            dataset_ids=tuple(value["dataset_ids"]),
            page_limit=value.get("page_limit"),
            seed=int(value.get("seed", 20260723)),
            split_ratios=tuple(value.get("split_ratios", [0.8, 0.1, 0.1])),
            base_model=str(value.get("base_model", "provider/model@revision")),
            training_method=str(value.get("training_method", "adapter")),
            augmentation_profiles=tuple(value.get("augmentation_profiles", [])),
            curriculum=tuple(value.get("curriculum", ["clean", "light", "mixed"])),
            schema_version=int(value.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "task": self.task,
            "dataset_ids": list(self.dataset_ids),
            "page_limit": self.page_limit,
            "seed": self.seed,
            "split_ratios": list(self.split_ratios),
            "base_model": self.base_model,
            "training_method": self.training_method,
            "augmentation_profiles": list(self.augmentation_profiles),
            "curriculum": list(self.curriculum),
        }


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
