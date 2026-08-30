from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clouda_contracts.storage import StorageRoots
from clouda_training.config.models import TrainingConfig
from clouda_training.datasets.approved import load_approved_datasets


@dataclass(frozen=True)
class TrainingPlan:
    plan_id: str
    task: str
    dry_run: bool
    datasets: tuple[dict[str, Any], ...]
    estimated_examples: int
    estimated_storage_bytes: int
    seed: int
    split_ratios: tuple[float, float, float]
    training_enabled: bool = False
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "task": self.task,
            "dry_run": self.dry_run,
            "training_enabled": self.training_enabled,
            "datasets": list(self.datasets),
            "estimated_examples": self.estimated_examples,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "seed": self.seed,
            "split_ratios": list(self.split_ratios),
        }


def _local_inventory(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        return 0, 0
    files = [path for path in root.rglob("*") if path.is_file()]
    image_count = sum(
        path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
        for path in files
    )
    return image_count, sum(path.stat().st_size for path in files)


def plan_training(
    config: TrainingConfig,
    *,
    roots: StorageRoots,
    catalog_path: str | Path,
) -> TrainingPlan:
    datasets = load_approved_datasets(
        config.dataset_ids,
        roots=roots,
        catalog_path=catalog_path,
    )
    records: list[dict[str, Any]] = []
    available_examples = 0
    storage_bytes = 0
    for dataset in datasets:
        examples, size = _local_inventory(dataset.root)
        available_examples += examples
        storage_bytes += size
        records.append(
            {
                "dataset_id": dataset.dataset_id,
                "status": dataset.status,
                "root_uri": f"dataset://data/downloads/{dataset.dataset_id}/",
                "available_examples": examples,
                "available_bytes": size,
                "attribution_required": bool(dataset.attribution_text),
            }
        )
    estimate = (
        min(available_examples, config.page_limit)
        if config.page_limit is not None
        else available_examples
    )
    return TrainingPlan(
        plan_id=config.plan_id,
        task=config.task,
        dry_run=True,
        datasets=tuple(records),
        estimated_examples=estimate,
        estimated_storage_bytes=storage_bytes,
        seed=config.seed,
        split_ratios=config.split_ratios,
    )
