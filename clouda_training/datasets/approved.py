from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clouda_contracts.storage import StorageRoots
from clouda_data.datasets.license_gate import require_dataset_use


@dataclass(frozen=True)
class ApprovedDataset:
    dataset_id: str
    root: Path
    status: str
    attribution_text: str


def load_approved_datasets(
    dataset_ids: tuple[str, ...],
    *,
    roots: StorageRoots,
    catalog_path: str | Path,
) -> tuple[ApprovedDataset, ...]:
    approved: list[ApprovedDataset] = []
    for dataset_id in dataset_ids:
        record = require_dataset_use(
            dataset_id,
            purpose="commercial_training",
            catalog_path=catalog_path,
        )
        approved.append(
            ApprovedDataset(
                dataset_id=dataset_id,
                root=roots.resolve_uri(record["data_root_uri"]),
                status=str(record["status"]),
                attribution_text=str(record["attribution_text"]),
            )
        )
    return tuple(approved)
