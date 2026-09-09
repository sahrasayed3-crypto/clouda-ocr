from __future__ import annotations

from dataclasses import dataclass

from clouda_contracts.checksums import sha256_file
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.pretraining.schema import SplitName

from .config import DatasetSection

PROTECTED_SPLIT_NAMES = {
    SplitName.HOLDOUT.value,
    "protected_holdout",
    "benchmark_holdout",
    "private_holdout",
}
PROTECTED_ROLES = {"holdout", "protected_holdout", "benchmark", "evaluation_only"}


@dataclass(frozen=True)
class DatasetIdentity:
    manifest_hash: str
    row_count: int
    source_ids: tuple[str, ...]
    source_licenses: tuple[str, ...]


def validate_training_dataset(dataset: DatasetSection) -> DatasetIdentity:
    split = dataset.split.strip().lower()
    if split in PROTECTED_SPLIT_NAMES or "holdout" in split:
        raise PermissionError(
            f"Protected holdout split cannot be used for training: {split}"
        )
    manifest = dataset.manifest_path
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest}")
    header, rows = read_manifest(manifest)
    role = str(
        header.get("dataset_role", header.get("role", header.get("purpose", "")))
    ).lower()
    if header.get("protected") is True or role in PROTECTED_ROLES or "holdout" in role:
        raise PermissionError(
            "Dataset manifest is marked as protected and cannot train"
        )
    selected = [
        row
        for row in rows
        if str(row.get("target_split", row.get("split", split))).lower() == split
    ]
    if not selected:
        raise ValueError(f"Dataset manifest contains no rows for split {split!r}")
    selected_splits = {
        str(row.get("target_split", row.get("split", split))).lower()
        for row in selected
    }
    if selected_splits & PROTECTED_SPLIT_NAMES:
        raise PermissionError("Protected holdout rows cannot be used for training")
    if any(row.get("protected") is True for row in selected):
        raise PermissionError("Protected dataset rows cannot be used for training")
    return DatasetIdentity(
        manifest_hash=sha256_file(manifest),
        row_count=len(selected),
        source_ids=tuple(
            sorted(
                {str(row.get("source_id")) for row in selected if row.get("source_id")}
            )
        ),
        source_licenses=tuple(
            sorted(
                {
                    str(row.get("source_license"))
                    for row in selected
                    if row.get("source_license")
                }
            )
        ),
    )
