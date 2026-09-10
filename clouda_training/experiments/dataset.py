from __future__ import annotations

from dataclasses import dataclass

from clouda_contracts.checksums import sha256_file
from clouda_contracts.protection import (
    PROTECTED_SPLIT_NAMES,
    normalize_marker,
    protection_metadata_is_malformed,
    record_is_protected,
)
from clouda_data.pretraining.manifest import read_manifest

from .config import DatasetSection


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
    for field, configured in (
        ("dataset_id", dataset.dataset_id),
        ("dataset_version", dataset.dataset_version),
    ):
        recorded = header.get(field)
        if recorded is not None and str(recorded).strip() != configured.strip():
            raise ValueError(
                f"Manifest {field} {recorded!r} does not match configured "
                f"value {configured!r}"
            )
    for row in rows:
        for field in ("target_split", "split", "source_split"):
            if (
                field in row
                and row[field] is not None
                and not isinstance(row[field], str)
            ):
                raise ValueError(f"Dataset row has malformed split metadata in {field}")
    if protection_metadata_is_malformed(header) or any(
        protection_metadata_is_malformed(row) for row in rows
    ):
        raise ValueError("Dataset row has malformed protection metadata")
    if record_is_protected(header):
        raise PermissionError(
            "Dataset manifest is marked as protected and cannot train"
        )
    if any(record_is_protected(row) for row in rows):
        raise PermissionError(
            "Dataset manifest contains protected holdout rows and cannot train"
        )
    selected = [
        row
        for row in rows
        if normalize_marker(row.get("target_split", row.get("split", split))) == split
    ]
    if not selected:
        raise ValueError(f"Dataset manifest contains no rows for split {split!r}")
    selected_splits = {
        normalize_marker(row.get("target_split", row.get("split", split)))
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
