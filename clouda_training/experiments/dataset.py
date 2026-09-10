from __future__ import annotations

from dataclasses import dataclass

from clouda_contracts.checksums import sha256_file
from clouda_contracts.protection import (
    PROTECTED_SPLIT_NAMES,
    normalize_marker,
    protection_metadata_is_malformed,
    record_is_protected,
)
from clouda_data.pretraining.manifest import iter_manifest

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
    header: dict = {}
    row_count = 0
    selected_count = 0
    source_ids: set[str] = set()
    source_licenses: set[str] = set()
    for line_index, row in enumerate(iter_manifest(manifest), start=1):
        if "_schema_version" in row and "sample_id" not in row:
            if header or line_index != 1:
                raise ValueError("Manifest must contain exactly one leading header.")
            header = row
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
            if protection_metadata_is_malformed(header):
                raise ValueError("Dataset row has malformed protection metadata")
            if record_is_protected(header):
                raise PermissionError(
                    "Dataset manifest is marked as protected and cannot train"
                )
            continue
        row_count += 1
        for field in ("target_split", "split", "source_split"):
            if (
                field in row
                and row[field] is not None
                and not isinstance(row[field], str)
            ):
                raise ValueError(f"Dataset row has malformed split metadata in {field}")
        if protection_metadata_is_malformed(row):
            raise ValueError("Dataset row has malformed protection metadata")
        if record_is_protected(row):
            raise PermissionError(
                "Dataset manifest contains protected holdout rows and cannot train"
            )
        row_split = normalize_marker(row.get("target_split", row.get("split", split)))
        if row_split != split:
            continue
        selected_count += 1
        source_id = row.get("source_id")
        if source_id:
            source_ids.add(str(source_id))
        source_license = row.get("source_license")
        if source_license:
            source_licenses.add(str(source_license))
    recorded_count = header.get("_row_count")
    if recorded_count is not None and recorded_count != row_count:
        raise ValueError(
            f"Manifest row count mismatch: header says {recorded_count}, "
            f"found {row_count}."
        )
    if not selected_count:
        raise ValueError(f"Dataset manifest contains no rows for split {split!r}")
    return DatasetIdentity(
        manifest_hash=sha256_file(manifest),
        row_count=selected_count,
        source_ids=tuple(sorted(source_ids)),
        source_licenses=tuple(sorted(source_licenses)),
    )
