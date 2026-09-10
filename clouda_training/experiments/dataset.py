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


def _marker(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _mapping_is_protected(value: object, *, nested: bool = False) -> bool:
    if not isinstance(value, dict):
        if nested:
            raise ValueError("Dataset row has malformed protection metadata")
        return False
    protected = value.get("protected")
    if "protected" in value and not (
        isinstance(protected, bool)
        or (
            isinstance(protected, str)
            and protected.strip().casefold()
            in {"true", "yes", "1", "protected", "false", "no", "0"}
        )
    ):
        raise ValueError("Dataset row has malformed protection metadata")
    if protected is True or (
        isinstance(protected, str)
        and protected.strip().casefold() in {"true", "yes", "1", "protected"}
    ):
        return True
    for field in ("target_split", "split", "source_split"):
        if (
            field in value
            and value[field] is not None
            and not isinstance(value[field], str)
        ):
            raise ValueError("Dataset row has malformed protection metadata")
        marker = _marker(value.get(field))
        if marker in PROTECTED_SPLIT_NAMES or "holdout" in marker:
            return True
    for field in ("dataset_role", "role", "purpose"):
        if (
            field in value
            and value[field] is not None
            and not isinstance(value[field], str)
        ):
            raise ValueError("Dataset row has malformed protection metadata")
        marker = _marker(value.get(field))
        if marker in PROTECTED_ROLES or "holdout" in marker:
            return True
    return False


def _row_is_protected(row: dict) -> bool:
    if _mapping_is_protected(row):
        return True
    return any(
        _mapping_is_protected(row[field], nested=True)
        for field in ("provenance", "metadata")
        if field in row and row[field] is not None
    )


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
    if _mapping_is_protected(header):
        raise PermissionError(
            "Dataset manifest is marked as protected and cannot train"
        )
    if any(_row_is_protected(row) for row in rows):
        raise PermissionError(
            "Dataset manifest contains protected holdout rows and cannot train"
        )
    selected = [
        row
        for row in rows
        if _marker(row.get("target_split", row.get("split", split))) == split
    ]
    if not selected:
        raise ValueError(f"Dataset manifest contains no rows for split {split!r}")
    selected_splits = {
        _marker(row.get("target_split", row.get("split", split))) for row in selected
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
