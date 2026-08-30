from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from clouda_data.locations import default_catalog_path

ALLOWED_STATUSES = {
    "approved",
    "approved_with_conditions",
    "research_only",
    "evaluation_only",
    "pending",
    "blocked",
    "expired",
}
COMMERCIAL_TRAINING_STATUSES = {"approved", "approved_with_conditions"}


@dataclass(frozen=True)
class LicenseDecision:
    dataset_id: str
    allowed: bool
    purpose: str
    status: str
    reasons: tuple[str, ...]


def _schema_path(catalog_path: Path) -> Path:
    return catalog_path.parent.parent / "schemas" / "dataset-record-v1.schema.json"


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path is not None else default_catalog_path()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported dataset catalog schema version.")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Dataset catalog must contain a datasets list.")
    schema_path = _schema_path(catalog_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    seen: set[str] = set()
    for dataset in datasets:
        errors = sorted(
            validator.iter_errors(dataset), key=lambda item: list(item.path)
        )
        if errors:
            details = "; ".join(error.message for error in errors)
            raise ValueError(f"Invalid dataset record: {details}")
        dataset_id = str(dataset["dataset_id"])
        if dataset_id in seen:
            raise ValueError(f"Duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        status = dataset.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid dataset status: {status}")
    return payload


def get_dataset(dataset_id: str, path: str | Path | None = None) -> dict[str, Any]:
    for dataset in load_catalog(path)["datasets"]:
        if dataset.get("dataset_id") == dataset_id:
            return dataset
    raise KeyError(f"Unknown dataset: {dataset_id}")


def decide_dataset_use(
    dataset_id: str,
    *,
    purpose: str,
    catalog_path: str | Path | None = None,
) -> LicenseDecision:
    dataset = get_dataset(dataset_id, catalog_path)
    status = str(dataset["status"])
    reasons: list[str] = []
    if purpose == "commercial_training":
        allowed = status in COMMERCIAL_TRAINING_STATUSES and bool(
            dataset.get("commercial_training_allowed")
        )
        if not allowed:
            reasons.append("commercial_training_not_approved")
    elif purpose == "evaluation":
        allowed = bool(dataset.get("evaluation_allowed")) and status not in {
            "pending",
            "blocked",
            "expired",
        }
        if not allowed:
            reasons.append("evaluation_not_approved")
    elif purpose == "redistribution":
        allowed = status in COMMERCIAL_TRAINING_STATUSES and bool(
            dataset.get("original_data_redistribution_allowed")
        )
        if not allowed:
            reasons.append("redistribution_not_approved")
    else:
        raise ValueError(f"Unsupported dataset-use purpose: {purpose}")
    return LicenseDecision(
        dataset_id=dataset_id,
        allowed=allowed,
        purpose=purpose,
        status=status,
        reasons=tuple(reasons),
    )


def require_dataset_use(
    dataset_id: str,
    *,
    purpose: str,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    decision = decide_dataset_use(
        dataset_id, purpose=purpose, catalog_path=catalog_path
    )
    if not decision.allowed:
        raise PermissionError(
            f"Dataset {dataset_id} is blocked for {purpose}: "
            f"{', '.join(decision.reasons)}"
        )
    return get_dataset(dataset_id, catalog_path)
