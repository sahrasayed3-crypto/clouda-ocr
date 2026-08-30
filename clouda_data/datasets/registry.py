from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clouda_data.locations import default_foundation_registry_path

APPROVED_STATUSES = {"approved", "approved_with_conditions"}
BLOCKED_STATUSES = {"research_only", "unclear_license", "rejected"}
COMMERCIAL_LICENSES = {"apache-2.0", "mit", "cc0-1.0", "cc-by-4.0"}


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    name: str
    classification: str
    license: str
    commercial_use_status: str
    total_size_bytes: int | None
    sample_size_bytes: int | None
    download_method: str
    sample_assets: list[dict[str, Any]]
    metadata: dict[str, Any]


def load_dataset_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = (
        Path(path) if path is not None else default_foundation_registry_path()
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if "sources" not in payload or not isinstance(payload["sources"], list):
        raise ValueError("Dataset registry must contain a sources list.")
    return payload


def list_sources(path: str | Path | None = None) -> list[dict[str, Any]]:
    return load_dataset_registry(path)["sources"]


def get_source(source_id: str, path: str | Path | None = None) -> dict[str, Any]:
    for source in list_sources(path):
        if source["source_id"] == source_id:
            return source
    raise KeyError(f"Unknown dataset source: {source_id}")


def verify_license(source: dict[str, Any]) -> dict[str, Any]:
    classification = source.get("classification")
    license_id = str(source.get("license", "")).lower()
    explicit = bool(source.get("license_verified", False))
    commercial_ok = (
        source.get("commercial_use_status") == "allowed"
        and license_id in COMMERCIAL_LICENSES
    )
    allowed_for_sample = (
        classification in APPROVED_STATUSES
        and explicit
        and source.get("commercial_use_status")
        in {"allowed", "allowed_with_attribution"}
    )
    return {
        "source_id": source["source_id"],
        "classification": classification,
        "license": source.get("license"),
        "license_verified": explicit,
        "commercial_use_allowed": commercial_ok,
        "sample_download_allowed": allowed_for_sample,
        "reasons": source.get("license_notes", []),
    }


def estimate_source_download(
    source: dict[str, Any], *, sample_only: bool = True
) -> dict[str, Any]:
    size = (
        source.get("sample_size_bytes")
        if sample_only
        else source.get("dataset_size_bytes")
    )
    return {
        "source_id": source["source_id"],
        "sample_only": sample_only,
        "estimated_bytes": size,
        "estimated_mb": round((size or 0) / (1024 * 1024), 3),
        "download_method": source.get("download_method"),
        "classification": source.get("classification"),
    }


def assert_source_download_allowed(
    source: dict[str, Any], *, full_dataset: bool = False, max_bytes: int
) -> None:
    license_result = verify_license(source)
    if full_dataset:
        raise PermissionError("Full dataset downloads are disabled in this stage.")
    if not license_result["sample_download_allowed"]:
        raise PermissionError(
            f"Sample download blocked by license classification: {source.get('classification')}"
        )
    if (
        source.get("requires_authentication")
        or source.get("requires_form")
        or source.get("requires_account")
    ):
        raise PermissionError(
            "This source requires authentication, account creation, or form/terms acceptance."
        )
    sample_size = source.get("sample_size_bytes")
    if sample_size is not None and sample_size > max_bytes:
        raise PermissionError("Estimated sample size exceeds requested byte limit.")
