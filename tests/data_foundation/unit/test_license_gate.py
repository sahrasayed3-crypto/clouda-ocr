from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_data.datasets.license_gate import (
    decide_dataset_use,
    load_catalog,
    require_dataset_use,
)

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "dataset_catalog" / "registry" / "datasets_v1.json"


def test_catalog_schema_and_unique_ids() -> None:
    catalog = load_catalog(CATALOG)
    identifiers = [record["dataset_id"] for record in catalog["datasets"]]
    assert len(identifiers) == 13
    assert len(identifiers) == len(set(identifiers))


def test_approved_with_conditions_dataset_is_explicitly_allowed() -> None:
    decision = decide_dataset_use(
        "rasam_dataset",
        purpose="commercial_training",
        catalog_path=CATALOG,
    )
    assert decision.allowed is True
    assert decision.status == "approved_with_conditions"


def test_pending_and_research_only_datasets_fail_closed() -> None:
    pending = decide_dataset_use(
        "pats_a01",
        purpose="commercial_training",
        catalog_path=CATALOG,
    )
    research = decide_dataset_use(
        "openiti_makhzan",
        purpose="commercial_training",
        catalog_path=CATALOG,
    )
    assert pending.allowed is False
    assert research.allowed is False
    with pytest.raises(PermissionError):
        require_dataset_use(
            "pats_a01",
            purpose="commercial_training",
            catalog_path=CATALOG,
        )


def test_invalid_record_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    source = json.loads(CATALOG.read_text(encoding="utf-8"))
    source["datasets"].append(dict(source["datasets"][0]))
    registry = tmp_path / "registry"
    schemas = tmp_path / "schemas"
    registry.mkdir()
    schemas.mkdir()
    (schemas / "dataset-record-v1.schema.json").write_text(
        (
            ROOT / "dataset_catalog" / "schemas" / "dataset-record-v1.schema.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    catalog_path = registry / "datasets_v1.json"
    catalog_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate dataset id"):
        load_catalog(catalog_path)


def test_unsupported_purpose_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        decide_dataset_use(
            "rasam_dataset",
            purpose="unspecified",
            catalog_path=CATALOG,
        )
