from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_contracts.storage import StorageRoots
from clouda_training.config.models import TrainingConfig, load_training_config
from clouda_training.planner import plan_training
from clouda_training.sampling.deduplication import deduplicate_records
from clouda_training.sampling.splits import (
    deterministic_document_split,
    deterministic_sample,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "dataset_catalog" / "registry" / "datasets_v1.json"


def _roots(tmp_path: Path) -> StorageRoots:
    return StorageRoots.from_env(
        {"CLOUDA_STATE_HOME": str(tmp_path)},
        read_only=False,
        create=True,
    )


def test_all_training_templates_validate() -> None:
    templates = list((ROOT / "configs" / "training").glob("*.json"))
    assert len(templates) == 6
    for path in templates:
        assert load_training_config(path).schema_version == 1


def test_split_is_document_level_and_deterministic() -> None:
    documents = ["doc-a", "doc-b", "doc-a", "doc-c"]
    first = deterministic_document_split(documents, seed=7)
    second = deterministic_document_split(reversed(documents), seed=7)
    assert first == second
    assert len(first) == 3


def test_exact_hash_deduplication() -> None:
    records = [
        {"image_checksum": "a"},
        {"image_checksum": "a"},
        {"image_checksum": "b"},
    ]
    kept, rejected = deduplicate_records(records)
    assert len(kept) == 2
    assert len(rejected) == 1


def test_sampling_is_deterministic() -> None:
    records = ["p3", "p1", "p2", "p1"]
    assert deterministic_sample(records, limit=2, seed=9) == deterministic_sample(
        reversed(records), limit=2, seed=9
    )


def test_planner_is_dry_run_and_counts_local_images(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    local = roots.resolve_uri("dataset://data/downloads/rasam_dataset/")
    local.mkdir(parents=True)
    (local / "one.jpg").write_bytes(b"one")
    config = TrainingConfig(
        plan_id="test",
        task="text_ocr",
        dataset_ids=("rasam_dataset",),
        page_limit=100,
        seed=1,
        split_ratios=(0.8, 0.1, 0.1),
        base_model="provider/model@revision",
        training_method="adapter",
    )
    result = plan_training(config, roots=roots, catalog_path=CATALOG)
    assert result.dry_run is True
    assert result.training_enabled is False
    assert result.estimated_examples == 1


def test_pending_dataset_is_blocked(tmp_path: Path) -> None:
    config = TrainingConfig(
        plan_id="blocked",
        task="text_ocr",
        dataset_ids=("pats_a01",),
        page_limit=1,
        seed=1,
        split_ratios=(0.8, 0.1, 0.1),
        base_model="provider/model@revision",
        training_method="adapter",
    )
    with pytest.raises(PermissionError):
        plan_training(config, roots=_roots(tmp_path), catalog_path=CATALOG)


def test_config_rejects_invalid_ratios(tmp_path: Path) -> None:
    config = {
        "plan_id": "bad",
        "task": "text_ocr",
        "dataset_ids": ["rasam_dataset"],
        "split_ratios": [0.5, 0.5, 0.5],
        "base_model": "provider/model@revision",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_config(path)
