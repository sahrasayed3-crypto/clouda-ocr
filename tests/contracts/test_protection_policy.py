from __future__ import annotations

import pytest

from clouda_contracts.protection import (
    is_training_split_eligible,
    record_is_protected,
)


@pytest.mark.parametrize(
    "record",
    [
        {"split": "holdout"},
        {"split": " HOLDOUT "},
        {"target_split": "train+holdout"},
        {"protected": True},
        {"protected": " yes "},
        {"dataset_role": "benchmark_holdout_v2"},
        {"role": "evaluation_only"},
        {"purpose": "protected"},
        {"metadata": {"protected": True}},
        {"metadata": {"canonical_manifest_row": {"protected": True}}},
        {"metadata": {"records": [{"protected": True}]}},
        {"provenance": {"source_split": "private_holdout"}},
        {"target_split": "train", "split": "holdout"},
        {"target_split": ["train"]},
        {"protected": 1},
        {"protected": "maybe"},
        {"metadata": "not-a-mapping"},
        {"provenance": {"role": {"name": "train"}}},
    ],
)
def test_protection_policy_fails_closed_for_bypass_shapes(record) -> None:
    assert record_is_protected(record) is True


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"target_split": "train"}, False),
        ({"split": " train ", "protected": "false"}, False),
        ({"split": "validation"}, False),
        ({"metadata": {"role": "training"}}, False),
        ({}, False),
    ],
)
def test_protection_policy_does_not_invent_markers(record, expected) -> None:
    assert record_is_protected(record) is expected


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ("train", True),
        (" TRAIN ", True),
        ("validation", False),
        ("test", False),
        ("unassigned", False),
        ("", False),
        (None, False),
        ("train+holdout", False),
        (["train"], False),
    ],
)
def test_training_eligibility_requires_an_explicit_train_split(split, expected) -> None:
    assert is_training_split_eligible(split) is expected


# ---------------------------------------------------------------------------
# Path-independence of holdout protection (regression proof)
#
# The canonical policy (clouda_contracts.protection) inspects record content
# and metadata only — it never receives or considers a filesystem path. These
# tests pin that property end to end through the canonical training gate
# (validate_training_dataset): an innocuous FILE NAME must not smuggle
# protected rows into training, and a scary FILE NAME must not block clean
# rows. Renames/moves cannot change the verdict because the verdict is a
# function of row content (and the content hash), not of the location.
# ---------------------------------------------------------------------------

from clouda_training.experiments.config import DatasetSection  # noqa: E402
from clouda_training.experiments.dataset import (  # noqa: E402
    validate_training_dataset,
)

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _dataset_config(manifest: Path) -> DatasetSection:
    return DatasetSection(
        dataset_id="synthetic-test",
        dataset_version="v1",
        manifest_path=manifest,
        split="train",
    )


_CLEAN_TRAIN_ROW = {
    "sample_id": "s-1",
    "target_split": "train",
    "source_id": "synthetic-test",
    "source_license": "Apache-2.0",
}

_HEADER = {
    "_schema_version": "clouda.pretraining.manifest.v1",
    "_row_count": 1,
    "dataset_id": "synthetic-test",
    "dataset_version": "v1",
}


def test_holdout_rows_blocked_despite_innocuous_file_name(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "totally_normal_train_data.jsonl",
        [
            dict(_HEADER),
            dict(_CLEAN_TRAIN_ROW),
            {"sample_id": "s-2", "target_split": "train", "source_split": "holdout"},
        ],
    )
    with pytest.raises(PermissionError, match="holdout"):
        validate_training_dataset(_dataset_config(manifest))


def test_clean_rows_allowed_despite_scary_file_name(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "BENCHMARK_HOLDOUT_DO_NOT_TRAIN.jsonl",
        [dict(_HEADER), dict(_CLEAN_TRAIN_ROW)],
    )
    identity = validate_training_dataset(_dataset_config(manifest))
    assert identity.row_count == 1


def test_rename_or_move_never_changes_protection_verdict(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        dict(_HEADER),
        dict(_CLEAN_TRAIN_ROW),
        {"sample_id": "s-2", "target_split": "train", "metadata": {"protected": True}},
    ]
    original = _write_manifest(tmp_path / "batch_a.jsonl", rows)
    with pytest.raises(PermissionError, match="protected"):
        validate_training_dataset(_dataset_config(original))

    # rename within the same directory and move to a different directory:
    # identical bytes, identical verdict — the block follows the content.
    renamed = tmp_path / "renamed_innocuous.jsonl"
    original.rename(renamed)
    with pytest.raises(PermissionError, match="protected"):
        validate_training_dataset(_dataset_config(renamed))

    moved_dir = tmp_path / "elsewhere"
    moved_dir.mkdir()
    moved = moved_dir / renamed.name
    renamed.rename(moved)
    with pytest.raises(PermissionError, match="protected"):
        validate_training_dataset(_dataset_config(moved))


def test_manifest_hash_is_content_identity_not_path_identity(
    tmp_path: Path,
) -> None:
    from clouda_contracts.checksums import sha256_file

    rows: list[dict[str, Any]] = [dict(_HEADER), dict(_CLEAN_TRAIN_ROW)]
    a = _write_manifest(tmp_path / "one.jsonl", rows)
    b = tmp_path / "sub"
    b.mkdir()
    c = _write_manifest(b / "two.jsonl", rows)
    assert sha256_file(a) == sha256_file(c)
    identity_a = validate_training_dataset(_dataset_config(a))
    identity_c = validate_training_dataset(_dataset_config(c))
    assert identity_a.manifest_hash == identity_c.manifest_hash
