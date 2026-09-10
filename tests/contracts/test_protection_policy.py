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
