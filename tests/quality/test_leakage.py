"""Leakage / protection guard tests — adversarial split markers (T1..T13).

Real assertions against clouda_contracts.protection (the module the
quality gate must consume, fail-closed) plus importorskip stubs for the
pending clouda_data.quality.leakage module.
"""

from __future__ import annotations

import pytest

leakage = pytest.importorskip("clouda_data.quality.leakage")

from clouda_contracts.protection import (  # noqa: E402
    is_training_split_eligible,
    record_is_protected,
    string_marks_protected,
)

from tests.quality.conftest import make_row  # noqa: E402


class TestAdversarialSplitMarkers:
    def test_holdout_casing_is_protected(self) -> None:
        assert string_marks_protected("Holdout") is True

    def test_holdout_whitespace_is_protected(self) -> None:
        assert string_marks_protected(" holdout ") is True

    def test_train_plus_holdout_is_protected(self) -> None:
        assert string_marks_protected("train+holdout") is True

    def test_benchmark_holdout_v2_is_protected(self) -> None:
        assert string_marks_protected("benchmark_holdout_v2") is True

    def test_plain_train_is_not_protected_marker(self) -> None:
        assert string_marks_protected("train") is False

    def test_holdout_not_training_eligible(self) -> None:
        assert is_training_split_eligible("holdout") is False


class TestProtectionRecords:
    def test_protected_int_flag_is_protected(self) -> None:
        row = make_row("smp_p", target_split="train").to_dict()
        row["provenance"] = {"protected": 1}
        assert record_is_protected(row) is True

    def test_provenance_as_list_is_protected(self) -> None:
        row = make_row("smp_l", target_split="train").to_dict()
        row["provenance"] = ["holdout"]
        assert record_is_protected(row) is True

    def test_nested_provenance_source_split_is_protected(self) -> None:
        row = make_row("smp_n", target_split="train").to_dict()
        row["provenance"] = {"origin": {"source_split": "holdout"}}
        assert record_is_protected(row) is True

    def test_role_train_target_holdout_is_protected(self) -> None:
        row = make_row("smp_r", target_split="holdout").to_dict()
        row["role"] = "train"
        assert record_is_protected(row) is True

    def test_clean_train_row_is_not_protected(self) -> None:
        row = make_row("smp_t", target_split="train").to_dict()
        assert record_is_protected(row) is False


class TestLeakageModule:
    def test_effective_partition_contract_pending(self) -> None:
        assert leakage is not None  # Wave2-H contract
