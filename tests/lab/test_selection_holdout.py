"""Tests: Dataset Selection + Holdout Safety (Phases 4-5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_lab.dataset_selection import (
    SelectionCriteria,
    select_samples,
    validate_derived_manifest_for_training,
    write_selection_manifest,
)
from clouda_lab.holdout_guard import (
    assert_selection_safe,
    filter_protected_rows,
    row_is_protected,
)

SCHEMA = "clouda.pretraining.manifest.v1"


def _write_manifest(path: Path, rows: list[dict], header: dict | None = None) -> Path:
    lines = [json.dumps({"_schema_version": SCHEMA, "_row_count": len(rows),
                         **(header or {})}, sort_keys=True)]
    lines.extend(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(sample_id: str, **extra) -> dict:
    row = {
        "sample_id": sample_id,
        "target_split": "train",
        "source_id": "synthetic-source",
        "source_license": "Apache-2.0",
        "document_type": "book",
    }
    row.update(extra)
    return row


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    rows = [
        _row("s-001", document_type="book", cer=0.05, wer=0.10),
        _row("s-002", document_type="book", cer=0.30, wer=0.40),
        _row("s-003", document_type="form", profile="bad_scan_heavy",
             provenance={"profile": "bad_scan_heavy", "seed": 1}, cer=0.60, wer=0.70),
        _row("s-004", document_type="form", source_id="source-b", cer=0.45, wer=0.50),
        _row("s-005", document_type="book", cer=0.20, wer=0.25,
             metadata={"tags": ["hard", "arabic"]}),
        _row("s-006", document_type="book", target_split="validation", cer=0.15),
    ]
    return _write_manifest(tmp_path / "base.jsonl", rows)


class TestSelectionFilters:
    def test_explicit_ids(self, manifest: Path):
        result = select_samples(
            str(manifest), SelectionCriteria(sample_ids=("s-002", "s-004"))
        )
        assert result.sample_ids == ("s-002", "s-004")

    def test_filter_document_type(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(document_type="form"))
        assert set(result.sample_ids) == {"s-003", "s-004"}

    def test_filter_split(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(split="validation"))
        assert result.sample_ids == ("s-006",)

    def test_filter_split_train(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(split="train"))
        assert set(result.sample_ids) == {"s-001", "s-002", "s-003", "s-004", "s-005"}

    def test_filter_profile_from_provenance(self, manifest: Path):
        result = select_samples(
            str(manifest), SelectionCriteria(profile="bad_scan_heavy")
        )
        assert result.sample_ids == ("s-003",)

    def test_filter_cer_range(self, manifest: Path):
        result = select_samples(
            str(manifest), SelectionCriteria(cer_min=0.25, cer_max=0.50)
        )
        assert set(result.sample_ids) == {"s-002", "s-004"}

    def test_filter_source(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(source="source-b"))
        assert result.sample_ids == ("s-004",)

    def test_filter_tags(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(tags=("hard",)))
        assert result.sample_ids == ("s-005",)

    def test_filter_error_type_and_model_and_bucket(self, manifest: Path):
        rows = [
            _row("e-1", error_type="whitespace", model_id="m1", failure_bucket="high_cer"),
            _row("e-2", error_type="diacritic", model_id="m2", failure_bucket="high_wer"),
            _row("e-3", error_types={"whitespace": 3}, model_id="m1",
                 failure_bucket="whitespace_heavy"),
        ]
        path = _write_manifest(manifest.parent / "errors.jsonl", rows)
        assert select_samples(str(path), SelectionCriteria(error_type="whitespace")).sample_ids == ("e-1", "e-3")
        assert select_samples(str(path), SelectionCriteria(model_id="m2")).sample_ids == ("e-2",)
        assert select_samples(str(path), SelectionCriteria(failure_bucket="high_wer")).sample_ids == ("e-2",)


class TestSelectionSampling:
    def test_deterministic_random(self, manifest: Path):
        first = select_samples(str(manifest), SelectionCriteria(random_sample_size=3), seed=7)
        second = select_samples(str(manifest), SelectionCriteria(random_sample_size=3), seed=7)
        different = select_samples(str(manifest), SelectionCriteria(random_sample_size=3), seed=8)
        assert first.sample_ids == second.sample_ids
        assert len(first.sample_ids) == 3
        assert first.sample_ids != different.sample_ids

    def test_top_n_hardest_with_scores(self, manifest: Path):
        scores = {f"s-00{i}": float(i) / 10 for i in range(1, 7)}
        result = select_samples(
            str(manifest),
            SelectionCriteria(top_n="hardest", top_n_count=2),
            scores=scores,
        )
        assert set(result.sample_ids) == {"s-005", "s-006"}  # scores 0.5, 0.6

    def test_top_n_easiest(self, manifest: Path):
        scores = {f"s-00{i}": float(i) / 10 for i in range(1, 7)}
        result = select_samples(
            str(manifest), SelectionCriteria(top_n="easiest", top_n_count=2), scores=scores
        )
        assert set(result.sample_ids) == {"s-001", "s-002"}

    def test_percentile_range(self, manifest: Path):
        scores = {f"s-00{i}": float(i) / 10 for i in range(1, 7)}
        result = select_samples(
            str(manifest), SelectionCriteria(percentile_range=(80, 100)), scores=scores
        )
        assert set(result.sample_ids) == {"s-005", "s-006"}  # top quintile

    def test_limit(self, manifest: Path):
        result = select_samples(str(manifest), SelectionCriteria(limit=2))
        assert len(result.sample_ids) == 2

    def test_selection_id_deterministic(self, manifest: Path):
        first = select_samples(str(manifest), SelectionCriteria(limit=2), seed=5)
        second = select_samples(str(manifest), SelectionCriteria(limit=2), seed=5)
        assert first.selection_id == second.selection_id


class TestSelectionProvenance:
    def test_derived_manifest_header(self, manifest: Path, tmp_path: Path):
        result = select_samples(
            str(manifest),
            SelectionCriteria(document_type="form", cer_min=0.3),
            seed=42,
        )
        out = tmp_path / "derived.jsonl"
        header = write_selection_manifest(result, str(out))
        assert header["selection_id"] == result.selection_id
        assert header["selection_seed"] == 42
        assert header["source_manifest_sha256"]
        assert header["source_manifest"].endswith("base.jsonl")
        assert header["selection_criteria"]["document_type"] == "form"
        assert header["manifest_role"] == "lab_selection"
        # derived rows are exactly the selected ones
        loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()[1:]]
        assert {row["sample_id"] for row in loaded} == set(result.sample_ids)

    def test_source_manifest_hash_changes_on_content_change(self, manifest: Path, tmp_path: Path):
        first = select_samples(str(manifest), SelectionCriteria(limit=1))
        rows = manifest.read_text(encoding="utf-8").splitlines()
        header = json.loads(rows[0])
        header["_row_count"] += 1
        rows[0] = json.dumps(header, sort_keys=True)
        rows.append(json.dumps(_row("s-999"), sort_keys=True))
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        second = select_samples(str(manifest), SelectionCriteria(limit=1))
        assert first.source_manifest_sha256 != second.source_manifest_sha256
        assert first.selection_id != second.selection_id


class TestHoldoutGuard:
    def test_protected_split_row_rejected(self, tmp_path: Path):
        rows = [_row("ok-1"), _row("h-1", target_split="holdout"),
                _row("h-2", target_split="protected_holdout")]
        safe, count = filter_protected_rows(rows)
        assert count == 2
        assert [r["sample_id"] for r in safe] == ["ok-1"]

    def test_holdout_alias_splits_rejected(self):
        for split in ("holdout", "protected_holdout", "benchmark_holdout",
                      "private_holdout", "eval_holdout", "HOLDOUT", " Holdout "):
            assert row_is_protected(_row("x", target_split=split)), split

    def test_protected_flag_variants(self):
        assert row_is_protected(_row("x", protected=True))
        assert row_is_protected(_row("x", protected="true"))
        assert row_is_protected(_row("x", protected="yes"))
        assert row_is_protected(_row("x", protected="Protected"))
        assert not row_is_protected(_row("x", protected=False))
        assert not row_is_protected(_row("x", protected="false"))

    def test_protected_role_rejected(self):
        for role in ("holdout", "protected_holdout", "benchmark", "evaluation_only"):
            assert row_is_protected(_row("x", dataset_role=role)), role
            assert row_is_protected(_row("x", role=role)), role
            assert row_is_protected(_row("x", purpose=role)), role

    def test_protected_nested_metadata_rejected(self):
        assert row_is_protected(_row("x", metadata={"protected": True}))
        assert row_is_protected(_row("x", provenance={"target_split": "holdout"}))
        assert row_is_protected(_row("x", metadata={"role": "benchmark"}))

    def test_composite_marker_strings_rejected(self):
        assert row_is_protected(_row("x", target_split="train+holdout"))
        assert row_is_protected(_row("x", dataset_role="benchmark_holdout_v2"))
        assert row_is_protected(_row("x", purpose="evaluation_only_export"))

    def test_malformed_metadata_fails_closed(self):
        assert row_is_protected(_row("x", protected=1))
        assert row_is_protected(_row("x", protected=["true"]))
        assert row_is_protected(_row("x", target_split={"split": "train"}))
        assert row_is_protected(_row("x", metadata={"protected": 3.5}))
        assert row_is_protected(_row("x", metadata="not-a-mapping"))
        assert row_is_protected(_row("x", provenance=42))

    def test_clean_rows_pass(self):
        assert not row_is_protected(_row("x"))
        assert not row_is_protected(_row("x", metadata={"page": 3}, provenance={"seed": 7}))
        assert not row_is_protected(_row("x", target_split="train"))

    def test_selection_never_returns_protected(self, tmp_path: Path):
        rows = [_row("ok-1", cer=0.1), _row("bad", target_split="holdout", cer=0.2),
                _row("ok-2", cer=0.3)]
        path = _write_manifest(tmp_path / "mixed.jsonl", rows)
        result = select_samples(str(path), SelectionCriteria(limit=10))
        assert "bad" not in result.sample_ids
        assert result.excluded_protected == 1
        assert set(result.sample_ids) == {"ok-1", "ok-2"}

    def test_random_sampling_skips_protected(self, tmp_path: Path):
        rows = [_row("h", target_split="holdout"), _row("ok-1"), _row("ok-2"), _row("ok-3")]
        path = _write_manifest(tmp_path / "mixed2.jsonl", rows)
        result = select_samples(str(path), SelectionCriteria(random_sample_size=10))
        assert set(result.sample_ids) == {"ok-1", "ok-2", "ok-3"}
        assert result.excluded_protected == 1
        assert "h" not in result.sample_ids

    def test_assert_selection_safe_raises(self):
        with pytest.raises(PermissionError):
            assert_selection_safe([_row("x", target_split="holdout")])
        assert_selection_safe([_row("ok")])  # does not raise

    def test_protected_header_manifest_rejected_for_training(self, tmp_path: Path):
        rows = [_row("ok-1")]
        path = _write_manifest(tmp_path / "prot.jsonl", rows,
                               header={"dataset_role": "protected_holdout"})
        with pytest.raises(PermissionError):
            validate_derived_manifest_for_training(str(path))

    def test_derived_manifest_from_clean_source_passes(self, manifest: Path, tmp_path: Path):
        result = select_samples(str(manifest), SelectionCriteria(limit=3))
        out = tmp_path / "derived.jsonl"
        write_selection_manifest(result, str(out))
        report = validate_derived_manifest_for_training(str(out))
        assert report["rows"] == 3
        assert report["protected_rows"] == 0
