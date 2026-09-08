"""Validation, deduplication, splitting, manifest, export tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from clouda_data.pretraining.dedupe import classify_duplicates
from clouda_data.pretraining.export import ExportConfig, get_exporter, select_exportable
from clouda_data.pretraining.hashing import sha256_text
from clouda_data.pretraining.manifest import read_manifest, write_manifest
from clouda_data.pretraining.schema import (
    DatasetSample,
    DuplicateState,
    SplitName,
    ValidationStatus,
    sort_key,
)
from clouda_data.pretraining.splitting import (
    assign_splits,
)
from clouda_data.pretraining.validation import (
    ValidationThresholds,
    apply_validation,
    validate_sample,
)


def _sample(sample_id: str, **kwargs: object) -> DatasetSample:
    base = {
        "sample_id": sample_id,
        "source_id": "src",
        "source_path": f"{sample_id}.png",
        "image_path": f"images/{sample_id}.png",
        "raw_text": "نص تجريبي",
        "text": "نص تجريبي",
        "file_sha256": f"hash_{sample_id}",
    }
    base.update(kwargs)
    return DatasetSample(**base)


# ------------------------------------------------------------- validation


def test_validation_flags_missing_image(tmp_path: Path):
    sample = _sample("a", image_path="images/gone.png")
    status, findings = validate_sample(sample, tmp_path)
    assert status == ValidationStatus.ERROR
    assert any(f.code == "missing_image" for f in findings)


def test_validation_flags_path_escape(tmp_path: Path):
    sample = _sample("a", image_path="../../etc/passwd.png")
    status, findings = validate_sample(sample, tmp_path)
    assert any(f.code == "path_escape" for f in findings)
    assert status == ValidationStatus.ERROR


def test_validation_flags_empty_and_long_text(tmp_path: Path):
    empty = _sample("a", raw_text="   ", text="  ")
    limits = ValidationThresholds(require_image=False)
    status, _ = validate_sample(empty, tmp_path, thresholds=limits)
    assert status == ValidationStatus.ERROR
    long_sample = _sample("b", raw_text="x" * 200, text="x" * 200)
    status, findings = validate_sample(
        long_sample,
        tmp_path,
        thresholds=ValidationThresholds(require_image=False, max_text_chars=100),
    )
    assert any(f.code == "text_too_long" for f in findings)
    assert status == ValidationStatus.ERROR


def test_validation_bad_sample_does_not_abort_dataset(tmp_path: Path):
    samples = [
        _sample("good1", image_path=None),
        _sample("bad", image_path="../escape.png"),
        _sample("good2", image_path=None),
    ]
    updated, report = apply_validation(
        samples, tmp_path, thresholds=ValidationThresholds(require_image=False)
    )
    assert report["counts"]["error"] == 1
    assert report["counts"]["ok"] == 2
    assert len(updated) == 3
    assert updated[1].exclusion_reason == "path_escape"


def test_validation_reports_malformed_metadata(tmp_path: Path):
    sample = _sample("a", provenance={"malformed_metadata": True})
    _, findings = validate_sample(
        sample, tmp_path, thresholds=ValidationThresholds(require_image=False)
    )
    assert any(f.code == "malformed_metadata" for f in findings)


# -------------------------------------------------------------- dedupe


def test_exact_file_duplicates_classified_with_provenance():
    samples = [
        _sample("a", file_sha256="same"),
        _sample("b", file_sha256="same"),
        _sample("c", file_sha256="unique"),
    ]
    updated, report = classify_duplicates(samples)
    states = {s.sample_id: s.duplicate_state for s in updated}
    assert states["a"] == DuplicateState.CANONICAL
    assert states["b"] == DuplicateState.DUPLICATE
    dup = next(s for s in updated if s.sample_id == "b")
    assert dup.duplicate_of == "a"
    assert dup.exclusion_reason == "duplicate_file_hash"
    assert report.duplicate_file_hash == 1


def test_conflicting_duplicates_share_text_but_not_content():
    text_hash = sha256_text("نص")
    samples = [
        _sample("a", normalized_text_sha256=text_hash, file_sha256="h1"),
        _sample("b", normalized_text_sha256=text_hash, file_sha256="h2"),
    ]
    updated, report = classify_duplicates(samples)
    states = {s.sample_id: s.duplicate_state for s in updated}
    assert states["a"] == DuplicateState.UNIQUE
    assert states["b"] == DuplicateState.CONFLICTING_DUPLICATE
    assert report.conflicting_duplicate == 1


def test_repeated_sample_ids_and_source_records_are_duplicates():
    samples = [
        _sample("a", source_record_id="rec1"),
        _sample("a2", source_record_id="rec1"),
        _sample("b"),
        _sample("b"),
    ]
    updated, report = classify_duplicates(samples)
    assert report.duplicate_source_record == 1
    assert report.duplicate_sample_id == 1
    excluded = [s for s in updated if s.duplicate_state == DuplicateState.DUPLICATE]
    assert all(s.exclusion_reason for s in excluded)


# ---------------------------------------------------------------- split


def _split_samples() -> list[DatasetSample]:
    samples = []
    for doc in range(60):
        for page in (1, 2):
            samples.append(
                _sample(
                    f"d{doc}_p{page}",
                    document_id=f"doc{doc}",
                    group_id=f"src:doc{doc}",
                    file_sha256=f"h_{doc}_{page}",
                    raw_text=f"نص {doc} {page}",
                    text=f"نص {doc} {page}",
                    normalized_text_sha256=sha256_text(f"نص {doc} {page}"),
                )
            )
    return samples


def test_split_is_deterministic_across_reruns():
    first = assign_splits(_split_samples(), seed=42)
    second = assign_splits(_split_samples(), seed=42)
    assert [s.target_split for s in first[0]] == [s.target_split for s in second[0]]
    other_seed = assign_splits(_split_samples(), seed=43)[0]
    assert [sample.target_split for sample in first[0]] != [
        sample.target_split for sample in other_seed
    ]


def test_pages_of_same_document_never_cross_splits():
    updated, report = assign_splits(_split_samples(), seed=42)
    assert report.passed
    by_doc: dict[str, set[SplitName]] = {}
    for sample in updated:
        if sample.target_split != SplitName.UNASSIGNED:
            by_doc.setdefault(str(sample.document_id), set()).add(sample.target_split)
    assert all(len(splits) == 1 for splits in by_doc.values())


def test_identical_file_hashes_cannot_leak_across_splits():
    samples = _split_samples()
    # force a shared file hash across two documents
    samples[0] = samples[0].evolve(file_sha256="leak_hash")
    samples[10] = samples[10].evolve(file_sha256="leak_hash")
    _, report = assign_splits(samples, seed=42)
    check = next(
        c
        for c in report.leakage_checks
        if c["check"] == "file_hash_not_shared_across_splits"
    )
    assert check["passed"]


def test_identical_normalized_text_cannot_leak_across_splits():
    samples = _split_samples()
    shared = sha256_text("نص مشترك")
    samples[0] = samples[0].evolve(normalized_text_sha256=shared)
    samples[10] = samples[10].evolve(normalized_text_sha256=shared)
    _, report = assign_splits(samples, seed=42)
    check = next(
        c
        for c in report.leakage_checks
        if c["check"] == "normalized_text_hash_not_shared_across_splits"
    )
    assert check["passed"]


def test_holdout_is_protected_and_reported():
    updated, report = assign_splits(_split_samples(), seed=42)
    holdout = [s for s in updated if s.target_split == SplitName.HOLDOUT]
    assert holdout, "expected holdout samples with default ratios"
    assert report.counts["holdout"] > 0
    selected = select_exportable(updated, ExportConfig())
    assert all(s.target_split != SplitName.HOLDOUT for s in selected)
    selected_with_holdout = select_exportable(
        updated, ExportConfig(include_holdout=True)
    )
    assert any(s.target_split == SplitName.HOLDOUT for s in selected_with_holdout)


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        assign_splits(_split_samples(), seed=1, ratios={"train": 2.0})


# ------------------------------------------------------------- manifest


def test_manifest_is_byte_identical_across_writes(tmp_path: Path):
    samples = sorted(_split_samples(), key=sort_key)
    rows = [s.to_dict() for s in samples]
    first = write_manifest(tmp_path / "one.jsonl", rows)
    second = write_manifest(tmp_path / "two.jsonl", rows)
    assert first.read_bytes() == second.read_bytes()
    header, read_rows = read_manifest(first)
    assert header["_schema_version"] == "clouda.pretraining.manifest.v1"
    assert len(read_rows) == len(rows)
    assert read_rows == rows


def test_manifest_atomic_write_leaves_no_tmp(tmp_path: Path):
    write_manifest(
        tmp_path / "m.jsonl", [DatasetSample(sample_id="s", source_id="x").to_dict()]
    )
    assert not (tmp_path / "m.jsonl.tmp").exists()


# --------------------------------------------------------------- export


def test_export_filters_excluded_duplicates_and_holdout(tmp_path: Path):
    samples = _split_samples()
    samples, _ = assign_splits(samples, seed=42)
    samples[0] = samples[0].evolve(
        validation_status=ValidationStatus.ERROR, exclusion_reason="missing_text"
    )
    samples[1] = samples[1].evolve(duplicate_state=DuplicateState.DUPLICATE)
    result = get_exporter("jsonl").export(samples, tmp_path / "export", ExportConfig())
    assert set(result.counts) <= {"train", "validation", "test"}
    total = sum(result.counts.values())
    # error + duplicate + holdout samples filtered out
    assert total == len(
        [
            s
            for s in samples
            if s.validation_status == ValidationStatus.OK
            and s.duplicate_state != DuplicateState.DUPLICATE
            and s.target_split != SplitName.HOLDOUT
        ]
    )


def test_export_rows_are_deterministic_jsonl(tmp_path: Path):
    samples, _ = assign_splits(_split_samples(), seed=42)
    config = ExportConfig()
    first = get_exporter("jsonl").export(samples, tmp_path / "a", config)
    second = get_exporter("jsonl").export(samples, tmp_path / "b", config)
    for path_a, path_b in zip(first.files, second.files):
        assert Path(path_a).read_bytes() == Path(path_b).read_bytes()
    import json

    row = json.loads(Path(first.files[0]).read_text(encoding="utf-8").splitlines()[0])
    assert {"image", "text", "sample_id", "split"} <= set(row)


def test_unknown_exporter_raises():
    with pytest.raises(KeyError):
        get_exporter("parquet")
