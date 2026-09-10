"""Adversarial regression tests for pre-training dataset invariants."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from clouda_data.pretraining.config import (
    PreparationConfig,
    PreparationConfigError,
    load_preparation_config,
)
from clouda_data.pretraining.dedupe import classify_duplicates
from clouda_data.pretraining.discovery import draft_samples, scan_source
from clouda_data.pretraining.export import ExportConfig, get_exporter
from clouda_data.pretraining.hashing import HashCache, sha256_file, sha256_text
from clouda_data.pretraining.manifest import read_manifest, write_manifest
from clouda_data.pretraining.normalize import NormalizationPolicy, normalize_text
from clouda_data.pretraining.schema import (
    DatasetSample,
    DuplicateState,
    SplitName,
    ValidationStatus,
    stable_sample_id,
)
from clouda_data.pretraining.sources import (
    SourceDefinition,
    SourceRegistryError,
    load_source_registry,
    register_source,
)
from clouda_data.pretraining.splitting import assign_splits
from clouda_data.pretraining.validation import (
    ValidationThresholds,
    apply_validation,
    validate_sample,
)
from clouda_data.pretraining.workflow import (
    WorkspacePaths,
    ensure_source_registered,
    index_source,
    normalize_workspace,
    prepare_dataset,
    validate_workspace,
)
from clouda_data.pretraining import workflow as workflow_module


def _png(path: Path, color: tuple[int, int, int] = (30, 60, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)


def _sample(sample_id: str, **changes: object) -> DatasetSample:
    values: dict[str, Any] = {
        "sample_id": sample_id,
        "source_id": "src",
        "source_path": f"{sample_id}.png",
        "image_path": f"{sample_id}.png",
        "raw_text": f"raw {sample_id}",
        "text": f"text {sample_id}",
        "file_sha256": f"file-{sample_id}",
        "normalized_text_sha256": f"text-{sample_id}",
    }
    values.update(changes)
    return DatasetSample(**values)


def test_sample_identity_canonicalizes_equivalent_relative_paths() -> None:
    expected = stable_sample_id("src", "pages/a.png", "labels/a.txt")
    assert stable_sample_id("src", "./pages\\a.png", ".\\labels/a.txt") == expected
    with pytest.raises(ValueError):
        stable_sample_id("src", "../outside.png")


def test_sample_reader_rejects_unknown_or_incompatible_schema() -> None:
    with pytest.raises(ValueError, match="Unknown sample fields"):
        DatasetSample.from_dict(
            {"sample_id": "s", "source_id": "src", "typo_field": True}
        )
    with pytest.raises(ValueError, match="schema version"):
        DatasetSample.from_dict(
            {
                "sample_id": "s",
                "source_id": "src",
                "schema_version": "clouda.pretraining.sample.v999",
            }
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
def test_config_rejects_unsafe_split_ratios(bad: float) -> None:
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.05, "holdout": 0.05}
    ratios["train"] = bad
    with pytest.raises(PreparationConfigError):
        PreparationConfig.from_mapping({"split_ratios": ratios})


def test_config_rejects_truthy_string_for_holdout_export() -> None:
    with pytest.raises(PreparationConfigError):
        PreparationConfig.from_mapping({"include_holdout_in_export": "false"})
    with pytest.raises((PreparationConfigError, TypeError)):
        PreparationConfig.from_mapping(
            {"normalization": {"remove_diacritics": "false"}}
        )


def test_index_honors_configured_extension_allowlist(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "page.png")
    (source_root / "page.txt").write_text("text", encoding="utf-8")
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    workspace = tmp_path / "workspace"
    config = PreparationConfig.from_mapping({"allowed_text_extensions": []})
    report = index_source(workspace, source, config)
    assert report["files"] == 1


def test_preparation_config_fingerprint_covers_derived_semantics() -> None:
    base = PreparationConfig()
    changed = PreparationConfig.from_mapping({"split_seed": base.split_seed + 1})
    assert base.fingerprint() != changed.fingerprint()


def test_config_file_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(PreparationConfigError, match="version"):
        load_preparation_config(path)
    path.write_text(
        json.dumps({"_schema_version": "clouda.pretraining.config.v1"}),
        encoding="utf-8",
    )
    assert load_preparation_config(path) == PreparationConfig()


@pytest.mark.parametrize(
    "mapping",
    [
        {"workers": "2"},
        {"max_pixels": "100"},
        {"split_seed": True},
        {"allowed_image_extensions": ".png"},
    ],
)
def test_config_rejects_wrong_scalar_and_collection_types(
    mapping: dict[str, object],
) -> None:
    with pytest.raises(PreparationConfigError):
        PreparationConfig.from_mapping(mapping)


def test_source_registry_rejects_relative_local_root(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistryError, match="local_root"):
        register_source(
            tmp_path / "sources.jsonl",
            SourceDefinition(source_id="src", name="Source", local_root="relative"),
        )


def test_source_registry_rejects_bad_schema_and_language_shape(tmp_path: Path) -> None:
    registry = tmp_path / "sources.jsonl"
    registry.write_text(
        json.dumps({"_schema_version": "wrong", "_row_count": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceRegistryError, match="version"):
        load_source_registry(registry)
    with pytest.raises(SourceRegistryError, match="languages"):
        SourceDefinition.from_dict(
            {"source_id": "src", "name": "Source", "languages": "ar"}
        )
    with pytest.raises(SourceRegistryError, match="adapter"):
        SourceDefinition.from_dict(
            {"source_id": "src", "name": "Source", "adapter": "executable.py"}
        )


def test_safe_normalization_defaults_preserve_joiners_controls_and_spacing() -> None:
    raw = "می\u200cروم\t  now\u0007"
    assert normalize_text(raw, NormalizationPolicy()).value == raw


def test_compatibility_normalization_requires_explicit_presentation_composition() -> (
    None
):
    with pytest.raises(ValueError, match="presentation_forms='compose'"):
        normalize_text("ﻻ", NormalizationPolicy(unicode_form="NFKC"))
    assert (
        normalize_text(
            "ﻻ", NormalizationPolicy(unicode_form="NFKC", presentation_forms="compose")
        ).value
        == "لا"
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "../outside.png",
        "../../outside.png",
        "/etc/passwd",
        r"C:\\Windows\\win.ini",
        r"\\server\\share\\image.png",
        "bad\x00name.png",
        "image\u202epng.exe",
    ],
)
def test_record_adapter_rejects_paths_outside_source_root(
    tmp_path: Path, hostile: str
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "records.jsonl").write_text(
        json.dumps({"image": hostile, "text": "safe text"}) + "\n",
        encoding="utf-8",
    )
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    drafts = draft_samples(source, scan_source(source))
    assert len(drafts) == 1
    assert drafts[0].malformed_metadata
    assert drafts[0].image_rel_path is None
    assert drafts[0].source_path == "records.jsonl"


def test_record_adapter_normalizes_safe_relative_path(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "images" / "page.png")
    (source_root / "records.jsonl").write_text(
        json.dumps({"image": "./images/../images/page.png", "text": "text"}) + "\n",
        encoding="utf-8",
    )
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    drafts = draft_samples(source, scan_source(source))
    record_draft = next(d for d in drafts if d.source_record_id)
    assert record_draft.image_rel_path == "images/page.png"


def test_declared_record_id_survives_unrelated_line_insertion(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "page.png")
    record = {"id": "stable", "image": "page.png", "text": "text"}
    record_path = source_root / "records.jsonl"
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    first = next(
        d for d in draft_samples(source, scan_source(source)) if d.source_record_id
    )
    record_path.write_text("\n" + json.dumps(record) + "\n", encoding="utf-8")
    second = next(
        d for d in draft_samples(source, scan_source(source)) if d.source_record_id
    )
    assert first.sample_id == second.sample_id
    assert first.source_record_id == second.source_record_id


def test_scan_does_not_index_symlinked_file_outside_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.png"
    _png(outside)
    link = source_root / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable on this host")
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    assert scan_source(source) == []


def test_resume_rehashes_same_size_file_after_content_change(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = source_root / "sample.txt"
    payload.write_bytes(b"first!")
    source = SourceDefinition(
        source_id="src", name="Source", local_root=str(source_root)
    )
    workspace = tmp_path / "workspace"
    index_source(workspace, source)
    first_rows = [
        row for row in read_manifest(workspace / "sources" / "src" / "index.jsonl")[1]
    ]
    first_hash = first_rows[0]["file_sha256"]
    payload.write_bytes(b"second")
    stat = payload.stat()
    os.utime(payload, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    index_source(workspace, source, resume=True)
    second_rows = read_manifest(workspace / "sources" / "src" / "index.jsonl")[1]
    assert second_rows[0]["file_sha256"] != first_hash
    assert second_rows[0]["file_sha256"] == sha256_file(payload)


def test_hash_cache_is_namespaced_by_source_and_mtime(tmp_path: Path) -> None:
    cache = HashCache(tmp_path / "hashes.jsonl")
    digest = "a" * 64
    cache.put("source-a", "page.png", 10, 100, digest)
    assert cache.get("source-a", "page.png", 10, 100) == digest
    assert cache.get("source-b", "page.png", 10, 100) is None
    assert cache.get("source-a", "page.png", 10, 101) is None


def test_hash_cache_ignores_corrupted_digest(tmp_path: Path) -> None:
    cache_path = tmp_path / "hashes.jsonl"
    key = HashCache.make_key("source", "page.png", 10, 100)
    cache_path.write_text(
        json.dumps({"key": key, "sha256": "not-a-sha256"}) + "\n",
        encoding="utf-8",
    )
    assert HashCache(cache_path).get("source", "page.png", 10, 100) is None


def test_hash_cache_recovers_after_truncated_final_line(tmp_path: Path) -> None:
    cache_path = tmp_path / "hashes.jsonl"
    cache_path.write_text('{"truncated":', encoding="utf-8")
    cache = HashCache(cache_path)
    digest = "b" * 64
    cache.put("source", "new.png", 20, 200, digest)
    assert HashCache(cache_path).get("source", "new.png", 20, 200) == digest


def test_duplicate_canonical_is_independent_of_tied_input_order() -> None:
    first = _sample(
        "same-id",
        source_path="same.png",
        source_record_id="record-b",
        raw_text="B",
        file_sha256="same-hash",
    )
    second = first.evolve(source_record_id="record-a", raw_text="A")
    outputs = []
    for items in ([first, second], [second, first]):
        classified, _ = classify_duplicates(list(items))
        canonical = next(
            sample
            for sample in classified
            if sample.duplicate_state == DuplicateState.CANONICAL
        )
        outputs.append((canonical.source_record_id, canonical.raw_text))
    assert outputs == [("record-a", "A"), ("record-a", "A")]


def test_duplicate_reason_tracks_transitive_cluster_signal() -> None:
    samples = [
        _sample("shared-id", source_record_id="record-a", file_sha256="file-a"),
        _sample("shared-id", source_record_id="record-b", file_sha256="file-b"),
        _sample("third", source_record_id="record-b", file_sha256="file-c"),
    ]
    classified, _ = classify_duplicates(samples)
    third = next(sample for sample in classified if sample.sample_id == "third")
    assert third.exclusion_reason == "duplicate_source_record"


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": -0.1, "validation": 0.2, "test": 0.4, "holdout": 0.5},
        {"train": math.nan, "validation": math.nan, "test": 0.5, "holdout": 0.5},
        {"train": math.inf, "validation": 0.0, "test": 0.0, "holdout": -math.inf},
    ],
)
def test_split_rejects_non_finite_or_negative_ratios(
    ratios: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        assign_splits([_sample("a")], seed=1, ratios=ratios)


def test_split_merges_transitive_document_text_file_and_duplicate_links() -> None:
    shared_text = sha256_text("shared")
    samples = [
        _sample("a", document_id="doc", normalized_text_sha256="ta"),
        _sample(
            "b",
            document_id="doc",
            normalized_text_sha256=shared_text,
            file_sha256="fb",
        ),
        _sample(
            "c",
            document_id="other",
            normalized_text_sha256=shared_text,
            file_sha256="fc",
        ),
        _sample(
            "d", document_id="third", normalized_text_sha256="td", file_sha256="fc"
        ),
        _sample("e", document_id="fourth", duplicate_of="d", file_sha256="fe"),
    ]
    updated, report = assign_splits(samples, seed=73)
    assert report.passed
    assert len({sample.target_split for sample in updated}) == 1
    duplicate_check = next(
        check
        for check in report.leakage_checks
        if check["check"] == "duplicate_cluster_not_shared_across_splits"
    )
    assert duplicate_check["passed"]


def test_validation_rejects_image_before_decoding_above_pixel_limit(
    tmp_path: Path,
) -> None:
    _png(tmp_path / "large.png")
    sample = _sample("large", image_path="large.png")
    status, findings = validate_sample(
        sample,
        tmp_path,
        thresholds=ValidationThresholds(max_pixels=100),
    )
    assert status == ValidationStatus.ERROR
    assert any(finding.code == "oversized_image" for finding in findings)


def test_validation_is_idempotent_for_quality_flags(tmp_path: Path) -> None:
    sample = _sample("warning", image_path=None, raw_text="warning\x07")
    limits = ValidationThresholds(require_image=False)
    first, _ = apply_validation([sample], tmp_path, thresholds=limits)
    second, _ = apply_validation(first, tmp_path, thresholds=limits)
    assert second == first


def test_validation_exclusion_reason_uses_error_not_prior_warning(
    tmp_path: Path,
) -> None:
    _png(tmp_path / "tiny.png")
    sample = _sample("tiny", image_path="tiny.png", raw_text=None, text=None)
    updated, _ = apply_validation(
        [sample],
        tmp_path,
        thresholds=ValidationThresholds(min_width=30, min_height=30),
    )
    assert updated[0].validation_status == ValidationStatus.ERROR
    assert updated[0].exclusion_reason == "missing_text"


def test_default_export_removes_stale_holdout_file(tmp_path: Path) -> None:
    exporter = get_exporter("jsonl")
    out_dir = tmp_path / "export"
    holdout = _sample("holdout", target_split=SplitName.HOLDOUT)
    exporter.export([holdout], out_dir, ExportConfig(include_holdout=True))
    assert (out_dir / "holdout.jsonl").exists()
    exporter.export([], out_dir, ExportConfig())
    assert not (out_dir / "holdout.jsonl").exists()


def test_export_rejects_forged_escaping_image_path(tmp_path: Path) -> None:
    forged = _sample(
        "forged", image_path="../../secret.png", target_split=SplitName.TRAIN
    )
    with pytest.raises(ValueError, match="image path"):
        get_exporter("jsonl").export([forged], tmp_path / "export", ExportConfig())


def test_export_retains_source_provenance(tmp_path: Path) -> None:
    sample = _sample(
        "sample",
        source_path="records.jsonl",
        source_record_id="records.jsonl#id=sample",
        source_license="CC-BY-4.0",
        target_split=SplitName.TRAIN,
    )
    result = get_exporter("jsonl").export([sample], tmp_path / "export", ExportConfig())
    row = json.loads(Path(result.files[0]).read_text(encoding="utf-8"))
    assert row["source_path"] == "records.jsonl"
    assert row["source_record_id"] == "records.jsonl#id=sample"
    assert row["source_license"] == "CC-BY-4.0"


def test_manifest_writer_sorts_rows_and_reader_rejects_corruption(
    tmp_path: Path,
) -> None:
    path = write_manifest(
        tmp_path / "manifest.jsonl",
        [_sample("b").to_dict(), _sample("a").to_dict()],
    )
    _, rows = read_manifest(path)
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    path.write_text(
        '{"_schema_version":"clouda.pretraining.manifest.v1"}\n{bad\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="line 2"):
        read_manifest(path)


def test_manifest_bytes_are_stable_when_primary_sort_keys_tie(tmp_path: Path) -> None:
    first = _sample("same", source_path="same.png", raw_text="B").to_dict()
    second = _sample("same", source_path="same.png", raw_text="A").to_dict()
    left = write_manifest(tmp_path / "left.jsonl", [first, second])
    right = write_manifest(tmp_path / "right.jsonl", [second, first])
    assert left.read_bytes() == right.read_bytes()


def test_workspace_rejects_incompatible_manifest_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = WorkspacePaths(workspace).manifest
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"_schema_version": "wrong", "_row_count": 1})
        + "\n"
        + json.dumps(_sample("a").to_dict())
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest schema"):
        normalize_workspace(workspace)


def test_dry_run_with_unregistered_path_writes_nothing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "page.png")
    (source_root / "page.txt").write_text("text", encoding="utf-8")
    workspace = tmp_path / "workspace"
    prepare_dataset(workspace, source_root, dry_run=True)
    assert not workspace.exists()


def test_dry_run_hashes_each_discovered_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "page.png")
    (source_root / "page.txt").write_text("text", encoding="utf-8")
    calls = 0
    real_hash = sha256_file

    def counting_hash(path: str | Path) -> str:
        nonlocal calls
        calls += 1
        return real_hash(path)

    monkeypatch.setattr(workflow_module, "sha256_file", counting_hash)
    prepare_dataset(tmp_path / "workspace", source_root, dry_run=True)
    assert calls == 2


def test_prepare_manifest_records_configuration_fingerprint(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _png(source_root / "page.png")
    (source_root / "page.txt").write_text("text", encoding="utf-8")
    workspace = tmp_path / "workspace"
    config = PreparationConfig()
    prepare_dataset(workspace, source_root, config, seed=91)
    header, _ = read_manifest(WorkspacePaths(workspace).manifest)
    assert header["preparation_config_fingerprint"] == config.fingerprint()
    assert header["split_seed"] == 91


def test_incremental_prepare_recomputes_cross_source_invariants(tmp_path: Path) -> None:
    roots = []
    for source_index in range(2):
        root = tmp_path / f"source-{source_index}"
        for item in range(16):
            _png(root / f"doc{item}.png", (source_index * 80, item * 8, 30))
            (root / f"doc{item}.txt").write_text(
                f"shared text {item}", encoding="utf-8"
            )
        roots.append(root)

    workspace = tmp_path / "workspace"
    for index, root in enumerate(roots):
        ensure_source_registered(
            workspace,
            SourceDefinition(
                source_id=f"source-{index}",
                name=f"Source {index}",
                local_root=str(root),
            ),
        )
    prepare_dataset(workspace, "source-0", seed=19)
    report = prepare_dataset(workspace, "source-1", seed=19)
    _, rows = read_manifest(WorkspacePaths(workspace).manifest)

    by_text: dict[str, set[str]] = {}
    for row in rows:
        by_text.setdefault(row["normalized_text_sha256"], set()).add(
            row["target_split"]
        )
    assert all(len(splits) == 1 for splits in by_text.values())
    assert report["stats"]["total_samples"] == len(rows) == 32
    assert report["split"]["passed"]


def test_validate_workspace_uses_each_samples_source_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    samples = []
    for source_id in ("a", "b"):
        root = tmp_path / source_id
        _png(root / f"{source_id}.png")
        ensure_source_registered(
            workspace,
            SourceDefinition(source_id=source_id, name=source_id, local_root=str(root)),
        )
        samples.append(
            _sample(
                source_id,
                source_id=source_id,
                source_path=f"{source_id}.png",
                image_path=f"{source_id}.png",
            )
        )
    write_manifest(
        WorkspacePaths(workspace).manifest, [sample.to_dict() for sample in samples]
    )
    report = validate_workspace(workspace)
    assert report["counts"]["ok"] == 2
    assert report["counts"]["error"] == 0


def test_normalize_stage_clears_stale_derived_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sample = _sample(
        "sample",
        raw_text="unchanged",
        text="changed",
        transformations=["remove_zero_width_and_bidi_marks"],
        duplicate_state=DuplicateState.CONFLICTING_DUPLICATE,
        duplicate_of="other",
        target_split=SplitName.TRAIN,
    )
    no_text = sample.evolve(
        sample_id="no-text",
        source_path="no-text.png",
        raw_text=None,
        text=None,
    )
    write_manifest(
        WorkspacePaths(workspace).manifest, [sample.to_dict(), no_text.to_dict()]
    )
    export_path = workspace / "export" / "jsonl" / "train.jsonl"
    export_path.parent.mkdir(parents=True)
    export_path.write_text("stale\n", encoding="utf-8")
    normalize_workspace(workspace, PreparationConfig())
    _, rows = read_manifest(WorkspacePaths(workspace).manifest)
    assert all(row["transformations"] == [] for row in rows)
    assert all(row["duplicate_state"] == DuplicateState.UNIQUE.value for row in rows)
    assert all(row["duplicate_of"] is None for row in rows)
    assert all(row["target_split"] == SplitName.UNASSIGNED.value for row in rows)
    assert not export_path.exists()


def test_normalize_stage_refuses_text_without_immutable_raw_text(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sample = _sample("sample", raw_text=None, text="cannot be reconstructed")
    write_manifest(WorkspacePaths(workspace).manifest, [sample.to_dict()])

    with pytest.raises(ValueError, match="without raw_text"):
        normalize_workspace(workspace, PreparationConfig())
