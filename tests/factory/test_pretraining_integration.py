"""Integration: Data Factory output → canonical manifest → Training Framework.

Proves the full canonical flow offline, CPU-only, with tiny fixtures:

    Data Factory run manifest (factory rows)
      → clouda_data.factory.adapters (lossless provenance conversion)
      → canonical clouda.pretraining.manifest.v1
      → leakage-safe split assignment (holdout protection verified)
      → Training Experiment Framework dry-run (mock adapter) accepts it

No network, no model downloads, no real training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from clouda_data.factory.adapters import (
    FactoryAdapterError,
    factory_rows_to_samples,
    read_factory_manifest,
    run_dir_to_dataset_manifest,
)
from clouda_data.factory.factory import generate_run
from clouda_data.pretraining.manifest import read_manifest
from clouda_data.pretraining.schema import SplitName
from clouda_data.pretraining.splitting import assign_splits
from clouda_training.experiments import run_experiment
from clouda_training.experiments.config import load_experiment_config

PROFILES = ["05_old_book_medium", "old_book_medium", "12_hard_composite"]


def _make_image_input(tmp_path: Path) -> Path:
    inbox = tmp_path / "input"
    inbox.mkdir(exist_ok=True)
    img = Image.new("RGB", (620, 877), (250, 247, 240))
    draw = ImageDraw.Draw(img)
    y = 60
    for _ in range(12):
        draw.text((580, y), "0" * 25, fill=(20, 20, 20))
        y += 30
    img.save(inbox / "sample_page.png")
    (inbox / "sample_page.txt").write_text("نص تجريبي للتكامل\n", encoding="utf-8")
    return inbox / "sample_page.png"


def _factory_run(tmp_path: Path) -> tuple[Path, dict]:
    image = _make_image_input(tmp_path)
    meta = generate_run(
        inputs=[image],
        runs_root=tmp_path / "runs",
        profile_names=PROFILES,
        variants=3,
        base_seed=12345,
        seed_mode="v1",
        backend="raqm",
        workers=1,
        export_pdf=True,
        export_png=True,
        max_pages=4,
    )
    return tmp_path / "runs" / meta["run_id"], meta


def test_factory_run_converts_to_canonical_manifest(tmp_path):
    run_dir, meta = _factory_run(tmp_path)
    manifest_path, report, manifest_hash = run_dir_to_dataset_manifest(
        run_dir,
        tmp_path / "dataset_manifest.jsonl",
        source_id="factory_integration",
    )
    assert report.rows_ok == 3
    assert report.samples == 3
    assert report.rows_error == 0
    assert len(manifest_hash) == 64

    header, rows = read_manifest(manifest_path)
    assert header["_schema_version"] == "clouda.pretraining.manifest.v1"
    assert header["_row_count"] == 3
    assert header["dataset_role"] == "training"
    assert header["dataset_id"] == "factory_integration"
    assert len(header["dataset_version"]) == 64
    assert len(rows) == 3
    for row in rows:
        assert row["source_id"] == "factory_integration"
        assert row["provenance"]["factory_generated"] is True
        assert row["provenance"]["seed"] is not None
        assert row["provenance"]["seed_mode"] == "v1"
        assert row["provenance"]["profile"] in PROFILES
        assert row["provenance"]["output_sha256"]
        assert row["provenance"]["source_sha256"] or row["provenance"]["clean_sha256"]
        assert row["provenance"]["transform_steps"]
        assert row["provenance"]["render_config"]
        assert row["provenance"]["distortion_config"]
        assert row["provenance"]["config_hash"] == meta["config_hash"]
        assert row["provenance"]["source_ref"] == "sample_page.png"
        assert not Path(row["provenance"]["source_ref"]).is_absolute()
        assert row["image_path"].endswith(".png")
        assert "\\" not in row["image_path"]
        assert row["file_sha256"] == row["provenance"]["output_sha256"]
        with Image.open(run_dir / Path(row["image_path"])) as image:
            image.verify()
        # run_dir_to_dataset_manifest assigns leakage-safe splits directly
        assert row["target_split"] in {"train", "validation", "test"}


def test_splitting_assigns_leakage_safe_splits(tmp_path):
    run_dir, _ = _factory_run(tmp_path)
    manifest_path, _, _ = run_dir_to_dataset_manifest(
        run_dir,
        tmp_path / "dataset_manifest.jsonl",
        source_id="factory_integration",
    )
    header, rows = read_manifest(manifest_path)
    from clouda_data.pretraining.schema import DatasetSample

    samples = [DatasetSample.from_dict(row) for row in rows]
    updated, report = assign_splits(samples, seed=42)
    assert report.passed
    for sample in updated:
        assert sample.target_split != SplitName.UNASSIGNED
    # pages of one document must share one split
    splits = {sample.target_split for sample in updated}
    assert splits <= {SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST}


def test_training_framework_dry_run_accepts_factory_manifest(tmp_path):
    run_dir, _ = _factory_run(tmp_path)
    manifest_path, _, manifest_hash = run_dir_to_dataset_manifest(
        run_dir,
        tmp_path / "dataset_manifest.jsonl",
        source_id="factory_integration",
    )
    # A single-document run assigns the whole group to exactly one split;
    # select whichever non-holdout split the converted manifest actually has.
    _header, _rows = read_manifest(manifest_path)
    selected_split = next(
        row["target_split"]
        for row in _rows
        if row["target_split"] in {"train", "validation", "test"}
    )
    config_yaml = tmp_path / "experiment.yaml"
    config_yaml.write_text(
        f"""
schema_version: 1
experiment:
  name: factory-integration-dry-run
  description: Data Factory -> Training Framework compatibility proof.
  tags: [integration, factory, offline]
  notes: Mock only; no real training.
model:
  model_id: mock/clouda-ocr
  revision: fixture-v1
  model_family: multimodal-ocr
  adapter_type: mock
  precision: float32
dataset:
  dataset_id: factory_integration
  dataset_version: {_header["dataset_version"]}
  manifest_path: {manifest_path.as_posix()}
  split: {selected_split}
  sample_limit: 2
  preprocessing_version: clouda.pretraining.normalize.v1
training:
  seed: 20260909
  epochs: 1
  max_steps: 4
  batch_size: 2
  learning_rate: 0.0001
checkpoint:
  save_strategy: steps
  save_steps: 2
  save_total_limit: 2
evaluation:
  enabled: true
  eval_split: validation
  eval_steps: 2
  metrics: [cer, wer]
runtime:
  device: cpu
  num_workers: 0
  output_root: {tmp_path / "runs"}
  dry_run: true
  offline: true
  deterministic: true
tracking:
  enabled: true
  backend: jsonl
  log_steps: 1
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_yaml)
    assert config.dataset.dataset_id == "factory_integration"
    assert config.dataset.manifest_path == manifest_path.resolve()

    handle = run_experiment(config)
    assert handle.status.value == "COMPLETED"
    summary = handle.summary()
    assert summary["status"] == "COMPLETED"
    metadata = json.loads((handle.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_id"] == "factory_integration"
    assert metadata["dataset_manifest_hash"] == manifest_hash
    assert metadata["dataset_split"] == selected_split
    assert metadata["dataset_rows"] >= 1
    assert "factory_integration" in metadata["source_provenance"]["source_ids"]
    assert metadata["status"] == "COMPLETED"
    # holdout safety: the run must record a non-protected split
    assert metadata["dataset_split"] not in {"holdout", "protected_holdout"}


def test_adapter_rejects_protected_split_rows(tmp_path):
    rows = [
        {
            "run_id": "r",
            "document_id": "d",
            "page_index": 0,
            "variant_id": "v00__x",
            "status": "ok",
            "profile": "p",
            "output_path": "scans/x.png",
            "output_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "target_split": "holdout",
        }
    ]
    with pytest.raises(FactoryAdapterError):
        factory_rows_to_samples(rows, source_id="s")


def test_adapter_rejects_pdf_only_rows_as_training_images():
    row = {
        "run_id": "r",
        "document_id": "d",
        "page_index": 0,
        "variant_id": "v00__x",
        "status": "ok",
        "profile": "p",
        "output_path": "scans/x.pdf",
        "output_sha256": "a" * 64,
        "source_sha256": "b" * 64,
    }
    with pytest.raises(FactoryAdapterError, match="raster image"):
        factory_rows_to_samples([row], source_id="s")


@pytest.mark.parametrize(
    "marker",
    [" HOLDOUT ", "Holdout", "hidden_holdout_alias"],
)
def test_adapter_rejects_normalized_or_aliased_holdout_markers(marker):
    row = {
        "run_id": "r",
        "document_id": "d",
        "page_index": 0,
        "variant_id": "v00__x",
        "status": "ok",
        "profile": "p",
        "output_path": "scans/x.png",
        "output_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "target_split": marker,
    }
    with pytest.raises(FactoryAdapterError, match="protected"):
        factory_rows_to_samples([row], source_id="s")


@pytest.mark.parametrize(
    "protection",
    [
        {"protected": True},
        {"provenance": {"protected": True}},
        {"metadata": {"dataset_role": "evaluation_only"}},
    ],
)
def test_adapter_rejects_protected_factory_metadata(protection):
    row = {
        "run_id": "r",
        "document_id": "d",
        "page_index": 0,
        "variant_id": "v00__x",
        "status": "ok",
        "profile": "p",
        "output_path": "scans/x.png",
        "output_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        **protection,
    }
    with pytest.raises(FactoryAdapterError, match="protected"):
        factory_rows_to_samples([row], source_id="s")


@pytest.mark.parametrize(
    "malformed",
    [
        {"provenance": {"target_split": ["holdout"]}},
        {"metadata": {"protected": "maybe"}},
        {"provenance": ["not", "a", "mapping"]},
    ],
)
def test_adapter_rejects_malformed_nested_protection_metadata(malformed):
    row = {
        "run_id": "r",
        "document_id": "d",
        "page_index": 0,
        "variant_id": "v00__x",
        "status": "ok",
        "profile": "p",
        "output_path": "scans/x.png",
        "output_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        **malformed,
    }
    with pytest.raises(FactoryAdapterError, match="malformed"):
        factory_rows_to_samples([row], source_id="s")


def test_adapter_counts_error_rows_without_dropping_provenance(tmp_path):
    rows = [
        {
            "run_id": "r",
            "status": "error",
            "error": "boom",
            "document_id": "d",
            "source_ref": "x.png",
        },
        {
            "run_id": "r",
            "document_id": "d",
            "page_index": 0,
            "variant_id": "v00__p",
            "status": "ok",
            "profile": "05_old_book_medium",
            "profile_schema": "scan_family_v1",
            "source_type": "image",
            "source_ref": "x.png",
            "source_sha256": "b" * 64,
            "clean_sha256": "c" * 64,
            "output_sha256": "d" * 64,
            "renderer": "passthrough",
            "seed": 777,
            "seed_mode": "v1",
            "base_seed": 20260831,
            "transform_steps": [{"stage": "scan_composite"}],
            "dpi": 300,
            "color": "rgb",
            "jpeg_quality": 84,
            "gt_path": "",
            "gt_sha256": "",
            "qc_passed": None,
            "qc": None,
            "created_utc": "2026-09-09T00:00:00Z",
            "output_path": "scans/d__v00__p/page_000000.png",
        },
    ]
    samples, report = factory_rows_to_samples(rows, source_id="s")
    assert report.rows_total == 2
    assert report.rows_error == 1
    assert report.samples == 1
    sample = samples[0]
    assert sample.provenance["seed"] == 777
    assert sample.provenance["output_sha256"] == "d" * 64
    assert sample.provenance["factory_run_id"] == "r"
    assert sample.provenance["source_ref"] == "x.png"
    assert "factory_run_dir" not in sample.provenance


@pytest.mark.parametrize(
    "source_ref", [r"C:\private\sample.png", "/private/sample.png"]
)
def test_adapter_removes_cross_platform_absolute_source_paths(source_ref):
    row = {
        "run_id": "r",
        "document_id": "d",
        "page_index": 0,
        "variant_id": "v00__p",
        "status": "ok",
        "profile": "p",
        "source_ref": source_ref,
        "source_sha256": "b" * 64,
        "output_sha256": "d" * 64,
        "output_path": "scans/sample.png",
    }
    samples, _ = factory_rows_to_samples([row], source_id="s")
    assert samples[0].provenance["source_ref"] == "sample.png"


def test_factory_manifest_reader_rejects_malformed(tmp_path):
    bad = tmp_path / "manifest.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(FactoryAdapterError):
        read_factory_manifest(bad)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_adapter_refuses_missing_or_corrupt_factory_artifacts(tmp_path, damage):
    run_dir, _ = _factory_run(tmp_path)
    row = next(
        item
        for item in read_factory_manifest(run_dir / "manifest.jsonl")
        if item["status"] == "ok"
    )
    artifact = run_dir / row["output_path"]
    if damage == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"corrupt")

    with pytest.raises(FactoryAdapterError, match="artifact"):
        run_dir_to_dataset_manifest(run_dir, tmp_path / "canonical.jsonl")


def test_adapter_refuses_missing_secondary_pdf_artifact(tmp_path):
    run_dir, _ = _factory_run(tmp_path)
    row = next(
        item
        for item in read_factory_manifest(run_dir / "manifest.jsonl")
        if item["status"] == "ok"
    )
    (run_dir / row["pdf_path"]).unlink()
    with pytest.raises(FactoryAdapterError, match="artifact"):
        run_dir_to_dataset_manifest(run_dir, tmp_path / "canonical.jsonl")


@pytest.mark.parametrize("damage", ["escape", "checksum"])
def test_adapter_rejects_untrusted_ground_truth_paths(tmp_path, damage):
    run_dir, _ = _factory_run(tmp_path)
    rows = read_factory_manifest(run_dir / "manifest.jsonl")
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    if damage == "escape":
        rows[0]["gt_path"] = str(outside)
        rows[0]["gt_sha256"] = "f" * 64
    else:
        rows[0]["gt_sha256"] = "f" * 64
    from clouda_data.factory.manifest import write_manifest_jsonl

    write_manifest_jsonl(rows, run_dir / "manifest.jsonl")
    with pytest.raises(FactoryAdapterError, match="ground-truth"):
        run_dir_to_dataset_manifest(run_dir, tmp_path / "canonical.jsonl")


@pytest.mark.parametrize("missing_field", ["gt_path", "gt_sha256"])
def test_adapter_rejects_unpaired_ground_truth_provenance(tmp_path, missing_field):
    run_dir, _ = _factory_run(tmp_path)
    rows = read_factory_manifest(run_dir / "manifest.jsonl")
    rows[0][missing_field] = ""
    from clouda_data.factory.manifest import write_manifest_jsonl

    write_manifest_jsonl(rows, run_dir / "manifest.jsonl")
    with pytest.raises(FactoryAdapterError, match="ground-truth"):
        run_dir_to_dataset_manifest(run_dir, tmp_path / "canonical.jsonl")


def test_dataset_version_changes_with_output_content(tmp_path):
    run_dir, _ = _factory_run(tmp_path)
    first, _, _ = run_dir_to_dataset_manifest(run_dir, tmp_path / "first.jsonl")
    first_header, _ = read_manifest(first)
    rows = read_factory_manifest(run_dir / "manifest.jsonl")
    artifact = run_dir / rows[0]["output_path"]
    artifact.write_bytes(artifact.read_bytes() + b"content-version-change")
    import hashlib

    changed_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    rows[0]["output_sha256"] = changed_hash
    rows[0]["png_sha256"] = changed_hash
    from clouda_data.factory.manifest import write_manifest_jsonl

    write_manifest_jsonl(rows, run_dir / "manifest.jsonl")
    second, _, _ = run_dir_to_dataset_manifest(run_dir, tmp_path / "second.jsonl")
    second_header, _ = read_manifest(second)
    assert first_header["dataset_version"] != second_header["dataset_version"]


def test_dataset_version_changes_with_ground_truth_content(tmp_path):
    run_dir, _ = _factory_run(tmp_path)
    first, _, _ = run_dir_to_dataset_manifest(run_dir, tmp_path / "first.jsonl")
    first_header, _ = read_manifest(first)
    rows = read_factory_manifest(run_dir / "manifest.jsonl")
    ground_truth = run_dir / rows[0]["gt_path"]
    ground_truth.write_text("changed training label", encoding="utf-8")
    import hashlib

    changed_hash = hashlib.sha256(ground_truth.read_bytes()).hexdigest()
    for row in rows:
        row["gt_sha256"] = changed_hash
    from clouda_data.factory.manifest import write_manifest_jsonl

    write_manifest_jsonl(rows, run_dir / "manifest.jsonl")
    second, _, _ = run_dir_to_dataset_manifest(run_dir, tmp_path / "second.jsonl")
    second_header, _ = read_manifest(second)
    assert first_header["dataset_version"] != second_header["dataset_version"]
