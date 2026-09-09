"""Tiny offline CPU-only E2E for the integrated Data Factory.

Covers the whole canonical chain in one test with tiny local fixtures:

1. tiny Arabic source (image + ground-truth text, generated in-memory);
2. clean artifacts (searchable PDF via img2pdf page assembly on this host;
   WeasyPrint/RAQM text backends are covered separately where natives exist);
3. distorted raster output (atomic + composite profiles);
4. QC record;
5. exports (PNG + image-only PDFs);
6. JSONL + CSV manifests + run metadata;
7. SHA-256 provenance for source and outputs;
8. deterministic seed recorded per row;
9. post-run hash verification (factory-verify);
10. manifest-driven resume;
11. conversion to the canonical pre-training manifest;
12. Training Experiment Framework dry-run on the result.

No network, no model downloads, no paid APIs, no large artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from clouda_data.factory.adapters import run_dir_to_dataset_manifest
from clouda_data.factory.autorun import auto_run
from clouda_data.factory.factory import generate_run
from clouda_data.factory.manifest import read_manifest as read_factory_manifest
from clouda_data.pretraining.manifest import read_manifest as read_canonical_manifest

REPO = Path(__file__).resolve().parents[2]


def _tiny_source(tmp_path: Path) -> Path:
    inbox = tmp_path / "input"
    inbox.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (420, 594), (250, 247, 240))  # A6-ish
    draw = ImageDraw.Draw(img)
    y = 40
    for _ in range(8):
        draw.text((390, y), "0" * 20, fill=(20, 20, 20))
        y += 28
    img.save(inbox / "tiny_page.png")
    (inbox / "tiny_page.txt").write_text("هذا نص تجريبي قصير\n", encoding="utf-8")
    return inbox


def test_tiny_offline_e2e_full_chain(tmp_path):
    inbox = _tiny_source(tmp_path)

    # -- 1-5: one-command run (autorun picks sensible defaults) -------------
    summary = auto_run(inbox, tmp_path / "out", variants=2, workers=1, max_pages=2)
    assert summary["inputs"]["image"] == 1
    assert summary["outputs_ok"] >= 2
    assert summary["verification"]["passed"] is True
    run_dir = tmp_path / "out" / summary["run_id"]

    # -- 6: manifests + metadata exist --------------------------------------
    assert (run_dir / "manifest.jsonl").is_file()
    assert (run_dir / "manifest.csv").is_file()
    assert (run_dir / "metadata.json").is_file()
    rows = read_factory_manifest(run_dir / "manifest.jsonl")
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert ok_rows

    # -- 7/8: provenance + deterministic seeds recorded ----------------------
    import hashlib

    for row in ok_rows:
        assert (
            row["source_sha256"]
            == hashlib.sha256((inbox / "tiny_page.png").read_bytes()).hexdigest()
        )
        assert isinstance(row["seed"], int)
        assert row["seed_mode"] == "v1"
        assert row["profile"] in summary["profiles"]
        assert row["transform_steps"]
        out = run_dir / row["output_path"]
        assert out.is_file()
        assert hashlib.sha256(out.read_bytes()).hexdigest() == row["output_sha256"]

    # -- 9: CLI verification -------------------------------------------------
    verify = subprocess.run(
        [sys.executable, "-m", "clouda_data.factory", "verify", str(run_dir)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    payload = json.loads(verify.stdout)
    assert payload["mismatches"] == []

    # -- 10: resume is idempotent --------------------------------------------
    summary2 = auto_run(inbox, tmp_path / "out", variants=2, workers=1, max_pages=2)
    assert summary2["run_id"] == summary["run_id"]
    assert summary2["skipped_resumed"] == summary["outputs_ok"]
    assert summary2["new_this_run"] == 0

    # -- 11: canonical pre-training manifest ----------------------------------
    manifest_path, report, manifest_hash = run_dir_to_dataset_manifest(
        run_dir,
        tmp_path / "dataset_manifest.jsonl",
        source_id="factory_e2e",
    )
    assert report.samples == len(ok_rows)
    assert len(manifest_hash) == 64
    header, canonical_rows = read_canonical_manifest(manifest_path)
    assert header["_schema_version"] == "clouda.pretraining.manifest.v1"
    assert len(canonical_rows) == len(ok_rows)

    # -- 12: Training Experiment Framework dry-run ---------------------------
    from clouda_training.experiments import run_experiment
    from clouda_training.experiments.config import load_experiment_config

    config_yaml = tmp_path / "experiment.yaml"
    config_yaml.write_text(
        f"""
schema_version: 1
experiment:
  name: factory-e2e-dry-run
  description: Tiny offline factory E2E dry-run.
  tags: [e2e, factory, offline]
model:
  model_id: mock/clouda-ocr
  revision: fixture-v1
  model_family: multimodal-ocr
  adapter_type: mock
  precision: float32
dataset:
  dataset_id: factory-e2e-dataset
  dataset_version: v1
  manifest_path: {manifest_path.as_posix()}
  split: train
  sample_limit: 2
  preprocessing_version: clouda.pretraining.normalize.v1
training:
  seed: 20260909
  epochs: 1
  max_steps: 2
  batch_size: 1
  learning_rate: 0.0001
checkpoint:
  save_strategy: steps
  save_steps: 2
  save_total_limit: 1
evaluation:
  enabled: false
runtime:
  device: cpu
  num_workers: 0
  output_root: {tmp_path / "training_runs"}
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
    handle = run_experiment(load_experiment_config(config_yaml))
    assert handle.status.value == "COMPLETED"
    metadata = json.loads((handle.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_manifest_hash"] == manifest_hash
    assert metadata["dataset_split"] == "train"


def test_tiny_e2e_deterministic_across_repeat(tmp_path):
    """Same input + same seed + same profiles → same output hashes."""
    inbox1 = _tiny_source(tmp_path / "a")
    inbox2 = _tiny_source(tmp_path / "b")
    meta1 = generate_run(
        inputs=[inbox1 / "tiny_page.png"],
        runs_root=tmp_path / "runs1",
        profile_names=["05_old_book_medium"],
        variants=1,
        base_seed=777,
        seed_mode="v1",
        backend="raqm",
        workers=1,
        export_pdf=True,
        export_png=True,
        max_pages=2,
    )
    meta2 = generate_run(
        inputs=[inbox2 / "tiny_page.png"],
        runs_root=tmp_path / "runs2",
        profile_names=["05_old_book_medium"],
        variants=1,
        base_seed=777,
        seed_mode="v1",
        backend="raqm",
        workers=1,
        export_pdf=True,
        export_png=True,
        max_pages=2,
    )
    rows1 = sorted(
        (r["variant_id"], r["output_sha256"])
        for r in read_factory_manifest(
            (tmp_path / "runs1" / meta1["run_id"]) / "manifest.jsonl"
        )
        if r["status"] == "ok"
    )
    rows2 = sorted(
        (r["variant_id"], r["output_sha256"])
        for r in read_factory_manifest(
            (tmp_path / "runs2" / meta2["run_id"]) / "manifest.jsonl"
        )
        if r["status"] == "ok"
    )
    assert rows1 and rows1 == rows2
