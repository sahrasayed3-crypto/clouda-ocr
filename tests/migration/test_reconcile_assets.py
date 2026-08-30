from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.migration.reconcile_assets import main, reconcile_assets, summarize

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "dataset_catalog" / "registry" / "datasets_v1.json"


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    asset = source / "data" / "downloads" / "rasam_dataset" / "page.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"page")
    manifest = source / "data" / "manifests" / "rasam_first_batch_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"total_pages": 100, "valid_pages": 88, "rejected_pages": 12}),
        encoding="utf-8",
    )
    return source


def test_reconciliation_is_dry_run_until_apply(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    results = reconcile_assets(
        source_root=source,
        destination_root=destination,
        catalog_path=CATALOG,
        apply=False,
    )
    assert not destination.exists()
    assert results[0].copy_result == "planned"


def test_apply_copies_and_checksum_verifies_without_overwrite(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    reports = tmp_path / "reports"
    assert (
        main(
            [
                "--source-root",
                str(source),
                "--destination-root",
                str(destination),
                "--catalog",
                str(CATALOG),
                "--report-dir",
                str(reports),
                "--apply",
            ]
        )
        == 0
    )
    copied = destination / "data" / "downloads" / "rasam_dataset" / "page.jpg"
    assert (
        hashlib.sha256(copied.read_bytes()).hexdigest()
        == hashlib.sha256(b"page").hexdigest()
    )
    second = reconcile_assets(
        source_root=source,
        destination_root=destination,
        catalog_path=CATALOG,
        apply=True,
    )
    assert second[0].copy_result == "already_present"
    assert second[0].verification_result == "verified"


def test_rasam_baseline_is_preserved(tmp_path: Path) -> None:
    source = _source(tmp_path)
    results = reconcile_assets(
        source_root=source,
        destination_root=tmp_path / "destination",
        catalog_path=CATALOG,
        apply=False,
    )
    assert summarize(results, source_root=source)["rasam_baseline"] == {
        "source_pages": 100,
        "valid_pages": 88,
        "rejected_pages": 12,
    }
