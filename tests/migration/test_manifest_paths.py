from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.migration.manifest_paths import main, migrate_records, storage_uri


def _record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "original_path": str(path),
        "canonical_path": "data/raw/pages/example.png",
        "file_role": "page",
        "source_type": "image",
        "checksum": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "duplicate_of": None,
    }


def test_storage_uri_maps_data_and_outputs() -> None:
    assert storage_uri(r"E:\old\data\raw\page.png") == ("dataset://data/raw/page.png")
    assert storage_uri("outputs/reports/result.json") == (
        "artifact://reports/result.json"
    )


def test_migration_preserves_legacy_path_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "data" / "downloads" / "page.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    migrated, verification = migrate_records(
        [_record(source)],
        source_root=tmp_path,
    )
    assert migrated[0]["legacy_original_path"] == str(source)
    assert migrated[0]["original_path"] == "dataset://data/downloads/page.png"
    assert migrated[0]["path_schema_version"] == 1
    assert verification[0].status == "verified"


def test_cli_is_dry_run_by_default_and_apply_writes_sanitized_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data" / "raw" / "page.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps([_record(source)]), encoding="utf-8")
    output = tmp_path / "state" / "migrated.json"
    reports = tmp_path / "state" / "reports"

    assert (
        main(
            [
                "--input",
                str(manifest),
                "--source-root",
                str(tmp_path),
                "--output",
                str(output),
                "--report-dir",
                str(reports),
            ]
        )
        == 0
    )
    assert not output.exists()

    assert (
        main(
            [
                "--input",
                str(manifest),
                "--source-root",
                str(tmp_path),
                "--output",
                str(output),
                "--report-dir",
                str(reports),
                "--apply",
            ]
        )
        == 0
    )
    assert output.is_file()
    for report in reports.iterdir():
        assert str(tmp_path) not in report.read_text(encoding="utf-8")
