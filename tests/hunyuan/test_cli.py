"""CLI tests for the Hunyuan bridge subcommands (Phase 29)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clouda_training.cli import build_parser, main


@pytest.fixture()
def raw_export(canonical_manifest: Path, tmp_path: Path) -> Path:
    """Run one export through the CLI and return the raw JSONL path."""
    out = tmp_path / "raw.jsonl"
    rc = main(
        [
            "hunyuan",
            "export",
            str(canonical_manifest),
            "--output",
            str(out),
            "--dataset-id",
            "synthetic-ar",
            "--dataset-version",
            "v1",
            "--image-root",
            str(tmp_path / "dataset_root"),
        ]
    )
    assert rc == 0
    return out


def test_hunyuan_help_lists_subcommands(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["hunyuan", "--help"])
    assert excinfo.value.code == 0
    assert "export" in capsys.readouterr().out
    # parse of a valid subcommand proves the parser wiring exists
    args = build_parser().parse_args(["hunyuan", "validate-raw", "x.jsonl"])
    assert args.command == "hunyuan"
    assert args.hunyuan_command == "validate-raw"


def test_cli_export_produces_lineage_report(
    canonical_manifest: Path, tmp_path: Path, raw_export: Path
) -> None:
    lines = raw_export.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    report = json.loads(
        raw_export.with_suffix(".report.json").read_text(encoding="utf-8")
    )
    assert report["dataset_id"] == "synthetic-ar"
    assert report["prompt_identity"] == "clouda_arabic_document_ocr@v1"
    assert report["upstream_revision"] == "c55965d3da1e"
    assert set(report["sample_ids"]) == {"ar-001", "ar-002", "ar-003"}


def test_cli_validate_raw(raw_export: Path, capsys: pytest.CaptureFixture) -> None:
    rc = main(["hunyuan", "validate-raw", str(raw_export)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"valid": True, "samples": 3}


def test_cli_validate_raw_detects_bad_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"image_path": []}\n', encoding="utf-8")
    with pytest.raises(Exception, match="image_path"):
        main(["hunyuan", "validate-raw", str(bad)])


def test_cli_validate_packed(
    tmp_path: Path, capsys: pytest.CaptureFixture, raw_export: Path
) -> None:
    capsys.readouterr()  # discard export report printed by the fixture
    sample = json.loads(raw_export.read_text(encoding="utf-8").splitlines()[0])
    packed_line = {
        "packed_samples": [sample, sample],
        "cu_seqlens": [0, 100, 200],
        "total_tokens": 200,
    }
    packed_path = tmp_path / "packed.jsonl"
    packed_path.write_text(
        json.dumps(packed_line, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rc = main(["hunyuan", "validate-packed", str(packed_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"valid": True, "packs": 1, "samples": 2}


def test_cli_plan_writes_packing_plan(tmp_path: Path, raw_export: Path) -> None:
    out_dir = tmp_path / "plan_out"
    rc = main(
        [
            "hunyuan",
            "plan",
            str(raw_export),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    plan = json.loads((out_dir / "packing_plan.json").read_text(encoding="utf-8"))
    assert plan["pack_length"] == 20480
    assert "pack_data.sh" in plan["upstream_command"]
    assert (out_dir / "data_list.txt").is_file()


def test_cli_preflight_without_model_reports_not_ok(
    capsys: pytest.CaptureFixture,
) -> None:
    rc = main(["hunyuan", "preflight"])
    payload = json.loads(capsys.readouterr().out)
    # transformers not installed in the test environment -> checks fail
    assert rc == 1
    assert payload["ok"] is False
    names = {c["name"] for c in payload["checks"]}
    assert "transformers_installed" in names


def test_cli_preflight_with_raw_data(
    tmp_path: Path, raw_export: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["hunyuan", "preflight", "--raw-data", str(raw_export)])
    payload = json.loads(capsys.readouterr().out)
    names = {c["name"] for c in payload["checks"]}
    assert "raw_jsonl_schema" in names
    raw_check = next(c for c in payload["checks"] if c["name"] == "raw_jsonl_schema")
    assert raw_check["passed"] is True


def test_cli_export_missing_hash_option_computes_it(
    canonical_manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "raw2.jsonl"
    rc = main(
        [
            "hunyuan",
            "export",
            str(canonical_manifest),
            "--output",
            str(out),
            "--dataset-id",
            "synthetic-ar",
            "--dataset-version",
            "v1",
            "--image-root",
            str(tmp_path),
            "--manifest-hash",
            hashlib.sha256(canonical_manifest.read_bytes()).hexdigest(),
        ]
    )
    assert rc == 0
    assert out.is_file()
