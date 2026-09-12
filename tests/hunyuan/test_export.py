"""Export tests: Arabic preservation, protection, portability, lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.hunyuan.exporter import export_raw_jsonl


def test_export_produces_valid_upstream_schema(export_config, tmp_path: Path) -> None:
    out = tmp_path / "raw.jsonl"
    report = export_raw_jsonl(export_config, out)
    assert report.exported_count == 3  # holdout + protected rows rejected
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert list(first) == ["image_path", "conversations"]  # upstream wire format
    assert first["image_path"][0].startswith(str(tmp_path))  # absolutized
    assert first["conversations"][0]["from"] == "human"
    assert first["conversations"][0]["value"].startswith("<image>\n")
    assert first["conversations"][1]["from"] == "gpt"


def test_arabic_gt_preserved_exactly(export_config, tmp_path: Path) -> None:
    out = tmp_path / "raw.jsonl"
    export_raw_jsonl(export_config, out)
    gts = []
    for line in out.read_text(encoding="utf-8").splitlines():
        gts.append(json.loads(line)["conversations"][1]["value"])
    assert "# عنوان\n\nهذا نص عربي تجريبي للاختبار." in gts
    assert "Mixed نص عربي with English 123 and أرقام عربية ٤٥٦." in gts
    assert "سطر أول\nسطر ثانٍ\n\nفقرة جديدة مع |جدول|نصي|." in gts
    # diacritics survive the UTF-8 round-trip (checked via exact matches above)


def test_protected_and_holdout_never_exported(export_config, tmp_path: Path) -> None:
    out = tmp_path / "raw.jsonl"
    report = export_raw_jsonl(export_config, out)
    assert report.skipped_counts.get("protected", 0) >= 1
    all_text = out.read_text(encoding="utf-8")
    # the protected row's GT value must never appear in the export
    assert '"protected"' not in all_text
    assert "holdout" not in [sid for sid in report.sample_ids]


def test_holdout_split_export_fails_closed(export_config, tmp_path: Path) -> None:
    from dataclasses import replace

    bad_config = replace(export_config, split="holdout")
    with pytest.raises(PermissionError):
        export_raw_jsonl(bad_config, tmp_path / "bad.jsonl")


def test_manifest_untouched_and_lineage_recorded(
    export_config, canonical_manifest: Path, tmp_path: Path
) -> None:
    import hashlib

    before = hashlib.sha256(canonical_manifest.read_bytes()).hexdigest()
    out = tmp_path / "raw.jsonl"
    export_raw_jsonl(export_config, out)
    after = hashlib.sha256(canonical_manifest.read_bytes()).hexdigest()
    assert before == after  # canonical manifest untouched

    report_data = json.loads(
        out.with_suffix(".report.json").read_text(encoding="utf-8")
    )
    assert report_data["manifest_hash"] == before
    assert report_data["dataset_id"] == "synthetic-ar"
    assert report_data["prompt_identity"] == "clouda_arabic_document_ocr@v1"
    assert report_data["upstream_revision"] == "c55965d3da1e"
    assert report_data["output_sha256"]
    assert set(report_data["sample_ids"]) == {"ar-001", "ar-002", "ar-003"}


def test_absolute_paths_only_in_generated_artifact(
    export_config, canonical_manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "raw.jsonl"
    export_raw_jsonl(export_config, out)
    canonical_text = canonical_manifest.read_text(encoding="utf-8")
    assert str(tmp_path) not in canonical_text  # canonical stays relative/portable
    # JSON escapes backslashes, so check parsed image paths instead of raw text
    generated_paths = [
        json.loads(line)["image_path"][0]
        for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert generated_paths, "export produced no samples"
    assert all(p.startswith(str(tmp_path)) for p in generated_paths)


def test_missing_gt_or_image_skips_row(tmp_path: Path) -> None:
    from clouda_training.hunyuan.models import (
        ARABIC_DOCUMENT_OCR_PROMPT,
        HunyuanExportConfig,
    )
    import hashlib

    rows = [
        {"_schema_version": "clouda.pretraining.manifest.v1", "_row_count": 2},
        {
            "sample_id": "a",
            "target_split": "train",
            "image_path": "a.png",
            "ground_truth": "نص",
            "source_id": "s",
        },
        {
            "sample_id": "b",
            "target_split": "train",
            "image_path": "b.png",
            "source_id": "s",
        },  # missing GT
    ]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    config = HunyuanExportConfig(
        dataset_id="s",
        dataset_version="v1",
        manifest_path=str(manifest),
        manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        split="train",
        image_root=str(tmp_path),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )
    out = tmp_path / "raw.jsonl"
    report = export_raw_jsonl(config, out)
    assert report.exported_count == 1
    assert report.skipped_counts.get("missing_gt") == 1


def test_duplicate_sample_id_skipped(tmp_path: Path) -> None:
    from clouda_training.hunyuan.models import (
        ARABIC_DOCUMENT_OCR_PROMPT,
        HunyuanExportConfig,
    )
    import hashlib

    row = {
        "sample_id": "dup",
        "target_split": "train",
        "image_path": "a.png",
        "ground_truth": "نص",
        "source_id": "s",
    }
    rows = [
        {"_schema_version": "clouda.pretraining.manifest.v1", "_row_count": 2},
        row,
        dict(row),
    ]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    config = HunyuanExportConfig(
        dataset_id="s",
        dataset_version="v1",
        manifest_path=str(manifest),
        manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        split="train",
        image_root=str(tmp_path),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )
    out = tmp_path / "raw.jsonl"
    report = export_raw_jsonl(config, out)
    assert report.exported_count == 1
    assert report.skipped_counts.get("duplicate_sample_id") == 1


def test_manifest_hash_mismatch_refuses(
    export_config, canonical_manifest: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    bad = replace(export_config, manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="manifest_hash"):
        export_raw_jsonl(bad, tmp_path / "raw.jsonl")
