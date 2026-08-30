from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "benchmarks" / "ocr_arabic"
VALIDATOR = RELEASE_ROOT / "scripts" / "validate_release.py"
EXPECTED_MANIFEST_SHA256 = (
    "2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893"
)
EXPECTED_RANKING = [
    "HunyuanOCR-1.5",
    "MBZUAI/AIN-7B",
    "Qari OCR 0.4.0",
    "Qwen3-VL-4B-Instruct",
    "DeepSeek-OCR-2",
    "Arabic Nougat Large",
]
EXPECTED_FAMILY_COUNTS = {
    "KITAB-Bench-derived datasets": 30,
    "Arabic E-Book Corpus": 20,
    "CALFA datasets": 30,
    "Arabic-img2md": 15,
    "Craneset free samples": 5,
}


def _read_csv(name: str) -> list[dict[str, str]]:
    with (RELEASE_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(name: str) -> list[dict[str, object]]:
    with (RELEASE_ROOT / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_release_validator_accepts_canonical_metadata() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--release-root", str(RELEASE_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OCR Arabic benchmark metadata validation: PASS" in result.stdout


def test_canonical_manifest_identity_and_page_count() -> None:
    manifest_path = RELEASE_ROOT / "benchmark_manifest.jsonl"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = _read_jsonl("benchmark_manifest.jsonl")

    assert digest == EXPECTED_MANIFEST_SHA256
    assert len(rows) == 177
    assert len({str(row["distorted_id"]) for row in rows}) == 177


def test_source_manifest_has_one_hundred_valid_provenance_records() -> None:
    source_rows = _read_csv("source_manifest.csv")
    benchmark_rows = _read_jsonl("benchmark_manifest.jsonl")
    derivatives = Counter(str(row["benchmark_id"]) for row in benchmark_rows)

    assert len(source_rows) == 100
    assert len({row["asset_id"] for row in source_rows}) == 100
    for row in source_rows:
        assert derivatives[row["asset_id"]] == int(row["derivative_count"])
        assert len(row["source_sha256"]) == 64
        assert len(row["clean_sha256"]) == 64
        assert len(row["ground_truth_sha256"]) == 64
        assert row["evidence_id"]


def test_source_composition_preserves_precise_dataset_provenance() -> None:
    source_rows = _read_csv("source_manifest.csv")
    family_counts = Counter(row["source_family"] for row in source_rows)
    precise_sources = {row["dataset_source"] for row in source_rows}

    assert dict(family_counts) == EXPECTED_FAMILY_COUNTS
    assert len(precise_sources) == 14
    assert not any("muharaf" in source.lower() for source in precise_sources)
    assert not any(
        blocked in source.lower()
        for source in precise_sources
        for blocked in ("baseer", "misraj")
    )


def test_leaderboard_is_ranked_only_by_normalized_arabic_cer() -> None:
    rows = _read_csv("results.csv")
    ranked = [row for row in rows if row["rankable"] == "true"]

    assert [row["model"] for row in ranked] == EXPECTED_RANKING
    assert [int(row["rank"]) for row in ranked] == list(range(1, 7))
    scores = [float(row["normalized_arabic_cer"]) for row in ranked]
    assert scores == sorted(scores)
    assert ranked[0]["model"] == "HunyuanOCR-1.5"
    assert all(int(row["pages"]) == 177 for row in ranked)


def test_partial_and_failed_runs_are_not_ranked() -> None:
    rows = _read_csv("results.csv")
    excluded = {row["model"]: row for row in rows if row["rankable"] == "false"}

    assert excluded["dots.mocr"]["status"] == "PARTIAL"
    assert excluded["dots.mocr"]["pages"] == "30"
    assert excluded["dots.mocr"]["rank"] == ""
    assert excluded["PaddleOCR-VL-1.6"]["status"] == "FAILED_SMOKE"
    assert excluded["PaddleOCR-VL-1.6"]["pages"] == "0"
    assert excluded["PaddleOCR-VL-1.6"]["rank"] == ""


def test_rights_dimensions_and_model_permissions_are_separate() -> None:
    source_rows = _read_csv("source_manifest.csv")
    model_rows = _read_csv("models.csv")
    permission_values = {"ALLOWED", "NOT_ALLOWED", "NEEDS_REVIEW"}

    assert len(model_rows) == 8
    for row in source_rows:
        assert row["evaluation_permission"] in permission_values
        assert row["commercial_training_permission"] in permission_values
        assert row["asset_redistribution_permission"] in permission_values
        if row["asset_redistribution_permission"] == "NEEDS_REVIEW":
            assert row["rights_decision"] in {"LOCAL_ONLY", "NEEDS_REVIEW"}
    for row in model_rows:
        assert row["model_license_status"]
        assert row["raw_output_redistribution"] in permission_values
        assert row["training_label_permission"] in permission_values


def test_public_release_contains_metadata_only_and_no_personal_paths() -> None:
    forbidden_suffixes = {
        ".bmp",
        ".docx",
        ".eml",
        ".gif",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
    forbidden_text = (
        "c:" + "\\users\\" + "ahmed",
        "downloads" + "\\ocr",
        "downloads" + "/ocr",
        "legal" + "\\",
        "legal" + "/",
    )

    files = [
        path
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    assert files
    assert not any(path.suffix.lower() in forbidden_suffixes for path in files)
    for path in files:
        content = path.read_text(encoding="utf-8").lower()
        assert not any(marker in content for marker in forbidden_text), path


def test_public_metadata_is_internally_consistent() -> None:
    release = json.loads((RELEASE_ROOT / "release.json").read_text(encoding="utf-8"))
    results = _read_csv("results.csv")
    sources = _read_csv("source_manifest.csv")
    manifest = _read_jsonl("benchmark_manifest.jsonl")

    assert release["benchmark_id"] == "clouda-ocr-arabic-177-v1"
    assert release["clean_source_count"] == len(sources) == 100
    assert release["distorted_page_count"] == len(manifest) == 177
    assert release["complete_run_count"] == sum(
        row["status"] == "COMPLETE" for row in results
    )
    assert release["primary_metric"] == "normalized_arabic_cer"
    assert release["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
