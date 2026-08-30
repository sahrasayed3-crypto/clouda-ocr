"""Validate the public-safe Arabic OCR benchmark metadata release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
FORBIDDEN_SUFFIXES = {
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
FORBIDDEN_TEXT = (
    "c:" + "\\users\\ahmed",
    "downloads" + "\\ocr",
    "downloads" + "/ocr",
    "le" + "gal\\",
    "le" + "gal/",
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_release(release_root: Path) -> list[str]:
    """Return human-readable validation errors for a release root."""

    errors: list[str] = []
    required = {
        "README.md",
        "RESULTS.md",
        "results.csv",
        "methodology.md",
        "benchmark_manifest.jsonl",
        "source_manifest.csv",
        "source_summary.csv",
        "models.csv",
        "CITATIONS.md",
        "release.json",
    }
    missing = sorted(name for name in required if not (release_root / name).is_file())
    errors.extend(f"missing required file: {name}" for name in missing)
    if missing:
        return errors

    manifest_path = release_root / "benchmark_manifest.jsonl"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != EXPECTED_MANIFEST_SHA256:
        errors.append(f"unexpected benchmark manifest SHA-256: {digest}")

    manifest = _load_jsonl(manifest_path)
    if len(manifest) != 177:
        errors.append(f"expected 177 benchmark rows, found {len(manifest)}")
    if len({str(row.get("distorted_id")) for row in manifest}) != len(manifest):
        errors.append("distorted_id values are not unique")

    sources = _load_csv(release_root / "source_manifest.csv")
    if len(sources) != 100:
        errors.append(f"expected 100 source rows, found {len(sources)}")
    if len({row.get("asset_id", "") for row in sources}) != len(sources):
        errors.append("asset_id values are not unique")

    derivative_counts = Counter(str(row.get("benchmark_id")) for row in manifest)
    for row in sources:
        asset_id = row.get("asset_id", "")
        try:
            declared = int(row.get("derivative_count", ""))
        except ValueError:
            errors.append(f"invalid derivative_count for {asset_id}")
            continue
        if derivative_counts[asset_id] != declared:
            errors.append(f"derivative count mismatch for {asset_id}")
        for field in ("source_sha256", "clean_sha256", "ground_truth_sha256"):
            value = row.get(field, "")
            if len(value) != 64:
                errors.append(f"invalid {field} for {asset_id}")

    family_counts = Counter(row.get("source_family", "") for row in sources)
    if dict(family_counts) != EXPECTED_FAMILY_COUNTS:
        errors.append(f"unexpected source composition: {dict(family_counts)}")
    precise_sources = {row.get("dataset_source", "") for row in sources}
    if len(precise_sources) != 14:
        errors.append(
            f"expected 14 precise dataset identifiers, found {len(precise_sources)}"
        )
    if any(
        blocked in source.lower()
        for source in precise_sources
        for blocked in ("muharaf", "baseer", "misraj")
    ):
        errors.append("blocked dataset identifier present")

    results = _load_csv(release_root / "results.csv")
    ranked = [row for row in results if row.get("rankable") == "true"]
    if [row.get("model") for row in ranked] != EXPECTED_RANKING:
        errors.append("leaderboard order does not match the canonical ranking")
    try:
        scores = [float(row.get("normalized_arabic_cer", "")) for row in ranked]
    except ValueError:
        errors.append("ranked result has an invalid normalized Arabic CER")
    else:
        if scores != sorted(scores):
            errors.append("leaderboard is not sorted by normalized Arabic CER")
    if any(row.get("pages") != "177" for row in ranked):
        errors.append("ranked result does not cover 177 pages")
    excluded = {
        row.get("model"): row for row in results if row.get("rankable") == "false"
    }
    if excluded.get("dots.mocr", {}).get("status") != "PARTIAL":
        errors.append("dots.mocr partial status is missing")
    if excluded.get("PaddleOCR-VL-1.6", {}).get("status") != "FAILED_SMOKE":
        errors.append("PaddleOCR-VL-1.6 failed-smoke status is missing")
    if any(row.get("rank") for row in excluded.values()):
        errors.append("partial or failed run has a rank")

    models = _load_csv(release_root / "models.csv")
    if len(models) != 8:
        errors.append(f"expected 8 model records, found {len(models)}")

    release = json.loads((release_root / "release.json").read_text(encoding="utf-8"))
    expected_release_values = {
        "clean_source_count": 100,
        "distorted_page_count": 177,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "primary_metric": "normalized_arabic_cer",
        "complete_run_count": 6,
    }
    for key, expected in expected_release_values.items():
        if release.get(key) != expected:
            errors.append(f"release.json has unexpected {key}: {release.get(key)!r}")

    public_files = [
        path
        for path in release_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    for path in public_files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden asset type: {path.relative_to(release_root)}")
            continue
        text = path.read_text(encoding="utf-8").lower()
        if any(marker in text for marker in FORBIDDEN_TEXT):
            errors.append(f"private path marker: {path.relative_to(release_root)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    errors = validate_release(args.release_root.resolve())
    if errors:
        print("OCR Arabic benchmark metadata validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("OCR Arabic benchmark metadata validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
