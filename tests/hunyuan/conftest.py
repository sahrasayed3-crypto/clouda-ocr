"""Shared fixtures for Hunyuan bridge tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.hunyuan.models import (
    ARABIC_DOCUMENT_OCR_PROMPT,
    HunyuanExportConfig,
)


@pytest.fixture()
def canonical_manifest(tmp_path: Path) -> Path:
    """Synthetic canonical Arabic training manifest (rows use canonical relative paths)."""
    rows = [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": 5,
            "dataset_id": "synthetic-ar",
            "dataset_version": "v1",
        },
        {
            "sample_id": "ar-001",
            "target_split": "train",
            "source_id": "synthetic-ar",
            "image_path": "pages/page_001.png",
            "ground_truth": "# عنوان\n\nهذا نص عربي تجريبي للاختبار.",
            "source_license": "Apache-2.0",
        },
        {
            "sample_id": "ar-002",
            "target_split": "train",
            "source_id": "synthetic-ar",
            "image_path": "pages/page_002.png",
            "ground_truth": "Mixed نص عربي with English 123 and أرقام عربية ٤٥٦.",
            "source_license": "Apache-2.0",
        },
        {
            "sample_id": "ar-003",
            "target_split": "train",
            "source_id": "synthetic-ar",
            "image_path": "pages/page_003.png",
            "ground_truth": "سطر أول\nسطر ثانٍ\n\nفقرة جديدة مع |جدول|نصي|.",
            "source_license": "Apache-2.0",
        },
        {
            "sample_id": "holdout-001",
            "target_split": "holdout",
            "source_id": "synthetic-ar",
            "image_path": "pages/holdout.png",
            "ground_truth": "protected",
        },
        {
            "sample_id": "ar-prot",
            "target_split": "train",
            "source_id": "synthetic-ar",
            "image_path": "pages/prot.png",
            "ground_truth": "x",
            "protected": True,
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def export_config(canonical_manifest: Path, tmp_path: Path):
    import hashlib

    manifest_hash = hashlib.sha256(canonical_manifest.read_bytes()).hexdigest()
    return HunyuanExportConfig(
        dataset_id="synthetic-ar",
        dataset_version="v1",
        manifest_path=str(canonical_manifest),
        manifest_hash=manifest_hash,
        split="train",
        image_root=str(tmp_path / "dataset_root"),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )
