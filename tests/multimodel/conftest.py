"""Shared fixtures for the multimodel adapter tests.

Registration is EXPLICIT (no import side effects): this conftest performs the
idempotent registration both adapter packages require, mirroring what the
lead's CLI wiring will do in production.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from clouda_training.hunyuan.models import (
    ARABIC_DOCUMENT_OCR_PROMPT,
    HunyuanExportConfig,
)
from clouda_training.hunyuan.registration import register_hunyuan_adapters
from clouda_training.qwen.data_adapter import QwenExportConfig
from clouda_training.qwen.registration import register_qwen_adapters

register_hunyuan_adapters()
register_qwen_adapters()


CANONICAL_ROWS: list[dict[str, Any]] = [
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
        "ground_truth": "Mixed نص عربي with English 123.",
        "source_license": "Apache-2.0",
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


@pytest.fixture()
def canonical_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in CANONICAL_ROWS]


@pytest.fixture()
def qwen_export_config(tmp_path: Path) -> QwenExportConfig:
    return QwenExportConfig(image_root=str(tmp_path / "dataset_root"))


@pytest.fixture()
def hunyuan_export_config(tmp_path: Path) -> HunyuanExportConfig:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in CANONICAL_ROWS),
        encoding="utf-8",
    )
    return HunyuanExportConfig(
        dataset_id="synthetic-ar",
        dataset_version="v1",
        manifest_path=str(manifest),
        manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        split="train",
        image_root=str(tmp_path / "dataset_root"),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )


@pytest.fixture()
def mock_checkpoint_manager(tmp_path: Path):
    """Minimal CheckpointManager stand-in for the backend __new__ shim."""

    from types import SimpleNamespace

    class MockCheckpointManager:
        def __init__(self) -> None:
            self.root = tmp_path / "checkpoints"
            self.saved: list[Path] = []

        def save(self, **kw: Any) -> Any:
            directory = self.root / f"step-{kw['step']:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            self.saved.append(directory)
            return SimpleNamespace(path=directory)

        def latest(self) -> Any:
            return SimpleNamespace(path=self.saved[-1]) if self.saved else None

    return MockCheckpointManager()
