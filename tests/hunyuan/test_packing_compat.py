"""Packing plan + upstream drift detection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.hunyuan.compat import (
    DRIFT_WARNING,
    CompatibilityMetadata,
    check_drift,
)
from clouda_training.hunyuan.packing import build_packing_plan


def _raw(path: Path) -> Path:
    sample = {
        "image_path": ["img.png"],
        "conversations": [
            {"from": "human", "value": "<image>\nاستخرج"},
            {"from": "gpt", "value": "نص"},
        ],
    }
    path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_packing_plan_mode_a(tmp_path: Path) -> None:
    raw = _raw(tmp_path / "raw.jsonl")
    plan = build_packing_plan(raw, pack_length=20480)
    assert plan.raw_validation == {"valid": True, "samples": 1}
    assert plan.pack_length == 20480
    assert plan.env_overrides["PACK_LEN"] == "20480"
    assert "pack_data.sh" in plan.upstream_command
    # artifacts written beside the raw file
    assert Path(plan.input_list_path).is_file()
    plan_file = raw.parent / "packing_plan.json"
    assert plan_file.is_file()
    saved = json.loads(plan_file.read_text(encoding="utf-8"))
    assert saved["upstream_revision"] == "c55965d3da1e"
    assert saved["env_overrides"]["INPUT_LIST"] == plan.input_list_path
    assert raw.name in Path(saved["env_overrides"]["INPUT_LIST"]).read_text(
        encoding="utf-8"
    )


def test_packing_plan_requires_valid_raw(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"broken": true}\n', encoding="utf-8")
    from clouda_training.hunyuan.validators import RawSchemaError

    with pytest.raises(RawSchemaError):
        build_packing_plan(bad)


def test_drift_check_compatible() -> None:
    observed = {
        "model_symbols": [
            "HunYuanVLForConditionalGeneration",
            "AutoTokenizer",
            "AutoProcessor",
        ],
        "data_classes": ["VLDataset", "PackedVLDataCollator", "VLDataCollator"],
        "training_entry": "train/train_hunyuan.py",
        "pack_fields": ["packed_samples", "cu_seqlens", "total_tokens"],
        "raw_fields": ["image_path", "conversations"],
        "pack_length": 20480,
        "sft_profile": {
            "lr": 2e-5,
            "epochs": 5,
            "batch_size": 1,
            "grad_accum": 1,
            "save_steps": 200,
        },
    }
    result = check_drift(observed)
    assert result["compatible"] is True
    assert result["warning"] is None


def test_drift_check_detects_changes() -> None:
    observed = {
        "model_symbols": ["AutoTokenizer"],  # model class renamed upstream
        "pack_length": 16384,
        "training_entry": "train/some_new.py",
    }
    result = check_drift(observed)
    assert result["compatible"] is False
    assert result["warning"] == DRIFT_WARNING
    assert any("model symbols" in d for d in result["drift"])
    assert any("pack length" in d for d in result["drift"])
    assert any("training entry" in d for d in result["drift"])


def test_compatibility_metadata_roundtrip(tmp_path: Path) -> None:
    meta = CompatibilityMetadata()
    p = tmp_path / "compat.json"
    meta.save(p)
    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["upstream_revision"] == "c55965d3da1e"
    assert saved["expected_pack_length"] == 20480
    assert saved["expected_training_entry"] == "train/train_hunyuan.py"
