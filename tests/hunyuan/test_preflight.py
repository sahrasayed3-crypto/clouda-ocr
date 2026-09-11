"""Preflight tests (local-only, no downloads)."""

from __future__ import annotations

import json
from pathlib import Path

from clouda_training.hunyuan.preflight import run_preflight


def test_preflight_reports_missing_everything(tmp_path: Path) -> None:
    report = run_preflight(model_path=str(tmp_path / "nope"))
    assert report["ok"] is False
    names = {c["name"]: c for c in report["checks"]}
    assert names["model_dir_exists"]["passed"] is False


def test_preflight_with_local_model_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "HunyuanOCR-1.5"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    report = run_preflight(model_path=str(model_dir))
    names = {c["name"]: c for c in report["checks"]}
    assert names["model_dir_exists"]["passed"] is True
    assert names["model_config_json"]["passed"] is True
    # weights missing -> fails (no download!)
    assert names["model_weights_present"]["passed"] is False


def test_preflight_with_data(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    sample = {
        "image_path": ["img.png"],
        "conversations": [
            {"from": "human", "value": "<image>\nاستخرج"},
            {"from": "gpt", "value": "نص"},
        ],
    }
    raw.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    packed = tmp_path / "packed.jsonl"
    packed.write_text(
        json.dumps(
            {
                "packed_samples": [sample],
                "cu_seqlens": [0, 100],
                "total_tokens": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = run_preflight(
        raw_data_path=str(raw), packed_data_path=str(packed), pack_length=20480
    )
    names = {c["name"]: c for c in report["checks"]}
    assert names["raw_jsonl_schema"]["passed"] is True
    assert names["packed_jsonl_schema"]["passed"] is True
