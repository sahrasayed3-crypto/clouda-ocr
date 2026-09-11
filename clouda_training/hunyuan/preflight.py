"""Local-only compatibility preflight for HunyuanOCR-1.5 training.

Checks what CAN be checked without weights/GPU/network. Never downloads,
never starts training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clouda_training.hunyuan.adapter import (
    model_class_available,
    transformers_available,
)
from clouda_training.hunyuan.validators import validate_raw_jsonl


def run_preflight(
    *,
    model_path: str | None = None,
    raw_data_path: str | None = None,
    packed_data_path: str | None = None,
    pack_length: int = 20480,
) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "checks": []}

    def _add(name: str, passed: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            report["ok"] = False

    # transformers availability
    _add(
        "transformers_installed",
        transformers_available(),
        "pip install transformers to proceed",
    )
    # model class availability (verified upstream symbol)
    _add(
        "hunyuan_model_class_importable",
        model_class_available(),
        "requires transformers with HunYuanVLForConditionalGeneration "
        "(trust_remote_code environment)",
    )

    # model directory
    if model_path:
        path = Path(model_path)
        exists = path.is_dir()
        _add("model_dir_exists", exists, str(path))
        if exists:
            config_ok = (path / "config.json").is_file()
            _add("model_config_json", config_ok)
            weights_present = any(
                (path / name).is_file()
                for name in ("model.safetensors", "pytorch_model.bin")
            )
            _add(
                "model_weights_present",
                weights_present,
                "safetensors or bin required for real training",
            )
        else:
            _add("model_config_json", False, "skipped (dir missing)")
            _add("model_weights_present", False, "skipped (dir missing)")

    # training data
    if raw_data_path:
        try:
            summary = validate_raw_jsonl(raw_data_path)
            _add("raw_jsonl_schema", True, json.dumps(summary))
        except Exception as exc:  # noqa: BLE001 — report everything
            _add("raw_jsonl_schema", False, str(exc))
    if packed_data_path:
        try:
            from clouda_training.hunyuan.validators import validate_packed_jsonl

            summary = validate_packed_jsonl(packed_data_path, pack_length=pack_length)
            _add("packed_jsonl_schema", True, json.dumps(summary))
        except Exception as exc:  # noqa: BLE001
            _add("packed_jsonl_schema", False, str(exc))

    return report
