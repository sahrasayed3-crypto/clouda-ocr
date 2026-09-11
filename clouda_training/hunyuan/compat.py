"""Upstream drift-detection contract for HunyuanOCR-1.5 compatibility.

Records what the bridge was verified against; a validator warns
UPSTREAM COMPATIBILITY NEEDS REVALIDATION when expectations no longer hold.
Never auto-updates anything from the internet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clouda_training.hunyuan.models import (
    HUNYUAN_COMPAT_REPOSITORY,
    HUNYUAN_COMPAT_REVISION,
)

DRIFT_WARNING = "UPSTREAM COMPATIBILITY NEEDS REVALIDATION"

EXPECTED_MODEL_SYMBOLS = (
    "HunYuanVLForConditionalGeneration",
    "AutoTokenizer",
    "AutoProcessor",
)
EXPECTED_DATA_CLASSES = ("VLDataset", "PackedVLDataCollator", "VLDataCollator")
EXPECTED_TRAINING_ENTRY = "train/train_hunyuan.py"
EXPECTED_PACK_FIELDS = ("packed_samples", "cu_seqlens", "total_tokens")
EXPECTED_RAW_FIELDS = ("image_path", "conversations")
EXPECTED_PACK_LENGTH = 20480
EXPECTED_SFT_PROFILE = {
    "lr": 2e-5,
    "epochs": 5,
    "batch_size": 1,
    "grad_accum": 1,
    "save_steps": 200,
}


@dataclass
class CompatibilityMetadata:
    upstream_repository: str = HUNYUAN_COMPAT_REPOSITORY
    upstream_revision: str = HUNYUAN_COMPAT_REVISION
    expected_data_schema_version: int = 1
    expected_model_symbols: tuple[str, ...] = EXPECTED_MODEL_SYMBOLS
    expected_data_classes: tuple[str, ...] = EXPECTED_DATA_CLASSES
    expected_training_entry: str = EXPECTED_TRAINING_ENTRY
    expected_pack_fields: tuple[str, ...] = EXPECTED_PACK_FIELDS
    expected_raw_fields: tuple[str, ...] = EXPECTED_RAW_FIELDS
    expected_pack_length: int = EXPECTED_PACK_LENGTH
    expected_sft_profile: dict[str, Any] = field(
        default_factory=lambda: dict(EXPECTED_SFT_PROFILE)
    )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )


def check_drift(
    observed: dict[str, Any],
    expected: CompatibilityMetadata | None = None,
) -> dict[str, Any]:
    """Compare observed upstream facts against verified expectations.

    `observed` keys (all optional): model_symbols, data_classes, training_entry,
    pack_fields, raw_fields, pack_length, sft_profile.
    Returns {compatible: bool, drift: [...], warning: str | None}.
    """
    expected = expected or CompatibilityMetadata()
    drift: list[str] = []

    if "model_symbols" in observed:
        missing = [
            s
            for s in expected.expected_model_symbols
            if s not in observed["model_symbols"]
        ]
        if missing:
            drift.append(f"missing model symbols: {missing}")
    if "data_classes" in observed:
        missing = [
            s
            for s in expected.expected_data_classes
            if s not in observed["data_classes"]
        ]
        if missing:
            drift.append(f"missing data classes: {missing}")
    if (
        "training_entry" in observed
        and observed["training_entry"] != expected.expected_training_entry
    ):
        drift.append(
            f"training entry changed: {observed['training_entry']!r} != {expected.expected_training_entry!r}"
        )
    if "pack_fields" in observed:
        missing = [
            f for f in expected.expected_pack_fields if f not in observed["pack_fields"]
        ]
        if missing:
            drift.append(f"missing pack fields: {missing}")
    if "raw_fields" in observed:
        missing = [
            f for f in expected.expected_raw_fields if f not in observed["raw_fields"]
        ]
        if missing:
            drift.append(f"missing raw fields: {missing}")
    if (
        "pack_length" in observed
        and observed["pack_length"] != expected.expected_pack_length
    ):
        drift.append(
            f"pack length changed: {observed['pack_length']} != {expected.expected_pack_length}"
        )
    if "sft_profile" in observed:
        for key, value in expected.expected_sft_profile.items():
            if observed["sft_profile"].get(key) != value:
                drift.append(
                    f"SFT profile {key} changed: {observed['sft_profile'].get(key)!r} != {value!r}"
                )

    compatible = not drift
    return {
        "compatible": compatible,
        "drift": drift,
        "warning": None if compatible else DRIFT_WARNING,
    }
