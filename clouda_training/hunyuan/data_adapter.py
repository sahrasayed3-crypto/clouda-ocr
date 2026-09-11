"""HunyuanOCR-1.5 training data adapter (ModelTrainingDataAdapter implementation).

Wraps the existing (frozen) Hunyuan bridge export path
(:func:`clouda_training.hunyuan.exporter.build_raw_sample`) so the canonical
:data:`clouda_training.adapters.data_adapter.ModelTrainingDataAdapter`
protocol can be satisfied without duplicating protection/export logic.

Upstream wire format (verified, pinned at
Tencent-Hunyuan/HunyuanOCR@c55965d3da1e):

    {"image_path": ["<absolute path>"], "conversations": [
        {"from": "human", "value": "<image>\n<prompt>"},
        {"from": "gpt",   "value": "<ground truth>"}]}

Lineage (canonical sample id, prompt identity, manifest hash) is kept in a
sidecar report — never inside the upstream wire format.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from clouda_training.hunyuan.exporter import build_raw_sample
from clouda_training.hunyuan.models import (
    HUNYUAN_COMPAT_REPOSITORY,
    HUNYUAN_COMPAT_REVISION,
    HunyuanExportConfig,
    PromptProfile,
)

HUNYUAN_DATA_ADAPTER_TYPE = "hunyuanocr15_sft_data"


class HunyuanDataAdapterError(RuntimeError):
    """Raised when the hunyuan data adapter is misused or export fails."""


class HunyuanTrainingDataAdapter:
    """ModelTrainingDataAdapter for the HunyuanOCR-1.5 family.

    No torch/transformers needed: export/validate ride on the pure-Python
    bridge exporter and validators.
    """

    def __init__(self) -> None:
        self.last_lineage_report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ export
    def export(
        self, manifest_rows: list[dict[str, Any]], config: Any
    ) -> list[dict[str, Any]]:
        """Convert canonical manifest rows into Hunyuan raw OCR records.

        ``config`` must be a :class:`HunyuanExportConfig` (the frozen bridge
        config: manifest path/hash, split, image_root, prompt profile).
        Protected rows raise (fail-closed) — the bridge exporter's policy.
        """
        if not isinstance(config, HunyuanExportConfig):
            raise HunyuanDataAdapterError(
                "Hunyuan data adapter requires a HunyuanExportConfig; got "
                f"{type(config).__name__}"
            )
        records: list[dict[str, Any]] = []
        sample_ids: list[str] = []
        for row in manifest_rows:
            sample = build_raw_sample(row, config)
            records.append(sample.to_upstream_dict())
            sample_ids.append(sample.canonical_sample_id)
        if len({sid for sid in sample_ids}) != len(sample_ids):
            raise HunyuanDataAdapterError(
                "duplicate canonical sample ids in export batch — refusing "
                "to emit ambiguous records"
            )
        self.last_lineage_report = self._lineage(config, sample_ids)
        return records

    # ------------------------------------------------------------------ validate
    def validate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate records against the upstream raw schema WITHOUT training.

        Reuses the frozen bridge validator (validate_raw_line) per record and
        returns a summary dict listing every problem (training-blocking).
        """
        from clouda_training.hunyuan.validators import RawSchemaError
        from clouda_training.hunyuan.validators import validate_raw_line

        errors: list[str] = []
        seen_paths: set[str] = set()
        for index, record in enumerate(records):
            prefix = f"record {index}"
            try:
                validate_raw_line(record)
            except RawSchemaError as exc:
                errors.append(f"{prefix}: {exc}")
                continue
            for path in record["image_path"]:
                if path in seen_paths:
                    errors.append(f"{prefix}: duplicate image path {path!r}")
                seen_paths.add(path)
        return {
            "valid": not errors,
            "records": len(records),
            "errors": errors,
        }

    # ------------------------------------------------------------------ contract
    def describe_contract(self) -> dict[str, Any]:
        """Machine-readable record contract (assertable without the model)."""
        return {
            "data_adapter_type": HUNYUAN_DATA_ADAPTER_TYPE,
            "data_contract_version": "1",
            "upstream_repository": HUNYUAN_COMPAT_REPOSITORY,
            "upstream_revision": HUNYUAN_COMPAT_REVISION,
            "record_format": "hunyuan_raw_ocr_jsonl",
            "fields": {
                "image_path": {
                    "type": "list[string]",
                    "description": "Absolute local image file paths (list per upstream)",
                },
                "conversations": {
                    "type": "list[dict]",
                    "description": (
                        "Chat turns: first from 'human' containing the "
                        "<image> tag in the text; ground truth in a 'gpt' turn"
                    ),
                },
            },
            "image_tag": {
                "value": "<image>",
                "level": "text",
                "note": "Upstream model handles placeholder expansion internally",
            },
            "lineage": {
                "in_wire_format": False,
                "report": "exporter HunyuanExportReport sidecar (.report.json)",
            },
            "packing": "official pack_data pipeline (packed_samples/cu_seqlens/total_tokens)",
        }

    # ------------------------------------------------------------------ lineage
    @staticmethod
    def _lineage(config: HunyuanExportConfig, sample_ids: list[str]) -> dict[str, Any]:
        prompt: PromptProfile = config.prompt_profile
        return {
            "schema_version": 1,
            "upstream_repository": HUNYUAN_COMPAT_REPOSITORY,
            "upstream_revision": HUNYUAN_COMPAT_REVISION,
            "manifest_path": config.manifest_path,
            "manifest_sha256": config.manifest_hash,
            "dataset_id": config.dataset_id,
            "dataset_version": config.dataset_version,
            "split": config.split,
            "prompt_identity": prompt.identity,
            "prompt_sha256": hashlib.sha256(
                prompt.prompt_text.encode("utf-8")
            ).hexdigest(),
            "sample_ids": list(sample_ids),
        }


def prompt_identity_sha256(prompt_text: str) -> str:
    """Deterministic prompt-identity hash (lineage only, not wire format)."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def dump_lineage_report(report: dict[str, Any], path: str) -> None:
    """Persist a lineage sidecar next to an exported artifact."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


__all__ = [
    "HUNYUAN_DATA_ADAPTER_TYPE",
    "HunyuanDataAdapterError",
    "HunyuanTrainingDataAdapter",
    "dump_lineage_report",
    "prompt_identity_sha256",
]
