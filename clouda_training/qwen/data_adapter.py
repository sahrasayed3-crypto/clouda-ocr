"""Qwen3-VL training data adapter (ModelTrainingDataAdapter implementation).

Exports canonical manifest rows into the official upstream Qwen3-VL
qwen-vl-finetune format (verified, pinned at
QwenLM/Qwen3-VL@96588727e44c78b25ba03ea03b8e12f7e64fd0da):

    {"image": "<absolute path>", "conversations": [
        {"from": "human", "value": "<image>\n<prompt>"},
        {"from": "gpt",   "value": "<ground truth>"}]}

Media stay FILE PATHS; the text-level ``<image>`` tag appears only in the
human turn (the processor inserts model-level vision tokens). Every record
carries lineage (canonical sample id) in a sidecar report, never inside the
upstream wire format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clouda_contracts.protection import (
    protection_metadata_is_malformed,
    record_is_protected,
)

from clouda_training.qwen.models import (
    QWEN_COMPAT_REPOSITORY,
    QWEN_COMPAT_REVISION,
    QWEN_DATA_CONTRACT_VERSION,
    QWEN_IMAGE_PLACEHOLDER,
)

GT_FIELDS = ("ground_truth", "gt", "text", "transcript")
IMAGE_FIELDS = ("image_path", "image", "source_path")
LINEAGE_REPORT_VERSION = 1


class QwenDataAdapterError(RuntimeError):
    """Raised when a manifest row cannot be exported for Qwen3-VL training."""


def _first_field(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for name in fields:
        if row.get(name):
            return row[name]
    return None


def _absolute_image_path(image_root: str, canonical_path: str) -> str:
    """Absolutize a canonical relative path inside the generated artifact only."""
    candidate = Path(canonical_path)
    if candidate.is_absolute():
        return str(candidate)
    root = Path(image_root)
    if not root.is_absolute():
        raise QwenDataAdapterError(
            f"image_root must be absolute for Qwen export: {image_root!r}"
        )
    return str((root / candidate).resolve())


@dataclass(frozen=True)
class QwenExportConfig:
    """Family-specific export configuration owned by the qwen package."""

    image_root: str
    prompt_text: str = "استخرج كامل محتوى الصورة النصي بدقة واكتبه كما هو."
    model_max_length: int = 8192
    exporter_version: str = "1.0.0"


@dataclass
class QwenLineageReport:
    """Lineage of one export run (canonical ids -> exported records)."""

    prompt_text: str
    prompt_sha256: str
    manifest_sha256: str
    sample_ids: list[str] = field(default_factory=list)
    skipped_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LINEAGE_REPORT_VERSION,
            "upstream_repository": QWEN_COMPAT_REPOSITORY,
            "upstream_revision": QWEN_COMPAT_REVISION,
            "data_contract_version": QWEN_DATA_CONTRACT_VERSION,
            "prompt_sha256": self.prompt_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sample_ids": list(self.sample_ids),
            "skipped_counts": dict(self.skipped_counts),
        }


def build_qwen_record(
    row: dict[str, Any],
    config: QwenExportConfig,
    manifest_sha256: str = "",
) -> tuple[dict[str, Any], QwenLineageReport]:
    """Convert one canonical manifest row into one Qwen3-VL training record.

    Returns ``(record, lineage)`` — lineage stays OUT of the wire format.
    Fails closed on anything ambiguous or protected.
    """
    import hashlib

    sample_id = row.get("sample_id")
    if not sample_id or not isinstance(sample_id, str):
        raise QwenDataAdapterError(f"manifest row missing string sample_id: {row!r}")
    if protection_metadata_is_malformed(row):
        raise QwenDataAdapterError(f"row {sample_id}: malformed protection metadata")
    if record_is_protected(row):
        raise QwenDataAdapterError(
            f"row {sample_id}: protected rows cannot be exported"
        )

    raw_gt = _first_field(row, GT_FIELDS)
    if not raw_gt or not str(raw_gt).strip():
        raise QwenDataAdapterError(f"row {sample_id}: missing/empty ground truth")
    raw_image = _first_field(row, IMAGE_FIELDS)
    if not raw_image or not str(raw_image).strip():
        raise QwenDataAdapterError(f"row {sample_id}: missing/empty image path")

    image_abs = _absolute_image_path(config.image_root, str(raw_image))
    record: dict[str, Any] = {
        "image": image_abs,
        "conversations": [
            {
                "from": "human",
                "value": f"{QWEN_IMAGE_PLACEHOLDER}\n{config.prompt_text}",
            },
            {"from": "gpt", "value": str(raw_gt)},
        ],
    }
    lineage = QwenLineageReport(
        prompt_text=config.prompt_text,
        prompt_sha256=hashlib.sha256(config.prompt_text.encode("utf-8")).hexdigest(),
        manifest_sha256=manifest_sha256,
        sample_ids=[sample_id],
    )
    return record, lineage


class QwenTrainingDataAdapter:
    """ModelTrainingDataAdapter for the Qwen3-VL family (no torch needed)."""

    def export(
        self, manifest_rows: list[dict[str, Any]], config: Any
    ) -> list[dict[str, Any]]:
        """Convert canonical manifest rows into Qwen conversations records.

        ``config`` must be a :class:`QwenExportConfig`. Protected, malformed,
        and duplicate rows are skipped (counted in the lineage report) — the
        same fail-closed-per-row policy as the Hunyuan exporter.
        """
        if not isinstance(config, QwenExportConfig):
            raise QwenDataAdapterError(
                "Qwen data adapter requires a QwenExportConfig; got "
                f"{type(config).__name__}"
            )
        import hashlib

        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_rows, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()

        records: list[dict[str, Any]] = []
        report = QwenLineageReport(
            prompt_text=config.prompt_text,
            prompt_sha256=hashlib.sha256(
                config.prompt_text.encode("utf-8")
            ).hexdigest(),
            manifest_sha256=manifest_sha256,
        )
        seen_ids: set[str] = set()
        for row in manifest_rows:
            try:
                record, lineage = build_qwen_record(row, config, manifest_sha256)
            except QwenDataAdapterError as exc:
                key = (
                    "protected"
                    if "protected" in str(exc)
                    else (
                        "missing_gt"
                        if "ground truth" in str(exc)
                        else (
                            "missing_image" if "image path" in str(exc) else "malformed"
                        )
                    )
                )
                report.skipped_counts[key] = report.skipped_counts.get(key, 0) + 1
                continue
            sid = lineage.sample_ids[0]
            if sid in seen_ids:
                report.skipped_counts["duplicate_sample_id"] = (
                    report.skipped_counts.get("duplicate_sample_id", 0) + 1
                )
                continue
            seen_ids.add(sid)
            records.append(record)
            report.sample_ids.append(sid)
        self.last_lineage_report = report
        return records

    def validate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate records against the official Qwen format WITHOUT training.

        Returns a summary dict; errors list every problem found (training-
        blocking for the caller's orchestration).
        """
        errors: list[str] = []
        for index, record in enumerate(records):
            prefix = f"record {index}"
            if not isinstance(record, dict):
                errors.append(f"{prefix}: not a JSON object")
                continue
            image = record.get("image")
            if not isinstance(image, str) or not image.strip():
                errors.append(f"{prefix}: 'image' must be a non-empty file path")
            conversations = record.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                errors.append(
                    f"{prefix}: 'conversations' must be a list with >=2 turns"
                )
                continue
            for turn_index, turn in enumerate(conversations):
                if not isinstance(turn, dict):
                    errors.append(f"{prefix}: turn {turn_index} is not an object")
                    continue
                role = turn.get("from")
                if role not in {"human", "gpt"}:
                    errors.append(
                        f"{prefix}: turn {turn_index} has invalid role {role!r}"
                    )
                if not isinstance(turn.get("value"), str):
                    errors.append(f"{prefix}: turn {turn_index} value must be a string")
            first = conversations[0]
            if isinstance(first, dict) and isinstance(first.get("value"), str):
                value = first["value"]
                if first.get("from") == "human":
                    if QWEN_IMAGE_PLACEHOLDER not in value:
                        errors.append(
                            f"{prefix}: human turn must contain the "
                            f"{QWEN_IMAGE_PLACEHOLDER!r} tag"
                        )
                    if value.count(QWEN_IMAGE_PLACEHOLDER) != 1:
                        errors.append(
                            f"{prefix}: human turn must contain exactly one "
                            f"{QWEN_IMAGE_PLACEHOLDER!r} tag"
                        )
                else:
                    errors.append(f"{prefix}: first turn must be from 'human'")
            gpt_turns = [
                t
                for t in conversations
                if isinstance(t, dict) and t.get("from") == "gpt"
            ]
            if not gpt_turns:
                errors.append(f"{prefix}: missing a 'gpt' turn with ground truth")
            elif any(not t["value"].strip() for t in gpt_turns):
                errors.append(f"{prefix}: 'gpt' ground truth must be non-empty")
        return {
            "valid": not errors,
            "records": len(records),
            "errors": errors,
        }

    def describe_contract(self) -> dict[str, Any]:
        """Machine-readable record contract (assertable without the model)."""
        return {
            "data_adapter_type": "qwen_vl_sft_data",
            "data_contract_version": QWEN_DATA_CONTRACT_VERSION,
            "upstream_repository": QWEN_COMPAT_REPOSITORY,
            "upstream_revision": QWEN_COMPAT_REVISION,
            "record_format": "qwen_conversations_jsonl",
            "fields": {
                "image": {
                    "type": "string",
                    "description": "Absolute local image file path (never a URL/bytes)",
                },
                "conversations": {
                    "type": "list[dict]",
                    "description": (
                        "Chat turns: first from 'human' containing exactly one "
                        f"{QWEN_IMAGE_PLACEHOLDER!r} tag in the text; ground truth "
                        "in a 'gpt' turn"
                    ),
                },
            },
            "image_tag": {
                "value": QWEN_IMAGE_PLACEHOLDER,
                "level": "text",
                "note": (
                    "Model-level <|vision_start|>/<|image_pad|>/<|vision_end|> "
                    "tokens are inserted by the processor, never stored in data"
                ),
            },
            "lineage": {
                "in_wire_format": False,
                "report": "QwenLineageReport.to_dict() sidecar",
            },
            "loss_masking": "labels = -100 outside assistant (gpt) spans",
            "packing": "official --data_flatten / --data_packing contract",
        }


__all__ = [
    "QwenDataAdapterError",
    "QwenExportConfig",
    "QwenLineageReport",
    "QwenTrainingDataAdapter",
    "build_qwen_record",
]
