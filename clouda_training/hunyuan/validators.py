"""Offline validators for HunyuanOCR-1.5 training artifacts.

Raw-schema and packed-schema validation requiring no model, tokenizer, or
network. Upstream schema: Tencent-Hunyuan/HunyuanOCR@c55965d3da1e.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clouda_training.hunyuan.models import (
    DEFAULT_PACK_LENGTH,
    HUNYUAN_IMAGE_PLACEHOLDER,
)


class RawSchemaError(ValueError):
    pass


class PackedSchemaError(ValueError):
    pass


def validate_raw_line(payload: Any, *, check_image_exists: bool = False) -> None:
    """Validate one decoded raw OCR JSONL object against the upstream schema."""
    if not isinstance(payload, dict):
        raise RawSchemaError("raw sample must be a JSON object")
    image_path = payload.get("image_path")
    if not isinstance(image_path, list) or not image_path:
        raise RawSchemaError("image_path must be a non-empty list")
    for p in image_path:
        if not isinstance(p, str) or not p.strip():
            raise RawSchemaError("image_path entries must be non-empty strings")
        if check_image_exists and not Path(p).is_file():
            raise RawSchemaError(f"image file does not exist: {p}")
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise RawSchemaError("conversations must be a non-empty list (>=2 turns)")
    for turn in conversations:
        if not isinstance(turn, dict):
            raise RawSchemaError("conversation turns must be JSON objects")
    roles = [turn.get("from") for turn in conversations]
    if roles[0] != "human":
        raise RawSchemaError("first conversation turn must be from 'human' role")
    if "gpt" not in roles[1:]:
        raise RawSchemaError("conversation must contain a 'gpt' turn")
    for turn in conversations:
        if not isinstance(turn, dict):
            raise RawSchemaError("conversation turns must be JSON objects")
        if turn.get("from") not in {"human", "gpt"}:
            raise RawSchemaError(f"invalid conversation role: {turn.get('from')!r}")
        if not isinstance(turn.get("value"), str):
            raise RawSchemaError("conversation values must be strings")
    human_turn = conversations[0]["value"]
    if HUNYUAN_IMAGE_PLACEHOLDER not in human_turn:
        raise RawSchemaError(f"human turn must contain {HUNYUAN_IMAGE_PLACEHOLDER!r}")
    if not human_turn.strip():
        raise RawSchemaError("human turn value must be non-empty")
    gt = next((t["value"] for t in conversations if t.get("from") == "gpt"), None)
    if gt is None or not gt.strip():
        raise RawSchemaError("gpt ground truth must be non-empty")


def validate_raw_jsonl(
    path: str | Path, *, check_image_exists: bool = False
) -> dict[str, Any]:
    """Validate an entire raw JSONL file. Returns a summary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    total = 0
    seen_paths: set[str] = set()
    try:
        handle = file_path.open("r", encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RawSchemaError(f"file is not valid UTF-8: {exc}") from exc
    with handle:
        line_number = 0
        while True:
            try:
                line = handle.readline()
            except UnicodeDecodeError as exc:
                raise RawSchemaError(
                    f"line {line_number + 1}: invalid UTF-8: {exc}"
                ) from exc
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RawSchemaError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            try:
                validate_raw_line(payload, check_image_exists=check_image_exists)
            except RawSchemaError as exc:
                raise RawSchemaError(f"line {line_number}: {exc}") from exc
            for p in payload["image_path"]:
                if p in seen_paths:
                    raise RawSchemaError(
                        f"line {line_number}: duplicate image path {p!r}"
                    )
                seen_paths.add(p)
            total += 1
    if total == 0:
        raise RawSchemaError("raw JSONL contains no samples")
    return {"valid": True, "samples": total}


def validate_packed_line(
    payload: Any, *, pack_length: int = DEFAULT_PACK_LENGTH
) -> int:
    """Validate one decoded packed object; returns its embedded sample count."""
    if not isinstance(payload, dict):
        raise PackedSchemaError("packed record must be a JSON object")
    packed = payload.get("packed_samples")
    if not isinstance(packed, list) or not packed:
        raise PackedSchemaError("packed_samples must be a non-empty list")
    cu = payload.get("cu_seqlens")
    if not isinstance(cu, list) or len(cu) < 2:
        raise PackedSchemaError("cu_seqlens must be a list with >=2 entries")
    total = payload.get("total_tokens")
    if not isinstance(total, int) or total <= 0:
        raise PackedSchemaError("total_tokens must be a positive integer")
    if total > pack_length:
        raise PackedSchemaError(
            f"total_tokens {total} exceeds pack_length {pack_length}"
        )
    if cu[0] != 0:
        raise PackedSchemaError("cu_seqlens must start at 0")
    for prev, curr in zip(cu, cu[1:]):
        if not isinstance(curr, int) or curr <= prev:
            raise PackedSchemaError("cu_seqlens must be strictly monotonic integers")
    if cu[-1] != total:
        raise PackedSchemaError(
            f"final cu_seqlens {cu[-1]} does not match total_tokens {total}"
        )
    for sample in packed:
        try:
            validate_raw_line(sample)
        except RawSchemaError as exc:
            raise PackedSchemaError(f"malformed embedded raw sample: {exc}") from exc
    return len(packed)


def validate_packed_jsonl(
    path: str | Path, *, pack_length: int = DEFAULT_PACK_LENGTH
) -> dict[str, Any]:
    """Validate an entire packed JSONL file. Returns a summary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    packs = 0
    samples = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackedSchemaError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            try:
                samples += validate_packed_line(payload, pack_length=pack_length)
            except PackedSchemaError as exc:
                raise PackedSchemaError(f"line {line_number}: {exc}") from exc
            packs += 1
    if packs == 0:
        raise PackedSchemaError("packed JSONL contains no records")
    return {"valid": True, "packs": packs, "samples": samples}
