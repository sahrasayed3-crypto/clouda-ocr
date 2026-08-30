from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from PIL import Image

DATASET_LAYOUT = (
    "registry",
    "raw",
    "normalized",
    "candidates",
    "gold",
    "silver",
    "review",
    "rejected",
    "holdout",
)


def ensure_dataset_layout(root: str | Path) -> tuple[Path, ...]:
    base = Path(root)
    paths = tuple(base / item for item in DATASET_LAYOUT)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def perceptual_dhash(path: str | Path) -> str:
    with Image.open(path) as opened:
        image = opened.convert("L").resize((9, 8))
        pixels = cast(list[int], list(image.get_flattened_data()))
    bits = [
        pixels[row * 9 + column] > pixels[row * 9 + column + 1]
        for row in range(8)
        for column in range(8)
    ]
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def validate_manifest_items(
    items: list[dict[str, Any]], *, near_duplicate_distance: int = 5
) -> None:
    pages: dict[str, str] = {}
    documents: dict[str, str] = {}
    hashes: list[tuple[str, str]] = []
    for item in items:
        page = str(item["source_page_id"])
        split = str(item["split"])
        document = str(item["source_document_id"])
        if page in pages and pages[page] != split:
            raise ValueError("page_split_leakage")
        pages[page] = split
        if document in documents and documents[document] != split:
            raise ValueError("document_split_leakage")
        documents[document] = split
        candidate = str(item["perceptual_hash"])
        for existing, existing_split in hashes:
            if (
                existing_split != split
                and hamming_distance(candidate, existing) <= near_duplicate_distance
            ):
                raise ValueError("near_duplicate_split_leakage")
        hashes.append((candidate, split))
        if split == "holdout" and item.get("training_candidate"):
            raise ValueError("holdout_training_leakage")


def write_manifest(
    items: list[dict[str, Any]], target: str | Path, *, frozen: bool = False
) -> dict[str, Any]:
    validate_manifest_items(items)
    target_path = Path(target)
    if target_path.exists():
        existing = json.loads(target_path.read_text(encoding="utf-8"))
        if existing.get("frozen"):
            raise PermissionError("frozen_holdout_is_immutable")
    payload = {
        "schema_version": 1,
        "algorithm_version": "teacher-dataset-manifest-v1",
        "frozen": frozen,
        "items": sorted(items, key=lambda item: str(item["source_page_id"])),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(encoded + b"\n")
    return {
        "path": str(target_path),
        "sha256": hashlib.sha256(encoded + b"\n").hexdigest(),
        "items": len(items),
        "frozen": frozen,
    }
