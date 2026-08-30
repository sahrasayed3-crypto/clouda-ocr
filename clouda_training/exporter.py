from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clouda_contracts.storage import StorageRoots
from clouda_data.datasets.license_gate import decide_dataset_use
from clouda_training.sampling.deduplication import deduplicate_records
from clouda_training.sampling.splits import deterministic_document_split

SUPPORTED_FORMATS = {
    "generic_jsonl",
    "conversation_multimodal",
    "ocr_plain_text",
    "markdown_extraction",
    "layout_extraction",
    "word_bounding_boxes",
    "reading_order",
}

PROMPTS = {
    "generic_jsonl": "Extract the text from this document image.",
    "conversation_multimodal": "اقرأ النص العربي في الصورة مع الحفاظ على ترتيب القراءة.",
    "ocr_plain_text": "Return plain text only.",
    "markdown_extraction": "Return document content as Markdown.",
    "layout_extraction": "Return text and layout regions.",
    "word_bounding_boxes": "Return words with bounding boxes.",
    "reading_order": "Return text blocks in reading order.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export_training_data(
    manifest: str | Path,
    output: str | Path,
    *,
    export_format: str = "generic_jsonl",
    seed: int = 20260723,
    purpose: str = "commercial_training",
    benchmark_exclusions: set[str] | None = None,
    max_contribution_per_source: int = 100,
    balance: bool = True,
    perceptual_duplicate: (
        Callable[[dict[str, Any], dict[str, Any]], bool] | None
    ) = None,
) -> dict[str, Any]:
    if export_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {export_format}")
    if not 1 <= max_contribution_per_source <= 100_000:
        raise ValueError("max_contribution_per_source is outside safe limits")
    roots = StorageRoots.from_env()
    input_path = Path(manifest).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    try:
        input_path.relative_to(roots.dataset_root.resolve())
        output_path.relative_to(roots.artifact_root.resolve())
    except ValueError as exc:
        raise PermissionError("Training input/output escaped configured roots") from exc
    source_records = _read_jsonl(input_path)
    exclusions = benchmark_exclusions or set()
    eligible: list[dict[str, Any]] = []
    for record in source_records:
        if record.get("source_document_id") in exclusions:
            continue
        if not record.get("ground_truth_text"):
            continue
        dataset_id = str(record.get("dataset_id", ""))
        status = str(record.get("license_status", "pending"))
        if purpose == "commercial_training":
            if dataset_id == "synthetic_evaluation":
                raise PermissionError(
                    "Synthetic evaluation data cannot enter commercial training"
                )
            try:
                decision = decide_dataset_use(dataset_id, purpose=purpose)
            except KeyError as exc:
                raise PermissionError(
                    f"Dataset {dataset_id or '<missing>'} is not present in the "
                    "approved license catalog and is blocked for commercial training"
                ) from exc
            if not decision.allowed:
                raise PermissionError(
                    f"Dataset {dataset_id} is blocked for commercial training "
                    f"(status={decision.status}, reasons={','.join(decision.reasons)})"
                )
            status = decision.status
            commercial_allowed = True
        elif purpose == "evaluation":
            if dataset_id == "synthetic_evaluation":
                if status != "evaluation_only" or record.get(
                    "commercial_training_allowed", False
                ):
                    raise PermissionError("Invalid synthetic evaluation provenance")
            else:
                try:
                    decision = decide_dataset_use(dataset_id, purpose=purpose)
                except KeyError as exc:
                    raise PermissionError(
                        f"Dataset {dataset_id or '<missing>'} is not present in the "
                        "approved license catalog and is blocked for evaluation"
                    ) from exc
                status = decision.status
                if not decision.allowed:
                    raise PermissionError(
                        f"Dataset {dataset_id} is blocked for evaluation "
                        f"(status={decision.status}, reasons={','.join(decision.reasons)})"
                    )
            if status in {"blocked", "pending", "expired"}:
                raise PermissionError(f"Dataset {dataset_id} is blocked for evaluation")
            commercial_allowed = (
                False
                if dataset_id == "synthetic_evaluation"
                else decide_dataset_use(
                    dataset_id, purpose="commercial_training"
                ).allowed
            )
        else:
            raise ValueError("purpose must be commercial_training or evaluation")
        image_path = roots.resolve_uri(str(record.get("output_uri", "")))
        if not image_path.is_file():
            raise FileNotFoundError("Training image is missing")
        expected_checksum = str(record.get("output_checksum", ""))
        actual_checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if expected_checksum != actual_checksum:
            raise ValueError("Training image checksum does not match manifest")
        eligible.append(
            {
                **record,
                "license_status": status,
                "commercial_training_allowed": commercial_allowed,
            }
        )
    deduped, duplicates = deduplicate_records(
        eligible,
        checksum_key="output_checksum",
        perceptual_duplicate=perceptual_duplicate,
    )
    ranked = sorted(
        deduped,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.get('generated_page_id')}".encode()
        ).digest(),
    )
    source_counts: Counter[str] = Counter()
    capped: list[dict[str, Any]] = []
    for item in ranked:
        source_id = str(item["source_document_id"])
        if source_counts[source_id] >= max_contribution_per_source:
            continue
        source_counts[source_id] += 1
        capped.append(item)
    if balance:
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in capped:
            key = (
                str(item.get("dataset_id", "unspecified")),
                str(item.get("profile_id", "unspecified")),
                str(item.get("estimated_visual_difficulty", "unspecified")),
            )
            buckets.setdefault(key, []).append(item)
        balanced: list[dict[str, Any]] = []
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:
                    balanced.append(buckets[key].pop(0))
        deduped = balanced
    else:
        deduped = capped
    assignments = deterministic_document_split(
        [str(item["source_document_id"]) for item in deduped], seed=seed
    )
    records: list[dict[str, Any]] = []
    for item in deduped:
        target: Any = item["ground_truth_text"]
        if export_format in {
            "layout_extraction",
            "word_bounding_boxes",
            "reading_order",
        }:
            target = {
                "text": item["ground_truth_text"],
                "regions": item.get("regions", []),
                "reading_order": item.get("reading_order", []),
            }
        base = {
            "schema_version": 1,
            "image_uri": item["output_uri"],
            "instruction": PROMPTS[export_format],
            "target": target,
            "dataset_id": item["dataset_id"],
            "source_document_id": item["source_document_id"],
            "page_id": item["source_page_id"],
            "generated_page_id": item["generated_page_id"],
            "profile_id": item["profile_id"],
            "license_status": item["license_status"],
            "attribution": item.get("attribution", ""),
            "split": assignments[str(item["source_document_id"])],
            "checksum": item["output_checksum"],
            "export_format": export_format,
        }
        if export_format == "conversation_multimodal":
            base["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["output_uri"]},
                        {"type": "text", "text": PROMPTS[export_format]},
                    ],
                },
                {"role": "assistant", "content": str(item["ground_truth_text"])},
            ]
        records.append(base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(output_path)
    split_counts = Counter(record["split"] for record in records)
    documents_by_split = {
        split: {
            record["source_document_id"]
            for record in records
            if record["split"] == split
        }
        for split in ("train", "validation", "test")
    }
    leakage = (
        documents_by_split["train"] & documents_by_split["validation"]
        or documents_by_split["train"] & documents_by_split["test"]
        or documents_by_split["validation"] & documents_by_split["test"]
    )
    summary = {
        "schema_version": 1,
        "records": len(records),
        "duplicates_rejected": len(duplicates),
        "split_counts": dict(split_counts),
        "document_leakage": sorted(leakage),
        "output": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "purpose": purpose,
        "training_started": False,
        "balanced": balance,
        "max_contribution_per_source": max_contribution_per_source,
        "datasets": dict(Counter(str(item["dataset_id"]) for item in records)),
        "profiles": dict(Counter(str(item["profile_id"]) for item in records)),
    }
    return summary


def training_statistics(manifest: str | Path) -> dict[str, Any]:
    records = _read_jsonl(Path(manifest))
    return {
        "records": len(records),
        "datasets": dict(Counter(str(item.get("dataset_id")) for item in records)),
        "profiles": dict(Counter(str(item.get("profile_id")) for item in records)),
        "difficulty": dict(
            Counter(str(item.get("estimated_visual_difficulty")) for item in records)
        ),
        "bytes": sum(
            StorageRoots.from_env().resolve_uri(str(item["output_uri"])).stat().st_size
            for item in records
        ),
    }
