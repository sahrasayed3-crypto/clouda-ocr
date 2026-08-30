from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clouda_contracts.references import DatasetPageReference

SUPPORTED_TASKS = {
    "text_ocr",
    "markdown_extraction",
    "word_bounding_boxes",
    "layout_reading_order",
    "ocr_bounding_boxes",
    "ocr_markdown_layout",
}


@dataclass(frozen=True)
class TrainingExample:
    task: str
    reference: DatasetPageReference
    prompt: str
    target: dict[str, Any]
    provenance: dict[str, str]


def generate_task_example(
    *,
    task: str,
    reference: DatasetPageReference,
    text: str,
    layout: list[dict[str, Any]] | None = None,
) -> TrainingExample:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported training task: {task}")
    target: dict[str, Any] = {"text": text}
    if task != "text_ocr":
        target["layout"] = layout or []
    return TrainingExample(
        task=task,
        reference=reference,
        prompt=f"Perform {task} for this page.",
        target=target,
        provenance={
            "dataset_id": reference.dataset.dataset_id,
            "record_id": reference.dataset.record_id,
        },
    )
