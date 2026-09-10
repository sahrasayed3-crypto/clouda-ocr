"""Shared E2E fixture: tiny offline Arabic benchmark + two fake OCR models.

No network, no GPU, no model downloads. Deterministic across runs.
"""

from __future__ import annotations

import hashlib

from clouda_data.results.identity import ArtifactRef
from clouda_data.results.models import (
    GroundTruthRecord,
    OCRPrediction,
    PageRecord,
    Provenance,
    ProtectionInfo,
)

SCHEMA = "clouda.ocr.results.v1"

# Arabic pages with diacritics, tatweel, Arabic-Indic digits, alef variants.
FIXTURE_PAGES: list[dict[str, str]] = [
    {
        "page_id": "fx_page_001",
        "document_id": "fx_doc_1",
        "split": "train",
        "profile": "clean",
        "raw_text": "كِتَابٌ قَدِيمٌ فِي مَكْتَبَةِ القَاهِرَة",
        "model_a": "كتاب قديم في مكتبة القاهرة",
        "model_b": "كتاب قديم في مكتبة الاسكندرية",
    },
    {
        "page_id": "fx_page_002",
        "document_id": "fx_doc_1",
        "split": "train",
        "profile": "old_book_medium",
        "raw_text": "وَصَفَ الوَرَقَةُ تَارِيخَـهَا بِتَفْصِيل ٢٠٢٦",
        "model_a": "وصف الورقة تاريخها بتفصيل 2026",
        "model_b": "وصف الورقة تاريخها بتفصيل ٢٠٢٦",
    },
    {
        "page_id": "fx_page_003",
        "document_id": "fx_doc_2",
        "split": "holdout",
        "profile": "bad_scan_heavy",
        "raw_text": "هَذِهِ الوَقَايَا المُحَمِّيَّةُ لَا تُسْتَخْدَمُ فِي التَّدْرِيب",
        "model_a": "هذه الوقاية المحمية لا تستخدم في التدريب",
        "model_b": "",
    },
    {
        "page_id": "fx_page_004",
        "document_id": "fx_doc_2",
        "split": "validation",
        "profile": "phone_photo_light",
        "raw_text": "نَصٌّ مُخْتَلِطٌ English words 123",
        "model_a": "نص مختلط English words 123",
        "model_b": "نص مختلط English words 456",
    },
]

MODELS = [
    ("fake-model-a", "1.0"),
    ("fake-model-b", "0.9"),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_fixture_pages(dataset_id: str = "fx-arabic-demo") -> list[PageRecord]:
    pages: list[PageRecord] = []
    for index, spec in enumerate(FIXTURE_PAGES, start=1):
        protected = spec["split"] == "holdout"
        pages.append(
            PageRecord(
                page_id=spec["page_id"],
                document_id=spec["document_id"],
                dataset_id=dataset_id,
                dataset_version="v1",
                split=spec["split"],
                page_number=index,
                image_artifact=ArtifactRef(
                    artifact_id=f"art_{spec['page_id']}",
                    kind="page_image",
                    uri=f"dataset://images/{spec['page_id']}.png",
                    sha256=sha256_text(f"image-bytes-{spec['page_id']}"),
                    role="page_image",
                ),
                ground_truth_uri=f"dataset://ground_truth/{spec['page_id']}.txt",
                ground_truth_sha256=sha256_text(spec["raw_text"]),
                profile=spec["profile"],
                distortions=tuple(
                    [{"distortion": "gaussian_blur", "severity": "medium"}]
                    if spec["profile"] != "clean"
                    else []
                ),
                distortion_seed=1000 + index,
                protection=ProtectionInfo(
                    protected=protected,
                    reasons=("protected_split:holdout",) if protected else (),
                    split=spec["split"],
                ),
                provenance=Provenance(
                    source_format="tests.fixtures.results_store_e2e",
                    source_uri="fixtures/e2e/pages.jsonl",
                ),
            )
        )
    return pages


def build_fixture_ground_truth(
    pages: list[PageRecord],
) -> list[GroundTruthRecord]:
    records: list[GroundTruthRecord] = []
    by_id = {page.page_id: page for page in pages}
    for spec in FIXTURE_PAGES:
        page = by_id[spec["page_id"]]
        records.append(
            GroundTruthRecord(
                page_id=page.page_id,
                raw_text=spec["raw_text"],
                raw_text_sha256=sha256_text(spec["raw_text"]),
                dataset_id=page.dataset_id,
                split=page.split,
                protection=page.protection,
                provenance=page.provenance,
            )
        )
    return records


def build_fixture_predictions(
    run_id: str,
    model_id: str,
    model_revision: str,
    pages: list[PageRecord],
) -> list[OCRPrediction]:
    predictions: list[OCRPrediction] = []
    by_id = {page.page_id: page for page in pages}
    for spec in FIXTURE_PAGES:
        page = by_id[spec["page_id"]]
        text = spec["model_a"] if model_id == "fake-model-a" else spec["model_b"]
        predictions.append(
            OCRPrediction(
                prediction_id=hashlib.sha256(
                    f"{run_id}:{page.page_id}".encode("utf-8")
                ).hexdigest()[:20],
                run_id=run_id,
                page_id=page.page_id,
                model_id=model_id,
                model_revision=model_revision,
                text=text,
                text_sha256=sha256_text(text),
                dataset_id=page.dataset_id,
                split=page.split,
                inference_settings={"temperature": 0.0, "max_tokens": 512},
                provenance=page.provenance,
            )
        )
    return predictions
