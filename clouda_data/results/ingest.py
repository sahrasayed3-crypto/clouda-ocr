"""Ingestion adapters: existing Clouda formats -> canonical results records.

Adapters exist only for formats verified present in the repository:

1. ``benchmark_manifest`` — ``benchmarks/ocr_arabic/benchmark_manifest.jsonl``
   (177-row metadata manifest of the final Arabic OCR benchmark, with
   source/distortion/QC/provenance blocks and SHA-256 hashes).
2. ``ocr_arabic_results`` — ``benchmarks/ocr_arabic/results.csv`` (run-level
   leaderboard rows with CER/WER/normalized Arabic CER and evidence ids).
3. ``foundation_evaluation`` — JSONL rows consumed by
   ``clouda_data.evaluation.execution.evaluate_records``
   (``reference_text``/``prediction_text``/``page_id`` style).
4. ``training_run_summary`` — Training Experiment Framework run directories
   (``metadata.json`` + ``summary.json``), used to mint a ModelRecord with
   training lineage (optional linkage, no registry duplication).

Provenance survives: every record carries source format + hashes. Protection
markers (holdout splits, protected splits) fail closed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from clouda_data.ground_truth.checksums import sha256_text

from .identity import (
    ArtifactRef,
    prediction_identity,
    run_identity,
)
from .metrics import build_page_metric_records
from .models import (
    EvaluationRecord,
    GroundTruthRecord,
    InferenceRun,
    InferenceRunStatus,
    ModelRecord,
    OCRPrediction,
    PageRecord,
    Provenance,
    ProtectionInfo,
    TrainingLineage,
)

ADAPTER_VERSION = "clouda.results.ingest.v1"

BENCHMARK_MANIFEST_FORMAT = "benchmarks.ocr_arabic.benchmark_manifest.v1"
OCR_ARABIC_RESULTS_FORMAT = "benchmarks.ocr_arabic.results_csv.v1"
FOUNDATION_EVALUATION_FORMAT = "clouda_data.evaluation.records.v1"
TRAINING_RUN_SUMMARY_FORMAT = "clouda_training.experiments.run_summary.v1"


def _protection_from_split(split: str | None) -> ProtectionInfo:
    value = (split or "unassigned").strip().lower() or "unassigned"
    protected = value in {
        "holdout",
        "protected_holdout",
        "benchmark_holdout",
        "private_holdout",
    }
    reasons = (f"protected_split:{value}",) if protected else ()
    return ProtectionInfo(protected=protected, reasons=reasons, split=value)


# ---------------------------------------------------------------------------
# 1. Arabic OCR benchmark manifest (benchmarks/ocr_arabic)
# ---------------------------------------------------------------------------


def iter_benchmark_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed benchmark manifest line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Benchmark manifest line {line_number} is not an object."
                )
            yield payload


def benchmark_manifest_row_to_page(
    row: dict[str, Any],
    *,
    dataset_id: str = "clouda-ocr-arabic-177",
    dataset_version: str = "v1",
    source_path: str | None = None,
    manifest_sha256: str | None = None,
    private_root_note: str | None = None,
) -> PageRecord:
    """Convert one benchmark manifest row into a canonical PageRecord.

    ``distorted_id`` is the unique per-page identity in the canonical manifest
    (each clean source ``benchmark_id`` may have several distorted
    derivatives; all 177 ``distorted_id`` values are unique). ``benchmark_id``
    is preserved as the document/source identity. Provenance keeps all
    manifest hashes; the source URI stays portable (manifest-relative), and
    any machine-local root note is quarantined in ``source_private``.
    """

    benchmark_id = str(row.get("benchmark_id") or "").strip()
    distorted_id = str(row.get("distorted_id") or "").strip()
    if not distorted_id:
        raise ValueError(
            "Benchmark manifest row is missing distorted_id (canonical page key)."
        )
    source = row.get("source") or {}
    split = str(source.get("source_split") or "unassigned")
    source_identity = {
        key: source[key]
        for key in (
            "source_dataset",
            "source_split",
            "source_type",
            "original_sample_id",
            "repository_revision",
        )
        if source.get(key) is not None
    }
    if distorted_id:
        source_identity["distorted_id"] = distorted_id

    distorted_uri = str(row.get("distorted_image_path") or "")
    clean_uri = str(source.get("clean_image_path") or "")
    gt_path = str(source.get("ground_truth_path") or "")
    distorted_sha = str(row.get("distorted_sha256") or "0" * 64)
    clean_sha = str(source.get("clean_sha256") or "0" * 64)

    provenance = Provenance(
        source_format=BENCHMARK_MANIFEST_FORMAT,
        source_uri=Path(source_path).as_posix() if source_path else None,
        source_sha256=manifest_sha256,
        license_or_permission=source.get("license_or_permission_note"),
        adapter_version=ADAPTER_VERSION,
        extra={"row_distorted_sha256": distorted_sha},
        source_private=(
            {"source_private": private_root_note} if private_root_note else None
        ),
    )

    steps = row.get("steps") or []
    distortions = tuple(
        {
            "distortion": str(step.get("distortion", "")),
            "severity": str(step.get("severity", "")),
        }
        for step in steps
        if isinstance(step, dict)
    )

    return PageRecord(
        page_id=distorted_id,
        document_id=benchmark_id or distorted_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split=split,
        page_number=1,
        image_artifact=(
            ArtifactRef(
                artifact_id=f"art_{distorted_id or benchmark_id}",
                kind="page_image",
                uri=f"dataset://{distorted_uri}",
                sha256=distorted_sha,
                role="page_image",
            )
            if distorted_uri
            else None
        ),
        ground_truth_uri=f"dataset://{gt_path}" if gt_path else None,
        ground_truth_sha256=source.get("ground_truth_sha256"),
        profile=row.get("profile"),
        distortions=distortions,
        distortion_seed=row.get("seed"),
        source_artifact=(
            ArtifactRef(
                artifact_id=f"src_{benchmark_id}",
                kind="clean_image",
                uri=f"dataset://{clean_uri}",
                sha256=clean_sha,
                role="clean_source",
            )
            if clean_uri
            else None
        ),
        source_identity=source_identity,
        protection=_protection_from_split(split),
        provenance=provenance,
        metadata={"bucket": row.get("bucket"), "kind": row.get("kind")},
    )


def benchmark_manifest_to_pages(
    path: str | Path,
    *,
    dataset_id: str = "clouda-ocr-arabic-177",
    dataset_version: str = "v1",
    manifest_sha256: str | None = None,
    private_root_note: str | None = None,
) -> list[PageRecord]:
    """Convert a whole benchmark manifest into canonical PageRecords."""

    source = Path(path)
    pages: list[PageRecord] = []
    for row in iter_benchmark_manifest(source):
        pages.append(
            benchmark_manifest_row_to_page(
                row,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                source_path=str(source),
                manifest_sha256=manifest_sha256,
                private_root_note=private_root_note,
            )
        )
    return pages


# ---------------------------------------------------------------------------
# 2. Arabic OCR benchmark results.csv (run-level leaderboard)
# ---------------------------------------------------------------------------


def ocr_arabic_results_to_runs(
    results_csv: str | Path,
    *,
    dataset_id: str = "clouda-ocr-arabic-177",
    dataset_version: str = "v1",
    manifest_sha256: str | None = None,
    model_records_out: list[ModelRecord] | None = None,
) -> list[InferenceRun]:
    """Convert the leaderboard CSV into canonical InferenceRun records.

    Also emits ModelRecord entries for every model encountered (optional out
    list) — metadata only, respecting ``benchmarks/ocr_arabic/models.csv``
    rights boundaries.
    """

    source = Path(results_csv)
    runs: list[InferenceRun] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = (row.get("model") or "").strip()
            if not model_id:
                continue
            status_text = (row.get("status") or "").strip().upper()
            status = {
                "COMPLETE": InferenceRunStatus.COMPLETED,
                "PARTIAL": InferenceRunStatus.INTERRUPTED,
                "FAILED_SMOKE": InferenceRunStatus.FAILED,
            }.get(status_text, InferenceRunStatus.CREATED)
            revision = (row.get("version") or model_id).strip()
            evidence_id = (row.get("evidence_id") or "").strip()
            run_id = run_identity(
                model_id=model_id,
                model_revision=revision,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                manifest_sha256=manifest_sha256,
                created_at=evidence_id or "unknown-evidence",
            )
            environment: dict[str, Any] = {}
            gpu = (row.get("gpu") or "").strip()
            if gpu:
                environment["gpu"] = gpu
            seconds = (row.get("seconds_per_page") or "").strip()
            if seconds:
                environment["seconds_per_page"] = seconds
            pages_text = (row.get("pages") or "").strip()
            run = InferenceRun(
                run_id=run_id,
                model_id=model_id,
                model_revision=revision,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                split="unassigned",
                status=status,
                manifest_sha256=manifest_sha256,
                page_count=int(pages_text) if pages_text.isdigit() else None,
                environment=environment,
                provenance=Provenance(
                    source_format=OCR_ARABIC_RESULTS_FORMAT,
                    source_uri=source.name,
                    adapter_version=ADAPTER_VERSION,
                    extra={
                        "evidence_id": evidence_id,
                        "rankable": row.get("rankable"),
                    },
                ),
                metadata={"rank": row.get("rank"), "notes": row.get("notes")},
            )
            runs.append(run)
            if model_records_out is not None:
                model_records_out.append(
                    ModelRecord(
                        model_id=model_id,
                        display_name=model_id,
                        revision=revision,
                        model_family="external-benchmark",
                    )
                )
    return runs


# ---------------------------------------------------------------------------
# 3. Foundation evaluation records (clouda_data.evaluation.execution input)
# ---------------------------------------------------------------------------


def foundation_record_to_canonical(
    row: dict[str, Any],
    *,
    run_id: str,
    dataset_id: str = "unspecified",
    dataset_version: str = "1",
    source_path: str | None = None,
) -> tuple[PageRecord, GroundTruthRecord, OCRPrediction, list[EvaluationRecord]]:
    """Convert one foundation evaluation JSONL row to canonical records.

    Expected keys (foundation contract): ``page_id`` or ``generated_page_id``,
    ``reference_text``/``ground_truth_text``, ``prediction_text``/``ocr_text``,
    optional ``model_id``/``model_revision``/``dataset_id``/``profile``.
    """

    page_id = str(row.get("page_id") or row.get("generated_page_id") or "").strip()
    if not page_id:
        raise ValueError("Evaluation record is missing page_id.")
    reference = str(row.get("reference_text") or row.get("ground_truth_text") or "")
    prediction_text = str(row.get("prediction_text") or row.get("ocr_text") or "")
    model_id = str(row.get("model_id") or "unknown-model")
    model_revision = str(row.get("model_revision") or "unresolved")
    split = str(row.get("split") or "unassigned")
    dataset = str(row.get("dataset_id") or dataset_id)
    profile = row.get("profile_id") or row.get("profile")

    page = PageRecord(
        page_id=page_id,
        document_id=str(row.get("source_document_id") or page_id),
        dataset_id=dataset,
        dataset_version=dataset_version,
        split=split,
        profile=profile,
        protection=_protection_from_split(split),
        provenance=Provenance(
            source_format=FOUNDATION_EVALUATION_FORMAT,
            source_uri=Path(source_path).as_posix() if source_path else None,
            adapter_version=ADAPTER_VERSION,
        ),
        metadata={"severity": row.get("overall_severity")},
    )
    gt = GroundTruthRecord(
        page_id=page_id,
        raw_text=reference,
        raw_text_sha256=sha256_text(reference),
        dataset_id=dataset,
        split=split,
        protection=page.protection,
        provenance=page.provenance,
    )
    prediction = OCRPrediction(
        prediction_id=prediction_identity(run_id=run_id, page_id=page_id),
        run_id=run_id,
        page_id=page_id,
        model_id=model_id,
        model_revision=model_revision,
        text=prediction_text,
        text_sha256=sha256_text(prediction_text),
        dataset_id=dataset,
        split=split,
        inference_settings={
            key: row[key]
            for key in ("temperature", "max_new_tokens", "prompt")
            if key in row
        },
        provenance=page.provenance,
    )
    metrics_records = build_page_metric_records(
        run_id=run_id,
        record=gt,
        prediction=prediction,
        dataset_id=dataset,
        split=split,
    )
    return page, gt, prediction, metrics_records


def iter_foundation_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed evaluation JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Evaluation line {line_number} is not an object.")
            yield payload


# ---------------------------------------------------------------------------
# 4. Training Experiment Framework run summary -> ModelRecord with lineage
# ---------------------------------------------------------------------------


def training_run_to_model_record(
    run_dir: str | Path,
    *,
    model_id: str | None = None,
) -> ModelRecord:
    """Link a training run to a canonical ModelRecord (optional lineage).

    Reads only ``metadata.json`` and ``summary.json``; never touches training
    internals or checkpoints.
    """

    source = Path(run_dir)
    metadata_path = source / "metadata.json"
    summary_path = source / "summary.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    experiment = str(
        metadata.get("experiment_name")
        or metadata.get("experiment")
        or source.parent.name
    )
    training_run_id = str(metadata.get("run_id") or source.name)
    config_hash = metadata.get("config_hash")
    resolved_model_id = model_id or str(
        (metadata.get("model") or {}).get("model_id")
        or metadata.get("model_id")
        or f"trained-{experiment}"
    )
    revision = str(
        (metadata.get("model") or {}).get("revision")
        or metadata.get("model_revision")
        or training_run_id
    )
    dataset_hash = metadata.get("dataset_manifest_hash")
    return ModelRecord(
        model_id=resolved_model_id,
        display_name=resolved_model_id,
        revision=revision,
        model_family="clouda-trained",
        training_lineage=TrainingLineage(
            experiment_name=experiment,
            training_run_id=training_run_id,
            checkpoint_name=metadata.get("checkpoint"),
            config_hash=str(config_hash) if config_hash else None,
            manifest_sha256=str(dataset_hash) if dataset_hash else None,
        ),
        metadata={"best_metrics": summary.get("best_metrics", {})},
    )
