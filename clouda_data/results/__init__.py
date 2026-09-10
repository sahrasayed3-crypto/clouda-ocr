"""Clouda OCR Benchmark & Results Store.

Canonical, safe, indexed access layer between raw benchmark/model outputs and
higher-level analysis systems (error analysis, hard-example mining, training
selection, Clouda Lab UI).

Schema version: ``clouda.ocr.results.v1``.
"""

from __future__ import annotations

from .artifacts import ArtifactResolutionError, ArtifactResolver
from .ground_truth import (
    build_ground_truth_record,
    normalized_view,
    verify_ground_truth,
)
from .identity import (
    RESULTS_SCHEMA_VERSION,
    ArtifactRef,
    dataset_identity,
    page_identity,
    prediction_identity,
    run_identity,
    utc_now,
    validate_sha256,
)
from .metrics import (
    EVALUATOR_VERSION,
    NORMALIZATION_POLICY_ARABIC_FOLD,
    NORMALIZATION_POLICY_RAW,
    build_page_metric_records,
    build_run_summary_records,
    compute_page_metrics,
)
from .models import (
    BenchmarkDataset,
    EvaluationRecord,
    EvaluationScope,
    GroundTruthRecord,
    InferenceRun,
    InferenceRunStatus,
    ModelRecord,
    OCRPrediction,
    PageRecord,
    ProtectionInfo,
    Provenance,
    ResultBundle,
    TrainingLineage,
)
from .service import NewPrediction, ResultsService
from .store import ConflictingRecordError, ResultsStore, UnknownRecordError

__all__ = [
    "RESULTS_SCHEMA_VERSION",
    "ArtifactRef",
    "ArtifactResolutionError",
    "ArtifactResolver",
    "BenchmarkDataset",
    "ConflictingRecordError",
    "EVALUATOR_VERSION",
    "EvaluationRecord",
    "EvaluationScope",
    "GroundTruthRecord",
    "InferenceRun",
    "InferenceRunStatus",
    "ModelRecord",
    "NORMALIZATION_POLICY_ARABIC_FOLD",
    "NORMALIZATION_POLICY_RAW",
    "NewPrediction",
    "OCRPrediction",
    "PageRecord",
    "Provenance",
    "ProtectionInfo",
    "ResultBundle",
    "ResultsService",
    "ResultsStore",
    "TrainingLineage",
    "UnknownRecordError",
    "build_ground_truth_record",
    "build_page_metric_records",
    "build_run_summary_records",
    "compute_page_metrics",
    "dataset_identity",
    "normalized_view",
    "page_identity",
    "prediction_identity",
    "run_identity",
    "utc_now",
    "validate_sha256",
    "verify_ground_truth",
]
