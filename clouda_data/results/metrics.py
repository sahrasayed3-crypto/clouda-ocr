"""Metric computation and storage.

The Results Store does NOT create competing metric implementations: CER/WER
come from ``clouda_data.evaluation`` (the canonical foundation engine), and
Arabic normalization from ``clouda_data.ground_truth.normalization`` via the
foundation's ``normalize_ocr_text``. This module only *adapts and stores*.
"""

from __future__ import annotations

from typing import Any, Iterable

from clouda_data.evaluation.cer import cer as foundation_cer
from clouda_data.evaluation.normalization import normalize_ocr_text
from clouda_data.evaluation.wer import wer as foundation_wer

from .identity import evaluation_record_identity, utc_now
from .models import EvaluationRecord, EvaluationScope, GroundTruthRecord, OCRPrediction

EVALUATOR_VERSION = "clouda.results.metrics.v1"
NORMALIZATION_POLICY_RAW = "raw"
NORMALIZATION_POLICY_ARABIC_FOLD = "comparison_arabic_fold_digits"


def compute_page_metrics(
    record: GroundTruthRecord,
    prediction: OCRPrediction,
    *,
    normalization: str = NORMALIZATION_POLICY_ARABIC_FOLD,
) -> dict[str, Any]:
    """Compute CER/WER/exact-match for one page prediction.

    ``normalization`` selects the comparison view:
    - ``raw``: exact strings, no normalization;
    - ``comparison_arabic_fold_digits``: the foundation's Arabic normalization
      (NFC + alef/ya folds + diacritics/tatweel removal + digit folding);
    raw stored strings are never modified either way.
    """

    if normalization == NORMALIZATION_POLICY_RAW:
        reference = record.raw_text
        hypothesis = prediction.text
    elif normalization == NORMALIZATION_POLICY_ARABIC_FOLD:
        reference = normalize_ocr_text(record.raw_text)
        hypothesis = normalize_ocr_text(prediction.text)
    else:
        raise ValueError(f"Unknown normalization policy: {normalization!r}")

    cer_value = foundation_cer(reference, hypothesis)
    wer_value = foundation_wer(reference, hypothesis)
    return {
        "cer": cer_value,
        "wer": wer_value,
        "exact_match": record.raw_text == prediction.text,
        "normalized_exact_match": (
            normalize_ocr_text(record.raw_text) == normalize_ocr_text(prediction.text)
        ),
        "character_accuracy": max(0.0, 1.0 - cer_value),
        "word_accuracy": max(0.0, 1.0 - wer_value),
        "missing_text": bool(reference and not prediction.text),
        "hallucination": bool(not reference and prediction.text),
        "empty_output": not prediction.text.strip(),
        "normalization": normalization,
    }


def build_page_metric_records(
    *,
    run_id: str,
    record: GroundTruthRecord,
    prediction: OCRPrediction,
    dataset_id: str = "",
    split: str = "unassigned",
    normalizations: Iterable[str] = (NORMALIZATION_POLICY_ARABIC_FOLD,),
    computed_at: str | None = None,
) -> list[EvaluationRecord]:
    """Create canonical page-scope EvaluationRecords for one prediction."""

    computed_at = computed_at or utc_now()
    records: list[EvaluationRecord] = []
    for normalization in normalizations:
        metrics = compute_page_metrics(record, prediction, normalization=normalization)
        for metric_name in ("cer", "wer", "exact_match", "normalized_exact_match"):
            value = float(metrics[metric_name])
            records.append(
                EvaluationRecord(
                    record_id=evaluation_record_identity(
                        run_id=run_id,
                        page_id=record.page_id,
                        metric=f"{metric_name}@{normalization}",
                    ),
                    run_id=run_id,
                    metric_name=f"{metric_name}@{normalization}",
                    value=value,
                    scope=EvaluationScope.PAGE,
                    page_id=record.page_id,
                    dataset_id=dataset_id or record.dataset_id,
                    split=split or record.split,
                    normalization_policy=normalization,
                    evaluator_version=EVALUATOR_VERSION,
                    computed_at=computed_at,
                    details={
                        key: metrics[key]
                        for key in (
                            "character_accuracy",
                            "word_accuracy",
                            "missing_text",
                            "hallucination",
                            "empty_output",
                        )
                    },
                )
            )
    return records


def build_run_summary_records(
    *,
    run_id: str,
    page_records: Iterable[EvaluationRecord],
    dataset_id: str = "",
    split: str = "unassigned",
    computed_at: str | None = None,
) -> list[EvaluationRecord]:
    """Aggregate page-level CER/WER into run-scope summary records.

    Mean over page-scope values of the same metric name — matching the
    foundation's arithmetic-mean leaderboard semantics.
    """

    computed_at = computed_at or utc_now()
    grouped: dict[str, list[float]] = {}
    for record in page_records:
        grouped.setdefault(record.metric_name, []).append(float(record.value))
    summaries: list[EvaluationRecord] = []
    for metric_name in sorted(grouped):
        values = grouped[metric_name]
        mean_value = sum(values) / len(values)
        summaries.append(
            EvaluationRecord(
                record_id=evaluation_record_identity(
                    run_id=run_id, page_id="*run_summary*", metric=metric_name
                ),
                run_id=run_id,
                metric_name=metric_name,
                value=mean_value,
                scope=EvaluationScope.RUN_SUMMARY,
                page_id=None,
                dataset_id=dataset_id,
                split=split,
                normalization_policy=(
                    metric_name.split("@", 1)[1] if "@" in metric_name else None
                ),
                evaluator_version=EVALUATOR_VERSION,
                computed_at=computed_at,
                details={"pages": len(values)},
            )
        )
    return summaries
