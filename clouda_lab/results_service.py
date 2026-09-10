"""Bridge canonical Results Store records into Clouda Lab analysis services."""

from __future__ import annotations

import json
from typing import Any

from clouda_data.results.identity import sha256_text
from clouda_data.results.service import ResultsService
from clouda_data.results.store import UnknownRecordError

from .dataset_selection import (
    SelectionCriteria,
    SelectionResult,
    select_rows,
)
from .evaluation_service import EvaluationService
from .failure_analysis import SampleMetrics
from .models import ErrorAnalysis, FailureReport, OCRSample


class StoredResultsAnalysisService:
    """Analyze persisted OCR evidence without parsing source files directly."""

    def __init__(
        self,
        results: ResultsService,
        *,
        evaluation: EvaluationService | None = None,
    ) -> None:
        self.results = results
        self.evaluation = evaluation or EvaluationService()

    def sample(
        self, run_id: str, page_id: str, *, model_id: str | None = None
    ) -> OCRSample:
        page = self.results.get_page(run_id, page_id)
        ground_truth = self.results.get_ground_truth(run_id, page_id)
        predictions = self.results.get_predictions(run_id, page_id, model_id=model_id)
        if not predictions:
            raise UnknownRecordError(
                f"No prediction for page {page_id!r} in run {run_id!r}."
            )
        if len(predictions) != 1:
            raise ValueError(
                f"Ambiguous predictions for page {page_id!r} in run "
                f"{run_id!r}; provide model_id."
            )
        prediction = predictions[0]
        metadata = {
            **dict(page.metadata),
            "dataset_version": page.dataset_version,
            "document_id": page.document_id,
            "split": page.split,
            "page_number": page.page_number,
            "profile": page.profile,
            "distortions": [dict(item) for item in page.distortions],
            "distortion_seed": page.distortion_seed,
            "source_identity": dict(page.source_identity),
            "tags": list(page.tags),
            "protection": page.protection.to_dict(),
            "training_eligible": page.protection.is_training_eligible,
            "model_revision": prediction.model_revision,
        }
        if page.provenance is not None:
            metadata["provenance"] = page.provenance.to_dict()
        return OCRSample(
            sample_id=page.page_id,
            page_id=page.page_id,
            ground_truth=ground_truth.raw_text,
            prediction=prediction.text,
            model_id=prediction.model_id,
            run_id=prediction.run_id,
            dataset_id=page.dataset_id,
            metadata=metadata,
        )

    def analyze_page(
        self, run_id: str, page_id: str, *, model_id: str | None = None
    ) -> ErrorAnalysis:
        return self.evaluation.evaluate_sample(
            self.sample(run_id, page_id, model_id=model_id)
        )

    def analyze_run(
        self, run_id: str, *, model_id: str | None = None
    ) -> tuple[ErrorAnalysis, ...]:
        pages = sorted(self.results.list_pages(run_id), key=lambda page: page.page_id)
        return tuple(
            self.analyze_page(run_id, page.page_id, model_id=model_id) for page in pages
        )

    def analysis_rows(
        self, run_id: str, *, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for analysis in self.analyze_run(run_id, model_id=model_id):
            metadata = dict(analysis.metadata)
            row = {
                **analysis.to_dict(),
                "sample_id": analysis.page_id,
                "ncer": analysis.normalized_cer,
            }
            for key in (
                "dataset_version",
                "document_id",
                "split",
                "profile",
                "distortions",
                "distortion_seed",
                "source_identity",
                "protection",
                "training_eligible",
                "model_revision",
                "provenance",
                "tags",
            ):
                if key in metadata:
                    row[key] = metadata[key]
            rows.append(row)
        return rows

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        *,
        thresholds: dict[str, float] | None = None,
    ) -> FailureReport:
        def metric_rows(run_id: str) -> list[SampleMetrics]:
            return [
                SampleMetrics(
                    sample_id=item.page_id,
                    cer=item.cer,
                    wer=item.wer,
                    ncer=item.normalized_cer,
                    error_types=item.error_type_counts,
                    metadata=item.metadata,
                )
                for item in self.analyze_run(run_id)
            ]

        return self.evaluation.compare_models(
            metric_rows(baseline_run_id),
            metric_rows(candidate_run_id),
            baseline_id=baseline_run_id,
            candidate_id=candidate_run_id,
            thresholds=thresholds,
        )

    def selection_rows(self, run_id: str) -> list[dict[str, Any]]:
        run = self.results.get_run(run_id)
        rows: list[dict[str, Any]] = []
        for page in sorted(
            self.results.list_pages(run_id), key=lambda item: item.page_id
        ):
            canonical = page.metadata.get("canonical_manifest_row")
            row = dict(canonical) if isinstance(canonical, dict) else {}
            page_payload = page.to_dict()
            metadata = dict(row.get("metadata", {}))
            metadata.update(page_payload.get("metadata", {}))
            page_payload["metadata"] = metadata
            row.update(
                {
                    **page_payload,
                    "sample_id": page.page_id,
                    "source_document_id": page.document_id,
                    "target_split": page.split,
                    "protected": page.protection.protected,
                    "training_eligible": page.protection.is_training_eligible,
                    "model_id": run.get("model_id"),
                    "model_revision": run.get("model_revision"),
                    "run_id": run_id,
                }
            )
            rows.append(row)
        return rows

    def select_pages(
        self,
        run_id: str,
        criteria: SelectionCriteria,
        *,
        seed: int = 20260723,
        scores: dict[str, float] | None = None,
    ) -> SelectionResult:
        rows = self.selection_rows(run_id)
        material = json.dumps(
            rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return select_rows(
            rows,
            criteria,
            source_manifest=f"results://runs/{run_id}/pages",
            source_manifest_sha256=sha256_text(material),
            seed=seed,
            scores=scores,
        )

    def recommend_next(
        self,
        run_id: str,
        *,
        strategy: str = "balanced_hard",
        batch_size: int = 32,
        seed: int = 20260723,
        history: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Return explainable, training-eligible recommendations for a run."""
        candidates = [
            row
            for row in self.analysis_rows(run_id)
            if row.get("training_eligible") is True
        ]
        recommendation = self.evaluation.recommend_next(
            candidates,
            strategy=strategy,
            batch_size=batch_size,
            seed=seed,
            history=history,
        )
        bucket_groups = self.evaluation.failure_buckets(candidates)
        buckets_by_sample: dict[str, list[str]] = {}
        for bucket, assignments in bucket_groups.items():
            for assignment in assignments:
                buckets_by_sample.setdefault(assignment.sample_id, []).append(bucket)
        candidates_by_id = {str(row["sample_id"]): row for row in candidates}
        rows: list[dict[str, Any]] = []
        for selection in recommendation.selections:
            source = candidates_by_id[selection.sample_id]
            buckets = sorted(buckets_by_sample.get(selection.sample_id, ["unknown"]))
            rows.append(
                {
                    **selection.to_dict(),
                    "page_id": selection.sample_id,
                    "bucket": buckets[0],
                    "buckets": buckets,
                    "rationale": recommendation.rationale[selection.sample_id],
                    "prior_selection_state": (
                        "previously_selected"
                        if selection.sample_id in history
                        else "new"
                    ),
                    "dataset_id": source["dataset_id"],
                    "dataset_version": source["dataset_version"],
                    "split": source["split"],
                    "training_eligible": source["training_eligible"],
                    "source_metadata": {
                        key: source[key]
                        for key in (
                            "document_id",
                            "source_identity",
                            "provenance",
                            "profile",
                            "distortions",
                            "distortion_seed",
                        )
                        if key in source
                    },
                }
            )
        return rows


__all__ = ["StoredResultsAnalysisService"]
