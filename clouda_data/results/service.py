"""ResultsService: the backend service contract for the future Clouda Lab.

One facade over the canonical store + adapters + metrics. UI-specific
concerns (rendering, dashboards, HTTP) are explicitly out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import ArtifactResolver, ArtifactResolutionError
from .ground_truth import build_ground_truth_record, normalized_view
from .identity import (
    ArtifactRef,
    RESULTS_SCHEMA_VERSION,
    prediction_identity,
    run_identity,
    utc_now,
)
from .metrics import build_page_metric_records, build_run_summary_records
from .models import (
    EvaluationRecord,
    EvaluationScope,
    GroundTruthRecord,
    InferenceRun,
    InferenceRunStatus,
    OCRPrediction,
    PageRecord,
)
from .store import ResultsStore


@dataclass(frozen=True)
class NewPrediction:
    """Input payload for registering a prediction (pre-canonicalization)."""

    page_id: str
    text: str
    model_id: str
    model_revision: str = "unresolved"
    inference_settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ResultsService:
    """Canonical OCR results service (backend only, no UI)."""

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self.store = ResultsStore(root, read_only=read_only)

    # ----------------------------------------------------------- datasets

    def register_dataset(
        self,
        *,
        dataset_id: str,
        version: str = "1",
        name: str = "",
        description: str = "",
        manifest_sha256: str | None = None,
        splits: Iterable[str] = (),
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .models import BenchmarkDataset, Provenance
        from .store import ConflictingRecordError, UnknownRecordError

        requested = {
            "name": name,
            "description": description,
            "manifest_sha256": manifest_sha256,
            "splits": list(splits),
            "tags": list(tags),
            "metadata": dict(metadata or {}),
        }
        try:
            existing = self.store.load_dataset(dataset_id, version)
        except UnknownRecordError:
            existing = None
        if existing is not None:
            comparable = {
                key: existing.get(key, [] if key in {"splits", "tags"} else None)
                for key in requested
            }
            comparable["name"] = existing.get("name", "")
            comparable["description"] = existing.get("description", "")
            comparable["metadata"] = existing.get("metadata", {})
            if comparable != requested:
                raise ConflictingRecordError(
                    f"Conflicting dataset record for {dataset_id!r}@{version!r}."
                )
            return existing

        dataset = BenchmarkDataset(
            dataset_id=dataset_id,
            version=version,
            name=name,
            description=description,
            manifest_sha256=manifest_sha256,
            splits=tuple(requested["splits"]),
            tags=tuple(requested["tags"]),
            metadata=dict(metadata or {}),
            provenance=Provenance(source_format="results-service.manual"),
        )
        self.store.save_dataset(dataset.to_dict())
        return dataset.to_dict()

    def get_dataset(self, dataset_id: str, version: str = "1") -> dict[str, Any]:
        return self.store.load_dataset(dataset_id, version)

    def list_datasets(self) -> list[dict[str, Any]]:
        return list(self.store.list_datasets())

    # ------------------------------------------------------------- models

    def register_model(self, record: Any) -> dict[str, Any]:
        from .models import ModelRecord
        from .store import ConflictingRecordError, UnknownRecordError

        if isinstance(record, ModelRecord):
            model = record
        else:
            model = ModelRecord(**record)
        requested = model.to_dict()
        try:
            existing = self.store.load_model(model.model_id)
        except UnknownRecordError:
            existing = None
        if existing is not None:
            existing_identity = {
                k: v for k, v in existing.items() if k != "registered_at"
            }
            requested_identity = {
                k: v for k, v in requested.items() if k != "registered_at"
            }
            if existing_identity != requested_identity:
                raise ConflictingRecordError(
                    f"Conflicting model record for {model.model_id!r}."
                )
            return existing
        self.store.save_model(requested)
        return requested

    def list_models(self) -> list[dict[str, Any]]:
        return list(self.store.list_models())

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self.store.load_model(model_id)

    # -------------------------------------------------------------- runs

    def create_run(
        self,
        *,
        model_id: str,
        model_revision: str = "unresolved",
        dataset_id: str,
        dataset_version: str = "1",
        split: str = "unassigned",
        manifest_sha256: str | None = None,
        config_hash: str | None = None,
        environment: dict[str, Any] | None = None,
        training_lineage: Any = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> InferenceRun:
        run_id = run_identity(
            model_id=model_id,
            model_revision=model_revision,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            manifest_sha256=manifest_sha256,
            created_at=created_at or utc_now(),
        )
        run = InferenceRun(
            run_id=run_id,
            model_id=model_id,
            model_revision=model_revision,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            split=split,
            status=InferenceRunStatus.CREATED,
            manifest_sha256=manifest_sha256,
            config_hash=config_hash,
            started_at=created_at,
            environment=dict(environment or {}),
            training_lineage=training_lineage,
            metadata=dict(metadata or {}),
        )
        self.store.save_run_metadata(run.to_dict())
        return run

    def mark_run(self, run_id: str, status: InferenceRunStatus) -> dict[str, Any]:
        payload = self.store.load_run_metadata(run_id)
        payload["status"] = status.value
        if status is InferenceRunStatus.RUNNING and not payload.get("started_at"):
            payload["started_at"] = utc_now()
        if status in (
            InferenceRunStatus.COMPLETED,
            InferenceRunStatus.FAILED,
            InferenceRunStatus.INTERRUPTED,
        ):
            payload["ended_at"] = utc_now()
        run = InferenceRun.from_dict(payload)
        self.store.save_run_metadata(run.to_dict())
        return run.to_dict()

    def list_runs(
        self,
        *,
        model_id: str | None = None,
        dataset_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        runs = list(self.store.list_runs())
        if model_id is not None:
            runs = [run for run in runs if run.get("model_id") == model_id]
        if dataset_id is not None:
            runs = [run for run in runs if run.get("dataset_id") == dataset_id]
        if status is not None:
            runs = [run for run in runs if run.get("status") == status]
        return runs

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.store.load_run_metadata(run_id)

    # ------------------------------------------------------------- pages

    def add_page(self, run_id: str, page: PageRecord) -> int:
        return self.store.append_pages(run_id, [page])

    def add_pages(self, run_id: str, pages: Iterable[PageRecord]) -> int:
        return self.store.append_pages(run_id, pages)

    def get_page(self, run_id: str, page_id: str) -> PageRecord:
        return self.store.get_page(run_id, page_id)

    def list_pages(
        self,
        run_id: str,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
        profile: str | None = None,
        document_id: str | None = None,
    ) -> list[PageRecord]:
        return list(
            self.store.iter_pages(
                run_id,
                dataset_id=dataset_id,
                split=split,
                profile=profile,
                document_id=document_id,
            )
        )

    # ------------------------------------------------------- ground truth

    def add_ground_truth(
        self, run_id: str, page: PageRecord, raw_text: str
    ) -> GroundTruthRecord:
        record = build_ground_truth_record(page=page, raw_text=raw_text)
        self.store.append_ground_truth(run_id, [record])
        return record

    def get_ground_truth(self, run_id: str, page_id: str) -> GroundTruthRecord:
        return self.store.get_ground_truth(run_id, page_id)

    def get_ground_truth_text(self, run_id: str, page_id: str) -> str:
        return self.store.get_ground_truth(run_id, page_id).raw_text

    def get_ground_truth_normalized(
        self, run_id: str, page_id: str, *, fold_digits: bool = True
    ) -> str:
        """Normalized comparison view (default matches the metrics policy)."""

        return normalized_view(
            self.store.get_ground_truth(run_id, page_id), fold_digits=fold_digits
        )

    # -------------------------------------------------------- predictions

    def add_prediction(
        self,
        run_id: str,
        payload: NewPrediction,
        *,
        dataset_id: str = "",
        split: str = "unassigned",
    ) -> OCRPrediction:
        import hashlib

        page = self.store.get_page(run_id, payload.page_id)
        prediction = OCRPrediction(
            prediction_id=prediction_identity(run_id=run_id, page_id=payload.page_id),
            run_id=run_id,
            page_id=payload.page_id,
            model_id=payload.model_id,
            model_revision=payload.model_revision,
            text=payload.text,
            text_sha256=hashlib.sha256(payload.text.encode("utf-8")).hexdigest(),
            dataset_id=dataset_id or page.dataset_id,
            split=split if split != "unassigned" else page.split,
            inference_settings=dict(payload.inference_settings or {}),
            metadata=dict(payload.metadata or {}),
            provenance=page.provenance,
        )
        self.store.append_predictions(run_id, [prediction])
        return prediction

    def get_predictions(
        self,
        run_id: str,
        page_id: str,
        *,
        model_id: str | None = None,
    ) -> list[OCRPrediction]:
        return list(
            self.store.iter_predictions(run_id, page_id=page_id, model_id=model_id)
        )

    def list_predictions(
        self,
        *,
        run_id: str,
        model_id: str | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
    ) -> list[OCRPrediction]:
        predictions = list(
            self.store.iter_predictions(run_id, model_id=model_id, split=split)
        )
        if dataset_id is not None:
            predictions = [
                prediction
                for prediction in predictions
                if prediction.dataset_id == dataset_id
            ]
        return predictions

    # ------------------------------------------------------------ metrics

    def evaluate_prediction(
        self,
        run_id: str,
        prediction: OCRPrediction,
        *,
        normalizations: Iterable[str] = ("comparison_arabic_fold_digits",),
    ) -> list[EvaluationRecord]:
        gt = self.store.get_ground_truth(run_id, prediction.page_id)
        records = build_page_metric_records(
            run_id=run_id,
            record=gt,
            prediction=prediction,
            dataset_id=prediction.dataset_id,
            split=prediction.split,
            normalizations=normalizations,
        )
        self.store.append_metrics(run_id, records)
        return records

    def finalize_run(self, run_id: str) -> dict[str, Any]:
        """Compute run-scope summaries, save summary.json, mark COMPLETED."""

        page_records = list(
            self.store.iter_metrics(run_id, scope=EvaluationScope.PAGE.value)
        )
        run_payload = self.store.load_run_metadata(run_id)
        summary_records = build_run_summary_records(
            run_id=run_id,
            page_records=page_records,
            dataset_id=str(run_payload.get("dataset_id", "")),
            split=str(run_payload.get("split", "unassigned")),
        )
        if summary_records:
            self.store.append_metrics(run_id, summary_records)
        page_count = sum(1 for _ in self.store.iter_pages(run_id))
        prediction_count = sum(1 for _ in self.store.iter_predictions(run_id))
        summary = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "run_id": run_id,
            "pages": page_count,
            "predictions": prediction_count,
            "metrics": {record.metric_name: record.value for record in summary_records},
            "finalized_at": utc_now(),
        }
        self.store.save_summary(run_id, summary)
        self.store.save_run_metadata(
            {
                **run_payload,
                "status": InferenceRunStatus.COMPLETED.value,
                "page_count": page_count,
            }
        )
        return summary

    def get_metrics(
        self,
        run_id: str,
        *,
        page_id: str | None = None,
        metric_name: str | None = None,
        scope: str | None = None,
    ) -> list[EvaluationRecord]:
        return list(
            self.store.iter_metrics(
                run_id, page_id=page_id, metric_name=metric_name, scope=scope
            )
        )

    def get_worst_pages(
        self,
        run_id: str,
        *,
        metric: str = "cer@comparison_arabic_fold_digits",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Basic sorted metric access (worst = highest error first)."""

        rows: list[dict[str, Any]] = [
            {
                "page_id": record.page_id,
                "metric": record.metric_name,
                "value": float(record.value),
            }
            for record in self.store.iter_metrics(run_id, metric_name=metric)
        ]
        rows.sort(key=lambda item: (-item["value"], item["page_id"]))
        return rows[: max(0, limit)]

    # ----------------------------------------------------------- artifacts

    def resolve_artifact(
        self, artifact: ArtifactRef, resolver: ArtifactResolver
    ) -> Path:
        try:
            return resolver.resolve(artifact)
        except ArtifactResolutionError:
            raise

    # ---------------------------------------------------------- integrity

    def verify(self, run_id: str) -> dict[str, Any]:
        return self.store.verify_bundle(run_id)
