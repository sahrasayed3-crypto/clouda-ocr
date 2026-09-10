"""Storage layout, integrity checks, and the query layer for Results Store bundles.

Layout (canonical, ``clouda.ocr.results.v1``)::

    <store_root>/datasets/<dataset_id>@<version>.json
    <store_root>/models/<model_id>.json
    <store_root>/runs/<run_id>/
        metadata.json          InferenceRun
        pages.jsonl            PageRecord rows (sorted by page_id)
        predictions.jsonl      OCRPrediction rows (sorted by page_id)
        metrics.jsonl          EvaluationRecord rows
        summary.json           run-level aggregate metrics + counts
        artifacts.jsonl        ArtifactRef rows

Properties:
- JSONL rows are append-safe and streamed (no full-corpus loads on read paths);
- duplicates: exact re-ingestion is idempotent; conflicting content for the
  same identity raises :class:`ConflictingRecordError`;
- portable: no absolute paths inside canonical files;
- indexes (``<run>/.index/``) are rebuildable caches, never the source of truth.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

from .identity import (
    RESULTS_SCHEMA_VERSION,
    ArtifactRef,
    append_jsonl,
    atomic_write_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
)
from .models import (
    EvaluationRecord,
    GroundTruthRecord,
    OCRPrediction,
    PageRecord,
)

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._@:+-]")


def safe_component(value: str, *, what: str = "identifier") -> str:
    """Validate an identifier used as a single path component."""

    if not value or _UNSAFE_ID.search(value):
        raise ValueError(f"Unsafe {what}: {value!r}")
    return value


class ConflictingRecordError(ValueError):
    """Raised when an existing record's content conflicts with new content."""


class UnknownRecordError(KeyError):
    """Raised when a referenced record does not exist."""


class ResultsStore:
    """Filesystem-backed canonical results store.

    The store owns one root directory. ``read_only=True`` (default for query
    objects) refuses any write. Writes are atomic per file and append-only per
    JSONL.
    """

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self.root = Path(root)
        self.read_only = read_only
        self._datasets_dir = self.root / "datasets"
        self._models_dir = self.root / "models"
        self._runs_dir = self.root / "runs"
        if not read_only:
            for directory in (self._datasets_dir, self._models_dir, self._runs_dir):
                directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def dataset_path(self, dataset_id: str, version: str) -> Path:
        component = safe_component(f"{dataset_id}@{version}", what="dataset id")
        return self._datasets_dir / (component.replace("@", "__at__") + ".json")

    def model_path(self, model_id: str) -> Path:
        return self._models_dir / f"{safe_component(model_id, what='model id')}.json"

    def run_dir(self, run_id: str) -> Path:
        return self._runs_dir / safe_component(run_id, what="run id")

    # ------------------------------------------------------------- datasets

    def save_dataset(self, payload: dict[str, Any]) -> Path:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        dataset_id = str(payload.get("dataset_id", ""))
        version = str(payload.get("version", "1"))
        target = self._datasets_dir / (
            safe_component(f"{dataset_id}__{version}", what="dataset id") + ".json"
        )
        return atomic_write_json(target, payload)

    def load_dataset(self, dataset_id: str, version: str = "1") -> dict[str, Any]:
        target = self._datasets_dir / (
            safe_component(f"{dataset_id}__{version}", what="dataset id") + ".json"
        )
        if not target.exists():
            raise UnknownRecordError(f"Unknown dataset: {dataset_id}@{version}")
        import json

        return json.loads(target.read_text(encoding="utf-8"))

    def list_datasets(self) -> Iterator[dict[str, Any]]:
        import json

        if not self._datasets_dir.exists():
            return
        for path in sorted(self._datasets_dir.glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))

    # --------------------------------------------------------------- models

    def save_model(self, payload: dict[str, Any]) -> Path:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        model_id = str(payload.get("model_id", ""))
        target = self.model_path(model_id)
        return atomic_write_json(target, payload)

    def load_model(self, model_id: str) -> dict[str, Any]:
        target = self.model_path(model_id)
        if not target.exists():
            raise UnknownRecordError(f"Unknown model: {model_id}")
        import json

        return json.loads(target.read_text(encoding="utf-8"))

    def list_models(self) -> Iterator[dict[str, Any]]:
        import json

        if not self._models_dir.exists():
            return
        for path in sorted(self._models_dir.glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))

    # ----------------------------------------------------------------- runs

    def run_metadata_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metadata.json"

    def save_run_metadata(self, payload: dict[str, Any]) -> Path:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        return atomic_write_json(
            self.run_metadata_path(str(payload["run_id"])), payload
        )

    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        target = self.run_metadata_path(run_id)
        if not target.exists():
            raise UnknownRecordError(f"Unknown run: {run_id}")
        import json

        return json.loads(target.read_text(encoding="utf-8"))

    def list_runs(self) -> Iterator[dict[str, Any]]:
        import json

        if not self._runs_dir.exists():
            return
        for path in sorted(self._runs_dir.iterdir()):
            metadata = path / "metadata.json"
            if path.is_dir() and metadata.exists():
                yield json.loads(metadata.read_text(encoding="utf-8"))

    def run_exists(self, run_id: str) -> bool:
        return self.run_metadata_path(run_id).exists()

    # --------------------------------------------------------------- pages

    def pages_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "pages.jsonl"

    def append_pages(self, run_id: str, pages: Iterable[PageRecord]) -> int:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        existing = {
            str(row.get("page_id")): row for row in iter_jsonl(self.pages_path(run_id))
        }
        count = 0
        for page in pages:
            payload = page.to_dict()
            prior = existing.get(page.page_id)
            if prior is not None:
                # Wall-clock ingestion metadata is not content: identical
                # re-ingestion stays idempotent even when ingested_at differs.
                # Content conflicts (any field outside provenance timestamps)
                # are still rejected.
                prior_content = dict(prior)
                new_content = dict(payload)
                prior_content.get("provenance", {}).pop("ingested_at", None)
                new_content.get("provenance", {}).pop("ingested_at", None)
                if prior_content != new_content:
                    raise ConflictingRecordError(
                        f"Conflicting page record for {page.page_id!r} "
                        f"in run {run_id!r}."
                    )
                continue
            append_jsonl(self.pages_path(run_id), payload)
            existing[page.page_id] = payload
            count += 1
        return count

    def iter_pages(
        self,
        run_id: str,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
        profile: str | None = None,
        document_id: str | None = None,
    ) -> Iterator[PageRecord]:
        for row in iter_jsonl(self.pages_path(run_id)):
            if dataset_id is not None and row.get("dataset_id") != dataset_id:
                continue
            if split is not None and row.get("split") != split:
                continue
            if profile is not None and row.get("profile") != profile:
                continue
            if document_id is not None and row.get("document_id") != document_id:
                continue
            yield PageRecord.from_dict(row)

    def get_page(self, run_id: str, page_id: str) -> PageRecord:
        for row in iter_jsonl(self.pages_path(run_id)):
            if row.get("page_id") == page_id:
                return PageRecord.from_dict(row)
        raise UnknownRecordError(f"Unknown page {page_id!r} in run {run_id!r}")

    # --------------------------------------------------------- ground truth

    def ground_truth_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "ground_truth.jsonl"

    def append_ground_truth(
        self, run_id: str, records: Iterable[GroundTruthRecord]
    ) -> int:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        existing = {
            str(row.get("page_id")): row
            for row in iter_jsonl(self.ground_truth_path(run_id))
        }
        count = 0
        for record in records:
            payload = record.to_dict()
            prior = existing.get(record.page_id)
            if prior is not None:
                prior_content = dict(prior)
                new_content = dict(payload)
                prior_content.get("provenance", {}).pop("ingested_at", None)
                new_content.get("provenance", {}).pop("ingested_at", None)
                if prior_content != new_content:
                    raise ConflictingRecordError(
                        f"Conflicting ground truth for {record.page_id!r} "
                        f"in run {run_id!r}."
                    )
                continue
            append_jsonl(self.ground_truth_path(run_id), payload)
            existing[record.page_id] = payload
            count += 1
        return count

    def get_ground_truth(self, run_id: str, page_id: str) -> GroundTruthRecord:
        for row in iter_jsonl(self.ground_truth_path(run_id)):
            if row.get("page_id") == page_id:
                return GroundTruthRecord.from_dict(row)
        raise UnknownRecordError(
            f"Unknown ground truth for page {page_id!r} in run {run_id!r}"
        )

    # ---------------------------------------------------------- predictions

    def predictions_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "predictions.jsonl"

    def append_predictions(
        self, run_id: str, predictions: Iterable[OCRPrediction]
    ) -> int:
        """Append predictions; identical re-ingestion is idempotent.

        Conflicting content for the same (run_id, page_id) is rejected.
        """

        if self.read_only:
            raise PermissionError("Results store is read-only.")
        existing = {
            str(row.get("page_id")): row
            for row in iter_jsonl(self.predictions_path(run_id))
        }
        count = 0
        for prediction in predictions:
            if prediction.run_id != run_id:
                raise ValueError(
                    f"Prediction run id {prediction.run_id!r} does not match "
                    f"bundle run {run_id!r}."
                )
            payload = prediction.to_dict()
            prior = existing.get(prediction.page_id)
            if prior is not None:
                prior_content = dict(prior)
                new_content = dict(payload)
                prior_content.get("provenance", {}).pop("ingested_at", None)
                new_content.get("provenance", {}).pop("ingested_at", None)
                if prior_content != new_content:
                    raise ConflictingRecordError(
                        f"Conflicting prediction for page {prediction.page_id!r} "
                        f"in run {run_id!r}; duplicate handling is "
                        "reject-on-conflict."
                    )
                continue
            append_jsonl(self.predictions_path(run_id), payload)
            existing[prediction.page_id] = payload
            count += 1
        return count

    def iter_predictions(
        self,
        run_id: str,
        *,
        page_id: str | None = None,
        model_id: str | None = None,
        split: str | None = None,
    ) -> Iterator[OCRPrediction]:
        for row in iter_jsonl(self.predictions_path(run_id)):
            if page_id is not None and row.get("page_id") != page_id:
                continue
            if model_id is not None and row.get("model_id") != model_id:
                continue
            if split is not None and row.get("split") != split:
                continue
            yield OCRPrediction.from_dict(row)

    # -------------------------------------------------------------- metrics

    def metrics_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metrics.jsonl"

    def append_metrics(self, run_id: str, records: Iterable[EvaluationRecord]) -> int:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        existing: set[tuple[str, str, str]] = set()
        rows: list[dict[str, Any]] = []
        for row in iter_jsonl(self.metrics_path(run_id)):
            rows.append(row)
            existing.add(
                (
                    str(row.get("page_id") or ""),
                    str(row.get("metric_name")),
                    str(row.get("scope")),
                )
            )
        count = 0
        for record in records:
            key = (record.page_id or "", record.metric_name, record.scope.value)
            if key in existing:
                continue
            append_jsonl(self.metrics_path(run_id), record.to_dict())
            existing.add(key)
            count += 1
        return count

    def iter_metrics(
        self,
        run_id: str,
        *,
        page_id: str | None = None,
        metric_name: str | None = None,
        scope: str | None = None,
    ) -> Iterator[EvaluationRecord]:
        for row in iter_jsonl(self.metrics_path(run_id)):
            if page_id is not None and row.get("page_id") != page_id:
                continue
            if metric_name is not None and row.get("metric_name") != metric_name:
                continue
            if scope is not None and row.get("scope") != scope:
                continue
            yield EvaluationRecord.from_dict(row)

    # ------------------------------------------------------------- summary

    def summary_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "summary.json"

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> Path:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        return atomic_write_json(self.summary_path(run_id), summary)

    def load_summary(self, run_id: str) -> dict[str, Any]:
        target = self.summary_path(run_id)
        if not target.exists():
            raise UnknownRecordError(f"Run {run_id!r} has no summary.")
        import json

        return json.loads(target.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ artifacts

    def artifacts_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts.jsonl"

    def append_artifacts(self, run_id: str, artifacts: Iterable[ArtifactRef]) -> int:
        if self.read_only:
            raise PermissionError("Results store is read-only.")
        existing = {
            str(row.get("artifact_id")): row
            for row in iter_jsonl(self.artifacts_path(run_id))
        }
        count = 0
        for artifact in artifacts:
            payload = artifact.to_dict()
            prior = existing.get(artifact.artifact_id)
            if prior is not None:
                if prior != payload:
                    raise ConflictingRecordError(
                        f"Conflicting artifact record {artifact.artifact_id!r}."
                    )
                continue
            append_jsonl(self.artifacts_path(run_id), payload)
            existing[artifact.artifact_id] = payload
            count += 1
        return count

    def iter_artifacts(self, run_id: str) -> Iterator[ArtifactRef]:
        for row in iter_jsonl(self.artifacts_path(run_id)):
            yield ArtifactRef.from_dict(row)

    # ------------------------------------------------------------ integrity

    def verify_bundle(self, run_id: str) -> dict[str, Any]:
        """Verify bundle integrity; returns a report, raises on corruption."""

        issues: list[str] = []
        run_id = safe_component(run_id, what="run id")
        if not self.run_exists(run_id):
            raise UnknownRecordError(f"Unknown run: {run_id}")

        metadata = self.load_run_metadata(run_id)
        if metadata.get("schema_version") != RESULTS_SCHEMA_VERSION:
            issues.append(
                f"Unexpected metadata schema version: {metadata.get('schema_version')!r}"
            )
        if metadata.get("run_id") != run_id:
            issues.append("metadata run_id mismatch")

        page_ids: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(self.pages_path(run_id)):
            page_id = str(row.get("page_id"))
            if page_id in page_ids:
                issues.append(f"duplicate page id: {page_id}")
            page_ids[page_id] = row
            protection = row.get("protection") or {}
            if protection.get("protected") and protection.get("is_training_eligible"):
                issues.append(f"protected page marked training-eligible: {page_id}")

        gt_seen: set[str] = set()
        for row in iter_jsonl(self.ground_truth_path(run_id)):
            page_id = str(row.get("page_id"))
            if page_id in gt_seen:
                issues.append(f"duplicate ground truth row: {page_id}")
            gt_seen.add(page_id)
            expected = str(row.get("raw_text_sha256"))
            if sha256_text(str(row.get("raw_text", ""))) != expected:
                issues.append(f"ground truth hash mismatch: {page_id}")
            if page_id not in page_ids:
                issues.append(f"ground truth references unknown page: {page_id}")

        prediction_pages: set[str] = set()
        seen_predictions: set[tuple[str, str]] = set()
        for row in iter_jsonl(self.predictions_path(run_id)):
            page_id = str(row.get("page_id"))
            key = (str(row.get("run_id")), page_id)
            if key in seen_predictions:
                issues.append(f"duplicate prediction: {key}")
            seen_predictions.add(key)
            prediction_pages.add(page_id)
            expected = str(row.get("text_sha256"))
            if sha256_text(str(row.get("text", ""))) != expected:
                issues.append(f"prediction text hash mismatch: {page_id}")
            if page_id not in page_ids:
                issues.append(f"prediction references unknown page: {page_id}")
            if str(row.get("run_id")) != run_id:
                issues.append(f"prediction run mismatch: {page_id}")

        metric_pages: set[str] = set()
        for row in iter_jsonl(self.metrics_path(run_id)):
            if row.get("scope") == "page":
                metric_pages.add(str(row.get("page_id")))
                if str(row.get("page_id")) not in page_ids:
                    issues.append(
                        f"metric references unknown page: {row.get('page_id')}"
                    )

        missing_gt = sorted(set(page_ids) - gt_seen)
        if missing_gt:
            issues.append(f"pages missing ground truth: {missing_gt[:10]}")

        return {
            "run_id": run_id,
            "schema_version": RESULTS_SCHEMA_VERSION,
            "pages": len(page_ids),
            "predictions": len(seen_predictions),
            "metrics": len(metric_pages) + 0,
            "issues": issues,
            "ok": not issues,
        }

    # ------------------------------------------------------------ indexes

    def rebuild_index(self, run_id: str) -> Path:
        """Rebuild the per-run rebuildable index (cache only)."""

        if self.read_only:
            raise PermissionError("Results store is read-only.")
        index_dir = self.run_dir(run_id) / ".index"
        index_dir.mkdir(parents=True, exist_ok=True)
        by_page: dict[str, list[str]] = {}
        for row in iter_jsonl(self.predictions_path(run_id)):
            by_page.setdefault(str(row.get("page_id")), []).append(
                str(row.get("prediction_id"))
            )
        atomic_write_json(index_dir / "predictions_by_page.json", by_page)
        return index_dir

    def export_run_copy(self, run_id: str, destination: str | Path) -> Path:
        """Export a portable self-contained store copy of one run.

        The destination becomes a valid ResultsStore root (``runs/<run_id>/…``)
        so it can be reopened read-only elsewhere.
        """

        source = self.run_dir(run_id)
        if not source.exists():
            raise UnknownRecordError(f"Unknown run: {run_id}")
        target = Path(destination)
        target_runs = target / "runs"
        if target.exists():
            shutil.rmtree(target)
        target_runs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target_runs / safe_component(run_id, what="run id"))
        return target


def sha256_of_file(path: str | Path) -> str:
    return sha256_file(path)
