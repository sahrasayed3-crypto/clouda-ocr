"""Typed domain models for the Clouda OCR Benchmark & Results Store.

All records are versioned with ``clouda.ocr.results.v1``, frozen, and
JSON-serializable. Serialization is strict: unknown fields are rejected on
deserialization so silently-evolving producers cannot corrupt the store.

Holdout safety is part of the contract: every page-adjacent record carries a
:class:`ProtectionInfo` block, and ``is_training_eligible`` fails closed —
unknown protection state means *not* eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

from .identity import RESULTS_SCHEMA_VERSION, ArtifactRef, utc_now, validate_sha256

PROTECTED_SPLIT_MARKERS = frozenset(
    {
        "holdout",
        "protected_holdout",
        "benchmark_holdout",
        "private_holdout",
    }
)

TRAINING_ELIGIBLE_SPLITS = frozenset({"train", "validation", "test", "unassigned"})


class InferenceRunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class EvaluationScope(StrEnum):
    PAGE = "page"
    RUN_SUMMARY = "run_summary"


@dataclass(frozen=True)
class ProtectionInfo:
    """Fail-closed protection markers for a page.

    ``protected`` is True when the page belongs to a protected/holdout split or
    carries an explicit protection marker. ``is_training_eligible`` is derived
    from these markers — never stored as an independent writable fact.
    """

    protected: bool = False
    reasons: tuple[str, ...] = ()
    split: str = "unassigned"

    def __post_init__(self) -> None:
        if self.protected and not self.reasons:
            raise ValueError("Protected pages must record at least one reason.")

    @property
    def is_training_eligible(self) -> bool:
        if self.protected:
            return False
        split = self.split.strip().lower() or "unassigned"
        if split in PROTECTED_SPLIT_MARKERS:
            return False
        return split in TRAINING_ELIGIBLE_SPLITS

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected": self.protected,
            "reasons": list(self.reasons),
            "split": self.split,
            "is_training_eligible": self.is_training_eligible,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProtectionInfo":
        split = str(value.get("split", "unassigned"))
        reasons = tuple(str(reason) for reason in value.get("reasons", ()))
        protected = bool(value.get("protected", False))
        marker_hits = sorted(({split} | set(reasons)) & PROTECTED_SPLIT_MARKERS)
        if marker_hits and not protected:
            protected = True
            extra = tuple(f"protected_split_marker:{marker}" for marker in marker_hits)
            reasons = reasons + extra
        return cls(protected=protected, reasons=reasons, split=split)


@dataclass(frozen=True)
class Provenance:
    """Where a record came from, in portable form."""

    source_format: str
    source_uri: str | None = None
    source_sha256: str | None = None
    license_or_permission: str | None = None
    adapter_version: str | None = None
    ingested_at: str = field(default_factory=utc_now)
    extra: dict[str, Any] = field(default_factory=dict)
    source_private: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.source_format.strip():
            raise ValueError("Provenance source_format cannot be blank.")
        if self.source_sha256 is not None:
            object.__setattr__(
                self, "source_sha256", validate_sha256(self.source_sha256)
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"source_format": self.source_format}
        for key in ("source_uri", "license_or_permission", "adapter_version"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.source_sha256 is not None:
            payload["source_sha256"] = self.source_sha256
        payload["ingested_at"] = self.ingested_at
        if self.extra:
            payload["extra"] = dict(self.extra)
        if self.source_private:
            payload["source_private"] = dict(self.source_private)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Provenance":
        known = {item.name for item in fields(cls)}
        passthrough = {key: item for key, item in value.items() if key not in known}
        raw_extra = value.get("extra")
        extra_dict: dict[str, Any] = (
            dict(raw_extra) if isinstance(raw_extra, dict) else {}
        )
        extra: dict[str, Any] = {**passthrough, **extra_dict}
        return cls(
            source_format=str(value.get("source_format", "unknown")),
            source_uri=value.get("source_uri"),
            source_sha256=value.get("source_sha256"),
            license_or_permission=value.get("license_or_permission"),
            adapter_version=value.get("adapter_version"),
            ingested_at=str(value.get("ingested_at") or utc_now()),
            extra=extra,
            source_private=(
                dict(value["source_private"]) if value.get("source_private") else None
            ),
        )


def _strict_kwargs(
    cls: type, data: dict[str, Any], *, version_key: str = "schema_version"
) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    version = data.get(version_key, RESULTS_SCHEMA_VERSION)
    if version != RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {cls.__name__} schema version: {version!r}; "
            f"expected {RESULTS_SCHEMA_VERSION!r}."
        )
    kwargs = dict(data)
    kwargs.pop(version_key, None)
    return kwargs


@dataclass(frozen=True)
class BenchmarkDataset:
    """A versioned benchmark/dataset registered in the store."""

    dataset_id: str
    version: str = "1"
    name: str = ""
    description: str = ""
    manifest_artifact: ArtifactRef | None = None
    manifest_sha256: str | None = None
    page_count: int | None = None
    splits: tuple[str, ...] = ()
    provenance: Provenance | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be blank.")
        if not self.version.strip():
            raise ValueError("Dataset version cannot be blank.")
        if self.manifest_sha256 is not None:
            object.__setattr__(
                self, "manifest_sha256", validate_sha256(self.manifest_sha256)
            )

    @property
    def identity(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "splits": list(self.splits),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }
        if self.manifest_artifact is not None:
            payload["manifest_artifact"] = self.manifest_artifact.to_dict()
        if self.manifest_sha256 is not None:
            payload["manifest_sha256"] = self.manifest_sha256
        if self.page_count is not None:
            payload["page_count"] = self.page_count
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkDataset":
        kwargs = _strict_kwargs(cls, data)
        manifest_artifact = kwargs.get("manifest_artifact")
        if isinstance(manifest_artifact, dict):
            kwargs["manifest_artifact"] = ArtifactRef.from_dict(manifest_artifact)
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        kwargs["splits"] = tuple(kwargs.get("splits", ()))
        kwargs["tags"] = tuple(kwargs.get("tags", ()))
        return cls(**kwargs)


@dataclass(frozen=True)
class DocumentRecord:
    """A source document (logical grouping of pages)."""

    document_id: str
    dataset_id: str = ""
    dataset_version: str = "1"
    title: str | None = None
    document_type: str | None = None
    language: str = "ar"
    source_artifact: ArtifactRef | None = None
    provenance: Provenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be blank.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "language": self.language,
            "metadata": dict(self.metadata),
        }
        for key in ("title", "document_type"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.source_artifact is not None:
            payload["source_artifact"] = self.source_artifact.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        kwargs = _strict_kwargs(cls, data)
        source_artifact = kwargs.get("source_artifact")
        if isinstance(source_artifact, dict):
            kwargs["source_artifact"] = ArtifactRef.from_dict(source_artifact)
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        return cls(**kwargs)


@dataclass(frozen=True)
class PageRecord:
    """One benchmark page: identity, image artifact, GT reference, provenance."""

    page_id: str
    document_id: str
    dataset_id: str
    dataset_version: str = "1"
    split: str = "unassigned"
    page_number: int | None = None
    language: str = "ar"
    image_artifact: ArtifactRef | None = None
    ground_truth_uri: str | None = None
    ground_truth_sha256: str | None = None
    profile: str | None = None
    distortions: tuple[dict[str, Any], ...] = ()
    distortion_seed: int | None = None
    source_artifact: ArtifactRef | None = None
    source_identity: dict[str, Any] = field(default_factory=dict)
    protection: ProtectionInfo = field(default_factory=ProtectionInfo)
    provenance: Provenance | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.page_id.strip():
            raise ValueError("page_id cannot be blank.")
        if not self.document_id.strip():
            raise ValueError("PageRecord requires document_id.")
        if not self.dataset_id.strip():
            raise ValueError("PageRecord requires dataset_id.")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("Page numbers are one-based.")
        if self.ground_truth_sha256 is not None:
            object.__setattr__(
                self, "ground_truth_sha256", validate_sha256(self.ground_truth_sha256)
            )
        object.__setattr__(
            self, "distortions", tuple(dict(item) for item in self.distortions)
        )
        # Accept serialized forms (dicts) for nested records.
        if isinstance(self.protection, dict):
            object.__setattr__(
                self, "protection", ProtectionInfo.from_dict(self.protection)
            )
        if isinstance(self.provenance, dict):
            object.__setattr__(
                self, "provenance", Provenance.from_dict(self.provenance)
            )
        for key in ("image_artifact", "source_artifact"):
            value = getattr(self, key)
            if isinstance(value, dict):
                object.__setattr__(self, key, ArtifactRef.from_dict(value))
        # Fail-closed: a protected split marker must mark the page protected
        # even when the caller forgot to pass an explicit ProtectionInfo.
        split = self.split.strip().lower()
        if split in PROTECTED_SPLIT_MARKERS and not self.protection.protected:
            object.__setattr__(
                self,
                "protection",
                ProtectionInfo(
                    protected=True,
                    reasons=(f"protected_split:{split}",),
                    split=self.split,
                ),
            )

    @property
    def is_training_eligible(self) -> bool:
        return self.protection.is_training_eligible

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "page_id": self.page_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "language": self.language,
            "protection": self.protection.to_dict(),
            "source_identity": dict(self.source_identity),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        if self.page_number is not None:
            payload["page_number"] = self.page_number
        if self.image_artifact is not None:
            payload["image_artifact"] = self.image_artifact.to_dict()
        if self.ground_truth_uri is not None:
            payload["ground_truth_uri"] = self.ground_truth_uri
        if self.ground_truth_sha256 is not None:
            payload["ground_truth_sha256"] = self.ground_truth_sha256
        if self.profile is not None:
            payload["profile"] = self.profile
        if self.distortions:
            payload["distortions"] = [dict(step) for step in self.distortions]
        if self.distortion_seed is not None:
            payload["distortion_seed"] = self.distortion_seed
        if self.source_artifact is not None:
            payload["source_artifact"] = self.source_artifact.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageRecord":
        kwargs = _strict_kwargs(cls, data)
        for key in ("image_artifact", "source_artifact"):
            value = kwargs.get(key)
            if isinstance(value, dict):
                kwargs[key] = ArtifactRef.from_dict(value)
        protection = kwargs.get("protection")
        if isinstance(protection, dict):
            kwargs["protection"] = ProtectionInfo.from_dict(protection)
        elif not isinstance(protection, ProtectionInfo):
            kwargs["protection"] = ProtectionInfo()
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        kwargs["distortions"] = tuple(kwargs.get("distortions", ()))
        kwargs["tags"] = tuple(kwargs.get("tags", ()))
        return cls(**kwargs)


@dataclass(frozen=True)
class GroundTruthRecord:
    """Canonical ground truth for one page.

    ``raw_text`` is preserved exactly (Arabic Unicode intact, no
    normalization). Normalized comparison views are derived on explicit
    request only.
    """

    page_id: str
    raw_text: str
    raw_text_sha256: str
    dataset_id: str = ""
    split: str = "unassigned"
    source_artifact: ArtifactRef | None = None
    provenance: Provenance | None = None
    protection: ProtectionInfo = field(default_factory=ProtectionInfo)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.page_id.strip():
            raise ValueError("page_id cannot be blank.")
        object.__setattr__(
            self, "raw_text_sha256", validate_sha256(self.raw_text_sha256)
        )
        # Fail-closed: protected split marker marks the record protected.
        split = self.split.strip().lower()
        if split in PROTECTED_SPLIT_MARKERS and not self.protection.protected:
            object.__setattr__(
                self,
                "protection",
                ProtectionInfo(
                    protected=True,
                    reasons=(f"protected_split:{split}",),
                    split=self.split,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "page_id": self.page_id,
            "raw_text": self.raw_text,
            "raw_text_sha256": self.raw_text_sha256,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "protection": self.protection.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.source_artifact is not None:
            payload["source_artifact"] = self.source_artifact.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroundTruthRecord":
        kwargs = _strict_kwargs(cls, data)
        source_artifact = kwargs.get("source_artifact")
        if isinstance(source_artifact, dict):
            kwargs["source_artifact"] = ArtifactRef.from_dict(source_artifact)
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        protection = kwargs.get("protection")
        if isinstance(protection, dict):
            kwargs["protection"] = ProtectionInfo.from_dict(protection)
        elif not isinstance(protection, ProtectionInfo):
            kwargs["protection"] = ProtectionInfo(
                protected=bool(kwargs.get("protected", False)),
                split=str(kwargs.get("split", "unassigned")),
            )
        return cls(**kwargs)


@dataclass(frozen=True)
class OCRPrediction:
    """One model's OCR output for one page, inside one inference run.

    Multiple predictions per page are expected (different models/runs);
    identity is (run_id, page_id). Identical re-ingestion is idempotent;
    conflicting content for the same identity is rejected upstream.
    """

    prediction_id: str
    run_id: str
    page_id: str
    model_id: str
    model_revision: str
    text: str
    text_sha256: str
    dataset_id: str = ""
    split: str = "unassigned"
    inference_settings: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    output_artifact: ArtifactRef | None = None
    created_at: str = field(default_factory=utc_now)
    provenance: Provenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("prediction_id cannot be blank.")
        for name in ("run_id", "page_id", "model_id", "model_revision"):
            if not getattr(self, name).strip():
                raise ValueError(f"OCRPrediction requires non-blank {name}.")
        object.__setattr__(self, "text_sha256", validate_sha256(self.text_sha256))
        for key, value in self.metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Prediction metric {key!r} must be numeric.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "prediction_id": self.prediction_id,
            "run_id": self.run_id,
            "page_id": self.page_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "inference_settings": dict(self.inference_settings),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        if self.output_artifact is not None:
            payload["output_artifact"] = self.output_artifact.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRPrediction":
        kwargs = _strict_kwargs(cls, data)
        output_artifact = kwargs.get("output_artifact")
        if isinstance(output_artifact, dict):
            kwargs["output_artifact"] = ArtifactRef.from_dict(output_artifact)
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        return cls(**kwargs)


@dataclass(frozen=True)
class TrainingLineage:
    """Optional link back to the Training Experiment Framework."""

    experiment_name: str
    training_run_id: str
    checkpoint_name: str | None = None
    config_hash: str | None = None
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name.strip() or not self.training_run_id.strip():
            raise ValueError("Training lineage requires experiment and run id.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "training_run_id": self.training_run_id,
        }
        for key in ("checkpoint_name", "config_hash", "manifest_sha256"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingLineage":
        return cls(
            experiment_name=str(value["experiment_name"]),
            training_run_id=str(value["training_run_id"]),
            checkpoint_name=value.get("checkpoint_name"),
            config_hash=value.get("config_hash"),
            manifest_sha256=value.get("manifest_sha256"),
        )


@dataclass(frozen=True)
class ModelRecord:
    """Registry entry for a model identity (metadata only — no weights)."""

    model_id: str
    display_name: str = ""
    provider: str | None = None
    revision: str = "unresolved"
    model_family: str = "generic"
    adapter_name: str | None = None
    license: str | None = None
    training_lineage: TrainingLineage | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=utc_now)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be blank.")
        if not self.revision.strip():
            raise ValueError("Model revision cannot be blank.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "revision": self.revision,
            "model_family": self.model_family,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "registered_at": self.registered_at,
        }
        for key in ("provider", "adapter_name", "license"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.training_lineage is not None:
            payload["training_lineage"] = self.training_lineage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        kwargs = _strict_kwargs(cls, data)
        lineage = kwargs.get("training_lineage")
        if isinstance(lineage, dict):
            kwargs["training_lineage"] = TrainingLineage.from_dict(lineage)
        kwargs["tags"] = tuple(kwargs.get("tags", ()))
        return cls(**kwargs)


@dataclass(frozen=True)
class InferenceRun:
    """One OCR inference/evaluation run over one dataset."""

    run_id: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_version: str = "1"
    split: str = "unassigned"
    status: InferenceRunStatus = InferenceRunStatus.CREATED
    manifest_sha256: str | None = None
    config_hash: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    page_count: int | None = None
    bundle_uri: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    training_lineage: TrainingLineage | None = None
    provenance: Provenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "model_id", "model_revision", "dataset_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"InferenceRun requires non-blank {name}.")
        if self.manifest_sha256 is not None:
            object.__setattr__(
                self, "manifest_sha256", validate_sha256(self.manifest_sha256)
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "status": self.status.value,
            "environment": dict(self.environment),
            "metadata": dict(self.metadata),
        }
        for key in (
            "manifest_sha256",
            "config_hash",
            "started_at",
            "ended_at",
            "bundle_uri",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.page_count is not None:
            payload["page_count"] = self.page_count
        if self.training_lineage is not None:
            payload["training_lineage"] = self.training_lineage.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRun":
        kwargs = _strict_kwargs(cls, data)
        kwargs["status"] = InferenceRunStatus(
            str(kwargs.get("status", InferenceRunStatus.CREATED.value))
        )
        lineage = kwargs.get("training_lineage")
        if isinstance(lineage, dict):
            kwargs["training_lineage"] = TrainingLineage.from_dict(lineage)
        provenance = kwargs.get("provenance")
        if isinstance(provenance, dict):
            kwargs["provenance"] = Provenance.from_dict(provenance)
        return cls(**kwargs)


@dataclass(frozen=True)
class EvaluationRecord:
    """One stored metric value at page or run/dataset-summary scope."""

    record_id: str
    run_id: str
    metric_name: str
    value: float
    scope: EvaluationScope = EvaluationScope.PAGE
    page_id: str | None = None
    dataset_id: str = ""
    split: str = "unassigned"
    normalization_policy: str | None = None
    evaluator_version: str | None = None
    computed_at: str = field(default_factory=utc_now)
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("record_id", "run_id", "metric_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"EvaluationRecord requires non-blank {name}.")
        if self.scope is EvaluationScope.PAGE and not (self.page_id or "").strip():
            raise ValueError("Page-scope metrics require page_id.")
        if not isinstance(self.value, (int, float)):
            raise ValueError("Metric value must be numeric.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "scope": self.scope.value,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "computed_at": self.computed_at,
            "details": dict(self.details),
        }
        if self.page_id is not None:
            payload["page_id"] = self.page_id
        if self.normalization_policy is not None:
            payload["normalization_policy"] = self.normalization_policy
        if self.evaluator_version is not None:
            payload["evaluator_version"] = self.evaluator_version
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRecord":
        kwargs = _strict_kwargs(cls, data)
        kwargs["scope"] = EvaluationScope(
            str(kwargs.get("scope", EvaluationScope.PAGE.value))
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class ResultBundle:
    """In-memory view of a canonical result bundle directory."""

    run: InferenceRun
    pages: tuple[PageRecord, ...] = ()
    predictions: tuple[OCRPrediction, ...] = ()
    metrics: tuple[EvaluationRecord, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    schema_version: str = RESULTS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "summary": dict(self.summary),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
