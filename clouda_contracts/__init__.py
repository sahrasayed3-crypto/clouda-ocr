"""Dependency-light contracts shared by Clouda subsystems."""

from .dataset_identity import DatasetIdentity
from .identity import DocumentIdentity, ModelIdentity, ModelVersion
from .model_observation import ModelObservation
from .observations import EvaluationObservation, LayoutObservation
from .ocr_observation import OCRObservation
from .page_identity import PageIdentity
from .references import DatasetPageReference, RuntimeJobReference
from .storage import StorageRoots, StorageSecurityError
from .storage_uri import StorageURI

__all__ = [
    "DatasetIdentity",
    "DatasetPageReference",
    "DocumentIdentity",
    "EvaluationObservation",
    "LayoutObservation",
    "ModelIdentity",
    "ModelObservation",
    "ModelVersion",
    "OCRObservation",
    "PageIdentity",
    "RuntimeJobReference",
    "StorageRoots",
    "StorageSecurityError",
    "StorageURI",
]
