from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PDF_CONVERSION_QUEUE = "pdf_conversion"
DATASET_PIPELINE_QUEUE = "dataset_pipeline"
TRAINING_PREPARATION_QUEUE = "training_preparation"
MODEL_EVALUATION_QUEUE = "model_evaluation"


@dataclass(frozen=True)
class QueuePolicy:
    name: str
    timeout_seconds: int
    retry_count: int
    storage_capability: str

    def __post_init__(self) -> None:
        if self.timeout_seconds < 60 or self.retry_count < 0:
            raise ValueError("Invalid queue timeout or retry policy.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "storage_capability": self.storage_capability,
        }


QUEUE_POLICIES = {
    PDF_CONVERSION_QUEUE: QueuePolicy(
        PDF_CONVERSION_QUEUE,
        timeout_seconds=7200,
        retry_count=2,
        storage_capability="runtime",
    ),
    DATASET_PIPELINE_QUEUE: QueuePolicy(
        DATASET_PIPELINE_QUEUE,
        timeout_seconds=14400,
        retry_count=1,
        storage_capability="dataset",
    ),
    TRAINING_PREPARATION_QUEUE: QueuePolicy(
        TRAINING_PREPARATION_QUEUE,
        timeout_seconds=21600,
        retry_count=0,
        storage_capability="dataset_and_artifact",
    ),
    MODEL_EVALUATION_QUEUE: QueuePolicy(
        MODEL_EVALUATION_QUEUE,
        timeout_seconds=10800,
        retry_count=1,
        storage_capability="model_and_artifact",
    ),
}


@dataclass(frozen=True)
class WorkerCapability:
    role: str
    queues: tuple[str, ...]
    allowed_storage_roots: tuple[str, ...]
    may_train: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "queues": list(self.queues),
            "allowed_storage_roots": list(self.allowed_storage_roots),
            "may_train": self.may_train,
        }


WORKER_CAPABILITIES = {
    "runtime": WorkerCapability(
        "runtime",
        queues=(PDF_CONVERSION_QUEUE,),
        allowed_storage_roots=("runtime", "cache"),
        may_train=False,
    ),
    "dataset": WorkerCapability(
        "dataset",
        queues=(DATASET_PIPELINE_QUEUE,),
        allowed_storage_roots=("dataset", "artifact", "cache"),
        may_train=False,
    ),
    "training_preparation": WorkerCapability(
        "training_preparation",
        queues=(TRAINING_PREPARATION_QUEUE,),
        allowed_storage_roots=("dataset", "artifact", "model", "cache"),
        may_train=False,
    ),
    "model_evaluation": WorkerCapability(
        "model_evaluation",
        queues=(MODEL_EVALUATION_QUEUE,),
        allowed_storage_roots=("dataset", "artifact", "model", "cache"),
        may_train=False,
    ),
}


def capability_report(role: str) -> dict[str, Any]:
    try:
        capability = WORKER_CAPABILITIES[role]
    except KeyError as exc:
        raise ValueError(f"Unknown worker role: {role}") from exc
    return {
        **capability.to_dict(),
        "policies": [QUEUE_POLICIES[name].to_dict() for name in capability.queues],
    }
