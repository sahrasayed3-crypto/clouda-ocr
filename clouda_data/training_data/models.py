"""Typed domain models for the training data loader subsystem.

All configuration dataclasses are frozen and hashable-friendly; identity
hashes are derived through ``config_identity`` using stable JSON
canonicalization (never ``hash()``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ShardStrategy(str, Enum):
    COUNT = "count"
    SIZE_AWARE = "size_aware"


class ShuffleMode(str, Enum):
    NONE = "none"
    SHARD_ORDER = "shard_order"
    BUFFERED = "buffered"


class ValidationMode(str, Enum):
    NONE = "none"
    LIGHT = "light"
    STRICT = "strict"


class BadSamplePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    SKIP_AND_RECORD = "skip_and_record"


class TraceMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


class RemainderPolicy(str, Enum):
    DROP = "drop"
    KEEP = "keep"


@dataclass(frozen=True)
class ShardConfig:
    strategy: ShardStrategy = ShardStrategy.COUNT
    samples_per_shard: int = 1000
    max_shard_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.samples_per_shard < 1:
            raise ValueError("samples_per_shard must be >= 1")
        if self.max_shard_bytes < 1:
            raise ValueError("max_shard_bytes must be >= 1")


@dataclass(frozen=True)
class ShuffleConfig:
    mode: ShuffleMode = ShuffleMode.BUFFERED
    buffer_size: int = 1000

    def __post_init__(self) -> None:
        if self.mode is ShuffleMode.BUFFERED and self.buffer_size < 1:
            raise ValueError("buffer_size must be >= 1 for buffered shuffle")


@dataclass(frozen=True)
class PrefetchConfig:
    enabled: bool = False
    depth: int = 2

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("prefetch depth must be >= 1")


@dataclass(frozen=True)
class WorkerConfig:
    world_size: int = 1
    rank: int = 0
    num_workers: int = 1
    worker_id: int = 0

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if not 0 <= self.worker_id < self.num_workers:
            raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int = 8
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass(frozen=True)
class TrainingDataConfig:
    """Top-level loader configuration."""

    dataset_id: str
    dataset_version: str
    split: str = "train"
    global_seed: int = 20260723
    shard: ShardConfig = field(default_factory=ShardConfig)
    shuffle: ShuffleConfig = field(default_factory=ShuffleConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    validation_mode: ValidationMode = ValidationMode.LIGHT
    bad_sample_policy: BadSamplePolicy = BadSamplePolicy.SKIP_AND_RECORD
    trace_mode: TraceMode = TraceMode.SUMMARY

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be blank")
        if not self.dataset_version.strip():
            raise ValueError("dataset_version cannot be blank")


@dataclass(frozen=True)
class SampleReference:
    """Lightweight pointer to one training sample.

    Carries metadata only: artifact bytes are loaded lazily by the artifact
    resolution layer, never eagerly during iteration.
    """

    sample_id: str
    shard_id: str
    position: int
    image_path: str | None = None
    text: str | None = None
    row: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "shard_id": self.shard_id,
            "position": self.position,
            "image_path": self.image_path,
            "text": self.text,
        }


@dataclass(frozen=True)
class ShardRecord:
    """One row of a shard manifest (metadata only, no artifact bytes)."""

    sample_id: str
    shard_id: str
    position: int
    image_path: str | None
    text_sha256: str | None
    approx_bytes: int
    row: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "shard_id": self.shard_id,
            "position": self.position,
            "image_path": self.image_path,
            "text_sha256": self.text_sha256,
            "approx_bytes": self.approx_bytes,
            "row": self.row,
        }


@dataclass(frozen=True)
class ResumeCursor:
    """Persistable iteration position for deterministic resume."""

    schema_version: str
    dataset_id: str
    dataset_version: str
    manifest_sha256: str
    loader_config_hash: str
    global_seed: int
    epoch: int
    world_size: int
    rank: int
    num_workers: int
    worker_id: int
    shard_position: int
    sample_position: int
    yielded_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResumeCursor":
        required = {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "manifest_sha256",
            "loader_config_hash",
            "global_seed",
            "epoch",
            "world_size",
            "rank",
            "num_workers",
            "worker_id",
            "shard_position",
            "sample_position",
            "yielded_count",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Resume cursor missing fields: {sorted(missing)}")
        return cls(
            schema_version=str(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_version=str(payload["dataset_version"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            loader_config_hash=str(payload["loader_config_hash"]),
            global_seed=int(payload["global_seed"]),
            epoch=int(payload["epoch"]),
            world_size=int(payload["world_size"]),
            rank=int(payload["rank"]),
            num_workers=int(payload["num_workers"]),
            worker_id=int(payload["worker_id"]),
            shard_position=int(payload["shard_position"]),
            sample_position=int(payload["sample_position"]),
            yielded_count=int(payload["yielded_count"]),
        )


@dataclass
class LoaderStats:
    """Runtime loader statistics (mutable counters)."""

    samples_read: int = 0
    samples_yielded: int = 0
    samples_skipped: int = 0
    batches_produced: int = 0
    bytes_read: int = 0
    sample_load_time_total: float = 0.0
    batch_prep_time_total: float = 0.0
    prefetch_wait_time_total: float = 0.0
    queue_starvations: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def throughput(self) -> float:
        if self.sample_load_time_total <= 0:
            return 0.0
        return self.samples_yielded / self.sample_load_time_total


@dataclass(frozen=True)
class LoaderState:
    """Immutable snapshot combining cursor and counters for checkpointing."""

    cursor: ResumeCursor
    stats_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor.to_dict(),
            "stats": dict(self.stats_snapshot),
        }


def config_identity(*configs: Any) -> str:
    """Stable SHA-256 identity over one or more config payloads."""

    canonical = json.dumps(
        [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in configs],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
