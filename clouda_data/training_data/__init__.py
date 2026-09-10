"""Scalable training data loader + sharding engine.

This subsystem consumes the canonical Clouda pre-training manifest
(``clouda.pretraining.manifest.v1``) and streams already-selected training
samples to future trainer adapters. It never decides what should be trained
on: selection, deduplication, splitting and holdout protection happen
upstream (``clouda_data.pretraining``, ``clouda_training.experiments``).

The loader is the last data-policy boundary before training: it re-validates
the canonical manifest, rejects protected holdout data fail-closed, and only
then shards/streams/batches samples deterministically.
"""

from __future__ import annotations

from .models import (
    BadSamplePolicy,
    BatchConfig,
    LoaderState,
    LoaderStats,
    PrefetchConfig,
    RemainderPolicy,
    ResumeCursor,
    SampleReference,
    ShardConfig,
    ShardStrategy,
    ShuffleConfig,
    TraceMode,
    TrainingDataConfig,
    ValidationMode,
    WorkerConfig,
)

__all__ = [
    "BadSamplePolicy",
    "BatchConfig",
    "LoaderState",
    "LoaderStats",
    "PrefetchConfig",
    "RemainderPolicy",
    "ResumeCursor",
    "SampleReference",
    "ShardConfig",
    "ShardStrategy",
    "ShuffleConfig",
    "TraceMode",
    "TrainingDataConfig",
    "ValidationMode",
    "WorkerConfig",
]
