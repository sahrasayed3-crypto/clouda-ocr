"""Streaming, resumable training data loader over canonical shards.

Design:
- Metadata-only iteration: shards are read line-by-line; image bytes are
  loaded lazily by the artifact layer, never during iteration.
- Deterministic shuffle: shard visit order is permuted per epoch; within a
  shard, a bounded shuffle buffer (configurable) reorders samples. Same
  (seed, epoch, topology, config) always reproduces the same order.
- Worker/rank partitioning: each (rank, worker_id) gets a deterministic,
  disjoint subset of samples via stable modular assignment.
- Resume: a :class:`ResumeCursor` records (epoch, shard position, sample
  position, yielded count). Resume rejects incompatible state fail-closed.
  Resume within a shard works because the epoch's shuffle order is a pure
  function of (seed, epoch, topology, config) — replaying deterministically
  reproduces the exact ordering, so skipping by position is sound.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from clouda_data.training_data.input_contract import (
    ManifestIdentity,
    validate_canonical_manifest,
)
from clouda_data.training_data.models import (
    BadSamplePolicy,
    LoaderState,
    LoaderStats,
    ResumeCursor,
    SampleReference,
    TraceMode,
    TrainingDataConfig,
    ValidationMode,
    config_identity,
)
from clouda_data.training_data.ordering import (
    bounded_shuffle_stream,
    derive_loader_seed,
    permute_indices,
    shard_order,
)
from clouda_data.training_data.sharding import (
    ShardIndex,
    load_shard_index,
)

LOADER_STATE_SCHEMA_VERSION = "clouda.training_data.loader_state.v1"


class ResumeError(RuntimeError):
    """Raised when resume state is incompatible with the current loader."""


def loader_config_hash(config: TrainingDataConfig) -> str:
    return config_identity(config)


class StreamingTrainingDataLoader:
    """Lazy, deterministic, resumable sample/batch stream over shards."""

    def __init__(
        self,
        *,
        shard_index_path: str | Path,
        loader_config: TrainingDataConfig,
        manifest_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.index_path = Path(shard_index_path)
        self.index: ShardIndex = load_shard_index(self.index_path)
        self.config = loader_config
        self.root = Path(dataset_root) if dataset_root else self.index_path.parent
        self.config_hash = loader_config_hash(loader_config)
        self.stats = LoaderStats()
        self._epoch = 0
        self._shard_position = 0
        self._sample_position = 0
        self._yielded = 0
        self._delivered: list[str] = []
        self._identity: ManifestIdentity | None = None
        self._manifest_path = (
            Path(manifest_path) if manifest_path else self._default_manifest_path()
        )
        if (
            loader_config.dataset_id != self.index.dataset_id
            or loader_config.dataset_version != self.index.dataset_version
        ):
            raise ResumeError(
                "Loader config dataset identity does not match shard index: "
                f"config={loader_config.dataset_id}/{loader_config.dataset_version} "
                f"index={self.index.dataset_id}/{self.index.dataset_version}"
            )

    def _default_manifest_path(self) -> Path:
        return self.index_path.parent / "manifest.jsonl"

    # ------------------------------------------------------------------
    # Identity / validation
    # ------------------------------------------------------------------

    def open(self) -> ManifestIdentity:
        """Validate the canonical manifest once, fail-closed, then cache."""

        if self._identity is None:
            self._identity = validate_canonical_manifest(
                self._manifest_path,
                dataset_id=self.config.dataset_id,
                dataset_version=self.config.dataset_version,
                split=self.config.split,
            )
            expected = self.index.source_manifest_sha256
            if self._identity.manifest_sha256 != expected:
                raise ResumeError(
                    "Manifest hash changed since sharding: "
                    f"manifest={self._identity.manifest_sha256} "
                    f"index={expected}"
                )
        return self._identity

    # ------------------------------------------------------------------
    # Sample iteration
    # ------------------------------------------------------------------

    def iter_samples(self, *, epoch: int | None = None) -> Iterator[SampleReference]:
        """Iterate samples for the current (or given) epoch deterministically.

        Completing a full pass advances the epoch. If a previous partial pass
        was resumed via :meth:`restore`, iteration continues exactly where it
        stopped — no duplicates, no omissions.
        """

        if epoch is not None and epoch != self._epoch:
            self._epoch = epoch
            self._shard_position = 0
            self._sample_position = 0
        self.open()
        ordered = self._epoch_shard_order(self._epoch)
        for shard_idx, shard_id in enumerate(ordered):
            if shard_idx < self._shard_position:
                continue
            skipping = shard_idx == self._shard_position and self._sample_position > 0
            seen_in_shard = 0
            for sample in self._iter_shard_samples(shard_id, self._epoch):
                seen_in_shard += 1
                if skipping and seen_in_shard <= self._sample_position:
                    continue
                if not self._owned_by_topology(sample.sample_id):
                    continue
                self._sample_position = seen_in_shard
                if self.config.trace_mode is TraceMode.FULL:
                    self._delivered.append(sample.sample_id)
                # Count before yielding: once the consumer receives the
                # sample, it is delivered — even if the generator is
                # abandoned before resuming, so the cursor never undercounts.
                self._yielded += 1
                self.stats.samples_yielded += 1
                yield sample
            self._shard_position = shard_idx + 1
            self._sample_position = 0
        # Full epoch completed: advance deterministically.
        self._epoch += 1
        self._shard_position = 0
        self._sample_position = 0

    def _epoch_shard_order(self, epoch: int) -> list[str]:
        shard_ids = [entry.shard_id for entry in self.index.shards]
        seed = derive_loader_seed(
            self.config.global_seed,
            epoch=epoch,
            world_size=self.config.workers.world_size,
            rank=self.config.workers.rank,
            num_workers=self.config.workers.num_workers,
            worker_id=self.config.workers.worker_id,
            config_hash=self.config_hash,
        )
        return shard_order(shard_ids, seed)

    def iter_sample_ids(self, *, epoch: int | None = None) -> Iterator[str]:
        """Convenience view: sample ids only, in delivery order."""

        for sample in self.iter_samples(epoch=epoch):
            yield sample.sample_id

    def _iter_shard_samples(
        self, shard_id: str, epoch: int
    ) -> Iterator[SampleReference]:
        entry = next(e for e in self.index.shards if e.shard_id == shard_id)
        path = self.index_path.parent / "shards" / entry.path
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        seed = derive_loader_seed(
            self.config.global_seed,
            epoch=epoch,
            world_size=self.config.workers.world_size,
            rank=self.config.workers.rank,
            num_workers=self.config.workers.num_workers,
            worker_id=self.config.workers.worker_id,
            config_hash=self.config_hash,
        )
        shuffle = self.config.shuffle
        if shuffle.mode.value == "buffered":
            stream = ((i, lines[i]) for i in range(len(lines)))
            pairs = list(bounded_shuffle_stream(stream, shuffle.buffer_size, seed))
        elif shuffle.mode.value == "shard_order":
            order = permute_indices(len(lines), seed)
            pairs = [(i, lines[i]) for i in order]
        else:
            pairs = list(enumerate(lines))

        for position, line in pairs:
            if not line.strip():
                continue
            record = json.loads(line)
            row = record.get("row", {})
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                continue
            ref = SampleReference(
                sample_id=sample_id,
                shard_id=shard_id,
                position=position,
                image_path=row.get("image_path"),
                text=row.get("text"),
                row=row,
            )
            self.stats.samples_read += 1
            if not self._validate_sample(ref):
                self.stats.samples_skipped += 1
                continue
            yield ref

    def _validate_sample(self, ref: SampleReference) -> bool:
        mode = self.config.validation_mode
        if mode is ValidationMode.NONE:
            return True
        issues: list[str] = []
        if ref.image_path:
            resolved = self._resolve_artifact(ref.image_path)
            if resolved is None:
                issues.append("missing_image")
            elif mode is ValidationMode.STRICT:
                from clouda_data.training_data.artifacts import verify_image_decodable

                if not verify_image_decodable(resolved):
                    issues.append("corrupt_image")
        elif mode is ValidationMode.STRICT and not ref.text:
            issues.append("missing_payload")
        if mode is ValidationMode.STRICT and ref.text:
            text_hash = ref.row.get("normalized_text_sha256")
            if text_hash:
                from clouda_data.training_data.artifacts import verify_text_hash

                if not verify_text_hash(ref.text, str(text_hash)):
                    issues.append("text_hash_mismatch")
        if not issues:
            return True
        self._record_rejection(ref, issues)
        if self.config.bad_sample_policy is BadSamplePolicy.FAIL_FAST:
            raise ValueError(f"Bad sample {ref.sample_id}: {issues}")
        return False

    def _record_rejection(self, ref: SampleReference, issues: list[str]) -> None:
        record = {
            "sample_id": ref.sample_id,
            "shard_id": ref.shard_id,
            "position": ref.position,
            "issues": issues,
            "epoch": self._epoch,
        }
        self.stats.rejected.append(record)

    def _resolve_artifact(self, rel_path: str) -> Path | None:
        from clouda_data.training_data.artifacts import resolve_artifact

        try:
            return resolve_artifact(self.root, rel_path)
        except (ValueError, OSError):
            return None

    def _owned_by_topology(self, sample_id: str) -> bool:
        worker = self.config.workers
        total = worker.world_size * worker.num_workers
        if total == 1:
            return True
        from clouda_data.training_data.ordering import stable_hash

        slot = stable_hash(sample_id) % total
        flat = worker.rank * worker.num_workers + worker.worker_id
        return slot == flat

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def iter_batches(self, *, epoch: int | None = None) -> Iterator[dict[str, Any]]:
        batch_size = self.config.batch.batch_size
        drop_last = self.config.batch.drop_last
        buffer: list[SampleReference] = []
        for sample in self.iter_samples(epoch=epoch):
            started = time.monotonic()
            buffer.append(sample)
            if len(buffer) >= batch_size:
                self.stats.batches_produced += 1
                yield self._make_batch(buffer)
                self.stats.batch_prep_time_total += time.monotonic() - started
                buffer = []
        if buffer and not drop_last:
            self.stats.batches_produced += 1
            yield self._make_batch(buffer)
            self.stats.batch_prep_time_total += time.monotonic() - started

    def _make_batch(self, batch: list[SampleReference]) -> dict[str, Any]:
        return {
            "sample_ids": [s.sample_id for s in batch],
            "samples": list(batch),
            "shard_ids": sorted({s.shard_id for s in batch}),
            "epoch": self._epoch,
            "size": len(batch),
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def get_cursor(self) -> ResumeCursor:
        worker = self.config.workers
        return ResumeCursor(
            schema_version=LOADER_STATE_SCHEMA_VERSION,
            dataset_id=self.config.dataset_id,
            dataset_version=self.config.dataset_version,
            manifest_sha256=self._identity.manifest_sha256 if self._identity else "",
            loader_config_hash=self.config_hash,
            global_seed=self.config.global_seed,
            epoch=self._epoch,
            world_size=worker.world_size,
            rank=worker.rank,
            num_workers=worker.num_workers,
            worker_id=worker.worker_id,
            shard_position=self._shard_position,
            sample_position=self._sample_position,
            yielded_count=self._yielded,
        )

    def get_state(self) -> LoaderState:
        return LoaderState(
            cursor=self.get_cursor(),
            stats_snapshot=self.stats.to_dict(),
        )

    def restore(self, cursor: ResumeCursor) -> None:
        identity = self.open()
        if cursor.schema_version != LOADER_STATE_SCHEMA_VERSION:
            raise ResumeError("Unsupported resume cursor schema version")
        checks = (
            ("dataset", cursor.dataset_id, self.config.dataset_id),
            ("dataset version", cursor.dataset_version, self.config.dataset_version),
            ("manifest hash", cursor.manifest_sha256, identity.manifest_sha256),
            ("loader config", cursor.loader_config_hash, self.config_hash),
            ("global seed", str(cursor.global_seed), str(self.config.global_seed)),
            ("world_size", str(cursor.world_size), str(self.config.workers.world_size)),
            ("rank", str(cursor.rank), str(self.config.workers.rank)),
            (
                "num_workers",
                str(cursor.num_workers),
                str(self.config.workers.num_workers),
            ),
            ("worker_id", str(cursor.worker_id), str(self.config.workers.worker_id)),
        )
        for label, recorded, current in checks:
            if recorded != current:
                raise ResumeError(
                    f"Resume rejected: {label} changed "
                    f"({recorded!r} -> {current!r})"
                )
        self._epoch = cursor.epoch
        self._shard_position = cursor.shard_position
        self._sample_position = cursor.sample_position
        self._yielded = cursor.yielded_count

    # ------------------------------------------------------------------
    # Traceability
    # ------------------------------------------------------------------

    def trace_summary(self) -> dict[str, Any]:
        identity = self._identity
        return {
            "schema_version": "clouda.training_data.trace_summary.v1",
            "dataset_id": self.config.dataset_id,
            "dataset_version": self.config.dataset_version,
            "manifest_sha256": identity.manifest_sha256 if identity else None,
            "loader_config_hash": self.config_hash,
            "global_seed": self.config.global_seed,
            "epoch_completed_through": self._epoch,
            "topology": {
                "world_size": self.config.workers.world_size,
                "rank": self.config.workers.rank,
                "num_workers": self.config.workers.num_workers,
                "worker_id": self.config.workers.worker_id,
            },
            "stats": self.stats.to_dict(),
            "delivered_count": self._yielded,
        }

    def trace_full(self) -> dict[str, Any]:
        summary = self.trace_summary()
        summary["delivered_order"] = list(self._delivered)
        return summary


class EpochManager:
    """Explicit epoch bookkeeping for trainer adapters."""

    def __init__(self, loader: StreamingTrainingDataLoader) -> None:
        self.loader = loader
        self.current_epoch = 0

    def iter_epoch(self, epoch: int | None = None) -> Iterator[SampleReference]:
        if epoch is not None:
            self.current_epoch = epoch
        yield from self.loader.iter_samples(epoch=self.current_epoch)
        self.current_epoch += 1
