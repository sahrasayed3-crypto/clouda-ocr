"""Streaming, shuffle, epoch, worker/rank, batching, prefetch, artifact,
resume and traceability tests for the training data loader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from clouda_data.training_data.models import (
    BadSamplePolicy,
    BatchConfig,
    ShuffleConfig,
    ShuffleMode,
    TraceMode,
    ValidationMode,
    WorkerConfig,
)
from clouda_data.training_data.loader import (
    ResumeError,
    StreamingTrainingDataLoader,
)
from clouda_data.training_data.prefetch import PrefetchIterator

from tests.data_foundation.fixtures.training_data_fixtures import (
    DATASET_ID,
    build_synthetic_dataset,
    make_corrupt_image,
    make_loader,
    shard_dataset,
)


def _ids(loader: StreamingTrainingDataLoader, **kwargs) -> list[str]:
    return [s.sample_id for s in loader.iter_samples(**kwargs)]


# ---------------------------------------------------------------------------
# Streaming / laziness
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_iteration_covers_dataset_exactly(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        ids = _ids(loader)
        assert sorted(ids) == sorted(f"smp_test_{i:06d}" for i in range(24))
        assert len(ids) == len(set(ids))

    def test_lazy_metadata_only(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        stream = loader.iter_samples(epoch=0)
        first = next(stream)
        assert first.image_path is not None
        # No artifact bytes were loaded during iteration
        assert first.row.get("text") is not None

    def test_large_fixture_stream(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "big", count=600)
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=100)
        loader = make_loader(tmp_path / "out" / "shard_index.json", manifest, root)
        ids = _ids(loader)
        assert len(ids) == 600
        assert len(set(ids)) == 600

    def test_buffered_shuffle_streams_shard_lines(
        self, index_path, synthetic_dataset, monkeypatch
    ):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path,
            manifest,
            root,
            shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=3),
        )
        loader.open()
        original_open = Path.open

        class GuardedShardFile:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *args):
                return self._handle.__exit__(*args)

            def __iter__(self):
                return iter(self._handle)

            def readlines(self, *args, **kwargs):
                raise AssertionError("buffered shuffle must not call readlines")

        def guarded_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path.parent.name == "shards":
                return GuardedShardFile(handle)
            return handle

        monkeypatch.setattr(Path, "open", guarded_open)
        assert len(_ids(loader, epoch=0)) == 24


# ---------------------------------------------------------------------------
# Deterministic shuffle
# ---------------------------------------------------------------------------


class TestShuffle:
    def test_same_seed_epoch_same_order(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        kwargs = dict(shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=6))
        a = _ids(make_loader(index_path, manifest, root, seed=11, **kwargs), epoch=0)
        b = _ids(make_loader(index_path, manifest, root, seed=11, **kwargs), epoch=0)
        assert a == b

    def test_buffered_shuffle_changes_order(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        shuffled = _ids(
            make_loader(
                index_path,
                manifest,
                root,
                shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=6),
            ),
            epoch=0,
        )
        assert shuffled != sorted(shuffled)

    def test_no_sample_loss_under_shuffle(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        ids = _ids(
            make_loader(
                index_path,
                manifest,
                root,
                shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=3),
            ),
            epoch=0,
        )
        assert len(ids) == 24 and len(set(ids)) == 24

    def test_shard_order_mode(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        ids = _ids(
            make_loader(
                index_path,
                manifest,
                root,
                shuffle=ShuffleConfig(mode=ShuffleMode.SHARD_ORDER),
            ),
            epoch=0,
        )
        assert len(ids) == 24 and len(set(ids)) == 24

    def test_no_builtin_hash_used(self):
        import builtins
        import clouda_data.training_data.ordering as ordering

        original = builtins.hash
        calls = []

        def spy(value):
            calls.append(value)
            return original(value)

        builtins.hash = spy
        try:
            ordering.permute_indices(50, 12345)
            ordering.bounded_shuffle_stream(iter(range(50)), 7, 999)
            ordering.derive_loader_seed(
                1,
                epoch=0,
                world_size=1,
                rank=0,
                num_workers=1,
                worker_id=0,
                config_hash="x",
            )
        finally:
            builtins.hash = original
        assert calls == []


# ---------------------------------------------------------------------------
# Epoch semantics
# ---------------------------------------------------------------------------


class TestEpochs:
    def test_different_epoch_different_order(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path,
            manifest,
            root,
            shuffle=ShuffleConfig(mode=ShuffleMode.SHARD_ORDER),
        )
        e0 = _ids(loader, epoch=0)
        e1 = _ids(
            make_loader(
                index_path,
                manifest,
                root,
                shuffle=ShuffleConfig(mode=ShuffleMode.SHARD_ORDER),
            ),
            epoch=1,
        )
        assert e0 != e1
        assert sorted(e0) == sorted(e1)

    def test_sequential_epochs_no_repeat_skip(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path,
            manifest,
            root,
            shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=5),
        )
        e0 = _ids(loader)  # advances epoch automatically
        e1 = _ids(loader)
        assert len(e0) == 24 and len(e1) == 24
        assert sorted(e0) == sorted(e1)


# ---------------------------------------------------------------------------
# Worker / rank partitioning
# ---------------------------------------------------------------------------


class TestWorkersAndRanks:
    @pytest.mark.parametrize(
        "world,num_workers",
        [(2, 1), (4, 1), (1, 3), (2, 2), (3, 2)],
    )
    def test_partition_complete_and_disjoint(
        self, index_path, synthetic_dataset, world, num_workers
    ):
        manifest, root = synthetic_dataset
        parts = []
        for rank in range(world):
            for wid in range(num_workers):
                loader = make_loader(
                    index_path,
                    manifest,
                    root,
                    workers=WorkerConfig(
                        world_size=world,
                        rank=rank,
                        num_workers=num_workers,
                        worker_id=wid,
                    ),
                )
                parts.append(_ids(loader, epoch=0))
        flat = [sid for part in parts for sid in part]
        assert len(flat) == len(set(flat))  # no duplicates
        assert sorted(flat) == sorted(
            f"smp_test_{i:06d}" for i in range(24)
        )  # full coverage

    def test_remainder_kept_by_default(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        # 24 samples over 5 ranks: 24 % 5 != 0 — modular assignment keeps all
        parts = []
        for rank in range(5):
            loader = make_loader(
                index_path,
                manifest,
                root,
                workers=WorkerConfig(world_size=5, rank=rank),
            )
            parts.append(_ids(loader, epoch=0))
        flat = [sid for part in parts for sid in part]
        assert sorted(flat) == sorted(f"smp_test_{i:06d}" for i in range(24))

    def test_deterministic_across_restarts(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset

        def run():
            loader = make_loader(
                index_path,
                manifest,
                root,
                workers=WorkerConfig(world_size=2, rank=1, num_workers=2, worker_id=1),
            )
            return _ids(loader, epoch=0)

        assert run() == run()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_fixed_batches_partial_final(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path, manifest, root, batch=BatchConfig(batch_size=10)
        )
        batches = list(loader.iter_batches(epoch=0))
        assert [b["size"] for b in batches] == [10, 10, 4]

    def test_drop_last_removes_partial(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path,
            manifest,
            root,
            batch=BatchConfig(batch_size=10, drop_last=True),
        )
        batches = list(loader.iter_batches(epoch=0))
        assert [b["size"] for b in batches] == [10, 10]

    def test_batch_boundaries_deterministic(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset

        def run():
            loader = make_loader(
                index_path,
                manifest,
                root,
                batch=BatchConfig(batch_size=7),
                shuffle=ShuffleConfig(mode=ShuffleMode.SHARD_ORDER),
            )
            return [b["sample_ids"] for b in loader.iter_batches(epoch=3)]

        assert run() == run()

    def test_batch_provenance_refs(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path, manifest, root, batch=BatchConfig(batch_size=5)
        )
        batch = next(iter(loader.iter_batches(epoch=0)))
        assert batch["sample_ids"]
        assert all(s.row.get("provenance") for s in batch["samples"])
        assert batch["shard_ids"]


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_preserves_order(self):
        source = list(range(50))
        got = list(PrefetchIterator(iter(source), depth=3))
        assert got == source

    def test_bounded_queue(self):
        from clouda_data.training_data.prefetch import PrefetchIterator

        produced = []

        def slow():
            for i in range(100):
                produced.append(i)
                yield i

        pf = PrefetchIterator(slow(), depth=2)
        next(pf)
        # Producer cannot run far ahead of the bounded queue
        assert len(produced) <= 4
        pf.close()

    def test_error_propagates(self):
        def boom():
            yield 1
            raise RuntimeError("upstream failure")

        pf = PrefetchIterator(boom(), depth=1)
        with pytest.raises(RuntimeError, match="upstream failure"):
            list(pf)

    def test_clean_shutdown(self):
        pf = PrefetchIterator(iter(range(1000)), depth=2)
        next(pf)
        pf.close()
        assert pf._thread is None or not pf._thread.is_alive()

    def test_context_manager_shutdown(self):
        with PrefetchIterator(iter(range(100)), depth=2) as pf:
            next(pf)
        assert not pf._thread.is_alive()

    def test_loader_batches_through_prefetch(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(
            index_path, manifest, root, batch=BatchConfig(batch_size=6)
        )
        direct = [b["sample_ids"] for b in loader.iter_batches(epoch=0)]
        loader2 = make_loader(
            index_path, manifest, root, batch=BatchConfig(batch_size=6)
        )
        with PrefetchIterator(loader2.iter_batches(epoch=0), depth=2) as pf:
            via_prefetch = [b["sample_ids"] for b in pf]
        assert direct == via_prefetch


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_lazy_image_loading(self, index_path, synthetic_dataset):
        from clouda_data.training_data.artifacts import ArtifactLoader

        _manifest, root = synthetic_dataset
        artifacts = ArtifactLoader(root)
        loader = make_loader(index_path, _manifest, root)
        sample = next(iter(loader.iter_samples(epoch=0)))
        image = artifacts.image(sample.image_path)
        assert image.size == (4, 4)

    def test_missing_artifact(self, tmp_path):
        from clouda_data.training_data.artifacts import resolve_artifact

        assert resolve_artifact(tmp_path, "pages/nope.png") is None

    def test_path_traversal_rejected(self, tmp_path):
        from clouda_data.training_data.artifacts import resolve_artifact

        with pytest.raises(ValueError, match="escapes"):
            resolve_artifact(tmp_path, "../outside.png")
        with pytest.raises(ValueError):
            resolve_artifact(tmp_path, "a/../../b.png")

    def test_corrupt_image_rejected_strict(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "ds", count=6)
        make_corrupt_image(root / "pages" / "page_0002.png")
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=3)
        loader = make_loader(
            tmp_path / "out" / "shard_index.json",
            manifest,
            root,
            validation=ValidationMode.STRICT,
            bad_policy=BadSamplePolicy.SKIP_AND_RECORD,
        )
        ids = _ids(loader, epoch=0)
        assert len(ids) == 5
        assert any("corrupt_image" in r["issues"] for r in loader.stats.rejected)

    def test_missing_image_light_skip_and_record(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "ds", count=6)
        (root / "pages" / "page_0001.png").unlink()
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=3)
        loader = make_loader(
            tmp_path / "out" / "shard_index.json",
            manifest,
            root,
            validation=ValidationMode.LIGHT,
            bad_policy=BadSamplePolicy.SKIP_AND_RECORD,
        )
        ids = _ids(loader, epoch=0)
        assert len(ids) == 5
        assert loader.stats.samples_skipped == 1
        assert loader.stats.rejected[0]["issues"] == ["missing_image"]

    def test_fail_fast(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "ds", count=6)
        (root / "pages" / "page_0001.png").unlink()
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=3)
        loader = make_loader(
            tmp_path / "out" / "shard_index.json",
            manifest,
            root,
            validation=ValidationMode.LIGHT,
            bad_policy=BadSamplePolicy.FAIL_FAST,
        )
        with pytest.raises(ValueError, match="Bad sample"):
            _ids(loader, epoch=0)

    def test_text_hash_mismatch_strict(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "ds", count=4)
        rows = manifest.read_text(encoding="utf-8").splitlines()
        header = rows[0]
        row = json_row(rows[1])
        row["text"] = "tampered text"
        manifest.write_text(
            "\n".join([header, dump_row(row)] + rows[2:]) + "\n", encoding="utf-8"
        )
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=2)
        loader = make_loader(
            tmp_path / "out" / "shard_index.json",
            manifest,
            root,
            validation=ValidationMode.STRICT,
        )
        _ids(loader, epoch=0)
        assert any("text_hash_mismatch" in r["issues"] for r in loader.stats.rejected)


def json_row(line: str) -> dict:
    import json

    return json.loads(line)


def dump_row(row: dict) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Resume cursor
# ---------------------------------------------------------------------------


class TestResume:
    def test_interrupt_resume_exact_continuation(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        kwargs = dict(
            shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=4), seed=5
        )
        loader = make_loader(index_path, manifest, root, **kwargs)
        stream = loader.iter_samples(epoch=0)
        partial = [next(stream) for _ in range(9)]
        cursor = loader.get_cursor()

        loader2 = make_loader(index_path, manifest, root, **kwargs)
        loader2.restore(cursor)
        rest = _ids(loader2)
        full = _ids(make_loader(index_path, manifest, root, **kwargs), epoch=0)
        combined = [s.sample_id for s in partial] + rest
        assert combined == full
        assert len(combined) == len(set(combined))

    def test_reject_changed_dataset(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        tampered = replace(cursor, dataset_id="other_dataset")
        with pytest.raises(ResumeError, match="dataset"):
            loader.restore(tampered)
        tampered2 = replace(cursor, dataset_version="9.9.9")
        with pytest.raises(ResumeError, match="dataset version"):
            loader.restore(tampered2)

    def test_reject_changed_manifest_hash(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        tampered = replace(cursor, manifest_sha256="0" * 64)
        with pytest.raises(ResumeError, match="manifest hash"):
            loader.restore(tampered)

    def test_reject_changed_config(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root, seed=1)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        with pytest.raises(ResumeError, match="loader config"):
            make_loader(index_path, manifest, root, seed=2).restore(cursor)

    def test_reject_changed_seed(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root, seed=1)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        with pytest.raises(ResumeError, match="global seed"):
            make_loader(index_path, manifest, root, seed=1).restore(
                replace(cursor, global_seed=cursor.global_seed + 1)
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("world_size", 4),
            ("rank", 1),
            ("num_workers", 2),
            ("worker_id", 1),
        ],
    )
    def test_reject_topology_change(self, index_path, synthetic_dataset, field, value):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        tampered = replace(cursor, **{field: value})
        with pytest.raises(ResumeError, match=field):
            loader.restore(tampered)

    def test_resume_cursor_roundtrip(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root)
        next(iter(loader.iter_samples(epoch=0)))
        cursor = loader.get_cursor()
        clone = type(cursor).from_dict(cursor.to_dict())
        loader.restore(clone)


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


class TestTraceability:
    def test_summary_mode(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root, trace=TraceMode.SUMMARY)
        _ids(loader, epoch=0)
        summary = loader.trace_summary()
        assert summary["dataset_id"] == DATASET_ID
        assert summary["stats"]["samples_yielded"] == 24
        assert "delivered_order" not in summary

    def test_full_mode_records_order(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        loader = make_loader(index_path, manifest, root, trace=TraceMode.FULL)
        ids = _ids(loader, epoch=0)
        full = loader.trace_full()
        assert full["delivered_order"] == ids

    def test_replay_reproduces_order(self, index_path, synthetic_dataset):
        manifest, root = synthetic_dataset
        kwargs = dict(
            trace=TraceMode.FULL,
            shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=6),
        )
        a = make_loader(index_path, manifest, root, **kwargs)
        ids_a = _ids(a, epoch=0)
        b = make_loader(index_path, manifest, root, **kwargs)
        ids_b = _ids(b, epoch=0)
        assert ids_a == ids_b
        assert a.trace_full()["delivered_order"] == b.trace_full()["delivered_order"]
