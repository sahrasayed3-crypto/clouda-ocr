"""Checkpoint bridge + offline E2E: loader -> batches -> MockTrainer-style
loop -> CheckpointManager -> resume cursor -> no duplicate/skip."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from clouda_contracts.checksums import sha256_file
from clouda_data.training_data.checkpoint_bridge import (
    CursorMismatchError,
    LoaderCheckpointHook,
    read_cursor_from_checkpoint,
)
from clouda_data.training_data.models import (
    BadSamplePolicy,
    BatchConfig,
    ShuffleConfig,
    ShuffleMode,
    TraceMode,
    TrainingDataConfig,
)
from clouda_data.training_data.loader import ResumeError, StreamingTrainingDataLoader
from clouda_training.experiments.checkpoints import CheckpointManager
from clouda_training.experiments.config import (
    DatasetSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    TrainingSection,
)
from clouda_training.experiments.io import atomic_write_json

from tests.data_foundation.fixtures.training_data_fixtures import (
    DATASET_ID,
    DATASET_VERSION,
    build_synthetic_dataset,
    shard_dataset,
)

LOADER_SEED = 7


def _loader_config() -> TrainingDataConfig:
    return TrainingDataConfig(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        global_seed=LOADER_SEED,
        shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=5),
        batch=BatchConfig(batch_size=4),
        bad_sample_policy=BadSamplePolicy.SKIP_AND_RECORD,
        trace_mode=TraceMode.SUMMARY,
    )


def _make_loader(tmp_path: Path, count: int = 24):
    manifest, root = build_synthetic_dataset(tmp_path / "dataset", count=count)
    shard_dataset(manifest, tmp_path / "sharded", samples_per_shard=5)
    index_path = tmp_path / "sharded" / "shard_index.json"
    return manifest, root, index_path


def _experiment_config(manifest: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="loader_e2e"),
        model=ModelSection(model_id="mock-ocr", adapter_type="mock"),
        dataset=DatasetSection(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            manifest_path=manifest,
        ),
        training=TrainingSection(seed=LOADER_SEED, batch_size=4, epochs=1),
    )


def _save_checkpoint_with_cursor(
    manager: CheckpointManager,
    exp_config: ExperimentConfig,
    step: int,
    epoch: int,
    loader: StreamingTrainingDataLoader,
) -> Path:
    """Mirror CheckpointManager.save but bake the data cursor into state.json
    before the integrity digest is computed."""

    final = manager.root / f"step-{step:08d}"
    if final.exists():
        return final / "state.json"
    staging = manager.root / f".step-{step:08d}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    hook = LoaderCheckpointHook(loader)
    state = hook.attach_to_state({"step": step, "epoch": epoch})
    atomic_write_json(staging / "state.json", state)
    digest = sha256_file(staging / "state.json")
    payload = {
        "schema_version": 1,
        "step": step,
        "epoch": float(epoch),
        "timestamp": "2026-09-10T00:00:00Z",
        "metrics": {"loss": 1.0 / step},
        "config_hash": exp_config.hash,
        "run_id": manager.run_id,
        "integrity_sha256": digest,
        "experiment_name": exp_config.experiment.name,
        "model_id": exp_config.model.model_id,
        "model_revision": exp_config.model.revision,
        "dataset_id": exp_config.dataset.dataset_id,
        "dataset_version": exp_config.dataset.dataset_version,
        "is_best": False,
    }
    atomic_write_json(staging / "metadata.json", payload)
    staging.rename(final)
    return final


def _train_loop(
    loader: StreamingTrainingDataLoader,
    manager: CheckpointManager,
    exp_config: ExperimentConfig,
    *,
    resume_cursor=None,
    interrupt_at_step=None,
):
    """MockTrainer-style loop: consume batches, checkpoint each step."""

    if resume_cursor is not None:
        loader.restore(resume_cursor)
    step = resume_cursor.yielded_count if resume_cursor else 0
    last_state_path = None
    for batch in loader.iter_batches(epoch=None):
        step += 1
        ckpt = _save_checkpoint_with_cursor(
            manager, exp_config, step, batch["epoch"], loader
        )
        last_state_path = ckpt
        if interrupt_at_step is not None and step >= interrupt_at_step:
            cursor = read_cursor_from_checkpoint(ckpt)
            return step, cursor, ckpt, True
    return step, loader.get_cursor(), last_state_path, False


class TestCheckpointBridge:
    def test_cursor_roundtrip_through_state(self, tmp_path):
        manifest, root, index_path = _make_loader(tmp_path)
        loader = StreamingTrainingDataLoader(
            shard_index_path=index_path,
            loader_config=_loader_config(),
            manifest_path=manifest,
            dataset_root=root,
        )
        stream = loader.iter_samples(epoch=0)
        for _ in range(6):
            next(stream)
        hook = LoaderCheckpointHook(loader)
        state = hook.attach_to_state({"step": 2})
        restored_cursor = hook.restore_from_state(state)
        assert restored_cursor.yielded_count == 6

    def test_state_without_cursor_fails_closed(self, tmp_path):
        manifest, root, index_path = _make_loader(tmp_path)
        loader = StreamingTrainingDataLoader(
            shard_index_path=index_path,
            loader_config=_loader_config(),
            manifest_path=manifest,
            dataset_root=root,
        )
        hook = LoaderCheckpointHook(loader)
        with pytest.raises(CursorMismatchError, match="no data cursor"):
            hook.restore_from_state({"step": 1})

    def test_schema_mismatch_fails_closed(self, tmp_path):
        manifest, root, index_path = _make_loader(tmp_path)
        loader = StreamingTrainingDataLoader(
            shard_index_path=index_path,
            loader_config=_loader_config(),
            manifest_path=manifest,
            dataset_root=root,
        )
        hook = LoaderCheckpointHook(loader)
        state = hook.attach_to_state({"step": 1})
        state["data_cursor_schema"] = "ancient.v0"
        with pytest.raises(CursorMismatchError, match="schema"):
            hook.restore_from_state(state)


class TestEndToEnd:
    def test_full_flow_checkpoint_resume(self, tmp_path):
        manifest, root, index_path = _make_loader(tmp_path, count=24)
        exp_config = _experiment_config(manifest)
        run_path = tmp_path / "run"
        run_path.mkdir()
        manager = CheckpointManager(run_path, "run_e2e", exp_config)

        def fresh_loader():
            return StreamingTrainingDataLoader(
                shard_index_path=index_path,
                loader_config=_loader_config(),
                manifest_path=manifest,
                dataset_root=root,
            )

        # Phase 1: train, interrupt after 3 checkpoints
        loader = fresh_loader()
        step1, cursor1, ckpt1, interrupted = _train_loop(
            loader, manager, exp_config, interrupt_at_step=3
        )
        assert interrupted and step1 == 3
        assert cursor1.yielded_count == 12

        # Phase 2: resume from the checkpoint cursor, run to completion
        loader2 = fresh_loader()
        step2, cursor2, _, interrupted2 = _train_loop(
            loader2, manager, exp_config, resume_cursor=cursor1
        )
        assert not interrupted2
        assert cursor2.yielded_count == 24

        # Integrity: every sample delivered exactly once, none skipped
        full = fresh_loader()
        expected = [s.sample_id for s in full.iter_samples(epoch=0)]
        partial = fresh_loader()
        stream = partial.iter_samples(epoch=0)
        first = [next(stream) for _ in range(cursor1.yielded_count)]
        assert [s.sample_id for s in first] == expected[: cursor1.yielded_count]
        loader3 = fresh_loader()
        loader3.restore(read_cursor_from_checkpoint(ckpt1))
        rest = [s.sample_id for s in loader3.iter_samples()]
        assert first_ids(first) + rest == expected
        assert len(first_ids(first) + rest) == len(set(first_ids(first) + rest))

    def test_resume_rejects_foreign_checkpoint(self, tmp_path):
        manifest, root, index_path = _make_loader(tmp_path, count=8)
        config = _loader_config()
        loader = StreamingTrainingDataLoader(
            shard_index_path=index_path,
            loader_config=config,
            manifest_path=manifest,
            dataset_root=root,
        )
        next(iter(loader.iter_samples(epoch=0)))
        cursor = replace(loader.get_cursor(), loader_config_hash="deadbeef")
        with pytest.raises(ResumeError, match="loader config"):
            loader.restore(cursor)


def first_ids(samples):
    return [s.sample_id for s in samples]
