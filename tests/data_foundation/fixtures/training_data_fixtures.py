"""Shared helpers for training-data tests: synthetic canonical manifests,
tiny PNG artifacts, shard utilities. Offline, CPU-only, no downloads."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from clouda_data.pretraining.manifest import write_manifest
from clouda_data.training_data.models import (
    BadSamplePolicy,
    BatchConfig,
    ShardConfig,
    ShardStrategy,
    ShuffleConfig,
    ShuffleMode,
    TraceMode,
    TrainingDataConfig,
    ValidationMode,
    WorkerConfig,
)
from clouda_data.training_data.sharding import (
    ShardIndex,
    build_shards,
)
from clouda_data.training_data.loader import (
    StreamingTrainingDataLoader,
)

DATASET_ID = "synthetic_training_fixture"
DATASET_VERSION = "1.0.0"


def tiny_png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(200, 100, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_sample_row(index: int, *, split: str = "train") -> dict:
    image_rel = f"pages/page_{index:04d}.png"
    text = f"نص تجريبي {index}"
    payload = tiny_png()
    return {
        "sample_id": f"smp_test_{index:06d}",
        "source_id": "synthetic_fixture",
        "source_path": f"synthetic/doc_{index:04d}",
        "image_path": image_rel,
        "text": text,
        "language": "ar",
        "script": "arabic",
        "target_split": split,
        "validation_status": "ok",
        "duplicate_state": "unique",
        "file_size": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "provenance": {"origin": "test_fixture", "page": index},
    }


def build_synthetic_dataset(
    root: Path, count: int = 24, *, split: str = "train"
) -> tuple[Path, Path]:
    """Create a canonical manifest + tiny PNG artifacts under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    pages = root / "pages"
    pages.mkdir(exist_ok=True)
    png = tiny_png()
    rows = []
    for index in range(count):
        (pages / f"page_{index:04d}.png").write_bytes(png)
        rows.append(make_sample_row(index, split=split))
    manifest = root / "manifest.jsonl"
    write_manifest(
        manifest,
        rows,
        metadata={
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "manifest_sha256": "",
        },
    )
    return manifest, root


def make_corrupt_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-really-a-png")


def shard_dataset(
    manifest: Path,
    output: Path,
    *,
    samples_per_shard: int = 7,
    strategy: ShardStrategy = ShardStrategy.COUNT,
) -> ShardIndex:
    return build_shards(
        manifest,
        output,
        ShardConfig(strategy=strategy, samples_per_shard=samples_per_shard),
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
    )


def make_loader(
    index_path: Path,
    manifest: Path,
    root: Path,
    *,
    seed: int = 42,
    shuffle: ShuffleConfig | None = None,
    batch: BatchConfig | None = None,
    workers=None,
    validation: ValidationMode = ValidationMode.NONE,
    bad_policy: BadSamplePolicy = BadSamplePolicy.SKIP_AND_RECORD,
    trace: TraceMode = TraceMode.NONE,
) -> StreamingTrainingDataLoader:
    config = TrainingDataConfig(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        global_seed=seed,
        shuffle=shuffle or ShuffleConfig(mode=ShuffleMode.NONE),
        batch=batch or BatchConfig(batch_size=4),
        workers=workers or WorkerConfig(),
        validation_mode=validation,
        bad_sample_policy=bad_policy,
        trace_mode=trace,
    )
    return StreamingTrainingDataLoader(
        shard_index_path=index_path,
        loader_config=config,
        manifest_path=manifest,
        dataset_root=root,
    )


def default_loader_kwargs(manifest: Path, root: Path) -> dict:
    return {"manifest_path": manifest, "dataset_root": root}


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    manifest, root = build_synthetic_dataset(tmp_path / "dataset", count=24)
    return manifest, root


@pytest.fixture
def shard_index(synthetic_dataset, tmp_path):
    manifest, root = synthetic_dataset
    return shard_dataset(manifest, tmp_path / "sharded", samples_per_shard=7)


@pytest.fixture
def index_path(shard_index, tmp_path):
    return tmp_path / "sharded" / "shard_index.json"
