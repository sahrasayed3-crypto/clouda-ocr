"""CPU-only synthetic loader benchmark (metadata-only fixtures).

Measures sharding, loader initialization, iteration, and batching at 100 /
1,000 / 10,000 sample scales. Uses tiny 4x4 PNGs written once per unique
page and referenced repeatedly, so disk footprint stays small while record
counts exercise streaming behavior. Deterministic ordering is asserted.

Run directly::

    python -m clouda_data.training_data.benchmark --sizes 100 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

from clouda_data.training_data.loader import StreamingTrainingDataLoader
from clouda_data.training_data.models import (
    BatchConfig,
    ShardConfig,
    ShuffleConfig,
    ShuffleMode,
    TrainingDataConfig,
)
from clouda_data.training_data.sharding import build_shards

BENCHMARK_SCHEMA_VERSION = "clouda.training_data.benchmark.v1"
DEFAULT_DATASET_ID = "synthetic_training_fixture"
DEFAULT_DATASET_VERSION = "1.0.0"


def generate_manifest(root: Path, count: int) -> tuple[Path, Path]:
    """Create a canonical manifest with ``count`` rows over tiny PNGs.

    One unique PNG per 24 pages (repeated) keeps fixture size tiny; rows
    themselves are unique and carry full provenance.
    """

    import hashlib
    import io

    from PIL import Image

    from clouda_data.pretraining.manifest import write_manifest

    root.mkdir(parents=True, exist_ok=True)
    pages = root / "pages"
    pages.mkdir(exist_ok=True)
    unique_pngs = 24
    png_bytes: list[bytes] = []
    for variant in range(unique_pngs):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color=(variant, 100, 200)).save(buffer, format="PNG")
        png_bytes.append(buffer.getvalue())

    rows = []
    for index in range(count):
        rel = f"pages/page_{index:06d}.png"
        payload = png_bytes[index % unique_pngs]
        path = pages / f"page_{index:06d}.png"
        if not path.exists():
            path.write_bytes(payload)
        text = f"نص تجريبي {index}"
        rows.append(
            {
                "sample_id": f"smp_bench_{index:08d}",
                "source_id": "benchmark_fixture",
                "source_path": f"benchmark/doc_{index:06d}",
                "image_path": rel,
                "text": text,
                "language": "ar",
                "script": "arabic",
                "target_split": "train",
                "validation_status": "ok",
                "duplicate_state": "unique",
                "file_size": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "normalized_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "provenance": {"origin": "benchmark", "page": index},
            }
        )
    manifest = root / "manifest.jsonl"
    write_manifest(
        manifest,
        rows,
        metadata={
            "dataset_id": DEFAULT_DATASET_ID,
            "dataset_version": DEFAULT_DATASET_VERSION,
            "manifest_sha256": "",
        },
    )
    return manifest, root


def run_benchmark(count: int, workdir: Path) -> dict[str, Any]:
    """Shard + iterate ``count`` synthetic samples; return timings/mem."""

    dataset_dir = workdir / f"ds_{count}"
    sharded_dir = workdir / f"shards_{count}"

    t0 = time.monotonic()
    manifest, dataset_root = generate_manifest(dataset_dir, count)
    generate_seconds = time.monotonic() - t0

    t0 = time.monotonic()
    index = build_shards(
        manifest,
        sharded_dir,
        ShardConfig(samples_per_shard=max(1, count // 10)),
        dataset_id=DEFAULT_DATASET_ID,
        dataset_version=DEFAULT_DATASET_VERSION,
    )
    shard_seconds = time.monotonic() - t0

    config = TrainingDataConfig(
        dataset_id=DEFAULT_DATASET_ID,
        dataset_version=DEFAULT_DATASET_VERSION,
        global_seed=20260723,
        shard=ShardConfig(),
        shuffle=ShuffleConfig(mode=ShuffleMode.BUFFERED, buffer_size=256),
        batch=BatchConfig(batch_size=32),
    )

    tracemalloc.start()
    t0 = time.monotonic()
    loader = StreamingTrainingDataLoader(
        shard_index_path=sharded_dir / "shard_index.json",
        loader_config=config,
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    loader.open()
    init_seconds = time.monotonic() - t0

    t0 = time.monotonic()
    sample_ids: list[str] = []
    batches = 0
    for batch in loader.iter_batches(epoch=0):
        batches += 1
        sample_ids.extend(batch["sample_ids"])
    iterate_seconds = time.monotonic() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    deterministic = sample_ids == list(
        StreamingTrainingDataLoader(
            shard_index_path=sharded_dir / "shard_index.json",
            loader_config=config,
            manifest_path=manifest,
            dataset_root=dataset_root,
        ).iter_sample_ids(epoch=0)
    )

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "samples": count,
        "shards": index.total_shards,
        "batches": batches,
        "fixture_generation_seconds": round(generate_seconds, 4),
        "sharding_seconds": round(shard_seconds, 4),
        "loader_init_seconds": round(init_seconds, 4),
        "iteration_seconds": round(iterate_seconds, 4),
        "samples_per_second": (
            round(count / iterate_seconds, 1) if iterate_seconds > 0 else None
        ),
        "peak_traced_memory_mb": round(peak / (1024 * 1024), 2),
        "deterministic_replay": deterministic,
        "validation_mode": config.validation_mode.value,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only loader benchmark")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="sample counts to benchmark (default: 100 1000)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="working directory (default: temp dir)",
    )
    args = parser.parse_args(argv)
    import tempfile

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="clouda_loader_bench_"))
    results = [run_benchmark(count, workdir) for count in args.sizes]
    payload = {"schema_version": BENCHMARK_SCHEMA_VERSION, "results": results}
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
