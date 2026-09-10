"""``clouda-data training-data-*`` command implementations.

Follows the canonical unified CLI conventions: flat subcommands in
``clouda_data.pipeline.cli``, implementations in a feature module. This
module holds shard/inspect/verify/sample/dry-run behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clouda_data.training_data.models import (
    BadSamplePolicy,
    BatchConfig,
    ShardConfig,
    ShardStrategy,
    ShuffleConfig,
    TraceMode,
    TrainingDataConfig,
    ValidationMode,
)
from clouda_data.training_data.sharding import (
    load_shard_index,
    verify_shards,
)


def _emit(payload: Any) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _shard_config_from_args(args: argparse.Namespace) -> ShardConfig:
    strategy = ShardStrategy.SIZE_AWARE if args.size_aware else ShardStrategy.COUNT
    return ShardConfig(
        strategy=strategy,
        samples_per_shard=args.samples_per_shard,
        max_shard_bytes=args.max_shard_bytes,
    )


def command_shard(args: argparse.Namespace) -> int:
    from clouda_data.training_data.sharding import build_shards

    manifest = Path(args.manifest).resolve()
    header = _read_header(manifest)
    index = build_shards(
        manifest,
        Path(args.output).resolve(),
        _shard_config_from_args(args),
        dataset_id=str(header["dataset_id"]),
        dataset_version=str(header["dataset_version"]),
        sample_limit=args.limit,
    )
    return _emit(index.to_dict())


def _read_header(manifest: Path) -> dict[str, Any]:
    from clouda_data.pretraining.manifest import iter_manifest

    for payload in iter_manifest(manifest):
        if "_schema_version" in payload and "sample_id" not in payload:
            return payload
    raise SystemExit(
        f"Manifest has no canonical header with dataset_id/dataset_version: {manifest}"
    )


def command_inspect(args: argparse.Namespace) -> int:
    index = load_shard_index(Path(args.shard_index).resolve())
    payload = index.to_dict()
    if args.shard:
        payload["shards"] = [
            entry for entry in payload["shards"] if entry["shard_id"] == args.shard
        ]
    return _emit(payload)


def command_verify(args: argparse.Namespace) -> int:
    index = load_shard_index(Path(args.shard_index).resolve())
    root = (
        Path(args.root).resolve()
        if args.root
        else Path(args.shard_index).resolve().parent
    )
    report = verify_shards(index, root)
    return _emit(report)


def command_sample(args: argparse.Namespace) -> int:
    from clouda_data.training_data.loader import StreamingTrainingDataLoader
    from clouda_data.training_data.models import WorkerConfig

    index_path = Path(args.shard_index).resolve()
    index = load_shard_index(index_path)
    config = TrainingDataConfig(
        dataset_id=index.dataset_id,
        dataset_version=index.dataset_version,
        workers=WorkerConfig(),
        validation_mode=ValidationMode.NONE,
        trace_mode=TraceMode.FULL,
    )
    loader = StreamingTrainingDataLoader(
        shard_index_path=index_path,
        loader_config=config,
        manifest_path=(
            Path(args.manifest).resolve() if args.manifest else index_path.parent / "manifest.jsonl"
        ),
        dataset_root=Path(args.root).resolve() if args.root else index_path.parent,
    )
    samples = []
    for sample in loader.iter_samples(epoch=args.epoch):
        samples.append(sample.to_dict())
        if len(samples) >= args.count:
            break
    return _emit({"count": len(samples), "samples": samples})


def command_dry_run(args: argparse.Namespace) -> int:
    from clouda_data.training_data.loader import StreamingTrainingDataLoader
    from clouda_data.training_data.models import WorkerConfig

    index_path = Path(args.shard_index).resolve()
    index = load_shard_index(index_path)
    config = TrainingDataConfig(
        dataset_id=index.dataset_id,
        dataset_version=index.dataset_version,
        global_seed=args.seed,
        shard=ShardConfig(),
        shuffle=ShuffleConfig(buffer_size=args.shuffle_buffer),
        batch=BatchConfig(batch_size=args.batch_size, drop_last=args.drop_last),
        workers=WorkerConfig(),
        validation_mode=ValidationMode(args.validation_mode),
        bad_sample_policy=BadSamplePolicy(args.bad_sample_policy),
        trace_mode=TraceMode.FULL,
    )
    loader = StreamingTrainingDataLoader(
        shard_index_path=index_path,
        loader_config=config,
        manifest_path=(
            Path(args.manifest).resolve()
            if args.manifest
            else index_path.parent / "manifest.jsonl"
        ),
        dataset_root=Path(args.root).resolve() if args.root else index_path.parent,
    )
    identity = loader.open()

    produced_batches: list[dict[str, Any]] = []
    iterator: Any
    if args.prefetch_depth > 1:
        from clouda_data.training_data.prefetch import prefetch

        iterator = prefetch(
            loader.iter_batches(epoch=args.epoch), depth=args.prefetch_depth
        )
    else:
        iterator = loader.iter_batches(epoch=args.epoch)

    with iterator if hasattr(iterator, "__enter__") else _NullCtx(iterator) as ctx:
        for batch in ctx:
            produced_batches.append(
                {
                    "sample_ids": batch["sample_ids"],
                    "epoch": batch["epoch"],
                    "size": batch["size"],
                }
            )
            if args.batches and len(produced_batches) >= args.batches:
                break

    # Deterministic replay check: second pass must deliver identical order.
    replay: list[str] = []
    loader_replay = StreamingTrainingDataLoader(
        shard_index_path=index_path,
        loader_config=config,
        manifest_path=loader._manifest_path,
        dataset_root=loader.root,
    )
    for sample in loader_replay.iter_samples(epoch=args.epoch):
        replay.append(sample.sample_id)
        if len(replay) >= sum(len(b["sample_ids"]) for b in produced_batches):
            break
    delivered = [sid for b in produced_batches for sid in b["sample_ids"]]
    replay_ok = delivered == replay[: len(delivered)]

    report = {
        "schema_version": "clouda.training_data.dry_run.v1",
        "dataset_id": index.dataset_id,
        "dataset_version": index.dataset_version,
        "manifest_sha256": identity.manifest_sha256,
        "requested_batches": args.batches,
        "produced_batches": len(produced_batches),
        "delivered_samples": len(delivered),
        "replay_identical": replay_ok,
        "stats": loader.stats.to_dict(),
        "trace": {
            "delivered_order": delivered,
            "mode": "full",
        },
    }
    return _emit(report)


class _NullCtx:
    def __init__(self, iterable: Any) -> None:
        self.iterable = iterable

    def __enter__(self) -> Any:
        return self.iterable

    def __exit__(self, *_exc: Any) -> None:
        return None


def register_training_data_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``training-data-*`` subcommands on the unified CLI."""

    p = sub.add_parser(
        "training-data-shard",
        help="Shard a canonical training manifest into deterministic JSONL shards.",
    )
    p.add_argument("manifest", type=Path)
    p.add_argument(
        "--output", type=Path, required=True, help="output directory for shards/"
    )
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--size-aware", action="store_true")
    p.add_argument("--max-shard-bytes", type=int, default=256 * 1024 * 1024)
    p.add_argument("--limit", type=int, default=None, help="shard only first N rows")
    p.set_defaults(func=command_shard)

    p = sub.add_parser(
        "training-data-inspect",
        help="Show a shard index (dataset identity, shard ids, counts, hashes).",
    )
    p.add_argument("shard_index", type=Path)
    p.add_argument("--shard", default=None, help="filter to one shard id")
    p.set_defaults(func=command_inspect)

    p = sub.add_parser(
        "training-data-verify",
        help="Verify shard files against the index (existence, hashes, counts).",
    )
    p.add_argument("shard_index", type=Path)
    p.add_argument("--root", type=Path, default=None)
    p.set_defaults(func=command_verify)

    p = sub.add_parser(
        "training-data-sample",
        help="Print N sample references from a shard index (metadata only).",
    )
    p.add_argument("shard_index", type=Path)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    p.set_defaults(func=command_sample)

    p = sub.add_parser(
        "training-data-dry-run",
        help="Open a dataset, iterate batches, and verify deterministic replay.",
    )
    p.add_argument("shard_index", type=Path)
    p.add_argument("--batches", type=int, default=0, help="0 = all batches")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--prefetch-depth", type=int, default=1)
    p.add_argument(
        "--validation-mode", choices=["none", "light", "strict"], default="light"
    )
    p.add_argument(
        "--bad-sample-policy", choices=["fail_fast", "skip_and_record"], default="skip_and_record"
    )
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    p.set_defaults(func=command_dry_run)
