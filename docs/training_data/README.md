# Training Data Loader + Sharding Engine

Scalable, deterministic, resumable delivery of already-selected training
data to future trainer adapters. This subsystem consumes the canonical
Clouda pre-training manifest and streams shards/batches; it does **not**
select data, decide holdout policy, or train models.

```
Canonical Dataset / Derived Training Manifest
        ↓  (clouda_data.pretraining.manifest — clouda.pretraining.manifest.v1)
Input validation + holdout fail-closed gate   (training_data/input_contract.py)
        ↓
Sharding Engine                               (training_data/sharding.py)
        ↓  shards/shard-*.jsonl + shard_index.json
Streaming Dataset Loader                      (training_data/loader.py)
        ↓  deterministic shuffle · epoch · worker/rank partitioning
Batching + Prefetch                           (loader + training_data/prefetch.py)
        ↓
Transform / Collation hooks                   (training_data/collation.py)
        ↓
Training Experiment Framework / future HunyuanOCR adapter
        ↓  canonical framework checkpoint with sealed loader cursor
        ↓                                     (training_data/checkpoint_bridge.py)
GPU (deferred — see HARDWARE_VALIDATION.md)
```

Module map (`clouda_data/training_data/`):

| Module | Responsibility |
| --- | --- |
| `models.py` | Typed frozen configs: `TrainingDataConfig`, `ShardConfig`, `ShuffleConfig`, `PrefetchConfig`, `WorkerConfig`, `BatchConfig`, `ResumeCursor`, `SampleReference`, `LoaderState`, `LoaderStats` |
| `input_contract.py` | Canonical manifest validation, holdout fail-closed gate |
| `sharding.py` | Deterministic shard builder + canonical shard index + verifier |
| `loader.py` | `StreamingTrainingDataLoader`, `EpochManager`, resume cursor |
| `ordering.py` | Stable seed derivation, permutations, bounded shuffle |
| `artifacts.py` | Lazy, path-safe artifact resolution/loading |
| `batching semantics` | in `loader.py` (`iter_batches`) |
| `prefetch.py` | Bounded thread-based prefetch |
| `collation.py` | `SampleTransform` / `BatchCollator` protocols + no-op impls |
| `balancing.py` | Generic weighted/stratified runtime sampling + curriculum hook |
| `checkpoint_bridge.py` | Cursor ↔ Training Experiment Framework checkpoints |
| `torch_adapter.py` | Optional (lazy) PyTorch `IterableDataset` adapter |
| `cli.py` | `training-data-*` commands on the unified `clouda-data` CLI |
| `benchmark.py` | CPU-only synthetic benchmark |

## 1. Canonical input contract

The loader consumes **only** the canonical manifest written by
`clouda_data.pretraining.manifest.write_manifest`
(`clouda.pretraining.manifest.v1`): one header line +
one JSON object per sample, sorted, with `_row_count` integrity.
No competing format is defined.

Validation (`validate_canonical_manifest`) is fail-closed and checks:

- header `_schema_version` equals `clouda.pretraining.manifest.v1`;
- `dataset_id` / `dataset_version` in the header match the loader config;
- the file's actual SHA-256 is computed and later cross-checked against
  the shard index (`manifest hash changed → reject`);
- every row: non-empty `sample_id`, unique ids, an artifact reference
  (`image_path` or `text`), object-typed `provenance`;
- training eligibility: `validation_status != error`, `duplicate_state`
  not `duplicate`/`conflicting_duplicate`, no `exclusion_reason`;
- malformed split/role/protection metadata (non-string split fields,
  invalid `protected` values) → reject, never silently interpret.

## 2. Holdout / policy safety (fail closed)

Holdout protection uses the canonical `clouda_contracts.protection` policy,
shared by the Training Experiment Framework, Results, Lab, and this loader.
There is exactly one definition of "protected" in the repository.

The loader rejects (raises `ProtectedManifestError`, a `PermissionError`
subclass, or `ManifestInputError`):

- any row whose split alias is holdout-like: `holdout`, `HOLDOUT`,
  `Holdout`, `protected_holdout`, `benchmark_holdout`, `private_holdout`
  (case-insensitive, any of `target_split`/`split`/`source_split`);
- any row (or nested `provenance`/`metadata` object) with a truthy
  `protected` marker;
- roles in `{holdout, protected_holdout, benchmark, evaluation_only}`;
- malformed protection metadata (ambiguous values fail closed).

A manifest that mixes protected and training rows is rejected wholesale —
never filtered silently.

## 3. Shard format

```
<output_dir>/
    shard_index.json
    shards/
        shard-<16 hex>.jsonl
        ...
```

Each shard row (JSONL):

```json
{
  "schema_version": "clouda.training_data.shard_record.v1",
  "sample_id": "smp_…",
  "shard_id": "shard-0123456789abcdef",
  "position": 17,
  "row": { …the full canonical manifest row, provenance included… }
}
```

Shard files are written atomically. Rows within a shard keep the canonical
manifest order (sorted by source_id/source_path/sample_id).

## 4. Sharding strategies

- `COUNT` — `samples_per_shard` rows per shard (default).
- `SIZE_AWARE` — closes a shard when either the row count or
  `max_shard_bytes` (approximate bytes: `file_size` when recorded, else
  the serialized row length) is exceeded. Approximate by design; exact
  byte accounting would require reading artifact payloads, which the
  sharding engine refuses to do.

## 5. Shard index

`shard_index.json` (`clouda.training_data.shard_index.v1`) records:
`dataset_id`, `dataset_version`, `source_manifest_sha256`,
`shard_config_hash`, `total_samples`, `total_shards`, per-shard
`{shard_id, ordinal, sample_count, approx_bytes, sha256, path}`,
`created_by`. It is plain JSON (no database) and fully rebuildable:
rerunning `build_shards` on the same manifest + config reproduces it
byte-identically. `verify_shards` checks existence, hashes, per-shard
counts, cross-shard duplicate ids, and totals.

## 6. Deterministic identities

- **Shard id** = `blake2b(dataset_id ‖ dataset_version ‖ manifest_sha256 ‖
  shard_config_hash ‖ ordinal)` truncated to 16 hex chars.
- **Shard config hash** = SHA-256 over canonical JSON of `ShardConfig`.
- **Loader config hash** = SHA-256 over canonical JSON of the full
  `TrainingDataConfig` (includes shuffle/batch/workers/validation).
- **Epoch seed** = `blake2b("clouda.training_data.shuffle.v1" ‖ global_seed ‖
  epoch ‖ world_size ‖ rank ‖ num_workers ‖ worker_id ‖ config_hash)`
  masked to 63 bits (same convention as `clouda_data.factory.seed.derive`).

Python's built-in `hash()` is never used anywhere in this subsystem
(enforced by a test that patches `builtins.hash`).

## 7. Streaming behavior

`StreamingTrainingDataLoader.iter_samples` streams shard files line by
line and yields typed `SampleReference` objects (metadata only — the full
row dict, image path, text). The entire corpus is never resident: `NONE` and
`BUFFERED` modes are bounded by the shard index plus the configured shuffle
buffer; exact `SHARD_ORDER` mode holds at most one configured-bounded shard.
Shards are opened lazily in visit order and closed after use. Canonical
manifest validation and framework dataset validation are streaming as well.

## 8. Shuffle strategy

Three modes (`ShuffleConfig.mode`):

- `NONE` — canonical order.
- `SHARD_ORDER` — shards are visited in a seeded permutation; within a
  shard the order is a seeded permutation. O(shard) ints in RAM.
- `BUFFERED` (default) — shard-order permutation + a bounded shuffle
  buffer of `buffer_size` samples within each shard.

**Documented tradeoff**: an exact global shuffle would require holding all
N sample keys in RAM (fine for metadata, but the buffer approach keeps RAM
constant regardless of N). `BUFFERED` gives local uniformity within the
buffer window and global uniformity across shards; it is reproducible and
lossless (every sample appears exactly once per epoch) but is not a
uniform random permutation of the full dataset. Use `SHARD_ORDER` with
many small shards for stronger mixing at O(total keys) memory.

## 9. Epoch behavior

`iter_samples(epoch=E)` produces a deterministic order per
`(global_seed, E, topology, config)`. Completing a full pass automatically
advances the internal epoch. New epoch ⇒ new seed ⇒ new shard order +
buffer permutation — no repeats, no omissions (every epoch covers the
partition exactly once). Resume cursors store the epoch; restoring
continues mid-epoch exactly (the ordering is a pure function of the seed
inputs, so skipping to a stored position reproduces the same sequence).

## 10. Multi-worker semantics

`WorkerConfig(world_size, rank, num_workers, worker_id)` partitions via
`stable_hash(sample_id) mod (world_size × num_workers)`: slot
`rank × num_workers + worker_id` owns the sample. Properties: deterministic,
order-independent (works across restarts and process boundaries), disjoint
(no duplicates between workers), and complete (no sample dropped).

**Remainder policy**: modular assignment has no remainder to drop — every
sample belongs to exactly one slot regardless of divisibility. This is the
documented behavior (`RemainderPolicy.KEEP` semantics).

## 11. Distributed / rank semantics

For multi-GPU training, set `world_size`/`rank` per process; each rank
additionally splits across its DataLoader workers via `num_workers`/
`worker_id`. Uneven shard sizes and dataset sizes not divisible by
world_size are handled by the same modular assignment (no drop). A
`drop_last`-style policy for ranks is intentionally not implemented at the
data layer; trainers wanting strict per-rank step parity should use
`BatchConfig.drop_last=True`. Distributed resume across changed
`world_size` is rejected (see below) and must be handled explicitly by the
trainer (fresh epoch).

## 12. Batching

`iter_batches` yields dicts:
`{sample_ids, samples, shard_ids, epoch, size}` with deterministic
boundaries for a given config/epoch. Final partial batch is kept unless
`BatchConfig.drop_last=True`. No tensorization happens here — that is the
collator's job.

## 13. Prefetch

`PrefetchIterator` (or `prefetch(iterable, config=PrefetchConfig)`): a
bounded queue (depth = `PrefetchConfig.depth`) fed by a single daemon
thread. Guarantees: FIFO order identical to upstream, bounded memory,
exception propagation to the consumer, clean shutdown (`.close()` /
context manager / consumer exit), no runaway threads.

## 14. Artifact loading (lazy)

`artifacts.resolve_artifact` enforces canonical root-relative paths
(reusing `clouda_data.pretraining.schema.canonical_relative_path`) and
re-resolves beneath the dataset root (symlink/traversal defense; `..`,
absolute paths, Windows drives, control characters all rejected).
`ArtifactLoader.image/text/image_bytes` load payloads only on request —
metadata iteration never reads image bytes. STRICT validation additionally
decodes images and verifies text hashes.

## 15. Validation modes & bad-sample policy

`ValidationMode`: `NONE` (no per-sample checks), `LIGHT` (referenced
artifact exists), `STRICT` (+ image decodes, + text hash matches when a
`normalized_text_sha256` is recorded). Expensive checks are opt-in;
default is `LIGHT`-quality metadata validation at load time and manifest
validation at `open()`.

`BadSamplePolicy`: `FAIL_FAST` (raise on first bad sample) or
`SKIP_AND_RECORD` (skip + append a structured rejection record —
`{sample_id, shard_id, position, issues, epoch}` — to `LoaderStats.rejected`).
Silent skipping never happens.

## 16. Resume cursor

`ResumeCursor` (schema `clouda.training_data.loader_state.v1`) stores:
dataset_id/version, manifest_sha256, loader_config_hash, global_seed,
epoch, world_size, rank, num_workers, worker_id, shard_position,
sample_position, yielded_count. `restore()` rejects incompatible state
fail-closed: changed dataset, changed version, changed manifest hash,
changed loader config, changed seed, changed topology. Counters are
incremented **before** yielding so a cursor taken at any moment never
undercounts delivered samples.

## 17. Training checkpoint integration

`LoaderCheckpointHook` attaches the cursor to Training Experiment Framework
checkpoints. Loader-aware `run_experiment` saves the cursor beside trainer
state inside the existing `CheckpointManager` integrity envelope;
loader-aware `resume_run` verifies exact dataset/manifest/config/shard
identity and restores it before continuing without duplicates or gaps. Lab's
`TrainingOrchestrator` prepares deterministic shards, constructs the canonical
loader, and delegates run/resume to those framework APIs. Runtime-only local
paths are returned in the prepared descriptor but are never persisted as run
lineage; persisted lineage contains only stable identities and hashes. No new
run registry or checkpoint system is introduced. See
`tests/data_foundation/integration/test_training_data_e2e.py` for the
full flow.

## 18. Sample traceability

`TraceMode.NONE/SUMMARY/FULL`. SUMMARY (default): counts, shard stats,
dataset identity, epoch stats, rejections (`trace_summary()`). FULL:
additionally the exact delivery order of sample ids (`trace_full()`).
Combined with the deterministic seed derivation this makes any run's data
consumption auditable and replayable.

## 19. Balancing / curriculum hooks

Generic runtime mechanisms only — upstream systems decide what is
easy/hard. `stratum_of` reads recognized strata keys
(`document_type`, `profile`, `distortion`, `source_class`, `language`,
`difficulty_bucket`, `difficulty`) from row fields or nested
`metadata`/`provenance` when present; absent labels are never invented.
`SamplingSchedule(weights)` maps strata → keep-probability in [0,1];
`weighted_stream` applies it deterministically per (seed, epoch).
`CurriculumHook(schedule_for_epoch)` yields per-epoch schedules (e.g.
epoch 0: hard↓, later: hard↑). This is not a second Dataset Selection
Engine — it reweights an already-chosen dataset at runtime.

## 20. PyTorch optional adapter

`torch_adapter.py` is **never imported by the package `__init__`** and
imports torch lazily. Core loader + all tests run without torch installed.
`make_iterable_dataset(loader_or_factory)` returns a
`torch.utils.data.IterableDataset` that reads `rank/world_size` from
`torch.distributed` (when initialized) and `worker_id/num_workers` from
`torch.utils.data.get_worker_info`, so `DataLoader(dataset, num_workers=N)`
delivers each sample exactly once. `torch_available()` / `require_torch()`
probe cleanly.

## 21. CLI

On the unified `clouda-data` CLI (no second CLI framework):

```
clouda-data training-data-shard <manifest.jsonl> --output DIR
    [--samples-per-shard N] [--size-aware] [--max-shard-bytes B] [--limit N]
clouda-data training-data-inspect <shard_index.json> [--shard ID]
clouda-data training-data-verify <shard_index.json> [--root DIR]
clouda-data training-data-sample <shard_index.json> [--count N] [--epoch E]
    [--manifest M] [--root R]
clouda-data training-data-dry-run <shard_index.json> [--batches N]
    [--batch-size B] [--drop-last] [--seed S] [--epoch E]
    [--shuffle-buffer N] [--prefetch-depth D]
    [--validation-mode none|light|strict]
    [--bad-sample-policy fail_fast|skip_and_record] [--manifest M] [--root R]
```

`dry-run` opens the dataset, iterates N batches, and **verifies
deterministic replay** (a second pass must reproduce the delivered order
exactly) without any training.

## 22. Performance considerations

- Memory-resident structures: shard index, the bounded shuffle buffer, the
  current batch, and the prefetch queue. Exact `SHARD_ORDER` additionally
  holds one configured-bounded shard. Nothing scales with total dataset size.
- The benchmark (`python -m clouda_data.training_data.benchmark`) measures
  init/iteration/memory at 100/1,000/10,000-sample scales on tiny
  fixtures; CI runs the 100-sample case, the 1,000-sample case is marked
  slow, 10,000 is manual.
- Metadata-only iteration is deliberately separated from artifact
  decoding; the heavy cost lives in trainer transforms, not the loader.
- No GPU utilization numbers are claimed anywhere — see
  `HARDWARE_VALIDATION.md`.

## 23. Future HunyuanOCR adapter boundary

A future HunyuanOCR-1.5 trainer adapter implements exactly two hooks:

1. `SampleTransform` — receives `SampleReference`; loads the page image
   via `ArtifactLoader`, applies the HF processor/tokenizer, builds
   prompts/labels.
2. `BatchCollator` — combines transformed samples into model-ready
   tensors/batches.

Nothing else changes: the loader, sharding, resume and checkpoint bridge
are model-agnostic. No Hunyuan-specific preprocessing is implemented here
(verified APIs + installed dependencies required first); no model weights
are downloaded by this subsystem.

## 24. Real hardware validation still required

See `HARDWARE_VALIDATION.md` — all GPU/storage/throughput numbers remain
unmeasured until real training runs exist.
