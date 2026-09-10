# Hardware Validation Deferred Items — Training Data Loader

All items below are **unmeasured**. No GPU, multi-GPU, or real-storage
benchmarking has been performed for this subsystem; every number that will
eventually appear here must come from a real training run. Nothing in this
document may be presented as a measured result.

## GPU feeding / starvation

- [ ] Actual GPU starvation measurement (steps/sec with the loader in the
  loop vs. trainer-only ceiling) on the target training GPU.
- [ ] Optimal `num_workers` for page-image decode + preprocessing on the
  target CPU/GPU pairing.
- [ ] Optimal `PrefetchConfig.depth` (queue too deep wastes RAM, too
  shallow starves; must be measured, not assumed).
- [ ] Batch preparation wall time vs. GPU step time (is batch prep
  overlapped at all?).
- [ ] CUDA transfer overlap: pinned-memory effectiveness for
  image-tensor batches (pinned buffers are not yet used by the loader).
- [ ] `torch.utils.data` worker behavior with the
  `torch_adapter` IterableDataset under real DataLoader prefetch
  (loader-level prefetch may need to be disabled inside workers to avoid
  double buffering).

## Storage throughput

- [ ] NVMe throughput for shard streaming (sequential JSONL reads).
- [ ] HDD throughput — expected bottleneck for large shards; may require
  larger `samples_per_shard` or a different visit order.
- [ ] Network storage (Wasabi/S3-mounted/HF) throughput and latency
  hiding; shard streaming is local-file based today.

## Distributed behavior

- [ ] Multi-GPU rank behavior under NCCL (assignment layer is tested by
  simulation only).
- [ ] Distributed resume across `world_size` changes (deliberately
  rejected by the cursor; a trainer-side epoch-restart strategy must be
  validated).
- [ ] Long-running worker stability (multi-hour runs, worker restarts,
  transient file-lock behavior on Windows).

## Scale

- [ ] Very-large-dataset throughput (10⁵–10⁷ rows): manifest validation
  currently streams the full canonical manifest at `open()` — its cost at
  tens of millions of rows must be measured and, if needed, amortized.
- [ ] Actual HunyuanOCR preprocessing cost per sample (unknown until the
  adapter exists with verified APIs and installed dependencies).
- [ ] Shard size tuning on real data (approximate-bytes accounting in
  SIZE_AWARE mode uses `file_size` when present, else serialized row
  length — real image sizes will shift the optimum).

## Memory

- [ ] Peak RSS under real image payloads (fixtures are tiny 4×4 PNGs;
  decoded real pages are 100–1000× larger and bounded only by the
  transform/batch layer, not tested here).
