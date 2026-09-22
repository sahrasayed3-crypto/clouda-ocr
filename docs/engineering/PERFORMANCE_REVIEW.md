# Performance Review — Current Session (2026-09-21)

No expensive workloads were run (per session rules). CPU/local-code review
only. No optimization was applied this session — nothing met the
real-bottleneck + safe-change + demonstrated-improvement bar.

## Observations (potential, none actionable enough to change now)

| ID | Location | Observation | Why not changed |
|---|---|---|---|
| P1 | `worker_api.py:533` `_validate_docx_file` | reads the whole ≤100 MiB file to inspect 2 magic bytes | trivial fix but touches a hot validation path mid-release-prep; noted as H7 hardening |
| P2 | `conversion_service.py:436-444` | budget lock held across a SQLite round-trip (`daily_cost()`) in the local path — contended DB would serialize all page workers | local path is not a production dispatch mode (hard-rejected at `worker_api.py:1116`); worker path already correct |
| P3 | `pdfword/job_queue.py:27` `JobQueue._jobs` | unbounded dict growth | latent: `submit()` has no production callers |
| P4 | `benchmarks/scripts/run_local_benchmark.py` | full fixture regeneration per run | benchmark-only, intentional determinism trade |
| P5 | `clouda_data` scans (`rglob`-based size/scan loops) | can pick up stale `*.tmp` siblings (pre-fix sites, U9) | correctness-adjacent rather than performance; batch with U9 cleanup |
| P6 | `evaluate`/`quality` pipelines | repeated full-manifest JSON parses across stages | each stage re-reads for isolation/resumability; restructuring is a design change, not a safe local optimization |

## Verified non-issues

- Page OCR pipeline is sequential by design (`max_parallel_pages` removed;
  `ocr_pipeline.py:236,251`) — the checkpoint/progress single-writer
  assumption this session's fixes rely on.
- SQLite access opens short-lived connections with WAL + 30 s busy_timeout
  (`database.py:108-116,112-114`); no long-lived connection contention found.
- Results store / key router use indexed sqlite access and per-run locks;
  no O(n²) scans found in the hot ingestion paths.
- Streaming everywhere on the network paths (uploads, downloads, backup
  extraction) with bounded buffers.
