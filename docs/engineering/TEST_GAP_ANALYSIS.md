# Test Gap Analysis — Current Session (2026-09-21)

Baseline: 2141 passed / 16 skipped (pre-session), suite green. This session
added 21 tests (2 new files). High-value gaps that remain open are listed at
the end.

## Gaps filled this session

| New test | Protects |
|---|---|
| `tests/test_job_lifecycle_wiring.py::test_retry_clears_stale_rq_job_id_and_redispatches` | retry actually re-enqueues (P1: jobs pending forever) |
| `::test_retry_survives_redis_outage_and_maintenance_redispatches` | retry → deferred → maintenance recovery chain |
| `::test_cancel_attempts_to_stop_the_rq_job` | cancel reaches the queue (P1: cancelled jobs kept running) |
| `::test_cancel_survives_redis_outage` | cancel degrades gracefully |
| `::test_worker_start_cannot_resurrect_a_cancelled_job` | user cancel is final against RQ retries |
| `::test_enqueue_dispatch_lock_prevents_duplicate_queue_entries` / `::test_enqueue_dispatch_lock_rejects_concurrent_dispatch` | TOCTOU duplicate execution (P2) |
| `::test_single_use_guest_result_token_is_concurrent_safe` | guest token double-download race (P2, deterministic) |
| `tests/test_atomic_publication.py::test_atomic_write_publishes_content_and_cleans_staging` / `::test_atomic_write_text_utf8` | atomic publication primitive (no staging residue, Arabic UTF-8) |
| `::test_checkpoint_roundtrip_leaves_no_tmp_residue` | checkpoint atomicity + clean job root |
| `::test_same_second_backups_do_not_overwrite_each_other` | backup same-second collision + both archives valid |
| `::test_safe_component_neutralizes_windows_reserved_devices` | CON/NUL/COM1/LPT filenames (Windows) |

Also strengthened indirectly: the quality-gate suite (249 tests) now runs
with the `revalidate_derived` fail-closed backstop active.

## Open gaps (highest value first)

1. **Text near-dup leakage detection** (D1): once `classify_text_pairs` is
   wired into the gate, add golden tests: same text with diacritic/tatweel
   variants across splits must FAIL (currently passes).
2. **FAIL→clean-manifest semantics** (D2): tests that a FAIL verdict either
   refuses the artifact or that ERROR-artifact rows (HASH_MISMATCH,
   IMAGE_DECODE) are excluded.
3. **Stale-recovery requeue** (U6): after `abandon_stale_processing`, a
   maintenance pass must re-dispatch (currently depends on RQ startup
   maintenance).
4. **Concurrent `transition_conversion` claim** (F4): the guest-token test
   covers the BEGIN IMMEDIATE fix indirectly; a direct two-thread
   pending→processing claim test would pin the state machine itself
   (single-attempt races are flaky as negative tests; use N iterations with
   an upper bound).
5. **Windows surface**: reserved-name handling is tested at
   `safe_component` level, not through `TenantStorage.write_upload` with
   `CON.pdf` (integration-level); trailing-dot/space names
   (`"report."`) also untested end-to-end.
6. **Atomic-write interruption**: no fault-injection test kills a writer
   mid-`atomic_write_bytes` to prove the old file survives (the helper's
   try/except covers the API, but a simulated `os.replace` failure asserting
   old-content-intact would pin the contract).
7. **Worker download-before-heartbeat window**: no test for the slow-download
   stale-abandon interaction noted in RELIABILITY_AUDIT.
8. **pypdf CVE regression**: no test asserts the configured pypdf floor
   (constraints pin) meets a known-good version — a cheap metadata test
   could parse constraints/py311.txt and compare against an advisory floor
   constant.
