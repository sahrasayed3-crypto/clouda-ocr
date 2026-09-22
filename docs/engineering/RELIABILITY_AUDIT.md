# Reliability & Failure-Injection Review — Current Session (2026-09-21)

Evidence base: existing failure-path tests exercised during this session
plus targeted review of crash windows. No real user data was touched; all
injection scenarios reviewed use isolated fixtures.

## Crash windows — status after this session's fixes

| Scenario | Before | After |
|---|---|---|
| Kill during final DOCX write | truncated deliverable published, row marked completed (C2) | atomic publish: user sees old file or new file, never partial (`atomic_write_bytes`) |
| Kill during backup zip write | truncated `clouda_backup_*.zip` visible to retention/restore (C1) | staged `.part` + fsync + rename; restore/validate never sees a partial archive |
| Kill during checkpoint save | fixed `.tmp` interleaving across writers; silent empty reload (C3) | unique staging + fsync + rename; malformed JSON still tolerated as documented |
| Kill during lifecycle audit write | partial/overwritten audit for a destructive op (C9) | unique name + atomic write |
| Worker crash mid-task (Windows SimpleWorker) | row stuck `processing` until stale recovery; stale recovery left `rq_job_id` set so maintenance could not requeue (concurrency F2/U6) | user retry path fixed (F2); stale-recovery-on-start self-heals when RQ requeues; U6 residual documented |
| Redis outage at dispatch | deferred (logged), maintenance re-dispatches | unchanged + enqueue lock loser degrades to deferred instead of double-enqueueing (F3) |
| Redis outage at cancel/heartbeat | cancel row-only; worker kept running (F1) | cancel also attempts RQ stop; outage logged as `distributed_cancel_deferred`, user sees cancelled row; worker's next start now 409s (F1) |
| Concurrent duplicate result upload | safe before (token fencing) | still safe (regression tests green) |
| Concurrent guest token consume | both could win (deferred-tx race F4) | exactly one wins (deterministic, tested) |
| Corrupt checkpoint / corrupt result file | tolerant (documented) | unchanged; corrupt-result stale recovery tested |
| Malformed metadata on result upload | safe before | still safe (`test_result_upload_rejects_non_numeric_score_metadata`) |
| os.replace failure mid-promotion | rollback + retry design (existing) | verified still green after conversion-service changes |

## Remaining failure-path gaps (documented, not injected here)

1. **U6**: `abandon_stale_processing` leaves `rq_job_id` set — single-worker
   deployments depend on RQ's own startup maintenance to requeue orphaned
   started jobs. Recommend clearing the id in the same UPDATE (small follow-up).
2. **S1/U9**: `provider_router` / `ai_model_router` / `self_learning` stores
   silently reset to defaults on corrupt JSON (`_safe_load`). If these ever
   gain production callers, corrupt-state must quarantine + log, not reset.
3. **D1** (dead text-near-dup tier): the quality gate's FAIL surface is the
   reliability boundary for training data; wiring it in (with golden-run
   comparison) is the highest-value reliability fix in the data plane.
4. Worker download-before-heartbeat window (worker_tasks.py:46-48): a very
   slow input download (> stale threshold) can in principle be abandoned
   while running; mitigated by the DB claim machine, but consider starting
   the heartbeat before `download_input`.

## What was verified green under injection

`tests/test_distributed_worker.py` (503 rollback + recovery paths),
`tests/test_atomic_publication.py` (new), `tests/test_job_lifecycle_wiring.py`
(new), `tests/security/test_adversarial_hardening.py`,
`tests/test_runtime_features.py` (backup restore failure rollback), quality
gate suite (249 tests) — all passing after the fixes.
