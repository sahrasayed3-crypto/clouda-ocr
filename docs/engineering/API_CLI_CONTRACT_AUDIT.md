# API / CLI Contract Audit — Current Session (2026-09-21)

## FastAPI worker API (`pdfword/worker_api.py`, ~60 routes)

- **Input validation**: strict job-id regex `^[A-Za-z0-9_-]{1,64}$`
  (`:57`) returning 400 on malformed ids; Pydantic models on bodies; worker
  identifiers/provider enums validated (422 on unsafe values, tested);
  result upload validates claim token, DOCX magic+zip integrity, size caps,
  metadata types (400 before any state or temp file is touched — regression-
  tested).
- **Error semantics**: 404 not-found, 403 containment/symlink escapes,
  409 state-machine conflicts (duplicate claim, cancel resurrection, final
  states), 413 size, 503 recoverable publish failures. Errors are
  deterministic and actionable; worker-facing failure messages sanitized to
  `[redacted]`.
- **JSON stability**: health endpoints have pinned payloads (regression-
  tested for exact equality and absence of the API key); document payloads
  exclude storage paths (`assert_safe_document_payload` in tests).
- **Contract versioning**: no explicit version prefix on routes (internal
  single-deployment API); the worker job protocol is exercised end-to-end by
  `tests/test_distributed_worker.py`.
- **Auth**: every mutating route carries CSRF/session/admin/internal-key
  guards (verified in the security audit); internal endpoints fail closed
  without `WORKER_API_KEY`.

## Console CLIs

- `clouda-data` (`clouda_data/pipeline/cli.py`): argparse hub; JSON-report
  emitting subcommands; doctor subcommand for stage readiness. Quality gate
  CLI (`clouda-quality`): documented exit contract 0 (pass/warnings-clean) /
  1 (fail) / 2 (error) — enforced in `quality/cli.py:241-276` and tested;
  this session added the `revalidate_derived` fail-closed backstop to
  `clean-manifest` (violations now raise instead of silently passing).
- `clouda-training`: `validate-config` exit 2 on invalid config (tested);
  plan/export/runs/resume/checkpoints; hunyuan sub-CLI wired, qwen
  library-only (documented gap — TRAINING_READINESS.md).
- `clouda-lab`: loopback-only serve plus read-only analysis subcommands.

## Environment variables

Centralized in `settings.py` / `StorageRoots.from_env()` /
`clouda_contracts/storage.py`; docs (.env.example) match the double-gated dev
defaults. Rate-limiter multi-worker limitation is loudly logged at startup.

## Findings

| ID | Sev | Finding |
|---|---|---|
| A1 | P3 | `Guest` result token travels as a URL query parameter (`worker_api.py:1519`) — single-use and unlogged, but header/body placement would be cleaner |
| A2 | P3 | Legacy `clouda_training/planner.py` `plan` command (`training_enabled=False` hardcoded) coexists with the new planner — two "plan" surfaces; consider deprecating the legacy path in docs |
| A3 | P3 | `retry`/`cancel` semantics changed this session (F1/F2): retry now re-dispatches synchronously; `start_job` refuses cancelled jobs with a dedicated 409 detail. Any external client scripting on the old behavior should treat 409 "Job was cancelled by its owner" as terminal pending explicit retry |

No contract violations found against the documented behaviors; exit codes and
error payloads match their tests.
