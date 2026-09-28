# Clouda Lab Audit — Current Session (2026-09-21)

Scope: `clouda_lab/` CLI + loopback dashboard + analysis services. No servers
were started publicly during this session.

## Network posture (verified)

- Bind: CLI argparse type rejects non-loopback `--host` (`clouda_lab/cli.py:30-38`);
  `serve` forces `local_only=True` (`cli.py:49-51`).
- Defense in depth: `require_loopback` checks the **client IP** as a global
  dependency (`dashboard/app.py:101`, `security.py:94-99`), so even a
  misconfigured `0.0.0.0` bind 403s remote clients.
- Debug exposure: `docs_url=None, redoc_url=None, openapi_url=None`
  (`dashboard/app.py:96-102`); strict CSP headers (`app.py:33-37`).
- No CORSMiddleware anywhere.

## Task storage & services (verified)

- `dashboard/tasks.py`: per-task atomic JSON writes under an `RLock`;
  worker-thread exceptions recorded as FAILED (never swallowed); orphan
  recovery on restart fails unfinished tasks. Minor: `_futures` dict never
  pruned (memory-only leak).
- `confirmations.py:71-91`: single-use confirmation consume is
  lock-protected.
- `models.py`: model-asset state file read+whole-write **without a lock**
  (`models.py:99-101,281,327,367`) — concurrent dashboard register/remove can
  lose updates (P3, SUSPECTED; single-user loopback tool).
- `io.py`: `export_json/jsonl/csv` are direct non-atomic writes despite the
  docstring claim (`io.py:146-181`); loaders are strict (no tolerance).
  EvaluationService is advertised as share-across-requests
  (`evaluation_service.py:37`), so two concurrent exports to the same target
  can interleave (P3). Same class as the fixed pdfword issues — candidate for
  the shared atomic-write helper.
- `selection_history.py:63-64`: single-`write()` JSONL appends — atomic per
  write on POSIX, weaker guarantee on Windows. Note only.

## Input handling (verified)

- IDs validated by `safe_identifier` (`security.py:13-26`) before any
  filesystem/registry use; uploads bounded (10 MiB / 25 pages, contract-
  tested); responses pass `browser_safe` redaction (secrets + absolute paths)
  before any browser-facing payload (`security.py:15-80`).
- Dashboard static JS builds DOM exclusively via `textContent`
  (`static/app.js:32-35`) — no innerHTML/document.write injection surface.

## Findings

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| L1 | Medium (SUSPECTED) | `/api/lab/session` returns the action token unauthenticated to any loopback request with no Host/Origin validation — a DNS-rebinding page in the host's browser could read the token and drive mutating endpoints (dataset removals, training starts, resume, doctor runs). Loopback dev-tool context; worker_api already has TrustedHostMiddleware, the Lab app does not | HARDENING: add TrustedHostMiddleware (loopback names) or same-origin check; low practical risk |
| L2 | P3 | `models.py` state file: unlocked read-modify-write | Add the same RLock pattern used by tasks.py |
| L3 | P3 | `io.py` exports non-atomic; docstring says "atomically-ish" | Route through a shared atomic writer |
| L4 | P3 | `_futures`/`_jobs` registries never pruned | Bound or clean up on completion |
