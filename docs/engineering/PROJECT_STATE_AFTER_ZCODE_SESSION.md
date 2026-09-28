# Project State After ZCode Session — 2026-09-21 (stopped early at user request)

## Git state

- Branch `main`, HEAD at session start `18994155b2c26f1b082e444356ecd58e9a467eee` ("chore: prepare metadata for v0.2.0 release"), tag `v0.2.0`.
- Working tree now carries TWO change sets (see `ZCODE_DEEP_SESSION.md`):
  1. **A concurrent v0.2.1 brand migration this session did NOT author** (README/CITATION.cff/.zenodo.json/CHANGELOG/app.py/doctor hints/launch scripts/deploy service, new `docs/BRAND_MIGRATION.md`, `docs/RELEASE_V0.2.1_PLAN.md`, `docs/REMOTE_METADATA_RECONCILIATION.md`, pyproject name/version). Left untouched.
  2. **This session's audit + fixes** (below). No commit, no push, no tag, no release, no Zenodo/benchmark/website modification.

## This session's files changed (code)

- `pdfword/worker_api.py` — F1/F2: cancel→RQ cancel; retry clears `rq_job_id` + re-dispatches; `start_job` refuses cancelled jobs (409)
- `pdfword/job_queue.py` — F3: Redis SETNX dispatch lock in `enqueue`
- `pdfword/database.py` — F4: `transaction(immediate=True)` (BEGIN IMMEDIATE) for `transition_conversion` + `consume_guest_result_token`
- `pdfword/atomic.py` (new) — atomic_write_bytes/text (unique staging + fsync + os.replace)
- `pdfword/backup.py` — F5: staged+fsynced zip publish, same-second collision suffix
- `pdfword/conversion_service.py` — F6/F8: DOCX + progress.json atomic publishes
- `pdfword/checkpoints.py` — F7: atomic checkpoint save
- `pdfword/storage.py` — F11: Windows reserved device names in `safe_component`
- `pdfword/tenant_storage.py` — F12: fixed dead symlink guard
- `clouda_data/lifecycle.py` — F9: unique+atomic audit report
- `clouda_data/quality/cli.py` — F10 + SP1: `revalidate_derived` wired into clean-manifest; clean exit codes for revalidation failure
- `pyproject.toml` / `requirements-base.txt` — pypdf floor `>=6.16.1` (CVE-2026-84309/84310/84311; only the pypdf lines — the rest of the file was the concurrent migration)
- venv: pypdf upgraded 6.15.0 → 6.16.1

## Tests

- `tests/test_job_lifecycle_wiring.py` (new, 8 tests), `tests/test_atomic_publication.py` (new, 5 tests) — all passing.
- Targeted suites re-run green after each fix: distributed worker, tenant isolation, maintenance, runtime features, quality (249), atomic/wiring, backup/restore, conversion paths, security adversarial.
- Full-suite re-run was started but **stopped mid-run at user request** — last complete full-suite result: baseline 2141 passed / 16 skipped (pre-fix) plus all targeted post-fix suites green. One full post-fix pytest pass is the remaining recommended step.

## Documentation created (docs/engineering/)

SESSION_BASELINE.md, ZCODE_DEEP_SESSION.md, ARCHITECTURE_MAP.md, CODE_AUDIT_CURRENT.md, TEST_GAP_ANALYSIS.md, SECURITY_AUDIT_CURRENT.md, DATA_PIPELINE_AUDIT.md, LICENSE_PROVENANCE_MATRIX.md, BENCHMARK_STATE_CURRENT.md, TRAINING_READINESS.md, PACKAGING_AUDIT.md, DOCUMENTATION_AUDIT.md, API_CLI_CONTRACT_AUDIT.md, CLOUDA_LAB_AUDIT.md, RELIABILITY_AUDIT.md, PERFORMANCE_REVIEW.md. (FINAL_VERIFICATION.md / SECOND_PASS_FINDINGS.md folded into this file and CODE_AUDIT_CURRENT.md due to early stop.)

## Top unresolved items (owner decisions, evidence in CODE_AUDIT_CURRENT.md)

U1 text near-dup tier is dead code (P2, quality gate), U2 FAIL still writes "clean" manifest (P2), U3 download authenticity conditional on registry digests (P2), U4/U5 CER/normalization divergence (P2), U6 stale-recovery leaves rq_job_id (P2).

## Security status

pip-audit pypdf CVEs fixed via floor bump; setuptools advisories are build-tooling-only (build backend requires >=83). Bandit high/medium findings triaged as false positive / hardening. No secrets in tracked files. Full detail: SECURITY_AUDIT_CURRENT.md.
