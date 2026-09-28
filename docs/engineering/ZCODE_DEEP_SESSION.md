# ZCode Deep Session Log

Continuous log. Factual entries only.

## Phase 0 — Preflight (complete)

- `git status`: clean, `main` @ `1899415`, tag `v0.2.0`, remote origin = clouda-ocr.
- Repo: ~74k LOC production (pdfword 22.5k, clouda_data 30.1k, clouda_training 11.7k, clouda_lab 8.5k, contracts 1.3k, models 0.2k), 185 test files.
- venv `.venv311` (Python 3.11.9, pytest 9.1.1) is the session interpreter.
- Baseline recorded in `SESSION_BASELINE.md`.

## Phase 1 — Test suite baseline (complete)

- Full pytest: **2141 passed, 16 skipped, 0 failed** in 728.9s.
- ruff: all checks passed.
- mypy: 1 error in `tests/test_maintenance.py:55` (return-value annotation; test file only, mypy exit 0 otherwise clean over 631 files).
- bandit (source packages, repo config): 0 High actionable / 0 Medium actionable. Findings reviewed:
  - B613 bidi control chars in `clouda_data/factory/render/_raqm/corpus.py:59` — intentional (RLM/LRM corpus constants for Arabic bidi). False positive.
  - B614 `torch.load(weights_only=False)` in `clouda_training/runtime/checkpoint_torch.py:47` — digest-guarded when `expected_sha256` provided; need to verify all callers pass digest (see Phase 3).
  - 62 Low findings (B110 try/except-pass etc.) — to triage in security audit.

## Concurrent working-tree activity (important)

Mid-session, the working tree changed from clean to carrying a **v0.2.1 brand
migration change set that this session did NOT author** (baseline at HEAD
`1899415` was clean): README/CITATION.cff/.zenodo.json/CHANGELOG branding,
`pyproject.toml` name `clouda-pdf`→`clouda-ocr` + version `0.2.1`, app.py UI
strings, launch scripts, `deploy/linux/clouda.service`, doctor hints, and new
untracked `docs/BRAND_MIGRATION.md`, `docs/RELEASE_V0.2.1_PLAN.md`,
`docs/REMOTE_METADATA_RECONCILIATION.md`. Per the baseline rules these edits
were left untouched. Session edits compose cleanly (in `pyproject.toml` /
`requirements-base.txt` only the pypdf floor lines were changed by this
session on top of the migrated file). Files changed by THIS session are
listed in `PROJECT_STATE_AFTER_ZCODE_SESSION.md`.

## Phase 4 — Fixes (complete)

Fixes F1–F12 implemented with 21 regression tests in
`tests/test_job_lifecycle_wiring.py` and `tests/test_atomic_publication.py`
(all passing). Full detail in `CODE_AUDIT_CURRENT.md`.

Dependency security: pip-audit found pypdf 6.15.0 with 3 DoS-class CVEs
(fixed upstream in 6.16.1; constraints/py311.txt already pinned 6.16.1 but
the venv and the minimum floors were stale). Raised `pypdf>=6.16.1` in
`pyproject.toml` + `requirements-base.txt` and upgraded the venv; PDF-path
tests pass. setuptools advisories (PYSEC-2025-49, PYSEC-2026-3447) are
build-tooling only; pyproject build requires setuptools>=83,<84 so builds
fetch a fixed version regardless of the dev venv's 78.1.0.

## Phase 3 — Deep static audit (complete)

- Filesystem/persistence, concurrency/state, input/security, data-pipeline
  audits complete. Consolidated confirmed/fixed/unresolved findings in
  `CODE_AUDIT_CURRENT.md` (F1–F12 fixed, U1–U11 documented as needing owner
  decisions, plus verified-safe negative checks).



## Continuation session (2026-09-22) — v0.2.1 release readiness

- Task A: text near-dup tier wired into `quality/gate.py` Stage 4; new L3b
  `LEAK_NEAR_TEXT` critical check in `leakage.py` + `IssueCode.LEAK_NEAR_TEXT`;
  same-partition pairs emit `DUP_TEXT_NEAR` warnings; `text_near_s` timing.
- Task B: `write_clean_manifest` refuses FAIL verdicts
  (`FailedVerdictRefusedError`, forensic opt-in `allow_failed_verdict=True`);
  `clean-manifest` CLI returns 1 with `clean_manifest: not_written`; dashboard
  `catalog.derive` raises a clean ValueError.
- Task C: digest trust model in `datasets/downloader.py` — `digest_source`
  (`registry_pinned`/`self_reported`), unpinned sample assets fail closed
  unless `CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS=1`.
- Tests: `tests/quality/test_leakage_integration.py` (15); downloader tests
  updated for the opt-in; `test_edge_manifests` asserts the refusal; dashboard
  fixture made gate-clean so the derive flow stays covered.
- Verification: ruff/black/mypy all clean; bandit unchanged (2 triaged);
  pip-audit 0 vulns (setuptools 83.x in dev venv); wheel+sdist built
  (`clouda_ocr-0.2.1`), clean install from BOTH wheel and sdist in a fresh
  venv, import/CLI/runtime smoke passed. Full pytest re-run for the record.
- Report: `docs/engineering/RELEASE_READINESS_V0.2.1.md`.
