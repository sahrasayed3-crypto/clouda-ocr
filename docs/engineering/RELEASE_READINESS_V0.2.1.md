# Release Readiness — Clouda OCR v0.2.1

Date: 2026-09-22 (continuation session). Prior session: see
`docs/engineering/PROJECT_STATE_AFTER_ZCODE_SESSION.md` and the
`docs/engineering/*` audit set.

## 1. Starting repository state

- Branch `main`, HEAD `18994155b2c26f1b082e444356ecd58e9a467eee`
  ("chore: prepare metadata for v0.2.0 release"), tag `v0.2.0`.
- Working tree: 37 dirty/untracked paths at start — the prior session's
  fixes/audits plus the concurrent (not authored by the sessions)
  v0.2.1 brand migration (README/CITATION/.zenodo/CHANGELOG/app.py,
  `pyproject` name `clouda-ocr` v0.2.1, launch scripts, brand docs).
- Nothing was reset, stashed, rebased, or rewritten at any point.

## 2. Changes made in this session

### A. Near-duplicate text leakage — integrated into the gate (was dead code)

- `clouda_data/quality/gate.py` Stage 4 now runs
  `text_dup.classify_text_pairs` (MinHash 128 perms, banded LSH 16×8,
  exact 4-gram Jaccard over the shared `DEDUPE_TEXT_POLICY` — NFKC,
  diacritic/tatweel/alef/ya folding, digits preserved) over every
  sample's ground-truth text. Deterministic (blake2b-derived parameters,
  sorted iteration). Skipped under `--no-near-duplicates` like the image
  tier; stage timing recorded as `text_near_s` (observability only).
- `clouda_data/quality/leakage.py` gains check **L3b
  `near_text_cross_split`** (`IssueCode.LEAK_NEAR_TEXT`, new constant in
  `models.py`): a confirmed near-text pair spanning TRAIN vs
  EVAL/PROTECTED partitions is CRITICAL → gate verdict FAIL, and the
  existing keep/exclude policy automatically excludes the TRAIN-side
  members (it keys off any critical `LEAK_*` finding). Same-partition
  pairs surface as `DUP_TEXT_NEAR` warnings (dedup signal, non-failing).
- Arabic safety: the dedupe policy folds diacritics/tatweel/alef/ya but
  preserves digits and does not collapse unrelated texts — the 0.85
  Jaccard threshold on 4-grams only fires on near-identical pages.
  Verified by tests: distinct Arabic texts pass; diacritic/tatweel/
  whitespace variants crossing the boundary fail.
- Tests (`tests/quality/test_leakage_integration.py::TestTextNearDupGate`,
  8 tests): exact duplicates across train/eval FAIL; Arabic diacritic
  variants FAIL; tatweel/whitespace variants FAIL; train-vs-holdout
  FAIL; legitimate distinct texts do NOT trigger (no LEAK/DUP codes);
  same-partition near-text is warning-only; `--no-near-duplicates`
  disables the tier; timings field present. **The gate provably runs the
  tier** (a two-sample train/validation manifest with identical Arabic
  text now ends FAIL with `LEAK_NEAR_TEXT`).

### B. FAIL verdict can no longer produce a "clean" manifest

- `clouda_data/quality/derived.py`: `write_clean_manifest` now refuses a
  FAIL-verdict run with `FailedVerdictRefusedError` unless the caller
  explicitly passes `allow_failed_verdict=True` (forensic/rejected
  artifact). Diagnostics are unaffected: exclusion report, leakage
  findings, severity counts all remain in the scan payload; the header
  still records `verdict` for allowed forensic artifacts.
- `clouda_data/quality/cli.py` (`clean-manifest`): on refusal prints a
  JSON payload `{clean_manifest: "not_written", verdict, severity_counts,
  reason_codes, excluded, quarantined}` and returns exit 1 (the
  documented fail exit). Revalidation errors (`DerivedManifestValidation
  Error`) map to exit 1 instead of a traceback.
- `clouda_lab/dashboard/catalog.py::derive`: FAIL-verdict datasets raise
  a clear `ValueError` (dashboard converts to a clean 4xx) before any
  write.
- Tests: `TestFailVerdictCleanManifestGuard` (3) — refusal leaves no
  artifact on disk; forensic opt-in writes with `verdict: FAIL`; CLI
  returns 1 with `not_written` payload.

### C. Dataset authenticity — explicit digest trust model

- Trust chain traced (`docs/engineering/DATA_PIPELINE_AUDIT.md` D3 and
  below): registry JSON (reviewed in-repo) → HTTPS + SSRF-guarded fetch
  → conditional sha256 check → self-computed digest recorded. Previously
  an asset without a registry `sha256` was "verified" against nothing.
- `clouda_data/datasets/downloader.py`:
  - `DownloadedFile.digest_source` records `registry_pinned` (bytes
    verified against a registry-pinned digest: integrity **and**
    publisher authenticity relative to the reviewed registry) or
    `self_reported` (integrity only).
  - Sample downloads **fail closed** on unpinned assets: the asset is
    skipped with an `unpinned_digest` issue unless
    `CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS=1` is explicitly set.
  - rasam batch downloads are labeled `self_reported` honestly.
- Residual trust assumption (documented, not eliminated): a pinned
  digest is only as trustworthy as the reviewed registry commit; there
  is no detached publisher signature infrastructure. Pinned digests in
  the shipped registry: currently **0 of 4 sample assets** — operators
  must either pin digests per source (recommended) or consciously opt in
  to self-reported digests. Manifests now make the difference visible.
- Tests: `TestDownloadDigestTrustModel` (4) — unpinned fails closed;
  opt-in records `self_reported`; pinned records `registry_pinned`;
  mismatched pin is rejected (issue recorded, no file kept).

### D. Hygiene fixes found during verification

- `pdfword/backup.py`: mypy-strict handling of the zip's Optional file
  handle around the new fsync.
- `tests/test_maintenance.py`: pre-existing mypy return-annotation error
  fixed (release checks run `mypy .`).
- Dev venv setuptools upgraded 78.1.0 → 83.x (clears PYSEC-2025-49 /
  PYSEC-2026-3447; matches the build backend requirement). Dev venv
  pypdf already at 6.16.1+ (wheel installs resolve 6.19.0).
- Session-touched files black-formatted (5 files); no repo-wide churn.

## 3. Tests added

- `tests/quality/test_leakage_integration.py` — 15 tests (A: 8, B: 3,
  C: 4).
- Updated: `tests/data_foundation/unit/test_dataset_downloader.py` (3
  env mocks declare the new unpinned opt-in; plus the new fail-closed
  tests), `tests/quality/adversarial/test_edge_manifests.py` (asserts
  the FAIL refusal; keeps the unicode round-trip via forensic opt-in).

## 4. Verification commands and results

| Check | Command | Result |
|---|---|---|
| Ruff | `python -m ruff check .` | All checks passed |
| Black | `python -m black --check .` | 635 files unchanged (0 need reformatting) |
| mypy | `python -m mypy .` | Success: no issues in 635 files |
| Bandit | `python -m bandit -q -r <packages> -c pyproject.toml` | 1 High + 1 Medium, both previously triaged: B613 `_raqm/corpus.py` intentional Arabic bidi chars (false positive); B614 digest-guarded `torch.load` (hardening note, callers always pass `expected_sha256`) |
| pip-audit | `python -m pip_audit -f json` | **0 known vulnerabilities** (pypdf floor 6.16.1 from prior session; dev venv setuptools upgraded) |
| Build | `python -m build --outdir <tmp>` | `clouda_ocr-0.2.1-py3-none-any.whl` (2.09 MB) + `clouda_ocr-0.2.1.tar.gz` (2.01 MB) |
| Clean install (wheel) | fresh `python -m venv` + `pip install <wheel>` | exit 0 |
| Clean install (sdist) | `pip install --force-reinstall <tar.gz>` | exit 0 (build backend resolves setuptools>=83) |
| Import smoke | import all 6 packages incl. quality gate/derived/downloader; console entry points resolved | OK; `clouda-ocr 0.2.1`; scripts `clouda-data/-lab/-quality/-training` present; session changes present in wheel |
| CLI smoke | `clouda-quality --help`, `clean-manifest --help`, `clouda-data --help`, `clouda-training validate-config configs/training/mock-experiment.yaml` | OK; config valid (same config hash as dev env) |
| Runtime smoke (fresh venv) | manifest → gate → assert FAIL + `LEAK_NEAR_TEXT` → assert clean-manifest refusal | SMOKE PASSED |
| Full pytest | `python -m pytest -q` | **2166 passed, 16 skipped, 0 failed** in ~759 s (complete run, no exclusions; up from 2141 baseline + 15 new + updated tests) |

## 5. Remaining risks (non-blocking, documented)

1. Residual authenticity trust: registry digests are review-trusted, not
   signature-verified (Section 2C). Recommend pinning digests for all
   registry assets before the next data-collection campaign.
2. Bandit B614 hardening: consider `weights_only=True` for
   `checkpoint_torch.load_torch_state` in a future change.
3. The gate's text near-dup tier is new: on real corpora expect new
   `DUP_TEXT_NEAR` warnings (non-failing) and possible new FAILs from
   previously-invisible cross-split variants — run the gate on the
   current canonical corpora before consuming derived manifests.
4. `docs/TESTING.md` "latest verified result" was refreshed in the prior
   session to 2141; now 2166 — doc updated by this session.
5. Unrelated-but-open items from the prior audit set remain tracked in
   `docs/engineering/CODE_AUDIT_CURRENT.md` (U6 stale-recovery requeue,
   U9 shared atomic-writer promotion, Lab models.py lock).

## 6. Release readiness verdict

**READY FOR v0.2.1 RELEASE** — with the standing caveat that the release
itself (tag/publish/Zenodo) is out of scope for this session and remains
unexecuted.
