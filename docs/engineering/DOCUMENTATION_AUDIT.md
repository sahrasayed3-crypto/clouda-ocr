# Documentation Consistency Audit — Current Session (2026-09-21)

Compared tracked documentation against code and release metadata at HEAD
`1899415` + working-tree state (which includes the concurrent v0.2.1 brand
migration).

## Verified ACCURATE

- **README quick-start**: venv + `pip install -r requirements-dev.txt` +
  constrained editable install with extras — matches pyproject extras and
  `constraints/py311.txt`. Run/test/quality command blocks match reality.
- **CLI examples verified by execution**: `clouda_training.cli
  validate-config configs/training/mock-experiment.yaml` → valid + config
  hash (exit 0); `plan`, `dry-run` subcommands exist with those names;
  `clouda_data.pipeline.cli doctor --deep` exists (`pipeline/cli.py:1324,1331`).
- **Benchmark framing**: 177-page v0.1.0 published vs 462-page expansion
  "in progress / paused" is consistent across README, ROADMAP, docs/ROADMAP,
  docs/MODEL_INTEGRATION.md — no inflated claims, explicit "historical
  benchmark-specific result" caveat for the published HunyuanOCR N-CER.
- **Capability caveats**: README/ROADMAP correctly state no trained model
  exists, OCR inference disabled by default, training planning is CPU-only
  mock — matches code (FutureOcrEngine placeholder, disabled legacy trainer,
  BLOCKED real-training path).
- **Release metadata (post brand migration)**: `CITATION.cff` (1.2.0),
  `.zenodo.json`, `pyproject.toml` (clouda-ocr, 0.2.1), `CHANGELOG.md`
  (0.2.1 unreleased entry) are mutually consistent; v0.2.0 correctly
  described as historical under "Clouda PDF"; author/ORCID unchanged.
- **Brand note**: README banner + `docs/BRAND_MIGRATION.md` explain the
  rename; import packages and console script names documented as unchanged —
  verified true in pyproject.
- **DATA_LICENSES.md / NOTICE / THIRD_PARTY_NOTICES.md**: consistent with
  `dataset_catalog/licenses/REVIEW_STATUS.json` fail-closed policy.

## Fixed this session

| Doc | Issue | Action |
|---|---|---|
| `docs/TESTING.md` | "Latest verified result" claimed **145 passed / 81% coverage** from 2026-07-14 — materially stale (actual baseline this session: 2141 passed / 16 skipped) | Refreshed to the current verified result after this session's full verification run (see FINAL_VERIFICATION.md) |

## Remaining observations (P3, left alone deliberately)

1. `docs/TESTING.md` quotes a `--cov=pdfword`-only coverage figure; the repo
   coverage config (`pyproject [tool.coverage.run]`) actually sources five
   packages — the doc's command measures less than the configured scope.
2. README's "External tools" section says do not commit Poppler — the repo
   tree contains `tools/poppler`, `tools/tesseract`, `tools/python` locally
   (untracked? gitignored) — packaging excludes them either way; the README
   wording is about commits, consistent with `.gitignore`.
3. `docs/BRAND_MIGRATION.md` / release plan are new and unreleased; no
   contradictions found in them.
4. ROADMAP still describes the merged-repo milestones accurately; the
   "10-model" expansion plan is not recorded anywhere in-repo (see
   BENCHMARK_STATE_CURRENT.md) — recommend adding it to ROADMAP when the
   expansion resumes.
