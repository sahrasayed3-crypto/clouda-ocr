# Packaging Audit — Current Session (2026-09-21)

Built locally into a scratch directory (`python -m build`):
`clouda_ocr-0.2.1-py3-none-any.whl` (452 files, 2.0 MiB) and
`clouda_ocr-0.2.1.tar.gz` (571 files). Note: name/version reflect the
concurrent v0.2.1 brand migration in the working tree (see session log).

## Wheel contents — clean

- Top-level packages exactly match `pyproject` include list: `pdfword`,
  `clouda_contracts`, `clouda_data`, `clouda_lab`, `clouda_models`,
  `clouda_training` (+ dist-info). No tests, no `tools/`, no vendored
  poppler/tesseract binaries.
- `LICENSE` + `NOTICE` included; metadata `License-Expression: Apache-2.0`,
  `Requires-Python: <3.12,>=3.11`, deps resolve (incl. the raised
  `pypdf>=6.16.1`).
- Package data present: 6 OFL fonts (~2 MiB total, largest 600 KB),
  `clouda_lab/dashboard/static/{index.html,app.js,styles.css}`,
  `pdfword/streamlit_firebase_auth/index.html`, quality/resources JSON/YAML.
- **No** datasets, model weights, `_state/`, logs, backups, outputs, or
  absolute local paths (scanned every text file for `F:\PROJECT` /
  `C:\Users\Ahmed`). No secrets (name-based scan hits are false positives:
  `key_router/credential_refs.py` is a source module).
- Console scripts unchanged: `clouda-data`, `clouda-lab`, `clouda-training`,
  `clouda-quality` (brand migration correctly left them alone).

## sdist contents — clean

- Includes tests (good for downstream packagers), excludes docs/ and all
  private/runtime trees (`.private`, `_state`, outputs, backups, conversions:
  name-scan only hit two false positives, `run_state.py` /
  `output_validation.py`).
- Name-based scan for env/secret/credential files: none.

## Installation

- Not installed into a fresh venv in this session (Windows venv churn
  avoided during an active session); the wheel is a pure-py3 wheel with
  standard metadata, and the repo's own release plan
  (`docs/RELEASE_V0.2.1_PLAN.md`) carries the fresh-venv install checklist.
- Existing dev environment: package installed editable
  (`clouda_pdf.egg-info` is stale relative to the rename — it will refresh on
  the next editable/pip install; harmless).

## Findings

| ID | Sev | Finding |
|---|---|---|
| PKG1 | P3 | `pyproject` `[project] name` changed to `clouda-ocr` mid-session (brand migration). The `dist/` directory still contains older `clouda_pdf-*` artifacts from the previous build — stale but harmless; refresh on next release build. |
| PKG2 | P3 | setuptools advisories (PYSEC-2025-49, PYSEC-2026-3447) exist in the dev venv (78.1.0); the build backend itself requires `setuptools>=83,<84` (fixed line), so built artifacts are unaffected. Upgrade the dev venv at convenience. |
