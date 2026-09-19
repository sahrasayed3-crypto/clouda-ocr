# Testing

Clouda PDF uses `pytest` with deterministic, copyright-free fixtures under `tests/fixtures`.

## Coverage

The readiness tests cover page analysis, categorical digital-text trust gating, typed direct-extraction authorization, post-extraction digest mismatch handling, scanned-page routing, structurally evidenced blank and near-blank classification, review placeholders, mixed page order, corrupt and empty PDFs, non-PDF handling, DOCX validity, OCR result schemas, engine registration, mock engines, Arabic/RTL fixtures, storage, API, Lab upload bounds, and Streamlit smoke tests.

Run the suite and coverage report:

```powershell
.\.venv311\Scripts\python.exe -m pytest --cov=pdfword --cov-report=term-missing
```

The report intentionally does not treat `pending_ocr_model` as OCR success or as a final processing failure.

OCR self-review tests use deterministic local fakes. They cover optional native
confidence, exact render-bound crop geometry, conservative replacement
provenance, review boundaries, and the opt-in CUDA plumbing smoke. The smoke
skips unless `CLOUDA_CUDA_SMOKE=1`; it does not establish OCR quality or future
model fit.

Routing tests assert the canonical sequence `Page Analyzer -> Trusted Digital
Text Gate -> Page Decision Engine`. A legitimate short-text page must not be
classified as `near_blank` from character count alone. Direct extraction uses
`confidence=None`; trust is categorical and backed by reason codes. If the
post-extraction text digest differs from the trusted context, the page must be
`review_required` with the `manual_review` next path, must retain a visible placeholder and page boundary, and must
not emit the extracted text.

Lab contract tests require explicit action-token-protected submission, enforce
the 10 MiB / 25-page bounds, and verify that the response and UI contain no
uploaded text, local paths, hashes, or user-facing accuracy, confidence, or
quality percentages.

## Test suite size

The suite currently contains **183 test files** with roughly **1,900 test
functions** (static count of `def test_` definitions as of 2026-09-20). Run
the suite on the current checkout for live pass counts rather than quoting a
snapshot; CI reports the authoritative result for each commit.

## Historical verified result

On 2026-07-14, an earlier, much smaller suite completed with **145 passed**
and **81%** overall `pdfword` coverage (`80.89%` measured by pytest-cov). This
is a dated historical snapshot of the digital-text conversion subsystem, not
the current suite. Coverage came from meaningful tests for invalid/empty
inputs, engine metadata, DOCX generation, cleanup safety, local key-store
behavior, CLI startup guards, OpenRouter/provider error handling,
conversion-service recovery, settings, storage, and worker guards.

## Quality checks

```powershell
.\.venv311\Scripts\python.exe -m ruff check .
.\.venv311\Scripts\python.exe -m black --check .
.\.venv311\Scripts\python.exe -m mypy .
.\.venv311\Scripts\python.exe -m compileall -q app.py pdfword scripts tests
```

Ruff, Black, and full `mypy .` pass across the checked project. `mypy.ini` excludes only local runtime/generated directories such as virtual environments, Poppler bundles, runtime data, conversions, logs, backups, and generated sample outputs.

## Fixture policy

`tests/fixtures/generate_fixtures.py` programmatically creates the small fixtures. They contain no third-party documents or copyrighted text. Regenerate them with:

```powershell
.\.venv311\Scripts\python.exe tests\fixtures\generate_fixtures.py
```
