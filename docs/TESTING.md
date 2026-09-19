# Testing

Clouda PDF uses `pytest` with deterministic, copyright-free fixtures under `tests/fixtures`.

## Coverage

The readiness tests cover page analysis, categorical digital-text trust gating, typed direct-extraction authorization, post-extraction digest mismatch handling, scanned-page routing, structurally evidenced blank and near-blank classification, review placeholders, mixed page order, corrupt and empty PDFs, non-PDF handling, DOCX validity, OCR result schemas, engine registration, mock engines, Arabic/RTL fixtures, storage, API, Lab upload bounds, and Streamlit smoke tests.

Run the suite and coverage report:

```powershell
.\.venv311\Scripts\python.exe -m pytest --cov=pdfword --cov-report=term-missing
```

The report intentionally does not treat `pending_ocr_model` as OCR success or as a final processing failure.

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

## Latest verified result

On 2026-07-14, the full suite completed with **145 passed** and **81%** overall `pdfword` coverage (`80.89%` measured by pytest-cov). Coverage improved through meaningful tests for invalid/empty inputs, engine metadata, DOCX generation, cleanup safety, local key-store behavior, CLI startup guards, OpenRouter/provider error handling, conversion-service recovery, settings, storage, and worker guards.

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
