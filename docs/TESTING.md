# Testing

Clouda PDF uses `pytest` with deterministic, copyright-free fixtures under `tests/fixtures`.

## Coverage

The readiness tests cover digital-text extraction, scanned-page routing, blank and near-blank classification, mixed page order, corrupt and empty PDFs, non-PDF handling, DOCX validity, OCR result schemas, engine registration, mock engines, Arabic/RTL fixtures, storage, API, and Streamlit smoke tests.

Run the suite and coverage report:

```powershell
.\.venv311\Scripts\python.exe -m pytest --cov=pdfword --cov-report=term-missing
```

The report intentionally does not treat `pending_ocr_model` as OCR success or as a final processing failure.

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
