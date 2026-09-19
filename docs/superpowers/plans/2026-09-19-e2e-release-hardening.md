# E2E Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical OCR pipeline verifiable end-to-end by one bounded CPU-only Doctor check.

**Architecture:** A new test-only-sized release self-test calls the existing pipeline and exporter; Doctor exposes its result in deep mode. CI runs the same command plus a JavaScript syntax check.

**Tech Stack:** Python 3.11, pypdf/pypdfium2, python-docx, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-19-e2e-release-hardening-design.md`

## Global Constraints

- Use `process_pdf` and `markdown_to_docx`; do not add routing or output paths.
- CPU-only, offline, deterministic; no model/data downloads or provider calls.
- Emit categorical state and reason evidence only; no user-facing percentages.
- Preserve page boundaries and refuse untrusted text.

## Review Focus

- Image-only pages must be pending/review, never successful text.
- Short title pages must not become near-blank merely because they are short.
- Review/pending pages must retain a visible DOCX boundary.
- Self-test must not require source checkout paths once installed.
- Doctor must redact implementation exceptions and document contents.

---

### Task 1: Canonical deterministic release self-test

**Files:**
- Create: `pdfword/release_self_test.py`
- Test: `tests/test_release_self_test.py`

**Interfaces:**
- Produces: `run_release_self_test() -> ReleaseSelfTestResult`
- Consumes: `process_pdf(...)` and `markdown_to_docx(...)`

- [ ] **Step 1: Write failing tests**

```python
def test_release_self_test_exercises_categorical_pipeline():
    result = run_release_self_test()
    assert result.ok
    assert {"digital_text", "blank_page", "pending_ocr_model"} <= set(result.states)
    assert result.page_boundaries_preserved
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_release_self_test.py -q`
Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement minimal self-test**

Create deterministic PDFs in `TemporaryDirectory`, invoke only canonical
pipeline/export APIs, and return categorical, non-sensitive diagnostics.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_release_self_test.py -q`
Expected: PASS.

### Task 2: Doctor and CI integration

**Files:**
- Modify: `clouda_data/doctor/report.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Test: `tests/doctor/test_release_self_test.py`

**Interfaces:**
- Consumes: `run_release_self_test() -> ReleaseSelfTestResult`
- Produces: Doctor deep check `deep.pdfword-release-self-test`.

- [ ] **Step 1: Write failing Doctor test**

```python
def test_deep_doctor_reports_pdfword_release_self_test(monkeypatch):
    report = collect_report(deep=True)
    assert any(check.id == "deep.pdfword-release-self-test" for check in report.all_checks())
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/doctor/test_release_self_test.py -q`
Expected: FAIL because the check is absent.

- [ ] **Step 3: Implement integration and documentation**

Add a required deep Doctor check with concise status, add the Doctor command
to release instructions, and make CI run JavaScript syntax plus deep Doctor.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/doctor/test_release_self_test.py tests/test_release_self_test.py -q`
Expected: PASS.

### Task 3: Release verification

**Files:**
- Test: existing full suite and package smoke

- [ ] **Step 1: Run focused E2E and Doctor checks**

Run: `python -m pytest tests/test_release_self_test.py tests/doctor/test_release_self_test.py -q; python -m clouda_data.pipeline.cli doctor --deep`
Expected: PASS / exit 0.

- [ ] **Step 2: Run project gates**

Run: `python -m pytest -q; python -m ruff check .; python -m black --check .; python -m mypy .; node --check clouda_lab/dashboard/static/app.js; python -m build --wheel --no-isolation --skip-dependency-check; git diff --check`
Expected: every command exits 0.

- [ ] **Step 3: Clean-install smoke**

Run: create a temporary virtual environment, install the built wheel without
the source checkout on `PYTHONPATH`, import `pdfword` and `clouda_lab`, and
run `clouda-data doctor --deep` when dependencies are locally available.

- [ ] **Step 4: Commit**

Run: `git add pdfword/release_self_test.py tests/test_release_self_test.py clouda_data/doctor/report.py tests/doctor/test_release_self_test.py .github/workflows/ci.yml README.md && git commit -m "feat: add offline OCR release self-test"`
Expected: committed feature with focused and full verification recorded.
