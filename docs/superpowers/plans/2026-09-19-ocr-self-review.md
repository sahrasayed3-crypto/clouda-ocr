# OCR Self-Review and Selective Re-Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded, categorical OCR self-review, verified selective re-read, and conservative reconciliation to the canonical OCR runtime.

**Architecture:** `pdfword/ocr_self_review.py` owns immutable contracts and deterministic policy. `ocr_pipeline.py` owns one-render execution and maps reconciliation to the existing page boundary; engine/provider contracts carry explicit current-render geometry only when available. Lab projects the canonical metadata safely.

**Tech Stack:** Python 3.11, frozen dataclasses, `StrEnum`, Pillow, pypdfium2, FastAPI, vanilla JavaScript, pytest, Ruff, Black, MyPy.

**Spec:** `docs/superpowers/specs/2026-09-19-ocr-self-review-design.md`

## Global Constraints

- Preserve Page Analyzer, Digital Text Gate, Page Decision Engine, and direct-text trust boundaries unchanged.
- No synthetic OCR quality/confidence score may accept/reject or override a categorical OCR review/reconciliation result.
- Native confidence is optional and internal; no user/Lab accuracy, confidence, or quality percentage.
- Re-read crop geometry must be explicitly image-pixel bound to the exact current render; invalid/stale/unverified boxes fail closed to `review_required`.
- Render once, crop in memory, reuse the configured local engine, and bound regions, attempts, pixels, and bytes.
- No model/dataset download, hosted OCR, training, benchmark, or GPU requirement.
- Preserve review page boundaries and categorical DOCX placeholders.
- Capture real exit status for the full baseline and final verification suite before completion or merge.

## Review Focus

- An OCR result without confidence but with clean observable evidence must be accepted; Task 2 covers it.
- A coordinate-valid-looking box from another render must never generate a crop; Task 3 covers it.
- A local re-read must not replace whole-page text merely because it is longer; Task 4 covers it.
- Existing conversion normalization must not overwrite an OCR categorical review state from a missing legacy score; Task 5 covers it.
- Lab serialization must not leak text, geometry, hashes, native confidence, or percentages; Task 6 covers it.

---

### Task 1: Baseline capture and canonical contracts

**Files:**
- Create: `pdfword/ocr_self_review.py`
- Create: `tests/test_ocr_self_review.py`

**Interfaces:**
- Produces `OCRReviewVerdict`, `OCRIssueCode`, `OCRPageState`, `RenderedPageContext`, `ReviewRegion`, `OCRReviewResult`, `ReReadResult`, and `OCRReconciliationResult`.
- Produces `make_rendered_page_context(page_no: int, image_bytes: bytes) -> RenderedPageContext`.

- [ ] **Step 1: Capture the baseline with a real exit status**

Run: `python -m pytest -q`

Expected: a captured exit code and complete pytest summary; do not continue on a failing baseline without a root-cause ruling.

- [ ] **Step 2: Write failing immutable-contract tests**

```python
def test_render_context_binds_identity_and_exact_png_dimensions():
    context = make_rendered_page_context(3, _png_bytes(40, 60))
    assert context.page_no == 3
    assert (context.width_px, context.height_px) == (40, 60)
    assert context.coordinate_space == "image_pixels"

def test_safe_diagnostics_omit_text_hash_and_geometry():
    region = _region_for(_context())
    assert "secret text" not in json.dumps(region.to_diagnostics())
    assert "bbox" not in json.dumps(region.to_diagnostics())
```

- [ ] **Step 3: Run the tests and verify RED**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: import/collection failure because the module does not exist.

- [ ] **Step 4: Implement the smallest typed domain records**

```python
@dataclass(frozen=True)
class RenderedPageContext:
    page_no: int
    render_identity: str = field(repr=False)
    width_px: int
    height_px: int
    coordinate_space: str = "image_pixels"
```

Keep raw text internal and add only safe `to_diagnostics()` projections.

- [ ] **Step 5: Run the focused tests and commit**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: PASS.

Commit: `git commit -am "feat: add OCR self-review contracts"`

### Task 2: Deterministic first-pass self-review

**Files:**
- Modify: `pdfword/ocr_self_review.py`
- Modify: `tests/test_ocr_self_review.py`

**Interfaces:**
- Produces `review_first_pass(result: OCRResult, analysis: PageAnalysis, render: RenderedPageContext) -> OCRReviewResult`.
- Consumes existing Arabic/integrity fields from `PageAnalysis` without duplicating routing logic.

- [ ] **Step 1: Write failing review tests**

```python
def test_clean_first_pass_without_confidence_is_accepted():
    review = review_first_pass(_ocr("ordinary readable text"), _analysis(), _context())
    assert review.verdict is OCRReviewVerdict.ACCEPTED

def test_empty_unicode_and_duplicate_text_emit_categorical_reasons():
    review = review_first_pass(_ocr("x\ufffd\nx\ufffd"), _analysis(), _context())
    assert OCRIssueCode.UNICODE_CORRUPTION in review.issue_codes
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: FAIL because `review_first_pass` is missing.

- [ ] **Step 3: Implement observable review checks**

Use normalized lines, canonical Arabic evidence, Unicode checks, explicit engine
error, optional documented confidence, and verified box evidence only. Missing
optional confidence must not create a failure. Make non-localized ambiguity
`review_required`.

- [ ] **Step 4: Verify focused GREEN and commit**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: PASS.

Commit: `git commit -am "feat: review OCR output categorically"`

### Task 3: Verified regions, crop budgets, and crop helper

**Files:**
- Modify: `pdfword/ocr_self_review.py`
- Modify: `pdfword/engines.py`
- Modify: `tests/test_ocr_self_review.py`

**Interfaces:**
- Produces `validate_and_bound_regions(...) -> tuple[ReviewRegion, ...]` and `crop_review_region(image_bytes: bytes, region: ReviewRegion, render: RenderedPageContext) -> bytes`.
- Adds backward-compatible `OCRBox` metadata convention: `coordinate_space`, `render_identity`, `image_width_px`, `image_height_px`.

- [ ] **Step 1: Write failing geometry and budget tests**

```python
def test_stale_or_unverified_box_fails_closed_without_crop():
    result = _ocr_with_box(_box(render_identity="old-render"))
    review = review_first_pass(result, _analysis(), _context())
    assert review.verdict is OCRReviewVerdict.REVIEW_REQUIRED
    assert not review.suspicious_regions

def test_overlap_is_deduplicated_and_budget_exhaustion_requires_review():
    bounded = validate_and_bound_regions(_overlapping_regions(), _context(), _tight_budget())
    assert len(bounded) == 1
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: FAIL because geometry validator/crop helper is missing.

- [ ] **Step 3: Implement strict validation and in-memory crops**

Validate finite numbers and exact current render bindings before clamping. Reject
rather than repairing an unverified coordinate frame. Decode/crop/encode PNG
with Pillow only after validation. Enforce count, attempts, crop pixels, and
byte budgets deterministically.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_ocr_self_review.py -q`

Expected: PASS.

Commit: `git commit -am "feat: bound OCR reread regions"`

### Task 4: Runtime orchestration and reconciliation

**Files:**
- Modify: `pdfword/ocr_pipeline.py`
- Modify: `pdfword/models.py`
- Modify: `tests/test_model_agnostic_engines.py`
- Modify: `tests/test_readiness_pipeline.py`

**Interfaces:**
- Produces `reconcile_ocr_results(...) -> OCRReconciliationResult` and page metadata `ocr_self_review`.
- `process_pdf()` maps categorical result to `accepted_first_pass`, `accepted_after_selective_reread`, `review_required`, `pending_ocr_model`, or `ocr_failed`.

- [ ] **Step 1: Write failing runtime tests using a deterministic fake engine**

```python
def test_clean_ocr_without_confidence_is_accepted_first_pass():
    rows, _ = _process_scanned(_engine(first_pass="clear", confidence=None))
    assert rows[0].metadata["page_state"] == "accepted_first_pass"
    assert rows[0].quality_score is None

def test_reread_disagreement_preserves_review_placeholder():
    rows, output = _process_scanned(_engine(first_pass="bad", reread="different"))
    assert rows[0].route_used == "review_required"
    assert "different" not in output
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_model_agnostic_engines.py tests/test_readiness_pipeline.py -q`

Expected: FAIL because the old path requires confidence and has no review metadata.

- [ ] **Step 3: Implement one-render orchestration**

Build `RenderedPageContext` immediately after rendering. Run first pass, review,
crop/re-read only returned verified regions, reconcile, and map failures to
categorical review placeholders. Remove confidence-required acceptance and all
synthetic quality generation from this local OCR path. Preserve cancellation
checks and direct/blank/pending behavior.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_model_agnostic_engines.py tests/test_readiness_pipeline.py tests/test_ocr_self_review.py -q`

Expected: PASS.

Commit: `git commit -am "feat: integrate OCR self-review runtime"`

### Task 5: Conversion compatibility and categorical boundaries

**Files:**
- Modify: `pdfword/conversion_service.py`
- Modify: `pdfword/checkpoints.py`
- Modify: `pdfword/docx_export.py`
- Modify: `tests/test_quality_acceptance_policy.py`
- Modify: `tests/test_runtime_features.py`

**Interfaces:**
- Existing numeric fields remain compatible but cannot change a categorical OCR page state.

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_conversion_does_not_reclassify_categorical_ocr_review_from_none_score():
    page = _review_page_with_state("review_required")
    _normalize_manual_review_flags([page])
    assert page.route_used == "review_required"
    assert page.text_quality_score is None
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_quality_acceptance_policy.py tests/test_runtime_features.py -q`

Expected: FAIL because legacy score normalization fills or overrides the page.

- [ ] **Step 3: Implement state-preserving compatibility behavior**

Skip legacy synthetic-score acceptance for all known categorical runtime states.
Keep historic database shape but emit `None` for new-runtime legacy score fields.
Retain categorical DOCX review placeholders and never introduce percentage
language for OCR review.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_quality_acceptance_policy.py tests/test_runtime_features.py tests/test_readiness_pipeline.py -q`

Expected: PASS.

Commit: `git commit -am "fix: preserve categorical OCR review outcomes"`

### Task 6: Safe Lab projection, CUDA smoke, docs, and final verification

**Files:**
- Modify: `clouda_lab/dashboard/document_intelligence.py`
- Modify: `clouda_lab/dashboard/static/app.js`
- Modify: `tests/dashboard/test_document_intelligence.py`
- Create: `tests/test_ocr_cuda_smoke.py`
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `docs/runtime/LOCAL_OCR_INTEGRATION.md`, `docs/TESTING.md`

**Interfaces:**
- Lab serializes only `OCRReconciliationResult.to_diagnostics()` derived categorical fields.
- CUDA smoke is skipped unless `CLOUDA_CUDA_SMOKE=1` and CUDA is usable.

- [ ] **Step 1: Write failing Lab/CUDA/documentation contract tests**

```python
def test_lab_ocr_diagnostics_are_categorical_and_redacted():
    payload = _lab_payload_with_ocr_metadata()
    serialized = json.dumps(payload).lower()
    assert "accepted_first_pass" in serialized
    assert not {"accuracy", "confidence", "quality", "secret text"}.intersection(serialized.split())

def test_cuda_smoke_skips_without_opt_in(monkeypatch):
    monkeypatch.delenv("CLOUDA_CUDA_SMOKE", raising=False)
    assert run_cuda_smoke().state == "skipped"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/dashboard/test_document_intelligence.py tests/test_ocr_cuda_smoke.py -q`

Expected: FAIL because no safe OCR projection or smoke result exists.

- [ ] **Step 3: Implement safe projection and opt-in smoke**

Keep Lab’s existing bounded/action-token/no-persistence behavior. Add only
categorical metadata fields and render them with `textContent`. Implement CUDA
inspection defensively without imports/downloads until explicitly enabled.
Document bounds, no percentages, and that hardware smoke proves plumbing only.

- [ ] **Step 4: Verify focused GREEN, commit, then capture final full verification**

Run:

```powershell
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m mypy .
node --check clouda_lab/dashboard/static/app.js
python -m build --wheel --no-isolation --skip-dependency-check
git diff --check
```

Expected: every command returns exit code 0 with captured output. Record whether
the optional CUDA smoke skipped, ran, or failed; do not enable it implicitly.

Commit: `git commit -am "docs: document bounded OCR self-review"`

