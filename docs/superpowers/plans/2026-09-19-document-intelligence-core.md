# Document Intelligence Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every PDF page through deterministic analysis, an explainable digital-text trust gate, and a closed decision engine before any direct extraction or OCR execution.

**Architecture:** `pdfword/page_routing.py` owns immutable analysis, gate, decision, and trusted-context contracts. `pdfword/ocr_pipeline.py` computes one document digest, invokes that canonical layer, and executes the selected engine path; `DirectPdfTextEngine` remains policy-free. Clouda Lab delegates to the same routing code and exposes only categorical decisions, reason codes, and sanitized scalar evidence.

**Tech Stack:** Python 3.11+, frozen dataclasses, string enums, `pypdf`, existing `pypdfium2` rendering, FastAPI, vanilla JavaScript, pytest, Ruff, Black, MyPy.

**Spec:** `docs/superpowers/specs/2026-09-19-document-intelligence-core-design.md`

## Global Constraints

- Compute the PDF SHA-256 once per `process_pdf` call and reuse it for every page context.
- `DirectPdfTextEngine` remains a pure extraction primitive and returns `confidence=None`.
- Direct text is executed only after `DigitalTextGateVerdict.TRUSTED` and its normalized digest is revalidated after extraction.
- Digest mismatch, unavailable mandatory evidence, or conflicting evidence must fail to `review_required`, never direct success.
- Short character count alone must never classify a page as near-blank.
- A review-required page preserves its document boundary and emits a categorical placeholder without untrusted extracted text.
- User-facing conversion output and Clouda Lab expose no routing accuracy, confidence, or quality percentages.
- Analysis performs no network calls, downloads, model selection, training, or GPU work.
- Existing loopback, action-token, path-redaction, engine-registry, local-model feature-flag, and pending-model boundaries remain intact.
- No heavy dependency is added; use the existing PDF stack.

## Review Focus

- A short centered title must route to review or trust based on structure, never near-blank solely because it has few characters; Task 2 pins this boundary.
- A page whose second extraction differs from analyzed text must emit the review placeholder and omit both text variants; Task 4 pins this failure mode.
- Image metadata or text geometry failure must not turn visible content into a blank page; Tasks 1 and 3 pin unavailable-evidence behavior.
- A trusted digital page with a small decorative image must stay eligible for direct extraction while a full-page image with sparse text must not; Tasks 1 and 2 pin both cases.
- Lab upload results must contain no extracted text, private path, secret, or percentage-like routing field; Task 5 pins the serialized contract.

---

### Task 1: Canonical page-analysis contracts and deterministic evidence

**Files:**
- Create: `pdfword/page_routing.py`
- Create: `tests/test_page_routing.py`

**Interfaces:**
- Produces: `DocumentRoutingContext.from_pdf(pdf_bytes: bytes) -> DocumentRoutingContext`
- Produces: `analyze_pdf_page(page: Any, page_number: int, document: DocumentRoutingContext) -> PageAnalysis`
- Produces: immutable `TextSpan`, `PageAnalysis`, `AnalysisWarningCode`, and `EvidenceAvailability` contracts.
- Produces: `normalize_embedded_text(text: str) -> str` and `digest_normalized_text(text: str) -> str` used by later runtime checks.

- [ ] **Step 1: Write failing analysis-contract tests**

Add tests that construct small PDFs in memory and assert document digest reuse, safe diagnostics, unavailable fields, text-span evidence, image evidence, Arabic evidence, and content-operation evidence. Patch `hashlib.sha256` through a narrow wrapper to prove `DocumentRoutingContext.from_pdf` hashes the PDF once rather than once per page.

```python
def test_document_context_hashes_pdf_once_and_reuses_digest(monkeypatch):
    calls = 0
    real = page_routing._sha256_hex

    def counted(payload: bytes) -> str:
        nonlocal calls
        calls += 1
        return real(payload)

    monkeypatch.setattr(page_routing, "_sha256_hex", counted)
    context = DocumentRoutingContext.from_pdf(_two_page_pdf())
    reader = PdfReader(io.BytesIO(context.pdf_bytes))
    first = analyze_pdf_page(reader.pages[0], 1, context)
    second = analyze_pdf_page(reader.pages[1], 2, context)

    assert calls == 1
    assert first.document_sha256 == second.document_sha256 == context.pdf_sha256


def test_page_diagnostics_never_include_extracted_text():
    analysis = _analyze_fixture("digital_text.pdf")
    payload = analysis.to_diagnostics()
    assert "Digital PDF text" not in json.dumps(payload)
    assert payload["normalized_character_count"] > 0
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_page_routing.py -q`

Expected: collection fails because `pdfword.page_routing` and its contracts do not exist.

- [ ] **Step 3: Implement immutable contracts and analyzer**

Implement the enums and frozen dataclasses from the spec. Parse text with a `pypdf` visitor, normalize bounding boxes to page coordinates, count text/image/path operators conservatively, and make unavailable metrics `None` plus warnings.

```python
@dataclass(frozen=True)
class DocumentRoutingContext:
    pdf_bytes: bytes = field(repr=False)
    pdf_sha256: str

    @classmethod
    def from_pdf(cls, pdf_bytes: bytes) -> "DocumentRoutingContext":
        return cls(pdf_bytes=pdf_bytes, pdf_sha256=_sha256_hex(pdf_bytes))


@dataclass(frozen=True)
class PageAnalysis:
    document_sha256: str
    page_number: int
    page_width_pt: float
    page_height_pt: float
    embedded_text: str = field(repr=False)
    normalized_text: str = field(repr=False)
    normalized_character_count: int
    word_count: int
    span_count: int
    spans: tuple[TextSpan, ...] = field(repr=False)
    text_coverage_ratio: float | None
    image_count: int | None
    visible_text_operations: int | None
    visible_image_operations: int | None
    visible_path_operations: int | None
    blank_evidence: bool
    near_blank_evidence: bool
    suspicious_sparse_text: bool
    suspicious_fragmentation: bool
    hybrid_page: bool
    arabic_character_count: int
    arabic_integrity_risk: bool
    reading_order_risk: bool | None
    multi_column_risk: bool | None
    warnings: tuple[AnalysisWarningCode, ...]

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "page_width_pt": round(self.page_width_pt, 3),
            "page_height_pt": round(self.page_height_pt, 3),
            "normalized_character_count": self.normalized_character_count,
            "word_count": self.word_count,
            "span_count": self.span_count,
            "text_coverage_ratio": self.text_coverage_ratio,
            "image_count": self.image_count,
            "blank_evidence": self.blank_evidence,
            "near_blank_evidence": self.near_blank_evidence,
            "suspicious_sparse_text": self.suspicious_sparse_text,
            "suspicious_fragmentation": self.suspicious_fragmentation,
            "hybrid_page": self.hybrid_page,
            "arabic_character_count": self.arabic_character_count,
            "arabic_integrity_risk": self.arabic_integrity_risk,
            "reading_order_risk": self.reading_order_risk,
            "multi_column_risk": self.multi_column_risk,
            "warnings": [warning.value for warning in self.warnings],
        }
```

`near_blank_evidence` must require both short text and structural localization or otherwise nearly empty visible content. A short centered title with ordinary font/span geometry must return `False`.

- [ ] **Step 4: Run analysis tests and verify GREEN**

Run: `python -m pytest tests/test_page_routing.py -q`

Expected: all Task 1 tests pass with no warnings.

- [ ] **Step 5: Commit the analyzer**

```powershell
git add pdfword/page_routing.py tests/test_page_routing.py
git commit -m "feat: add deterministic PDF page analysis"
```

### Task 2: Trusted Digital Text Gate

**Files:**
- Modify: `pdfword/page_routing.py`
- Modify: `tests/test_page_routing.py`
- Create or update deterministic PDF builders under: `tests/fixtures/generate_fixtures.py`

**Interfaces:**
- Consumes: `PageAnalysis` from Task 1.
- Produces: `evaluate_digital_text_trust(analysis: PageAnalysis) -> DigitalTextGateResult`.
- Produces: `DigitalTextGateVerdict`, `DigitalTextReasonCode`, `GateCheck`, and `DigitalTextGateResult`.

- [ ] **Step 1: Write failing gate tests for the required matrix**

Add one-behavior tests for complete digital text, no text, hidden sparse text over a page image, incomplete hybrid text, harmless decorative images, fragmented Arabic, pathological Arabic spacing, Unicode corruption, complex layout, unavailable evidence, and the short-title near-blank regression.

```python
def test_hidden_sparse_text_over_page_image_is_untrusted():
    result = evaluate_digital_text_trust(_analysis("hidden_partial_text.pdf"))
    assert result.verdict is DigitalTextGateVerdict.UNTRUSTED
    assert DigitalTextReasonCode.IMAGE_DOMINANT_PARTIAL_TEXT in result.reason_codes


def test_short_centered_title_is_not_near_blank():
    analysis = _analysis("short_title.pdf")
    result = evaluate_digital_text_trust(analysis)
    assert analysis.near_blank_evidence is False
    assert result.verdict in {
        DigitalTextGateVerdict.TRUSTED,
        DigitalTextGateVerdict.UNCERTAIN,
    }
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_page_routing.py -q -k "gate or hidden or hybrid or arabic or title"`

Expected: failures because the gate contracts and evaluation function do not exist.

- [ ] **Step 3: Implement explicit gate checks**

Use named threshold constants and independent checks. Do not let a diagnostic score select the verdict.

```python
class DigitalTextGateVerdict(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class DigitalTextGateResult:
    verdict: DigitalTextGateVerdict
    checks: tuple[GateCheck, ...]
    reason_codes: tuple[DigitalTextReasonCode, ...]


def evaluate_digital_text_trust(analysis: PageAnalysis) -> DigitalTextGateResult:
    checks = _gate_checks(analysis)
    if _has_definitive_untrusted_evidence(checks):
        verdict = DigitalTextGateVerdict.UNTRUSTED
    elif _has_missing_or_conflicting_evidence(checks):
        verdict = DigitalTextGateVerdict.UNCERTAIN
    elif _all_mandatory_checks_pass(checks) and _has_positive_completeness(checks):
        verdict = DigitalTextGateVerdict.TRUSTED
    else:
        verdict = DigitalTextGateVerdict.UNCERTAIN
    return DigitalTextGateResult(verdict, tuple(checks), _reason_codes(checks))
```

- [ ] **Step 4: Verify GREEN and run the full routing test file**

Run: `python -m pytest tests/test_page_routing.py -q`

Expected: all analyzer and gate cases pass.

- [ ] **Step 5: Commit the gate**

```powershell
git add pdfword/page_routing.py tests/test_page_routing.py tests/fixtures/generate_fixtures.py tests/fixtures
git commit -m "feat: gate embedded PDF text with explicit evidence"
```

### Task 3: Closed Page Decision Engine and trusted context

**Files:**
- Modify: `pdfword/page_routing.py`
- Modify: `tests/test_page_routing.py`

**Interfaces:**
- Consumes: `DocumentRoutingContext`, `PageAnalysis`, and `DigitalTextGateResult`.
- Produces: `decide_page_route(analysis: PageAnalysis, gate: DigitalTextGateResult, *, ocr_available: bool) -> PageDecisionResult`.
- Produces: `TrustedDigitalTextContext`, `PageDecision`, `PageNextPath`.
- Produces: `validate_trusted_context_before_extraction` and `validate_trusted_text_after_extraction`.

- [ ] **Step 1: Write failing decision and context tests**

Cover every closed outcome and next path, OCR availability, context reuse across the wrong page/document, and post-extraction digest mismatch.

```python
def test_untrusted_page_with_unavailable_ocr_preserves_pending_path():
    result = decide_page_route(_image_analysis(), _untrusted_gate(), ocr_available=False)
    assert result.decision is PageDecision.OCR_REQUIRED
    assert result.next_path is PageNextPath.PENDING_OCR_MODEL


def test_post_extraction_digest_mismatch_is_rejected():
    decision = decide_page_route(_trusted_analysis(), _trusted_gate(), False)
    assert decision.trusted_context is not None
    assert not validate_trusted_text_after_extraction(
        decision.trusted_context,
        "different extracted text",
    )
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_page_routing.py -q -k "decision or context or digest"`

Expected: failures because decision and trusted-context APIs do not exist.

- [ ] **Step 3: Implement decision precedence and context validation**

```python
def decide_page_route(
    analysis: PageAnalysis,
    gate: DigitalTextGateResult,
    *,
    ocr_available: bool,
) -> PageDecisionResult:
    if analysis.blank_evidence or analysis.near_blank_evidence:
        return _blank_decision(analysis, gate)
    if gate.verdict is DigitalTextGateVerdict.TRUSTED:
        return _trusted_decision(analysis, gate)
    if gate.verdict is DigitalTextGateVerdict.UNTRUSTED:
        return _ocr_decision(analysis, gate, ocr_available=ocr_available)
    return _review_decision(analysis, gate)
```

Context validation compares document digest and page number before execution,
then compares `digest_normalized_text(engine_result.text)` after execution.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_page_routing.py -q`

Expected: all routing-domain tests pass.

- [ ] **Step 5: Commit the decision engine**

```powershell
git add pdfword/page_routing.py tests/test_page_routing.py
git commit -m "feat: add closed page decision engine"
```

### Task 4: Runtime, engine, and DOCX compatibility integration

**Files:**
- Modify: `pdfword/engines.py`
- Modify: `pdfword/ocr_pipeline.py`
- Modify: `pdfword/docx_export.py`
- Modify: `pdfword/models.py` only if a small typed compatibility field is needed
- Modify: `tests/test_model_agnostic_engines.py`
- Modify: `tests/test_readiness_pipeline.py`
- Modify: `tests/test_conversion_service_paths.py`
- Modify: `tests/test_engine_boundaries.py`

**Interfaces:**
- Consumes all Task 1-3 routing interfaces.
- Preserves: `process_pdf(...) -> tuple[list[PageResult], str]`, `OCRResult`, registry selection, and local feature flags.
- Produces: `PageResult.metadata["document_intelligence"]` safe categorical diagnostics.

- [ ] **Step 1: Write failing engine and runtime integration tests**

Add assertions that direct confidence is `None`, untrusted text never reaches output, trusted extraction revalidates the digest, mismatch routes to review, review pages have visible placeholders in DOCX, and placeholders contain no percentage.

```python
def test_direct_engine_does_not_fabricate_confidence():
    result = DirectPdfTextEngine().extract_page(
        pdf_bytes=(FIXTURES / "digital_text.pdf").read_bytes(), page_no=1
    )
    assert result.success
    assert result.confidence is None


def test_direct_text_digest_mismatch_becomes_visible_review(monkeypatch):
    monkeypatch.setattr(
        DIRECT_TEXT_ENGINE,
        "extract_page",
        lambda **_: OCRResult(
            engine_name="direct_pdf_text",
            status=OCR_STATUS_SUCCEEDED,
            text="changed after analysis",
            confidence=None,
        ),
    )
    rows, _ = _process("digital_text.pdf")
    assert rows[0].route_used == "review_required"
    assert "changed after analysis" not in rows[0].markdown
    assert "REQUIRES REVIEW" in rows[0].markdown
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_model_agnostic_engines.py tests/test_readiness_pipeline.py tests/test_conversion_service_paths.py tests/test_engine_boundaries.py -q`

Expected: new confidence, routing, digest, and placeholder assertions fail.

- [ ] **Step 3: Refactor `process_pdf` around the canonical decision**

At function entry, create one `DocumentRoutingContext`. For each page, analyze,
gate, decide, and execute only `decision.next_path`. Map diagnostics without raw
text.

```python
document = DocumentRoutingContext.from_pdf(pdf_bytes)
reader = PdfReader(io.BytesIO(document.pdf_bytes))

analysis = analyze_pdf_page(reader.pages[page_no - 1], page_no, document)
gate = evaluate_digital_text_trust(analysis)
decision = decide_page_route(
    analysis,
    gate,
    ocr_available=local_model_engine is not None,
)
```

Before direct extraction validate the context; after extraction validate the
returned text digest. On any mismatch create the categorical review result.
Review markdown contains only a fixed placeholder such as:

```text
[PAGE 3 REQUIRES REVIEW - embedded text was not used; reasons: conflicting_evidence]
```

Update `_manual_review_marker` so it reports category, route, and reason only;
remove the existing `estimated quality NN%` user-facing text.

- [ ] **Step 4: Update `DirectPdfTextEngine` and status reporting**

Set direct result confidence to `None`. Keep extraction policy-free. Change
`available_engine_status` so direct extraction is available but conditionally
selected rather than globally active.

- [ ] **Step 5: Verify focused integration GREEN**

Run: `python -m pytest tests/test_page_routing.py tests/test_model_agnostic_engines.py tests/test_readiness_pipeline.py tests/test_conversion_service_paths.py tests/test_engine_boundaries.py -q`

Expected: all focused routing and compatibility tests pass.

- [ ] **Step 6: Run the repository Python test suite before moving on**

Run: `python -m pytest -q`

Expected: zero failures. Any unrelated pre-existing failure is recorded by exact test name and investigated before continuing.

- [ ] **Step 7: Commit runtime integration**

```powershell
git add pdfword/engines.py pdfword/ocr_pipeline.py pdfword/docx_export.py pdfword/models.py tests/test_model_agnostic_engines.py tests/test_readiness_pipeline.py tests/test_conversion_service_paths.py tests/test_engine_boundaries.py
git commit -m "feat: route PDF pages through document intelligence"
```

### Task 5: Safe Clouda Lab inspection API

**Files:**
- Create: `clouda_lab/dashboard/document_intelligence.py`
- Modify: `clouda_lab/dashboard/app.py`
- Create: `tests/dashboard/test_document_intelligence.py`
- Modify: `tests/dashboard/test_security_and_app.py`

**Interfaces:**
- Consumes: `DocumentRoutingContext`, `analyze_pdf_page`, `evaluate_digital_text_trust`, and `decide_page_route`.
- Produces: `DocumentIntelligenceService.analyze(pdf_bytes: bytes) -> dict[str, Any]`.
- Produces: action-token-protected `POST /api/lab/document-intelligence/analyze` with a bounded raw `application/pdf` body. Raw streaming avoids multipart temporary-file spooling.

- [ ] **Step 1: Write failing service and endpoint tests**

Test loopback/action-token enforcement, PDF type/size/page limits, categorical
payloads, absence of raw text/private paths/percentage fields, no persistence,
and no network calls.

```python
def test_lab_document_intelligence_is_categorical_and_sanitized(client):
    token = client.get("/api/lab/session").json()["action_token"]
    response = client.post(
        "/api/lab/document-intelligence/analyze",
        headers={"X-Clouda-Lab-Action": token},
        files={"file": ("sample.pdf", DIGITAL_PDF, "application/pdf")},
    )
    payload = response.json()
    serialized = json.dumps(payload).lower()
    assert response.status_code == 200
    assert payload["pages"][0]["gate_verdict"] in {"trusted", "untrusted", "uncertain"}
    assert "digital pdf text" not in serialized
    assert "accuracy" not in serialized
    assert "confidence" not in serialized
    assert "quality_percentage" not in serialized
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/dashboard/test_document_intelligence.py tests/dashboard/test_security_and_app.py -q`

Expected: endpoint and service imports fail because they do not exist.

- [ ] **Step 3: Implement bounded in-memory Lab analysis**

Enforce `%PDF-` input, a 10 MiB byte cap, and a 25-page cap. Do not accept path
parameters. Do not retain uploads. Serialize only `to_diagnostics`, verdicts,
reason codes, paths, and booleans. Pass the final value through `browser_safe`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/dashboard/test_document_intelligence.py tests/dashboard/test_security_and_app.py -q`

Expected: all Lab service, security, and sanitization tests pass.

- [ ] **Step 5: Commit Lab backend integration**

```powershell
git add clouda_lab/dashboard/document_intelligence.py clouda_lab/dashboard/app.py tests/dashboard/test_document_intelligence.py tests/dashboard/test_security_and_app.py
git commit -m "feat(lab): expose safe page routing diagnostics"
```

### Task 6: Clouda Lab Document Intelligence page

**Files:**
- Modify: `clouda_lab/dashboard/static/index.html`
- Modify: `clouda_lab/dashboard/static/app.js`
- Modify: `clouda_lab/dashboard/static/styles.css` only for existing-component layout gaps
- Modify: `tests/dashboard/test_ui_contract.py`

**Interfaces:**
- Consumes: `POST /api/lab/document-intelligence/analyze` from Task 5.
- Produces: `#/document-intelligence` route with explicit file submission and categorical tables.

- [ ] **Step 1: Write failing UI contract tests**

Assert navigation and renderer registration, bounded raw-PDF submission with the action token, safe `textContent` rendering, category/reason columns, and absence of accuracy/confidence/quality percentage labels.

```python
def test_document_intelligence_page_is_categorical_only():
    html = INDEX.read_text(encoding="utf-8")
    script = APP.read_text(encoding="utf-8")
    assert 'href="#/document-intelligence"' in html
    assert '"document-intelligence": renderDocumentIntelligence' in script
    assert "X-Clouda-Lab-Action" in script
    assert "Gate verdict" in script
    assert "Reason codes" in script
    assert "accuracy percentage" not in script.lower()
    assert "confidence percentage" not in script.lower()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/dashboard/test_ui_contract.py -q -k document_intelligence`

Expected: the new route and renderer assertions fail.

- [ ] **Step 3: Implement explicit upload and categorical rendering**

Extend `api` with a `rawBody` branch that sets `application/pdf`, sends the
action token, and retains the current abort signal. Render decision, gate
verdict, next path, reason-code badges, OCR pending state, and safe evidence.
Do not render internal diagnostic scores or OCR confidence fields.

- [ ] **Step 4: Verify UI GREEN and JavaScript syntax**

Run: `python -m pytest tests/dashboard/test_ui_contract.py -q`

Run: `node --check clouda_lab/dashboard/static/app.js`

Expected: UI contracts pass and Node exits 0.

- [ ] **Step 5: Commit Lab UI**

```powershell
git add clouda_lab/dashboard/static/index.html clouda_lab/dashboard/static/app.js clouda_lab/dashboard/static/styles.css tests/dashboard/test_ui_contract.py
git commit -m "feat(lab): add document routing inspection page"
```

### Task 7: Directly affected documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/runtime/LOCAL_OCR_INTEGRATION.md`
- Modify: `docs/lab/CLOUDA_LAB_BACKEND.md`
- Modify: `docs/TESTING.md`

**Interfaces:**
- Documents the implemented contracts and behavior from Tasks 1-6.
- Removes claims that embedded text is automatically trusted or direct extraction is unconditional.

- [ ] **Step 1: Search for inaccurate architectural claims**

Run:

```powershell
rg -n "selectable text|digital_text|direct extraction|text layer|pending_ocr_model|accuracy|confidence|quality" README.md docs/ARCHITECTURE.md docs/runtime/LOCAL_OCR_INTEGRATION.md docs/lab/CLOUDA_LAB_BACKEND.md docs/TESTING.md
```

- [ ] **Step 2: Update only affected documentation**

Document the canonical flow exactly:

```text
Page Analyzer
-> Trusted Digital Text Gate
-> Page Decision Engine
-> trusted direct extraction OR OCR/pending/review/blank path
```

State that Lab exposes categorical routing evidence, not accuracy/confidence/
quality percentages. State that no final OCR model or training method is selected.

- [ ] **Step 3: Verify documentation consistency**

Run:

```powershell
rg -n "automatically trusted|unconditional production path|final OCR model has been selected|routing accuracy|routing confidence" README.md docs/ARCHITECTURE.md docs/runtime/LOCAL_OCR_INTEGRATION.md docs/lab/CLOUDA_LAB_BACKEND.md docs/TESTING.md
git diff --check
```

Expected: no stale unconditional-trust claim and no whitespace errors.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md docs/ARCHITECTURE.md docs/runtime/LOCAL_OCR_INTEGRATION.md docs/lab/CLOUDA_LAB_BACKEND.md docs/TESTING.md
git commit -m "docs: document trusted PDF text routing"
```

### Task 8: Full verification, review, and safe Git integration

**Files:**
- Verify all changed files.
- Do not create product changes in this task unless a failing verification has a regression test first.

**Interfaces:**
- Produces the exact verification evidence and Git SHAs required by the final report.

- [ ] **Step 1: Run the complete required verification suite**

```powershell
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m mypy .
node --check clouda_lab/dashboard/static/app.js
python -m build --wheel --no-isolation --skip-dependency-check
git diff --check
```

Expected: every command exits 0. Record exact pytest counts and tool outputs.

- [ ] **Step 2: Review requirement coverage and repository state**

Run:

```powershell
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- pdfword clouda_lab tests README.md docs
```

Check every acceptance criterion in the spec against code or a named test. Confirm no model, dataset, provider call, training process, or GPU workload occurred.

- [ ] **Step 3: Fetch and detect upstream movement**

Run:

```powershell
git fetch origin --prune
git rev-parse origin/main
git merge-base --is-ancestor origin/main HEAD
```

If `origin/main` moved, merge it into the feature branch without force or history rewrite, resolve only scoped conflicts, then rerun the full verification suite.

- [ ] **Step 4: Push the feature branch**

Run: `git push -u origin feature/document-intelligence-core`

Expected: normal non-force push succeeds.

- [ ] **Step 5: Integrate safely into current main**

If repository permissions allow direct integration and branch protection does not require a PR:

```powershell
git switch main
git pull --ff-only origin main
git merge --no-ff feature/document-intelligence-core
git push origin main
```

If protection requires a PR or any state is ambiguous, create the normal PR instead and attach it to the task. Never force-push, reset away remote work, delete remote branches, or rewrite history.

- [ ] **Step 6: Confirm final remote state**

Run:

```powershell
git fetch origin --prune
git rev-parse feature/document-intelligence-core
git rev-parse origin/main
git status --short --branch
```

Expected: clean tree and the integrated commit visible from `origin/main`, or a clearly reported PR awaiting required review.
