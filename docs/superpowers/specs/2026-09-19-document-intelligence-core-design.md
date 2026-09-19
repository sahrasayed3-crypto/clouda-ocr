# Document Intelligence Core Design

## Purpose

Clouda OCR currently treats non-empty embedded PDF text as the normal successful
path. This change makes embedded text conditional: every page is analyzed first,
the text layer is evaluated by an explainable trust gate, and one canonical
decision engine selects direct extraction, OCR, manual review, or an explicit
blank state.

The change is routing infrastructure, not OCR model selection or training. It
must preserve the existing pending-model behavior whenever OCR is required but
no approved runtime is available.

Routing and trust are categorical, not accuracy estimates. User-facing and Lab
surfaces expose verdicts, decisions, and explicit reason codes. They do not show
an "accuracy", "confidence", or "quality percentage" unless a separate metric
has a documented definition and validation evidence.

## Scope and constraints

The implementation covers:

- deterministic page analysis using the repository's existing `pypdf` and PDF
  rendering primitives;
- a typed Trusted Digital Text Gate with closed verdicts and reason codes;
- a typed Page Decision Engine with closed routing outcomes;
- integration into the canonical `pdfword.ocr_pipeline.process_pdf` flow;
- safe inspection in the existing Clouda Lab;
- compatibility with the OCR engine registry, `OCRResult`, `PageResult`, local
  OCR feature flags, and existing conversion entry points;
- focused deterministic fixtures, regression tests, and directly affected
  documentation.

The implementation does not select, download, train, benchmark, or execute a
new model. It does not call external providers, add a hosted service, implement
OCR self-review, or redesign unrelated runtime/training systems.

## Chosen architecture

One new canonical module, `pdfword/page_routing.py`, owns the complete routing
domain. It exposes immutable typed contracts and three independently testable
operations:

1. `analyze_pdf_page` creates `PageAnalysis` from a PDF page.
2. `evaluate_digital_text_trust` creates `DigitalTextGateResult` from that
   analysis.
3. `decide_page_route` creates `PageDecisionResult` from the analysis, gate
   result, and current OCR runtime availability.

`pdfword.ocr_pipeline` remains the execution orchestrator. It calls those three
operations exactly once per selected page and executes only the selected next
path. There is no second page-routing pipeline in Clouda Lab or in the engine
registry.

`DirectPdfTextEngine` remains a pure extraction primitive. It does not own page
analysis, trust policy, or routing. The canonical runtime calls it only after a
trusted decision. A `TrustedDigitalTextContext` produced by the routing module
binds the authorization to the SHA-256 digest of the PDF bytes, page number,
gate verdict, and extracted-text digest. The orchestrator validates that context
before invoking the direct engine. This is an accidental-bypass guard, not a
security claim: the low-level extraction primitive remains importable for tests
and tooling.

Direct PDF extraction reports `confidence=None`. Embedded-text trust comes only
from `DigitalTextGateResult` and its explicit evidence. The implementation will
remove the fabricated direct-extraction confidence of `100.0`.

The existing document-level utilities in `pdfword.document_analysis` and
`pdfword.intelligence` remain available for compatibility, but they do not make
page routing decisions. Where they need the same facts, they may delegate to the
new analyzer rather than introducing competing heuristics.

## Typed contracts

All enums are string enums so their serialized values are stable and readable.
All result types are frozen dataclasses. Diagnostic serialization is explicit;
raw extracted text and PDF bytes are never included in browser-safe diagnostic
payloads.

### Analysis enums

`EvidenceAvailability` has the values `available` and `unavailable`.

`AnalysisWarningCode` includes:

- `text_extraction_failed`
- `text_geometry_unavailable`
- `image_metadata_unavailable`
- `content_stream_unavailable`
- `unsupported_page_geometry`

Warnings identify unavailable evidence. They do not fabricate replacement
values.

### Text geometry

`TextSpan` contains the internal facts needed for analysis:

- normalized span text;
- optional `(x0, y0, x1, y1)` bounding box in PDF points;
- optional font size;
- source sequence index.

Text is retained only in the in-memory analysis contract because the gate needs
Unicode structure and fragmentation evidence. `PageAnalysis.to_diagnostics()`
omits span text and returns counts, ratios, booleans, reason codes, and rounded
geometry only.

### `PageAnalysis`

`PageAnalysis` contains:

- `page_number`, `page_width_pt`, and `page_height_pt`;
- raw embedded text for the trusted execution path and normalized text for
  deterministic checks;
- `embedded_text_present`, raw character count, normalized character count,
  non-whitespace character count, alphanumeric character count, word count,
  line count, and span count;
- tuple of `TextSpan` records;
- text bounding-box coverage ratio when geometry is available;
- vertical and horizontal distribution bucket counts when geometry is
  available;
- image count, largest embedded-image pixel count, and the existing conservative
  image-density proxy when image metadata is available;
- visible content-operation evidence from the PDF content stream, limited to
  counts of text-showing, image-drawing, and path-painting operations;
- blank and near-blank evidence flags;
- sparse-text, fragmented-text, and hybrid-page indicators;
- Arabic codepoint count and ratio;
- isolated-Arabic-character ratio, pathological inter-letter spacing count,
  replacement/control/private-use character count, and Arabic integrity risk;
- layout-order risk and multi-column risk when span geometry supports those
  conclusions;
- warnings and machine-readable evidence codes.

Counts are integers, ratios are derived measurements between zero and one, and
unavailable measurements are `None`. The analyzer does not emit a model-style
confidence value.

### Gate verdict and evidence

`DigitalTextGateVerdict` has exactly:

- `trusted`
- `untrusted`
- `uncertain`

`DigitalTextReasonCode` includes positive and negative evidence:

- `complete_digital_text`
- `sufficient_text_distribution`
- `harmless_decorative_images`
- `no_embedded_text`
- `insufficient_usable_text`
- `suspicious_sparse_text`
- `suspicious_fragmentation`
- `arabic_isolated_character_runs`
- `arabic_pathological_spacing`
- `unicode_corruption_detected`
- `image_dominant_partial_text`
- `hybrid_page_incomplete_text`
- `layout_order_risk`
- `complex_layout_risk`
- `required_evidence_unavailable`
- `conflicting_evidence`

`GateCheck` records one named check, whether it passed, its reason code, and
small scalar evidence. `DigitalTextGateResult` contains the closed verdict, all
checks, and deduplicated reason codes. An internal engineering-only diagnostic
score may be retained if it helps calibration, but it only summarizes checks,
never determines the verdict by itself, and is excluded from end-user, Lab, and
conversion-result presentation.

The gate is fail-closed:

- no embedded or usable text is `untrusted`;
- clear partial/hidden/hybrid text evidence is `untrusted`;
- deterministic Arabic corruption is `untrusted` when severe and `uncertain`
  when borderline;
- missing structural evidence or conflicting signals is `uncertain`;
- `trusted` requires all mandatory checks to pass and at least one positive
  completeness/distribution signal;
- decorative images do not fail an otherwise complete digital page;
- an image-backed page with sparse or localized text is never trusted.

Thresholds are named module constants, documented next to their rationale, and
covered at the boundary by tests. No single character-count threshold can grant
trust. Character amount, fragmentation, distribution, image evidence, Unicode
integrity, and layout risk are evaluated as separate checks.

### Trusted context

`TrustedDigitalTextContext` is created only by the routing module when the gate
verdict is `trusted`. It contains:

- PDF SHA-256 digest;
- page number;
- normalized-text SHA-256 digest;
- the trusted gate result.

The runtime validates all four fields immediately before direct extraction and
rejects a mismatch as `review_required`. It does not silently reroute a context
mismatch to direct text.

### Page decision

`PageDecision` has exactly:

- `trusted_digital_text`
- `ocr_required`
- `review_required`
- `blank_or_near_blank`

`PageNextPath` has exactly:

- `direct_pdf_text`
- `local_ocr`
- `pending_ocr_model`
- `manual_review`
- `no_extraction`

`PageDecisionResult` contains the decision, next path, gate verdict, reason
codes, compact evidence, `ocr_required`, `ocr_available`, `review_required`, and
optional trusted context.

Decision precedence is deterministic:

1. Strong blank evidence selects `blank_or_near_blank` and `no_extraction`.
2. Strong near-blank evidence selects `blank_or_near_blank`; meaningful short
   text is preserved in `PageResult` for review compatibility.
3. A trusted gate selects `trusted_digital_text` and `direct_pdf_text`.
4. An untrusted gate selects `ocr_required`; the next path is `local_ocr` only
   when the existing feature-flagged registered OCR engine is available,
   otherwise `pending_ocr_model`.
5. An uncertain gate selects `review_required` and `manual_review`. It is not
   reported as successful direct extraction.

## Deterministic analysis

The analyzer uses `pypdf.PdfReader` page objects already created by the runtime.
It uses `extract_text` plus visitor callbacks for text sequence and geometry,
page media-box dimensions, embedded-image metadata, and read-only content-stream
operator inspection. It does not render unless the selected OCR path later
requires pixels.

Text normalization is Unicode-preserving and limited to line-ending
normalization, directional-mark removal already used by the runtime, whitespace
measurement, and an analysis-only normalized form. The text returned for a
trusted page remains the direct engine's extracted text after the existing
markdown cleanup; the analyzer does not rewrite document content.

Arabic checks are deliberately narrow:

- measure runs dominated by one-letter Arabic tokens;
- detect whitespace inserted between consecutive Arabic letters across a
  sustained run;
- count replacement characters, unexpected control characters, and private-use
  characters;
- compare usable Arabic amount with otherwise strong page-content evidence;
- mark uncertain layout when span ordering and coordinates conflict.

These checks identify obvious corruption risks. They do not claim to prove
Arabic reading order.

Multi-column risk is reported only when positioned spans form separated stable
horizontal clusters with overlapping vertical ranges. Reading-order risk is
reported only when extracted sequence repeatedly moves against the expected
vertical progression within a detected column or jumps between overlapping
columns. If geometry is absent, both metrics are unavailable rather than
guessed from whitespace.

Image dominance uses available PDF object metadata as a conservative proxy. The
field name and documentation state that it is not rendered pixel coverage. If
image metadata cannot be read, the gate treats image completeness evidence as
unavailable.

## Runtime integration

`process_pdf` will keep its public signature and return type. For each page it
will:

1. build `PageAnalysis`;
2. evaluate the digital-text gate;
3. inspect the already configured engine registry for an available local OCR
   engine;
4. obtain `PageDecisionResult`;
5. execute exactly the decision's next path;
6. map the typed result to the compatibility-facing `PageResult`.

Compatibility mappings are:

| Page decision | `PageResult.route_used` | Compatibility behavior |
|---|---|---|
| trusted digital text | `direct_pdf_text` | Extract and preserve text; direct confidence is `None` |
| OCR required, available | registered local engine name | Existing quality/review checks remain active |
| OCR required, unavailable | `pending_ocr_model` | Existing pending message and non-success state |
| review required | `review_required` | Preserve no untrusted text in the DOCX output; expose reasons in metadata |
| blank | `blank_page` | Preserve page boundary, no review required |
| near blank | `near_blank` | Preserve meaningful short text, require review |

Every `PageResult.metadata` gains a `document_intelligence` object containing
the decision, gate verdict, reason codes, safe analyzer diagnostics, next path,
and OCR availability. Existing top-level keys such as `page_state` and
`embedded_text_chars` remain during migration.

Failures are conservative:

- analyzer failure becomes `review_required` unless the content-stream evidence
  independently proves an image-only page requiring OCR;
- direct extraction failure after trusted authorization becomes
  `review_required`, never an unexamined OCR success;
- unavailable OCR preserves `pending_ocr_model`;
- local OCR errors preserve the existing failed/pending behavior and diagnostics;
- no untrusted embedded text is substituted merely to populate output.

## Direct extraction primitive

`DirectPdfTextEngine.extract_page` retains its current model-agnostic engine
signature for registry compatibility. It performs only page-bound validation,
embedded-text extraction, timing, and `OCRResult` construction. It does not call
the analyzer or gate.

For non-empty extraction it returns `OCR_STATUS_SUCCEEDED` with
`confidence=None`. Empty extraction retains a non-success status compatible with
current callers. The canonical runtime does not use that status to decide trust;
it invokes the primitive only after validating `TrustedDigitalTextContext`.

`available_engine_status` no longer labels the direct engine as the globally
active route. It identifies it as available but conditionally selected by the
page decision engine.

## Clouda Lab integration

The existing Lab gains a Document Intelligence page and no second dashboard.
A focused `DocumentIntelligenceService` delegates to the same canonical routing
module used by production.

The Lab endpoint accepts an uploaded PDF, not a filesystem path. It is protected
by the existing loopback dependency and action token, enforces a small byte and
page limit, performs no persistence, and makes no network call. The response is
passed through `browser_safe` and contains only:

- page number;
- page decision and next path;
- gate verdict;
- reason codes and scalar check evidence;
- safe analyzer diagnostics;
- OCR required/available/pending flags;
- review and blank flags.

It never returns extracted page text, PDF bytes, environment data, secrets, or
absolute paths. It also never presents an engineering diagnostic as an accuracy,
confidence, or quality percentage. The static page uses safe DOM text rendering
and shows loading, validation, empty, and failure states. Navigation performs no
analysis until a developer explicitly submits a local file.

## Tests and fixtures

Critical routing behavior follows red-green-refactor development. Small local
PDF fixtures are generated deterministically in tests with the repository's
existing fixture tools and libraries. No test downloads data or makes a network
request.

Focused coverage includes:

1. complete born-digital text routes to trusted direct extraction;
2. textless content routes to OCR or blank according to content evidence;
3. an image-only scan routes to OCR;
4. a tiny hidden/partial text layer is not trusted;
5. a hybrid image page with incomplete text is not trusted;
6. a complete digital page with decorative images can be trusted;
7. fragmented Arabic extraction is untrusted or uncertain with Arabic reason
   codes;
8. materially ambiguous or unavailable evidence routes to review;
9. a blank page is explicitly blank;
10. unavailable OCR preserves `pending_ocr_model`;
11. `OCRResult` and existing registry contracts remain compatible;
12. local-model feature flags and pinned-revision checks remain respected;
13. Lab exposes sanitized decisions and no page text or path;
14. analyzer and Lab tests prove no network/download function is invoked;
15. existing PDF conversion, DOCX, page-order, and near-blank behavior remains
    valid where compatible with the new trust gate;
16. `DirectPdfTextEngine` returns `confidence=None`;
17. trusted context cannot be reused for another document or page;
18. all enum and reason-code serialization is stable;
19. Lab and conversion-facing diagnostics contain no routing accuracy,
    confidence, or quality percentage.

Tests are split by responsibility:

- `tests/test_page_routing.py` covers analyzer, gate, decisions, and context;
- existing runtime/engine tests are updated for integration and compatibility;
- `tests/dashboard/test_document_intelligence.py` covers the Lab service and API;
- UI contract tests cover the new page without relying on browser-side HTML
  injection.

## Documentation changes

Only directly affected documentation changes:

- `README.md` describes conditional trusted extraction and the four semantic
  decisions;
- `docs/ARCHITECTURE.md` documents the analyzer, gate, decision engine, and
  execution paths;
- `docs/runtime/LOCAL_OCR_INTEGRATION.md` documents that untrusted text routes to
  local OCR or pending OCR rather than direct extraction;
- `docs/lab/CLOUDA_LAB_BACKEND.md` documents the bounded diagnostics endpoint and
  its security/data-handling rules;
- `docs/TESTING.md` lists the new routing and Lab coverage.

Claims that embedded text is automatically trusted or that direct extraction is
unconditional are removed. Existing benchmark results may remain factual, but
the documentation will not state that a final OCR model or training method has
been selected.

## Verification and Git integration

Focused tests run after each TDD cycle. Before integration, the exact required
commands run from the feature branch:

```powershell
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m mypy .
node --check clouda_lab/dashboard/static/app.js
python -m build --wheel --no-isolation --skip-dependency-check
git diff --check
```

After successful verification, the workflow fetches `origin/main` again. If it
changed, the feature branch is synchronized without rewriting published
history, affected and full verification are rerun, and the feature branch is
pushed. Integration into `main` uses a normal merge and push when allowed; if
branch protection requires a pull request or any operation is ambiguous, the
workflow stops and reports the safe next action. There is no force push, remote
branch deletion, hard reset, or history rewrite.

## Acceptance behavior

- A trustworthy digital page uses direct embedded text with a trusted gate
  result and no fabricated confidence.
- A scanned page requires OCR and remains pending when no approved OCR runtime
  is available.
- A partial hidden OCR layer is not trusted and cannot silently populate output.
- A hybrid page is trusted only when structural evidence supports complete
  digital text; otherwise it requires OCR or review.
- An ambiguous page is explicitly `review_required`.
- A blank or deterministically near-blank page has an explicit blank decision.
- OCR-required pages preserve the canonical pending-model state when OCR is
  unavailable.
