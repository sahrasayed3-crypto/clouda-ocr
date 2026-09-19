# OCR Self-Review and Selective Re-Read Design

## Purpose

Document Intelligence already decides whether a PDF page may use trusted
digital text or requires OCR. This stage makes the OCR route equally
conservative: a successful engine call is evidence, not proof. The runtime
self-reviews observable evidence, selectively re-reads only verified bounded
regions, compares both results, and reconciles categorically.

Conversion and Clouda Lab expose closed states and reason codes, never an OCR
accuracy, confidence, or quality percentage.

## Scope and boundaries

The change adds a canonical model-agnostic OCR self-review domain and integrates
it after actual local OCR execution. It reuses `PageAnalysis`, the Digital Text
Gate, Page Decision Engine, `render_pdf_page_to_png_bytes`, `ExtractionEngine`,
`OCRResult`, `OCRBox`, `PageResult`, and the existing bounded Lab endpoint.

It includes deterministic review, bounded regions/crops/re-reads,
reconciliation, runtime and Lab projection, DOCX review behavior, deterministic
fake-provider tests, and an opt-in CUDA plumbing smoke path. It does not select
or download a model/dataset, train, benchmark, call a hosted OCR service, alter
pre-OCR routing, or create parallel storage, task, conversion, or Lab systems.

`pdfword/ocr_self_review.py` owns all OCR self-review policy. `ocr_pipeline.py`
remains the executor: it renders once, runs first-pass OCR, delegates review,
runs justified crops through the same configured local engine, then maps the
reconciliation result to `PageResult`. Trusted digital, blank, and near-blank
pages bypass OCR self-review. An unavailable engine remains `pending_ocr_model`.

## Typed categorical contracts

All public enums are `StrEnum` values, result records are frozen dataclasses,
and serialized diagnostics exclude OCR text, crop/PDF bytes, coordinates,
paths, hashes, secrets, and provider payloads.

`OCRReviewVerdict`: `accepted`, `reread_required`, `review_required`, `failed`.

`OCRIssueCode` contains only observed conditions:

- `empty_ocr_output`, `suspiciously_sparse_output`, `duplicate_line_sequence`,
  `repeated_text_block`, `reading_order_risk`, `layout_coverage_gap`,
  `low_engine_confidence`, `missing_expected_region`,
  `arabic_fragmentation`, `arabic_pathological_spacing`, `unicode_corruption`,
  `conflicting_ocr_evidence`, `engine_error`, `reread_disagreement`, and
  `required_evidence_unavailable`.

`OCRReviewResult` contains verdict, issue codes, bounded suspicious regions,
safe diagnostics, `selective_reread_justified`, and `manual_review_required`.
`ReReadResult` contains region and engine identity/revision, categorical
success/failure, internal OCR text, optional boxes/order, optional documented
native confidence, and safe diagnostics. `OCRReconciliationResult` contains
final state, selected text, accepted replacements, unresolved regions, reason
codes, review flag, and first-pass/re-read provenance.

`OCRPageState`: `accepted_first_pass`, `accepted_after_selective_reread`,
`review_required`, `pending_ocr_model`, and `ocr_failed`.

An engine-native confidence is optional. If supplied finite and documented it
may produce `low_engine_confidence`; absent confidence is not a failure and is
never converted into a universal Clouda score.

## Exact rendered-image coordinate binding

`RenderedPageContext` contains page number, a private digest identity of the
in-memory PNG, exact pixel width/height, and image-pixel coordinate space. A
`ReviewRegion` contains a stable ID, page number, integer pixel box, source
render identity/dimensions/coordinate-space declaration, category, priority,
reasons, and safe evidence.

A crop is permitted only when its page number, render identity, dimensions, and
explicit image-pixel coordinate system all match the current context; every
coordinate is finite/non-negative/ordered; conservative clamping keeps it
within bounds; it satisfies minimum crop dimensions/area; and it stays inside
per-page region, attempts, pixels, and bytes limits.

`OCRBox` currently has no guaranteed coordinate-system contract. A box may
localize a re-read only after its metadata explicitly binds it to the current
rendered image. Invalid, stale, mismatched, unverified, negative, NaN,
infinite, or ambiguous boxes create no crop. They fail closed to
`review_required` with `required_evidence_unavailable`; Clouda never infers or
invents a crop.

Regions are normalized to current image pixels, stably ordered, deduplicated by
high overlap, and merged only when nearby compatible reasons justify it.
Defaults are finite and centralized: at most four regions, one attempt per
region, four total attempts, and bounded total crop pixels/bytes per page.
Exceeding a bound becomes explicit review, never a retry loop.

## Review, re-read, and reconciliation

First-pass OCR is reviewed only after an attempt, using available evidence:
empty/sparse output; duplicate lines/blocks; Unicode replacement, control, or
private-use corruption; existing canonical Arabic fragmentation/pathological
spacing signals; explicit ordering anomalies; verified box coverage gaps;
documented native confidence; engine error; and contradictions with pre-OCR
analysis. Unavailable evidence remains unavailable, not an optimistic score.

A localized defect is `reread_required` only if a verified bounded region
exists. Non-localized defects and unavailable required localization become
`review_required`. The executor decodes the one rendered page in memory,
creates crops with Pillow, and passes crop PNG bytes to the existing configured
engine. It checks cancellation before first pass and each crop. No region is
re-rendered and no automatic download/network provider is added.

Reconciliation accepts only normalized exact agreement or a replacement
strictly local to a verified suspicious region when it removes a known
corruption and preserves supported geometry/order. It never chooses the longer
text. Failed re-read, disagreement, unsupported boundaries, and unresolved
issues become `review_required`. Provenance records sources and accepted
replacement reasons internally. Review output preserves its page boundary and a
categorical placeholder while withholding disputed OCR text.

## Legacy score compatibility

Synthetic OCR quality/confidence scores are no longer authoritative for OCR
acceptance. `estimate_quality_components()` and `final_acceptance_decision()`
must not gate, overwrite, or reclassify any page produced by the new review and
reconciliation runtime.

Legacy numeric `PageResult` and database/conversion fields remain only for
backward compatibility with unrelated historical records. New-runtime pages
set them to `None` unless a separate defined and validated metric is approved
later. Any retained legacy value is diagnostic-only, never user-facing or Lab
visible, and cannot override the categorical reconciliation state or its
accepted/review-required outcome. Conversion normalization preserves the
categorical result rather than fabricating a score from missing data.

## Runtime, Lab, and DOCX integration

The canonical path is:

```text
Page Analyzer -> Digital Text Gate -> Page Decision Engine
  -> trusted direct text | blank/near-blank | pending OCR
  -> first-pass OCR -> self-review -> selective re-read -> reconciliation
```

`PageResult.metadata` gains a safe self-review projection. Existing DOCX review
placeholders continue to preserve page boundaries. Clouda Lab stays 10 MiB/25
pages, loopback-only, action-token protected, and browser-sanitized. It may
show only first-pass state, review verdict/reasons, suspicious/re-read counts,
accepted/rejected re-read counts, final state, unresolved flag, and canonical
engine identity/revision. It must not return OCR/page text, crop data,
coordinates, hashes, paths, environment values, secrets, native confidence, or
accuracy/confidence/quality percentages.

## CUDA and verification

An optional `CLOUDA_CUDA_SMOKE=1` path may inspect CUDA availability, allocate
a tiny tensor, test fake-provider crop wiring, release memory, and map CUDA
OOM/device errors to bounded failure states. It skips without CUDA, downloads
nothing, does no benchmark or real inference by default, and proves neither OCR
accuracy nor future model/VRAM/multi-GPU fit.

Tests use small deterministic PDFs and fake providers to cover clean first
pass, empty/duplicate/corrupt Arabic/Unicode output, verified/missing geometry,
dedupe and budgets, replacement/disagreement/provenance, missing confidence,
pending/bypass paths, Lab redaction, DOCX boundaries, no-network/default-no-
download behavior, and CUDA skip/device errors.

The prior dropped baseline output is unresolved verification. Before any
success, merge, or push claim, capture real exit status for a full baseline and
final run of:

```powershell
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m mypy .
node --check clouda_lab/dashboard/static/app.js
python -m build --wheel --no-isolation --skip-dependency-check
git diff --check
```

No partial or dropped output counts as verification.
