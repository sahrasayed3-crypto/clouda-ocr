# Clouda OCR E2E Release Hardening Design

## Goal

Provide a tiny, deterministic, CPU-only release check for the existing PDF
conversion pipeline. It proves the public categorical outcomes and generated
DOCX boundaries without selecting a model, downloading assets, contacting a
provider, or exposing numeric quality claims.

## Scope and constraints

- The canonical runtime remains `pdfword.ocr_pipeline.process_pdf`; no second
  router, review system, or output generator is introduced.
- The check creates its own small PDFs in a temporary directory and uses only
  the built-in pending OCR engine plus deterministic test-local OCR engines.
- Observable output is categorical: `digital_text`, `blank_page`,
  `near_blank`, `ocr_required`/`pending_ocr_model`, and `review_required`.
  User-facing accuracy, confidence, and quality percentages are prohibited.
- Each page remains represented in the DOCX, including pending and review
  pages. Untrusted embedded text is never emitted.
- Doctor remains loopback-safe and diagnostic-only. CUDA is informational and
  no model, dataset, GPU inference, training, benchmark, or external provider
  is invoked.

## Design

Add `pdfword.release_self_test`, a narrow library function returning a small
structured result. It creates one deterministic PDF containing a trusted
digital page, a genuine blank page, a short title page, and an image-only
page. It runs `process_pdf`, builds DOCX through the existing exporter, and
asserts page order, categorical routes, no fabricated scores, and presence of
the pending-page boundary. A second deterministic mock OCR invocation covers
the existing self-review/reread/reconciliation path without altering runtime
selection.

The existing `clouda-data doctor --deep` command calls that self-test as an
optional deep check and reports pass/fail text without raw document contents.
Its JSON schema uses existing Doctor check records. CLI remains a delegate to
the canonical service; Lab continues to invoke the same Doctor report.

Focused tests validate the release self-test result, Doctor integration, and
wheel package data. Existing conversion, routing, self-review, Lab security,
and full-suite tests remain the regression net. CI gains a JavaScript syntax
gate and invokes the deep Doctor check; it does not run a benchmark.

## Failure policy

Any malformed fixture, routing discrepancy, missing DOCX boundary, failed
mock reconciliation, or unexpected numeric score causes a deterministic
failed check. The self-test does not substitute untrusted text, retry without
a bounded policy, or claim successful OCR.

## Deferred work

GPU/model validation, model selection, data downloads, training, and
performance benchmarking remain a later phase.
