# Architecture

## Active Pipeline

1. A user uploads a PDF through `app.py`.
2. The file is stored under the configured storage root.
3. A conversion job calls `pdfword.conversion_service`.
4. `pdfword.ocr_pipeline.process_pdf()` validates page selection and reads each page.
5. `Page Analyzer` gathers bounded structural, visible-content, text-integrity, and layout evidence. The PDF SHA-256 is computed once per document and shared by every page context.
6. `Trusted Digital Text Gate` returns a categorical `trusted`, `untrusted`, or `uncertain` verdict with explicit reason codes.
7. `Page Decision Engine` selects trusted direct extraction, OCR, pending OCR, review, blank, or near-blank handling. A short character count alone can never establish `near_blank`.
8. `DirectPdfTextEngine` is a pure extraction primitive. It can run only with a typed trusted context issued by the canonical decision layer, and its extracted-text digest is checked again afterward. A mismatch becomes `review_required` with the `manual_review` next path; the untrusted text is not emitted.
9. `pdfword.docx_export` writes an editable text-only DOCX. Review pages retain their page boundary and contain an explicit non-text review placeholder.

## Engine Interface

`pdfword.engines.ExtractionEngine` is the extension point for future OCR models.

Current engines:

- `DirectPdfTextEngine`: conditionally available extraction primitive; it owns no trust or routing policy.
- `FutureOcrEngine`: placeholder, inactive, represents scanned-page OCR after the selected primary candidate passes licensing, integration, and evaluation gates.

The engine interface is intentionally model-agnostic. It can accept page images as bytes, page image paths, or PDF bytes with a page number. It returns a unified `OCRResult` schema with optional fields for model name, internal confidence, processing time, layout boxes, reading order, metadata, and error message. Direct extraction reports `confidence=None`; its trust comes only from `DigitalTextGateResult`. Internal diagnostic values are not user-facing accuracy, confidence, or quality percentages. Engines are registered through `EngineRegistry`, so future model integrations can be added without hard-coding model names in the main router.

## Explicit Non-Goals For Current Version

- No legacy OCR integration.
- The selected primary model candidate is not integrated, trained/adapted, or final.
- No ROCm/GPU support claim.
- No CUDA-specific or ROCm-specific engine contract.
- No layout-perfect PDF reconstruction.

## Storage

Runtime files are kept outside Git:

- `data/`
- `conversions/`
- `logs/`
- `backups/`
- `.secrets/`

## Page-state contract

- `digital_text`: embedded text passed the trust gate and its post-extraction digest matched the trusted context.
- `blank_page`: no embedded text or image signal was found. The page boundary is retained.
- `near_blank`: structural and visible-content evidence establishes that the page is nearly empty. Short text by itself is insufficient, protecting legitimate title and other short pages.
- `pending_ocr_model`: an image-only or scanned page requires a future approved OCR engine. It is neither a successful OCR result nor a final processing failure.
- `failed`: input validation or conversion failed and is recorded at job level with the error message.
- `manual_review`: a job-level outcome when one or more pages require review.

The canonical routing flow is:

```text
Page Analyzer
-> Trusted Digital Text Gate
-> Page Decision Engine
-> trusted direct extraction OR OCR/pending/review/blank path
```
