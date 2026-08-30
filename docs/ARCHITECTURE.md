# Architecture

## Active Pipeline

1. A user uploads a PDF through `app.py`.
2. The file is stored under the configured storage root.
3. A conversion job calls `pdfword.conversion_service`.
4. `pdfword.ocr_pipeline.process_pdf()` validates page selection and reads each page.
5. `pdfword.engines.DirectPdfTextEngine` extracts embedded text from born-digital PDF pages.
6. Blank pages are marked `blank_page`; pages with only a short embedded fragment or small image signal are marked `near_blank`; image-only scanned pages are marked `pending_ocr_model`.
7. `pdfword.docx_export` writes an editable text-only DOCX.

## Engine Interface

`pdfword.engines.ExtractionEngine` is the extension point for future OCR models.

Current engines:

- `DirectPdfTextEngine`: active, extracts embedded PDF text.
- `FutureOcrEngine`: placeholder, inactive, represents scanned-page OCR after the selected primary candidate passes licensing, integration, and evaluation gates.

The engine interface is intentionally model-agnostic. It can accept page images as bytes, page image paths, or PDF bytes with a page number. It returns a unified `OCRResult` schema with optional fields for model name, confidence, processing time, layout boxes, reading order, metadata, and error message. Engines are registered through `EngineRegistry`, so future model integrations can be added without hard-coding model names in the main router.

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

- `digital_text`: embedded/selectable text was extracted without OCR.
- `blank_page`: no embedded text or image signal was found. The page boundary is retained.
- `near_blank`: only a short text fragment (such as a page number) or a small visual signal was found. The content is preserved for manual review.
- `pending_ocr_model`: an image-only or scanned page requires a future approved OCR engine. It is neither a successful OCR result nor a final processing failure.
- `failed`: input validation or conversion failed and is recorded at job level with the error message.
- `manual_review`: a job-level outcome when one or more pages require review.
