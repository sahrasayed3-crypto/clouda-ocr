# Model Integration Policy

## Current state

The published 177-page benchmark v0.1.0 is complete. A larger 462-page held-out
evaluation is in progress and currently paused pending additional compute
capacity, so production model selection remains open. On the published
177-page benchmark, HunyuanOCR-1.5 ranked first by Normalized Arabic CER
(0.391497); this is a benchmark-specific historical result, not a final
production-model selection, and no model is installed or integrated as a final
OCR engine. Runtime integration and validation of candidate OCR/VLM models are
the next major technical stage, currently limited primarily by access to
suitable GPU compute. A dedicated Clouda-trained model may be pursued only if
benchmark evidence shows a meaningful gap that existing open and self-hostable
models do not adequately close. Dataset and model rights remain governed
separately by the fail-closed licensing and provenance process. The runtime
performs direct extraction only; scanned or image-only pages remain
`pending_ocr_model`.

## Integration contract

Add a candidate engine, or any future fallback engine, by implementing `pdfword.engines.ExtractionEngine`, returning `OCRResult`, and registering it in `EngineRegistry`. The contract supports optional confidence, layout boxes, reading order, timing, error details, and metadata without assuming a vendor, framework, CPU, GPU, CUDA, ROCm, or model family.

## Required gate before activation

1. Document the selected primary candidate, licence, supported languages, hardware, and dependencies.
2. Add a separate optional dependency group; do not make it required for direct extraction.
3. Implement CPU-safe unavailable-model handling that keeps pages `pending_ocr_model` rather than crashing.
4. Evaluate against a versioned, consented ground-truth set containing Arabic, English, mixed RTL/LTR, digital, scanned, old, and low-quality pages.
5. Publish CER/WER only with the dataset definition, sample counts, methodology, hardware, and uncertainty.
6. Add regression tests for success, failure, cancellation, resource cleanup, page order, and metadata.

## AMD/ROCm readiness

The interface is hardware-neutral. AMD/ROCm support is not claimed until an integrated engine is tested on documented hardware and software versions. No GPU job is required by the current CI workflow.
