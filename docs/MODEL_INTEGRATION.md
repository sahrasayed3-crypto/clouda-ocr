# Model Integration Policy

## Current state

A primary model candidate has been selected, but it is not installed or integrated as a final OCR engine. Training has not started yet because dataset licensing and written-permission verification are still in progress. The runtime performs direct extraction only; scanned or image-only pages remain `pending_ocr_model`.

## Integration contract

Add the selected primary candidate, or any future fallback engine, by implementing `pdfword.engines.ExtractionEngine`, returning `OCRResult`, and registering it in `EngineRegistry`. The contract supports optional confidence, layout boxes, reading order, timing, error details, and metadata without assuming a vendor, framework, CPU, GPU, CUDA, ROCm, or model family.

## Required gate before activation

1. Document the selected primary candidate, licence, supported languages, hardware, and dependencies.
2. Add a separate optional dependency group; do not make it required for direct extraction.
3. Implement CPU-safe unavailable-model handling that keeps pages `pending_ocr_model` rather than crashing.
4. Evaluate against a versioned, consented ground-truth set containing Arabic, English, mixed RTL/LTR, digital, scanned, old, and low-quality pages.
5. Publish CER/WER only with the dataset definition, sample counts, methodology, hardware, and uncertainty.
6. Add regression tests for success, failure, cancellation, resource cleanup, page order, and metadata.

## AMD/ROCm readiness

The interface is hardware-neutral. AMD/ROCm support is not claimed until an integrated engine is tested on documented hardware and software versions. No GPU job is required by the current CI workflow.
