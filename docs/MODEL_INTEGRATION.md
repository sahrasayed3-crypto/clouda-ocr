# Model Integration Policy

## Current state

The model-selection benchmark v1.0 is complete and published in the separate
[clouda-ocr-model-selection-benchmark](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark)
repository: 462 Arabic document pages, 10 candidate OCR/VLM models, 5 complete
ranked runs, 3 partial runs, and 2 failed runs; the primary ranking metric is
Normalized Arabic CER, and 7,198 / 7,198 release checksums were verified. Model
selection was completed from these published results, and the current project
phase is selected-model development and training. The earlier 177-page
benchmark v0.1.0 (historical) remains published as a separate record; on it,
HunyuanOCR-1.5 ranked first by Normalized Arabic CER (0.391497), a
benchmark-specific historical result. No model is installed or integrated in
this repository as a final OCR engine, and no final trained OCR model exists
yet. Dataset and model rights remain governed separately by the fail-closed
licensing and provenance process. The runtime performs direct extraction only;
scanned or image-only pages remain
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
