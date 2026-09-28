# Benchmarks

## Arabic OCR benchmarks

The metadata-only 177-page Arabic OCR benchmark v0.1.0 (historical) is
documented in [`ocr_arabic/`](ocr_arabic/README.md). It covers 100 clean-source
records and a common 177-page distorted cohort. Its public area contains
results, methodology, hashes, sanitized provenance, rights classifications,
model-status metadata, and validation code only; all image, ground-truth,
raw-output, and permission evidence remains private and local.

The current 462-page model-selection benchmark v1.0 is published in the
separate
[clouda-ocr-model-selection-benchmark](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark)
repository: 10 candidate OCR/VLM models, 5 complete ranked runs, 3 partial
runs, and 2 failed runs, ranked by Normalized Arabic CER with 7,198 / 7,198
release checksums verified.

## Local PDF-to-DOCX benchmark

This benchmark measures only local PDF text extraction, local OCR engines that are actually available, the current local router, preprocessing effects, and DOCX output validation.

Run from the project root:

```powershell
.\.venv311\Scripts\python.exe benchmarks\scripts\run_local_benchmark.py
```

Outputs:

- `benchmarks/manifest.json`
- `benchmarks/results/local_benchmark_results.json`
- `benchmarks/results/local_benchmark_results.csv`
- `benchmarks/results/engine_summary.json`
- `benchmarks/results/engine_summary.csv`
- `benchmarks/results/router_summary.json`
- `benchmarks/results/docx_validation.json`
- `reports/local_benchmark/*.md`

The cloud route is intentionally not implemented, mocked, or tested by this benchmark.
