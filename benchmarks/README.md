# Benchmarks

## Final Arabic OCR model benchmark

The metadata-only final Arabic OCR benchmark is documented in
[`ocr_arabic/`](ocr_arabic/README.md). It covers 100 clean-source records and a
common 177-page distorted cohort. Its public area contains results, methodology,
hashes, sanitized provenance, rights classifications, model-status metadata,
and validation code only; all image, ground-truth, raw-output, and permission
evidence remains private and local.

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
