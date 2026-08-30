# Local PDF-to-DOCX Benchmark

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
