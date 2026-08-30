# Final Arabic OCR Benchmark

This directory is the public-safe, metadata-only representation of the final
Clouda Arabic OCR benchmark. It does not contain source images, clean images,
distorted images, ground-truth text, model outputs, correspondence, or other
dataset assets.

## Canonical identity

- Benchmark ID: `clouda-ocr-arabic-177-v1`
- Clean source records: 100
- Distorted benchmark pages: 177
- Canonical manifest SHA-256:
  `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893`
- Primary ranking metric: Normalized Arabic CER, lower is better

The exact canonical metadata manifest is `benchmark_manifest.jsonl`.
The benchmark assets and raw result evidence remain private and local.

## Source composition

| Source family | Clean records |
|---|---:|
| KITAB-Bench-derived datasets | 30 |
| Arabic E-Book Corpus | 20 |
| CALFA datasets | 30 |
| Arabic-img2md | 15 |
| Craneset free samples | 5 |
| **Total** | **100** |

`source_manifest.csv` preserves all 14 precise dataset identifiers, source item
identifiers, file hashes, derivative counts, and fail-closed rights decisions.
`source_summary.csv` provides the corresponding dataset-level summary. No
Muharaf or Baseer/Misraj material is present.

## Rights boundary

Evaluation permission, commercial-training permission, and public asset
redistribution permission are separate fields. Training permission never
implies redistribution permission. Package-local license labels are retained
only as raw provenance evidence; the verified rights review controls the
published classifications.

Model licenses, raw-output redistribution, and permission to use outputs as
training labels are separate again. See `models.csv`. An open or available
model must not be assumed to grant training-label rights.

Unknown or unresolved rights fail closed as `LOCAL_ONLY` or `NEEDS_REVIEW`.

## Files

- `RESULTS.md` — human-readable leaderboard and exclusions.
- `results.csv` — machine-readable complete, partial, and failed-run records.
- `methodology.md` — benchmark identity, metric implementation, selection
  rules, hardware caveat, and publication boundary.
- `benchmark_manifest.jsonl` — exact 177-row canonical metadata manifest.
- `source_manifest.csv` — sanitized 100-row source provenance and rights data.
- `source_summary.csv` — precise dataset-level composition and rights summary.
- `models.csv` — model, evaluation, output, and training-label status.
- `CITATIONS.md` — attribution and source-identifier guidance.
- `release.json` — compact release identity.
- `scripts/validate_release.py` — local metadata integrity validator.

## Validation

From the repository root:

```powershell
.\.venv311\Scripts\python.exe benchmarks\ocr_arabic\scripts\validate_release.py
.\.venv311\Scripts\python.exe -m pytest tests\benchmarks\test_ocr_arabic_release.py -q
```

This integration does not run OCR inference and does not change application or
runtime behavior.
