# Clouda OCR — Due-Diligence Source Map

This map links the current-state claims in
`INVESTOR_TECHNICAL_DUE_DILIGENCE.md` to maintained evidence. It intentionally
does not convert historical test snapshots or benchmark records into current
production claims.

| Claim | Primary evidence |
| --- | --- |
| Canonical project name and current runtime boundaries | `README.md`, `docs/BRAND_MIGRATION.md` |
| v0.2.1 package version and release metadata | `pyproject.toml`, `CITATION.cff`, [v0.2.1 release](https://github.com/sahrasayed3-crypto/clouda-ocr/releases/tag/v0.2.1) |
| Trusted digital-text PDF-to-DOCX route and review routing | `pdfword/`, `docs/ARCHITECTURE.md`, `tests/test_page_routing.py` |
| Scanned pages route to `pending_ocr_model` | `README.md`, `pdfword/page_routing.py`, `pdfword/ocr_pipeline.py` |
| Model-agnostic engine boundary | `pdfword/engines.py`, `pdfword/model_registry.py`, `tests/test_model_agnostic_engines.py` |
| Data, quality, provenance, and training-planning subsystems | `clouda_data/`, `clouda_training/`, `docs/pretraining_dataset_infrastructure.md`, `docs/training/` |
| Loopback-only Lab boundaries | `clouda_lab/`, `docs/lab/`, `tests/lab/` |
| Current v1.0 benchmark status and ranked results | [model-selection benchmark v1.0 release](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark/releases/tag/v1.0.0), `README.md`, `benchmarks/README.md` |
| Historical v0.1.0 benchmark | `benchmarks/ocr_arabic/README.md`, `benchmarks/ocr_arabic/RESULTS.md`, `README.md` |
| CI platforms, commands, required gates, and Python version | `.github/workflows/ci.yml` |
| Historical and current test-reporting policy | `docs/TESTING.md`, `docs/engineering/RELEASE_READINESS_V0.2.1.md` |
| Security and license boundaries | `SECURITY.md`, `LICENSE`, `NOTICE`, `DATA_LICENSES.md`, `THIRD_PARTY_NOTICES.md` |

## Interpretation rules

- Benchmark quality results are not production application accuracy claims.
- A current benchmark winner is not a final integrated production model.
- The v0.1.0 177-page benchmark is historical; v1.0 is the current
  462-page model-selection benchmark.
- CI status is specific to a commit and workflow run. Verify the relevant run
  instead of reusing a past green result.
- Release, dataset, model, and deployment rights must be verified separately;
  no model-license conclusion follows from this source map.
