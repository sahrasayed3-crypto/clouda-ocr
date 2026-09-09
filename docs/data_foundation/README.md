# Arabic OCR Distortion Data Foundation

> Migration note: this is the preserved Project B overview. The controlled
> RASAM acquisition described in `reports/RASAM_FIRST_BATCH_REPORT.md`
> occurred after some “foundation only” wording below was written. The merged
> system keeps all datasets outside Git and treats this subsystem as
> `clouda_data`.

This repository prepares the foundation for a later Arabic OCR training-data pipeline. The future pipeline will distort clean Arabic and Arabic-English document pages while preserving exact ground-truth text and layout metadata.

Current stage: functioning local CPU data-generation and evaluation system.

Implemented:

- Project folder structure for raw, staging, processed, synthetic-test, manifest, output, log, cache, temp, backup, notebook, source, scripts, configs, and docs areas.
- Validated configuration models.
- Formal clean-page input contract.
- Ground-truth preservation helpers and checksum checks.
- Real deterministic pixel-changing distortion engine; metadata-only mode is
  retained only for explicit tests.
- Distortion profiles for clean, modern, historical, scanner, phone, binding, faded, compressed, small-text, mixed-layout, and footnote-heavy scenarios.
- Layout-region abstractions.
- Manifest, status, checkpoint, JSONL, and SQLite storage foundations.
- Quality validation foundations.
- CER/WER evaluation foundations.
- CLI entry point.
- Tiny synthetic fixtures for tests only.

Ingestion preparation now adds copy-only clean source registration, dry-run validation, source-document manifests, page manifests, file registry checksums, and duplicate detection.

Dataset acquisition preparation adds a verified candidate-source registry, license matrix, safe sample downloader, download manifests, and sample-to-ingestion validation.

External or intentionally disabled:

- Dataset download.
- OCR model download.
- OCR training.
- Real model-weight selection and GPU training.
- AWS resource creation.

Useful commands after a real Python 3.10+ interpreter is available:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python -m clouda_data.pipeline.cli validate-project
.\.venv\Scripts\python -m clouda_data.pipeline.cli validate-source tests/fixtures/ingestion/source_manifest.json
.\.venv\Scripts\python -m clouda_data.pipeline.cli ingest tests/fixtures/ingestion/source_manifest.json --dry-run
.\.venv\Scripts\python -m clouda_data.pipeline.cli list-dataset-sources
.\.venv\Scripts\python -m clouda_data.pipeline.cli download-dataset-sample rasam_dataset --max-bytes 104857600
```

## Data Factory

The integrated synthetic data factory lives in `clouda_data.factory`. See [DATA_FACTORY.md](DATA_FACTORY.md).
