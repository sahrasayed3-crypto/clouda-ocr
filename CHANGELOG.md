# Changelog

All notable changes are documented here.

## Unreleased

- Integrated the Clouda Data Factory as `clouda_data.factory`: deterministic
  synthetic Arabic OCR data generation (ingest, Arabic RTL rendering, atomic +
  composite distortions, QC, exports, JSONL/CSV manifests, seeds, resume,
  multiprocessing) with `factory-*` commands in the canonical CLI, a
  lossless factory-to-canonical-manifest adapter, SIL OFL 1.1 bundled fonts,
  and Training Experiment Framework dry-run integration. The standalone
  `clouda-data-factory` repository is superseded.
- Added explicit `blank_page` and `near_blank` page states with per-page metadata.
- Added deterministic readiness fixtures, tests, demo, Windows CI, and publication documentation.
- Kept image-only scanned pages in `pending_ocr_model`; no OCR model was added.
