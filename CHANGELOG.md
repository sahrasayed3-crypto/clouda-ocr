# Changelog

All notable changes are documented here.

## Unreleased

## 0.2.1 - Unreleased (local preparation - not published)

- Brand migration: the canonical project and product name is now Clouda OCR.
  The v0.2.0 release remains historical evidence under its original title
  "Clouda PDF"; no tags, release assets, or git history were rewritten.
- Distribution metadata renamed `clouda-pdf` -> `clouda-ocr`. Python import
  packages (`pdfword`, `clouda_data`, `clouda_lab`, `clouda_models`,
  `clouda_training`, `clouda_contracts`), console script names
  (`clouda-data`, `clouda-lab`, `clouda-training`, `clouda-quality`), and the
  `~/.clouda_pdf_word` state directory are unchanged for compatibility.
  See `docs/BRAND_MIGRATION.md`.
- Public-facing metadata updated to the canonical name: README, CITATION.cff,
  Zenodo deposition metadata, application UI strings, launch scripts, service
  description, and install/remediation hints.


## 0.2.0 - 2026-09-21

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
