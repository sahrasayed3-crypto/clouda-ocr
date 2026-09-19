# Changelog

All notable changes are documented here.

## Unreleased

- Documentation hardening: unified public identity as Clouda OCR with a
  product-first README (status matrix, engineering-evidence section, CI badge),
  relabeled historical test results in `docs/TESTING.md` as dated snapshots,
  scrubbed a local path from development plan docs, added the investor-facing
  GitHub audit (`docs/investor-github-audit.md`), the technical due-diligence
  pack (`docs/INVESTOR_TECHNICAL_DUE_DILIGENCE.md`) and its source map
  (`docs/INVESTOR_TECHNICAL_DUE_DILIGENCE_SOURCES.md`), corrected the static
  test-suite count in the README, and repointed a dead deployment-doc reference
  in `deploy/linux/README.md`. No runtime code was modified.

## 0.2.0 - 2026-09-20

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
