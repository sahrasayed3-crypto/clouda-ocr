# Data Factory Integration Provenance

This file records the provenance of the Clouda Data Factory integration into
the canonical Clouda OCR repository, without rewriting Git history.

## Source

- Repository name: `clouda-data-factory`
- Local path (read-only archive, never modified):
  `Downloads/clouda-data-factory` relative to the owner's Windows user directory
- Remote: `https://github.com/sahrasayed3-crypto/clouda-data-factory.git`
- Branch: `master`
- Integrated HEAD: `4663f17c3416e970633574712d432b5b92cd5b9a`
  ("docs: prepare repository for public release")
- Source status at integration: clean (no uncommitted changes)

## Target

- Repository: `clouda-ocr` (`https://github.com/sahrasayed3-crypto/clouda-ocr`)
- Local canonical path: the unified Clouda project checkout on the owner's workstation
- Branch: `main`
- Pre-integration HEAD: `63ff57fdd5c4c7504a4140626e9111f3bb043d34`
  ("docs(training): document experiment workflow and hardware validation")

## Integration date

2026-09-09

## Major subsystems migrated

| Subsystem | Source location | Target location |
|---|---|---|
| Package shell | `src/clouda_data_factory/` | `clouda_data/factory/` |
| Ingest (text/image/HF) | `ingest/` | `clouda_data/factory/ingest/` |
| Render (weasyprint + raqm + vendored stack) | `render/` | `clouda_data/factory/render/` |
| Distortions (atomic/composite/QC) | `distort/` | `clouda_data/factory/distort/` |
| Profiles + legacy adapters | `profiles/` | `clouda_data/factory/profiles/` |
| Seeds (v1 + legacy) | `seed/` | `clouda_data/factory/seed/` |
| Export/manifest/provenance | `export/`, `manifest/`, `provenance/` | same, under `clouda_data/factory/` |
| Orchestration/autorun/CLI | `factory.py`, `autorun.py`, `cli.py` | same, under `clouda_data/factory/` |
| Distortion catalogue config | `configs/distortion_profiles/ocr_benchmark.yaml` | `clouda_data/resources/data_factory/ocr_benchmark.yaml` |
| Render layout config | `configs/render_profiles/layout.yaml` | `clouda_data/resources/data_factory/render_layout.yaml` |
| Arabic fonts (SIL OFL 1.1) | `assets/fonts/*.ttf` | `clouda_data/resources/fonts/` |
| Tests + fixture | `tests/` | `tests/factory/` (retargeted imports) |

## New modules written for the integration

- `clouda_data/factory/profiles/scan_families.py` — the 12 System B
  scan-family profiles as a first-class module (replaces the source repo's
  `importlib` hack against its vendored legacy tree; parameters verified
  field-for-field, `variant_seed` contract pinned by tests).
- `clouda_data/factory/adapters.py` — factory run manifest → canonical
  `clouda.pretraining.manifest.v1` conversion with leakage-safe split
  assignment (the pre-training ↔ factory integration seam).
- `docs/data_foundation/DATA_FACTORY.md`, `docs/data_foundation/../..` docs
  updates, this file.

## Deferred / not migrated

- The source repo's `legacy/` frozen trees (`ocrbench_benchmark`,
  `arabic_scan_factory`) were NOT copied: they exist only for bit-exact
  historical reproduction inside the standalone repo, which remains untouched
  as the local archive. All engine math they contain was already vendored into
  the portable `src/clouda_data_factory` package (byte-parity tested).
- `IMPLEMENTATION_REPORT.md` / `MERGE_PLAN.md` (source-repo process docs).
- The `sample_corpus/` texts (rights not established for redistribution).

## License handling

- Target repository license (Apache-2.0) applies to the integrated code.
- Fonts: SIL OFL 1.1 (verified from each TTF name table); attribution added
  to `NOTICE`.
- No license text was carried from the standalone repo (it had none).
