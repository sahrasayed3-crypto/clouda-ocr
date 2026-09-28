# Brand Migration — Clouda PDF → Clouda OCR

Date prepared: 2026-09-21 (local, uncommitted)

## Summary

`Clouda PDF` was the old release and runtime name. As of this migration the
canonical project, product, and brand name is **Clouda OCR**. PDF-to-DOCX
conversion is a capability/runtime path inside Clouda OCR, not a separate
product identity.

- Repository: `clouda-ocr` (unchanged)
- Domain: `cloudaocr.xyz` (unchanged)
- Benchmark: `Clouda OCR Arabic OCR Benchmark` (unchanged, separate artifact)
- Software/runtime: **Clouda OCR**

## Historical constraint

Release **v0.2.0** was published on 2026-09-21 under the title **"Clouda
PDF v0.2.0"** (GitHub release `v0.2.0`, Zenodo software record
`10.5281/zenodo.22880981`, record title "Clouda PDF").

This history was **not rewritten**:

- the git tag `v0.2.0` and its source archives are unchanged;
- the published Zenodo record keeps its original title until remote metadata
  is reconciled (see `docs/REMOTE_METADATA_RECONCILIATION.md`);
- v0.2.0 remains valid historical evidence of the project's progress.

Future releases use the canonical name **Clouda OCR**, starting with the
prepared (not published) **v0.2.1** branding-correction release
(see `docs/RELEASE_V0.2.1_PLAN.md`).

## What was changed locally (uncommitted)

| File | Change |
|---|---|
| `pyproject.toml` | distribution name `clouda-pdf` → `clouda-ocr`; version `0.2.0` → `0.2.1`; description updated |
| `CITATION.cff` | title → `Clouda OCR`; version → `0.2.1` (creator/ORCID/license unchanged) |
| `.zenodo.json` | deposition title/description → `Clouda OCR` (applies to the *next* Zenodo deposition; the published record was not edited remotely) |
| `README.md` | H1, description, and references → `Clouda OCR`; brand note added |
| `CHANGELOG.md` | 0.2.1 (unreleased) branding-migration entry |
| `app.py` | Streamlit page title, header, and footer strings → `Clouda OCR` |
| `run_server.ps1`, `start_clouda_all.ps1` | launch messages → `Clouda OCR` |
| `deploy/linux/clouda.service` | service description → `Clouda OCR document runtime` |
| `CONTRIBUTING.md`, `SECURITY.md`, `docs/TESTING.md` | naming → `Clouda OCR` |
| `clouda_training/*`, `clouda_data/doctor/*` | install/remediation hints `pip install clouda-pdf[...]` → `pip install clouda-ocr[...]` (required: extras exist under the new distribution name) |
| `clouda_training/experiments/environment.py` | installed-distribution check `clouda-pdf` → `clouda-ocr` |

## What was intentionally NOT changed (compatibility / history)

| Item | Location | Reason |
|---|---|---|
| Python import packages (`pdfword`, `clouda_data`, `clouda_lab`, `clouda_models`, `clouda_training`, `clouda_contracts`) | repository layout | renaming would break imports for no branding benefit |
| Console scripts (`clouda-data`, `clouda-lab`, `clouda-training`, `clouda-quality`) | `pyproject.toml [project.scripts]` | already `clouda-*`; no `clouda-pdf` CLI command ever existed, so no alias is required |
| State directory `~/.clouda_pdf_word` | `pdfword/constants.py` | compatibility surface: renaming would orphan existing local state |
| Redis queue names `clouda:pdf_conversion` | worker/tests | compatibility surface for deployed workers |
| Sample/ground-truth fixture text ("Clouda PDF يحول المستند…") | `samples/`, `benchmarks/ground_truth/`, `tools/generate_test_pdfs.py` | deterministic fixtures must stay byte-consistent with generated PDFs; benchmark ground truth is read-only evidence |
| `SBOM.json` self-entry `clouda-pdf` | `SBOM.json` | point-in-time record of the v0.2.0 artifact; regenerate at the v0.2.1 release |
| Provenance labels `"software_version": "clouda-pdf/0.2.0"` / `created_by: "clouda-pdf 0.2.0"` | `clouda_data/distortion/workflow.py`, `clouda_data/training_data/sharding.py` | provenance lineage of existing generated data; update at the v0.2.1 release if desired |
| `docs/engineering/*` session notes | uncommitted pre-existing work | historical session records, not branding surfaces |
| Published release/record titles | GitHub v0.2.0 release, Zenodo record 22880981 | remote metadata; reconciliation listed separately |

## Compatibility notes

- `pip install clouda-pdf` continues to refer to the historical v0.2.0
  distribution (sdist/wheel from that tag). New installs should use
  `pip install clouda-ocr` from v0.2.1 onward.
- No Python API changed: imports, classes, and console commands are identical.
- No state, queue, or fixture format changed.
