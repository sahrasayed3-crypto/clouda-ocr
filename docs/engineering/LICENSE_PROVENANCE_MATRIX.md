# License & Provenance Matrix — Current Session (2026-09-21)

Repository evidence only; no legal conclusions beyond what the tracked files
state. Unknowns are marked explicitly.

## Code & bundled assets

| Resource | Source | License | Redistribution | Training use | Commercial use | Attribution | Provenance status | Uncertainty |
|---|---|---|---|---|---|---|---|---|
| Clouda OCR source (pdfword, clouda_*) | original | Apache-2.0 (`LICENSE`, pyproject, CITATION) | permitted (Apache-2.0) | permitted | permitted | LICENSE + NOTICE | Verified: wheel carries LICENSE+NOTICE; metadata `License-Expression: Apache-2.0` | none |
| Vendored RAQM render stack (`clouda_data/factory/render/_raqm/`) | legacy engine, byte-parity copy (comment in `pyproject.toml` per-file-ignores) | **not stated in-file** | unknown | — | — | none found in directory | Upstream-compat documented in `UPSTREAM_COMPATIBILITY.md` (read, not modified this session) | **uncertain: license of the vendored stack should be confirmed against its upstream** |
| Bundled Arabic fonts: Amiri, Scheherazade New, Noto Naskh Arabic, Cairo | third parties | SIL OFL 1.1 (`NOTICE`) | permitted with license/copyright notices | permitted (OFL) | permitted (OFL, incl. embedding) | copyright lines in NOTICE | Verified present in wheel (`clouda_data/resources/fonts/`, 6 ttf) | none |
| `pdfword/streamlit_firebase_auth/` | original/adapted (index.html packaged) | Apache-2.0 (project) | permitted | — | permitted | NOTICE | packaged | none |
| `tools/poppler`, `tools/tesseract`, `tools/python` | vendored binaries, **not packaged** (excluded from wheel/sdist and bandit) | local toolchain only | not redistributed via package | — | — | — | Verified absent from built artifacts | runtime-only use; distribution of the repo tree itself would carry their licenses — marked in `.gitignore`? **uncertain: check before publishing the repo tree as a distributable archive** |

## Datasets (dataset_catalog)

Gate: `dataset_catalog/licenses/REVIEW_STATUS.json` — fail-closed policy:
pending / research_only / blocked / unknown are blocked from commercial
training and production; enforcement in
`clouda_data/datasets/registry.py:93-113` (license gate on downloads).

| Dataset | Review status | License (registry) | Notes |
|---|---|---|---|
| rasam_dataset | approved_with_conditions | — | conditions apply; see catalog |
| sard_synthetic_arabic_recognition_dataset | approved_with_conditions | — | conditions apply |
| craneset_arabic_ocr_hf_sample | approved_with_conditions | — | conditions apply |
| humans_in_the_loop_arabic_documents_ocr | pending | — | blocked until reviewed |
| openiti_makhzan | research_only | — | non-commercial training only |
| muharaf_public | research_only | — | non-commercial training only |
| hicma_dataset | research_only | — | non-commercial training only |
| qnl_arabic_ocr_corpus_v2 | pending | QNL no-copyright-claim statement (scans); metadata CC0-1.0 | blocked until reviewed |
| pats_a01, kafd_ldc2016t21, mssqpi_arabic_ocr_dataset | pending | — | blocked |
| loay_arabic_ocr_synthetic_scans_faker_300k | blocked | — | blocked outright |
| noisy_ocr_dataset_nod | blocked | — | blocked outright |

`foundation_sources_v1.json` (13 sources) carries per-source `license_id`
values incl. Apache-2.0, MIT, CC0-1.0, CC-BY-NC-SA-4.0, CC-BY-NC-4.0 and a
QNL statement — NC variants are compatible with the fail-closed policy
(research/non-commercial only).

## Benchmark materials (published v0.1.0, read-only)

| Resource | License posture | Notes |
|---|---|---|
| `benchmarks/ocr_arabic/` metadata (manifests, results, methodology) | published by this project; raw evidence `retained_privately` (`release.json:13`) | metadata-only release; per-row source/permission fields recorded |
| Underlying 100 sources / 14 dataset ids | respective third-party licenses; publication of results grants no rights to underlying data (`NOTICE`) | explicitly restated in NOTICE |

## Models

| Resource | Status |
|---|---|
| `configs/models/registry.v1.json` | single placeholder entry (`future-vlm-adapter`): license `license-review-required`, deployment `disabled`, commercial `pending` — consistent with "no model selected" |
| HunyuanOCR-1.5 / Qwen3-VL descriptors (`clouda_training/*/descriptor.py`) | pinned revisions for planning only; `real_weights_validated=False`, `gpu_validated=False`; no weights in repo |

## Citations / release metadata

- `CITATION.cff` (1.2.0) and `.zenodo.json`: author Wahbah, ORCID
  0009-0003-1237-7665, Apache-2.0, version 0.2.1 (local, unpublished),
  consistent with `pyproject` after the concurrent brand migration.
- `NOTICE` / `THIRD_PARTY_NOTICES.md`: cover fonts, datasets, benchmark
  source materials, model names.

## Explicit unknowns

1. License of the vendored `_raqm` render stack (not labeled in-file).
2. Redistribution terms for the vendored `tools/` binaries if the repo tree
   itself is ever distributed as an archive (they are excluded from packages).
3. Dataset "approved_with_conditions" condition details live in the catalog
   and PERMISSION_EVIDENCE_INDEX (referenced, not re-verified this session).
