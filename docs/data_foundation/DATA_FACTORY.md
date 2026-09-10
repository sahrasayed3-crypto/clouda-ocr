# Clouda Data Factory (integrated)

The Data Factory is now a **first-class subsystem of Clouda OCR**: the package
`clouda_data.factory` inside this canonical repository. The standalone
repository (`sahrasayed3-crypto/clouda-data-factory`, kept untouched as a local
archive copy) is **functionally superseded**
by this integration and remains untouched locally as an archive/fallback.

## What it produces

From Arabic/mixed text files, images, and PDFs:

- clean documents — searchable PDFs (WeasyPrint/Pango/HarfBuzz) and/or clean
  page images (PNG);
- synthetically degraded pages (PNG) and image-only scan PDFs (12 scan-family
  + 8 benchmark degradation profiles, 20 atomic distortion ops);
- ground-truth text files (the untouched source text);
- JSONL + CSV manifests with SHA-256 hashes, seeds, transform parameters,
  QC results, and per-page provenance for every artifact.

Batch processing, multiprocessing, and manifest-driven resume are built in.
Output is byte-deterministic for a fixed (input, seed, profile, environment)
and identical for 1 worker or N workers.

## One command

```bash
python -m clouda_data.pipeline.cli factory-run <input> <output>
```

`factory-run` automatically detects input types, selects the best available
text-render backend, renders clean + degraded variants, exports PNGs and
image-only PDFs, writes manifests, resumes interrupted runs, isolates
per-document failures, and verifies every output hash after the run.

## CLI

All capabilities are commands of the canonical `clouda-data` CLI
(`clouda_data.pipeline.cli`):

| Command | Purpose |
|---|---|
| `factory-run <input> <output>` | zero-configuration one-command data factory |
| `factory-generate <inputs…> --output` | explicit-control generation |
| `factory-profiles` | list all 20 profiles + backends |
| `factory-verify <run_dir>` | re-check every recorded SHA-256 |
| `factory-seeds` | seed-derivation vectors for all modes |
| `factory-manifest <run_dir> <output>` | convert a run into the canonical pre-training manifest |

The module also runs standalone: `python -m clouda_data.factory <command>`.

## Architecture

```
clouda_data/factory/
  ingest/       text_file · image_file · hf_datasets (optional)
  render/       weasyprint_backend · raqm_page_backend · _raqm/ (vendored engine)
  distort/      atomic (20 ops) · scan_composite · qc readability gate
  profiles/     unified schema + lossless legacy adapters + scan_families
  seed/         unified v1 derivation · byte-identical legacy derivations
  export/       image-only PDFs (pinned dates) · atomic PNG writer
  manifest/     JSONL+CSV writers
  provenance/   hashing · atomic writes
  adapters.py   factory → canonical pre-training manifest conversion
  factory.py    orchestration: multiprocessing, resume, failure isolation
  autorun.py    the zero-configuration `factory-run` command
  cli.py        standalone command surface (mirrored into the canonical CLI)
```

Configs ship in `clouda_data/resources/data_factory/` (`ocr_benchmark.yaml` distortion
catalogue, `render_layout.yaml` page-layout options). Arabic fonts
(SIL OFL 1.1: Amiri, Scheherazade New, Noto Naskh Arabic, Cairo) live in
`clouda_data/resources/fonts/`.

## Seed modes

| mode | derivation | default seed |
|---|---|---|
| `v1` | BLAKE2b-8 over (global_seed, source_sha256, document_id, page_index, variant_index, profile, distortion_stage, severity) | 20260831 |
| `ocr_benchmark` | legacy System A derivation, verbatim | 20260825 |
| `arabic_scan_factory` | legacy System B derivation, verbatim; per-page = variant seed + page index | 20260831 |

Nothing depends on wall-clock time, machine state, or iteration order. The
legacy modes replay the recorded seeds of the historical benchmark manifest
(test-proven). The vendored engines in `render/_raqm/` and the atomic /
composite distortion engines preserve the original math exactly; their byte
parity is protected by tests (`tests/factory/`).

## Provenance contract

Every run records: source bytes + SHA-256 (sources are never modified and are
re-verified after generation), clean/output SHA-256s, seeds, full transform
parameters, renderer used, GT hashes, QC readability metrics, and
success/failure status. Atomic writes everywhere; run directories are never
silently overwritten.

## Canonical manifest contract

The canonical dataset manifest is `clouda.pretraining.manifest.v1`
(`clouda_data/pretraining/manifest.py`): JSONL with one header line
(`_schema_version`, `_row_count`, metadata) then one row per sample sorted by
`(source_id, source_path, sample_id)`.

`adapters.py` converts a factory run manifest into this canonical format
**losslessly**: source identity, source hash, output hash, seed, seed mode,
profile, transform steps, renderer, QC record, and ground-truth reference all
travel inside the sample's `provenance` dict. Split assignment is performed by
the standard leakage-safe splitter (`assign_splits`) with a seed derived from
the converted content, so identical factory content always converts to
identical splits. The canonical dataset version is a digest of the factory
configuration and generated artifact hashes, so content changes produce a
new version. Protected markers are rejected at both conversion and training
boundaries. Canonical conversion requires a raster output, so runs
created with `--no-png` remain valid archival/PDF runs but are not
training-manifest inputs.

```bash
# Data Factory run → canonical manifest (training-ready, leakage-checked)
python -m clouda_data.pipeline.cli factory-manifest <run_dir> dataset_manifest.jsonl
```

## Training Experiment Framework integration

The converted manifest plugs directly into the Training Experiment Framework
(`clouda_training.experiments`): dataset id/version, manifest hash, split,
source provenance, and holdout safety are validated by
`validate_training_dataset` before any run; dry-run mock experiments consume
factory-generated datasets exactly like any other manifest. See
`tests/factory/test_pretraining_integration.py` and
`tests/factory/test_e2e.py` for the tested end-to-end path:

```
raw sources → Data Factory (render/distort/export)
            → factory run manifest (factory/v1 rows)
            → adapters.py → clouda.pretraining.manifest.v1
            → leakage-safe splits (holdout protected)
            → Training Experiment Framework dry-run (mock adapter)
```

## Native dependencies (optional extras)

| Capability | Requirement | Extra |
|---|---|---|
| Distortion engines, PNG/PDF export, QC | numpy, opencv-python-headless, img2pdf, pikepdf | `.[factory]` |
| Searchable clean PDFs | WeasyPrint + Pango/HarfBuzz natives | `.[factory-render]` |
| RAQM Arabic shaping | Pillow built with libraqm | host-level |
| PDF ingestion / rasterization | Poppler (`pdftoppm`) | host-level |

Without the native text-render libraries, text rendering and PDF ingestion are
skipped **per document** with explicit errors in the manifest and final
summary — the factory never silently produces broken Arabic text. Image
inputs always process.

## Testing

`tests/factory/` covers seeds (including replay of the historical benchmark
manifest), lossless profile adapters, distortion engines (byte-parity of
ported ops), manifests and atomicity, end-to-end runs, worker-count
determinism, automatic resume, failure isolation, CLI surface, the
factory → canonical-manifest conversion, and the Training Framework dry-run
integration. Renderer tests skip automatically on hosts without
libraqm/Pango.

## Provenance of this integration

- Source repository: `clouda-data-factory`
  (`https://github.com/sahrasayed3-crypto/clouda-data-factory`)
- Integrated HEAD: `4663f17c3416e970633574712d432b5b92cd5b9a` (master)
- Integration date: 2026-09-09
- Major subsystems migrated: ingest, render (incl. vendored RAQM stack),
  distort (atomic + composite + QC), profiles (incl. scan families), seed
  (v1 + legacy), export, manifest, provenance, orchestration, autorun, CLI,
  tests, configs, fonts.
- The source repository was not modified; its commit history was not merged.

## Licensing

The integrated factory code carries the repository's Apache-2.0 license.
The bundled fonts are SIL Open Font License 1.1 (redistribution permitted
with attribution; see NOTICE). The standalone source repository had not
selected a license; no license text was carried from it.
