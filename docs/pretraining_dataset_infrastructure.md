# Pre-Training Dataset Infrastructure

Developer documentation for the dataset-preparation layer in
`clouda_data/pretraining/`. This is the stage that runs **before** any
large-scale dataset download and **before** model training.

## 1. What it does

Turns registered raw sources into a deterministic, traceable,
leakage-safe, training-ready dataset:

```
raw sources -> source registration -> discovery/indexing -> hashing
-> duplicate detection -> text normalization -> sample validation
-> dataset manifest -> grouping -> leakage-safe splitting
-> QC -> training-ready export -> optional Data Factory handoff
```

Everything is validated on tiny synthetic fixtures today and is designed to
scale to hundreds of GB without architectural redesign (see §15).

## 2. What it intentionally does NOT do

- No dataset downloading (the existing `clouda_data.datasets` download gate
  stays the only network path, and it is license-gated).
- No dependency on a trained OCR model, no inference serving, no OCR routing.
- No distortion/rendering (that is the separate `clouda-data-factory`
  project; integration is a declarative file boundary, §13).
- No heavyweight services: plain files and standard library only (PIL for
  image validation, which is already a project dependency).

## 3. Workspace layout

```
<workspace>/
  sources.jsonl                    registered source definitions
  sources/<id>/index.jsonl         discovered files + hashes (resumable)
  cache/file_hashes.jsonl          incremental hash cache (relpath,size)->sha
  manifest/samples.v1.jsonl        canonical dataset manifest
  manifest/validation.json         validation report
  manifest/dedupe_report.json      duplicate classification report
  manifest/split_report.json       split + leakage report
  manifest/stats.json              dataset statistics
  export/jsonl/<split>.jsonl       training-ready export
  handoff/                         optional data-factory handoff artifacts
```

Source files are **never copied or modified**; every reference is a
dataset-root-relative path.

## 4. Data schema

`clouda_data/pretraining/schema.py`, version `clouda.pretraining.sample.v1`.
A frozen dataclass (`DatasetSample`) serializable to JSONL rows: stable
`sample_id` (SHA-256 of source id + path + record key), source provenance
(`source_id`, `source_dataset`, `source_path`, `source_record_id`,
`source_url`, `source_license`, `source_split`), document coordinates
(`document_id`, `page_id`, `page_index`, `group_id`), content
(`image_path`, `raw_text`, `text`, `language`, `script`, dimensions,
`file_sha256`, `normalized_text_sha256`), state (`validation_status`,
`duplicate_state`, `duplicate_of`, `exclusion_reason`, `target_split`),
and history (`provenance`, `transformations`, `quality_flags`).

`raw_text` and `text` (normalized) are always separate; normalization never
overwrites source ground truth.

## 5. Source registry

`sources.py`. A source is data, not code: JSONL rows with source id, name,
origin, license, languages, expected format, local root, optional remote
reference, adapter id, enabled flag, redistribution restrictions, and a
`training_only|public|private|restricted` classification. Registering a new
dataset never touches ingestion logic. Adapters today: image+sidecar text,
JSONL records, CSV/TSV records (fields configurable in `discovery.py`).

## 6. Normalization policy

`normalize.py`, version `clouda.pretraining.normalize.v1` plus a policy
fingerprint. Safe defaults: NFC, BOM strip, CRLF→LF, zero-width/bidi-mark
removal, control-character removal, whitespace collapse with line-break
preservation. **Opt-in only** (never default): tatweel removal, Arabic
diacritic removal, alef/ya folding, Arabic-Indic digit folding,
presentation-form composition (NFKC). The applied operations are recorded
per sample in `transformations`.

## 7. Deduplication policy

`dedupe.py`. Deduplication never deletes data. Samples are classified:

- `unique` / `canonical` (deterministic representative of a family),
- `duplicate` (exact file hash, repeated sample id, repeated source record)
  — excluded from exports but kept in the manifest with a `duplicate_of`
  link,
- `conflicting_duplicate` (identical normalized text over different
  images) — kept, and merged during splitting to prevent text leakage.

Perceptual image dedup can be added later as an additional equivalence
signal without changing the classification contract.

## 8. Leakage prevention

`splitting.py`. Three barriers:

1. pages of one document share a group (`group_id` → `document_id` →
   `source_record_id` → `sample_id`) and never cross splits;
2. groups sharing a file hash or normalized-text hash are union-merged
   before assignment;
3. the split report verifies all three invariants (plus holdout
   disjointness) after every split run.

## 9. Split strategy

Deterministic hash-based assignment: `sha256(seed + ":split:" + group)`
maps each merged group into cumulative ratio boundaries
(default train/validation/test/holdout = 0.8/0.1/0.05/0.05). Reproducible,
inspectable, and stable when new groups arrive (no global reshuffle). With
few groups, ratio targets are approximate by design — counts are reported.

## 10. Holdout protection

The `holdout` split is structurally protected: exports exclude it unless
explicitly enabled (`--include-holdout` / config), and the split report
records holdout counts and disjointness.

## 11. Manifest format

`manifest.py`, version `clouda.pretraining.manifest.v1`. Canonical format is
JSONL: one header line (`_schema_version`, `_row_count`) then one row per
sample, sorted by `(source_id, source_path, sample_id)`. Writes are atomic
(temp + `os.replace`), so interrupted runs never corrupt existing manifests.
A CSV mirror is available for human review. Image bytes are never embedded.

## 12. Training-ready export

`export.py`. A `TrainingExporter` protocol with a generic JSONL
implementation (`{"image": rel_path, "text": ..., provenance...}`).
Exports filter excluded/duplicate/holdout samples, preserve provenance, and
are byte-deterministic. Future formats (Hugging Face datasets, Parquet,
model-specific layouts) plug in via `register_exporter` without touching
the core pipeline.

## 13. Clouda Data Factory boundary

`handoff.py` writes a declarative request (`handoff/data_factory_handoff.json`)
plus a candidate manifest of clean samples: source root, sample ids,
requested profiles, seed, intended output location, and manifest SHA-256
provenance. This repository never imports or calls
`clouda-data-factory` (public repo: `sahrasayed3-crypto/clouda-data-factory`);
a future caller would feed the candidate manifest to that project's
`clouda-data-factory run <input> <output>` flow. The main project's tests
never require the external package.

## 14. Running on a tiny local dataset

```bash
# one command end to end (scan -> index -> normalize -> hash -> dedupe
# -> validate -> split -> manifests -> export -> summary)
python -m clouda_data.pipeline.cli dataset-prepare <source_dir> <workspace> --seed 42

# with a declarative Data Factory handoff request
python -m clouda_data.pipeline.cli dataset-prepare <source_dir> <workspace> \
    --seed 42 --data-factory-handoff --handoff-profiles old_book_medium

# per-stage control
... dataset-register-source <workspace> <id> --name ... --local-root ...
... dataset-scan <source> <workspace> [--dry-run] [--fresh]
... dataset-validate <workspace> / dataset-normalize / dataset-dedupe
... dataset-split <workspace> --seed 7 --ratios 0.8,0.1,0.05,0.05
... dataset-export <workspace> [--include-holdout]
... dataset-stats <workspace>
```

All commands are non-interactive; `--dry-run` writes nothing.

## 15. Scaling later

The design assumes large future datasets: streaming file hashing with a
size-keyed cache, incremental/resumable indexing, line-oriented manifests
and exports, generators over discovered files, relative paths, no source
copying, and bounded memory (no dataset is ever fully materialized in RAM).
When data grows beyond single-file JSONL ergonomics, the manifest can be
sharded per source (`sources/<id>/samples.v1.jsonl`) with the same schema —
a storage-layout change, not a schema change.

## Tests

`tests/test_pretraining_unit.py`, `tests/test_pretraining_dedupe_split.py`,
`tests/test_pretraining_workflow.py` (51 tests) cover stable ids,
deterministic hashes, normalization safety, duplicate classification and
provenance, discovery ordering and junk skipping, validation severity,
deterministic splitting, document/hash/text leakage prevention, holdout
protection, manifest reproducibility, resume idempotency, export filtering,
the Data Factory boundary, CLI smoke tests, and a full tiny-fixture
end-to-end run. The fixture is generated programmatically
(`tests/pretraining_fixture.py`) — no binary blobs are committed.
