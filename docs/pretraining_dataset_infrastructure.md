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

The implementation is validated on synthetic fixtures. File contents are
streamed while hashing, but dataset metadata is still processed in memory;
the practical scaling limits and migration point are described in §15.

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
  cache/file_hashes.jsonl          incremental hash cache (source,path,size,mtime)->sha
  manifest/samples.v1.jsonl        canonical dataset manifest
  manifest/validation.json         validation report
  manifest/dedupe_report.json      duplicate classification report
  manifest/split_report.json       split + leakage report
  manifest/stats.json              dataset statistics
  export/jsonl/<split>.jsonl       training-ready export
  handoff/                         optional data-factory handoff artifacts
```

Source files are **never copied or modified**; every sample reference is a
path relative to its registered source root.

## 4. Data schema

`clouda_data/pretraining/schema.py`, version `clouda.pretraining.sample.v1`.
A frozen dataclass (`DatasetSample`) serializable to JSONL rows: stable
`sample_id` (SHA-256 of source id + canonical relative path + record key), source provenance
(`source_id`, `source_dataset`, `source_path`, `source_record_id`,
`source_url`, `source_license`, `source_split`), document coordinates
(`document_id`, `page_id`, `page_index`, `group_id`), content
(`image_path`, `raw_text`, `text`, `language`, `script`, dimensions,
`file_sha256`, `normalized_text_sha256`), state (`validation_status`,
`duplicate_state`, `duplicate_of`, `exclusion_reason`, `target_split`),
and history (`provenance`, `transformations`, `quality_flags`).

Identity paths normalize `/` versus `\`, redundant `.` components, and safe
internal `..` components. Absolute, drive-letter, UNC, control-character, and
root-escaping paths are rejected. Path case remains significant because the
same manifest contract must also work on case-sensitive filesystems. For
record files, an explicit `id`/`record_id`/`sample_id` is preferred over a line
coordinate, so inserting an unrelated JSONL line does not renumber identities.
Changing a source root alone does not change sample ids.

`raw_text` and `text` (normalized) are always separate; normalization never
overwrites source ground truth. The dataclass is shallowly frozen: callers must
still treat nested provenance dictionaries and history lists as immutable.
Readers reject unknown fields and unsupported schema versions instead of
silently accepting misspellings or incompatible rows.

## 5. Source registry

`sources.py`. A source is data, not code: JSONL rows with source id, name,
origin, license, languages, expected format, local root, optional remote
reference, adapter id, enabled flag, redistribution restrictions, and a
`training_only|public|private|restricted` classification. Registering a new
dataset never touches ingestion logic. Accepted adapter identifiers are
`auto`, `image_sidecar`, `jsonl_records`, and `csv_records`; the current scanner
uses the same deterministic mixed discovery behavior for each identifier.
Supported sample inputs are image+sidecar text, JSONL records, and CSV/TSV
records using the fixed aliases declared in `discovery.py`. Plain `.json` files
are discovered as metadata but do not create samples.

Record image references are normalized and resolved beneath the source root.
Traversal, absolute paths, Windows drives/UNC paths, null/control characters,
and symlink escapes are recorded as malformed metadata rather than accessed.
JSONL is decoded and parsed one line at a time, duplicate JSON keys are rejected,
and a single record line is limited to 1 MiB. A malformed line produces an
excluded draft without aborting valid sibling lines.

## 6. Normalization policy

`normalize.py`, version `clouda.pretraining.normalize.v1` plus a policy
fingerprint. Conservative defaults are NFC, leading BOM removal, and CRLF/CR
to LF normalization. Whitespace collapse, zero-width/joiner/bidi removal,
control-character removal, tatweel removal, Arabic diacritic removal,
alef/ya folding, Arabic-Indic digit folding, line flattening, and presentation-
form composition are opt-in. Compatibility normalization (`NFKC`) is accepted
only with explicit `presentation_forms="compose"`. The operations used for the
current normalized value are recorded in order in `transformations`; re-running
with a less destructive policy removes stale transformation labels.

## 7. Deduplication policy

`dedupe.py`. Deduplication never deletes data. Samples are classified:

- `unique` / `canonical` (exactly one deterministic representative of each
  connected exact-duplicate family),
- `duplicate` (exact file hash, repeated sample id, repeated source record)
  — excluded from exports but kept in the manifest with a `duplicate_of`
  link,
- `conflicting_duplicate` (identical normalized text over different
  images) — kept, and merged during splitting to prevent text leakage.

Perceptual image dedup can be added later as an additional equivalence
signal without changing the classification contract.

## 8. Leakage prevention

`splitting.py`. Three barriers:

1. pages of one document share a source-scoped group (`group_id` → `document_id` →
   `source_record_id` → `sample_id`) and never cross splits;
2. groups sharing a file hash or normalized-text hash are union-merged
   before assignment;
3. explicit `duplicate_of` links are union-merged as a separate barrier; and
4. the split report verifies file, text, document, group, duplicate-cluster,
   and holdout disjointness after every split run.

## 9. Split strategy

Deterministic hash-based assignment: `sha256(seed + ":split:" + group)`
maps the full 256-bit digest into exact integer ratio boundaries
(default train/validation/test/holdout = 0.8/0.1/0.05/0.05). Reproducible,
inspectable, and stable when new groups arrive (no global reshuffle). With
few groups, ratio targets are approximate by design — counts are reported.

## 10. Holdout protection

The `holdout` split is structurally protected: exports and Data Factory
candidate manifests exclude it unless a training export explicitly enables
holdout (`--include-holdout` or a strictly typed boolean config value). A
default export removes a stale holdout file left by an earlier explicit export.
The split report records holdout counts and group disjointness.

## 11. Manifest format

`manifest.py`, version `clouda.pretraining.manifest.v1`. Canonical format is
JSONL: one header line (`_schema_version`, `_row_count`) then one row per
sample, sorted by `(source_id, source_path, sample_id)` plus a canonical JSON
tie-breaker. The header records row count and, after a full prepare, the
configuration fingerprint, effective seed, and enabled source ids. Readers
reject malformed JSONL, row-count mismatches, unknown sample fields, and
incompatible sample/manifest versions. Replacement uses a unique same-directory
temporary file, file flush/fsync, and `os.replace`; this protects the previous
file from partial replacement but is not a concurrent-writer lock or a promise
of filesystem-level durability after sudden hardware failure. A CSV mirror is
available for human review. Image bytes are never embedded.

## 12. Training-ready export

`export.py`. A `TrainingExporter` protocol with a generic JSONL
implementation (`{"image": rel_path, "text": ..., provenance...}`).
Exports filter excluded/duplicate/holdout samples, reject unsafe image paths,
retain source path/record/license provenance, and are byte-deterministic.
Each run replaces its selected split files and removes stale split files.
Future formats (Hugging Face datasets, Parquet,
model-specific layouts) plug in via `register_exporter` without touching
the core pipeline.

## 13. Clouda Data Factory boundary

`handoff.py` writes a declarative request (`handoff/data_factory_handoff.json`)
plus a candidate manifest of clean, non-holdout samples from the source targeted
by the prepare command: absolute source root, sample ids, sorted unique requested
profiles, seed, absolute intended output location, and manifest SHA-256
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

All commands are non-interactive. `--dry-run` scans, streams hashes, validates,
deduplicates, and splits in memory but creates no workspace files or directories.

## 15. Resume and scaling limits

Every full prepare rescans all enabled registered local sources. Unchanged
files reuse an index/cache entry only when source id, canonical relative path,
size, and nanosecond modification time match; content is streamed in 1 MiB
chunks when hashing, and a before/after stat check rejects files changed during
the read. Cache rows with malformed SHA-256 values are ignored, and a truncated
last cache line does not poison the next append. Because ordinary filesystem
metadata cannot prove content identity against an actor able to rewrite content
and restore timestamps, use `--fresh` for that adversarial case.

After indexes are refreshed, validation, exact dedupe, leakage grouping, split,
reports, manifest, and exports are recomputed over the complete enabled source
set. Adding/removing/disabling a source, changing normalization, seed, ratios,
or source content therefore cannot silently retain old derived fields. Standalone
normalize/validate/dedupe/split commands invalidate existing JSONL exports;
normalization and validation also reset downstream duplicate/split state so the
remaining stages must be re-run explicitly.

Hashing is streaming, JSONL record parsing is line-oriented, and the union-find
algorithms are near-linear. Discovery lists, indexes, manifests, samples,
duplicate maps, split maps, and export selections are nevertheless held in
memory, and canonical manifests are rewritten in full. The current design is
therefore not bounded-memory for hundreds of millions of records. Before
metadata volume exceeds a single-machine memory budget, move indexes/manifests
to a transactional store such as SQLite or shard by source while retaining the
same sample schema. Concurrent writers are not supported; workspace ownership
must be exclusive during a run.

## Tests

`tests/test_pretraining_unit.py`, `tests/test_pretraining_dedupe_split.py`,
`tests/test_pretraining_workflow.py`, and `tests/test_pretraining_hardening.py`
cover stable ids,
deterministic hashes, normalization safety, duplicate classification and
provenance, discovery ordering and junk skipping, validation severity,
deterministic splitting, document/hash/text leakage prevention, holdout
protection, manifest reproducibility, resume idempotency, export filtering,
the Data Factory boundary, CLI smoke tests, and a full tiny-fixture
end-to-end run. The fixture is generated programmatically
(`tests/pretraining_fixture.py`) — no binary blobs are committed.
