# Clouda Benchmark & OCR Results Store

Schema version: `clouda.ocr.results.v1`

## What it is and why it exists

The Results Store is the canonical backend data layer between raw
benchmark/model outputs and the higher-level analysis systems (error
analysis, failure analysis, hard-example mining, training selection, and the
future Clouda Lab UI). Today benchmark pages, ground truth, model outputs,
runs, and metrics live in different files and formats; the store provides one
indexed, integrity-checked, portable access layer over them.

It is **backend only**: no UI, no Streamlit, no React, no HTTP. It does not
run inference, download models, or duplicate the Training Experiment
Framework — it only records and links.

## Canonical layout

```
<store_root>/
    datasets/<dataset_id>__<version>.json
    models/<model_id>.json
    runs/<run_id>/
        metadata.json        InferenceRun
        pages.jsonl          PageRecord rows
        ground_truth.jsonl   GroundTruthRecord rows
        predictions.jsonl    OCRPrediction rows
        metrics.jsonl        EvaluationRecord rows
        summary.json         run-level aggregates
        artifacts.jsonl      ArtifactRef rows
        .index/              rebuildable cache (never source of truth)
```

JSONL files are append-only and streamed on read (no full-corpus loads).
Writes are atomic (temp file + `os.replace`).

## Identity

All identities are deterministic SHA-256 digests over domain-tagged logical
fields — never `hash()`, never absolute paths, never process state.

- **Dataset**: `dataset_id@version` (existing ids preserved verbatim).
- **Page**: `dataset@split:<page_key>` where `page_key` is the source's own
  stable id. For the Arabic OCR benchmark the canonical page key is
  `distorted_id` (177 unique values; each clean `benchmark_id` may have
  several distorted derivatives, so `benchmark_id` is stored as the
  document/source identity, not the page id).
- **Run**: `run_<sha256[:20]>` over (model, revision, dataset, manifest hash,
  created-at scope). Re-evaluating identical content with a different
  timestamp yields a distinct run; identical scope+timestamp yields the same
  id.
- **Prediction**: digest of (run, page) — multiple models/runs per page are
  first-class.
- **Metric**: digest of (run, page, metric name).

## Ground truth

- Raw text is stored **exactly**: Arabic Unicode, diacritics, tatweel, and
  digit forms are preserved byte-for-byte.
- `raw_text_sha256` proves immutability (verified on `verify`).
- Normalized comparison views are derived **only on explicit request**
  (`normalized_view()` / `get_ground_truth_normalized()`); raw and normalized
  are never conflated. The default normalized view matches the foundation
  evaluation policy (`comparison_arabic_fold_digits`).

## Predictions

Multiple predictions for the same page (different models, revisions, runs)
are expected and never overwritten. Duplicate handling:

- **Idempotent**: re-ingesting an identical record is a no-op
  (wall-clock `ingested_at` is excluded from the content comparison).
- **Reject-on-conflict**: different content under the same identity raises
  `ConflictingRecordError`.

## Models and runs

`ModelRecord` is metadata only (no weights, no external calls). Optional
`TrainingLineage` links a trained model back to an experiment, training run
id, checkpoint, and config hash — adapters read only `metadata.json` /
`summary.json` of the Training Experiment Framework.

`InferenceRun` captures model, revision, dataset, manifest hash, config hash,
status (`CREATED`/`RUNNING`/`COMPLETED`/`FAILED`/`INTERRUPTED`), page count,
environment, and bundle location. Runs from training checkpoints link via the
same optional lineage.

## Metrics

CER/WER come from `clouda_data.evaluation` (the canonical implementations) —
the store computes and stores, never reimplements. Page-level and
run-summary-level records are distinguished by `scope`; each record carries
the metric name, value, normalization policy, and evaluator version.
`get_worst_pages()` provides basic sorted access for downstream analysis.

## Artifacts and portability

Artifact references are portable `dataset://` / `artifact://` URIs, resolved
only against explicitly configured roots (`CLOUDA_DATASET_ROOT`, etc.).
Absolute Windows drives, UNC paths, `..` traversal, and boundary escapes are
rejected. Machine-local absolute paths are never written to canonical files;
when a source carries one it is quarantined under `source_private` in
provenance. Bundles are Windows/Linux portable (POSIX-style relative paths,
UTF-8, `ensure_ascii=False`).

## Holdout safety

`ProtectionInfo` is fail-closed: a page whose split is `holdout`,
`protected_holdout`, `benchmark_holdout`, or `private_holdout` is
automatically marked protected, and `is_training_eligible` is a **derived
property** — it cannot drift from the protection markers. Unknown splits are
not eligible. Adapters propagate protection from source splits; verification
rejects any record claiming eligibility while protected.

## Integrity

`verify_bundle()` / `results-verify` check: schema versions, run-id
consistency, duplicate ids, GT/prediction hash mismatches, predictions or
metrics referencing unknown pages, prediction run mismatches, and pages
missing ground truth. Corrupted data is reported, never silently accepted.

## Query API

`ResultsService` (see `clouda_data/results/service.py`) exposes datasets,
pages, ground truth (raw + normalized), predictions, models, runs, metrics,
worst-page access, and integrity verification. All read paths stream JSONL;
filtering happens during the stream.

## CLI

Commands follow the unified CLI convention (`python -m
clouda_data.pipeline.cli`, also `python -m clouda_data.results.cli`):

```
results-ingest <manifest> --store DIR       ingest the Arabic OCR benchmark manifest
results-ingest-runs <results.csv> --store   ingest leaderboard rows as runs/models
results-verify <run_id> --store DIR         verify bundle integrity
results-list-runs --store DIR [--model --dataset --status]
results-list-models --store DIR
results-show-page <run_id> <page_id> --store DIR
results-worst-pages <run_id> --store DIR [--metric --limit]
results-export <run_id> <output> --store DIR
```

## Ingestion adapters

Adapters exist only for formats verified in the repository:

1. `benchmarks/ocr_arabic/benchmark_manifest.jsonl` (177 rows) — pages with
   full source/distortion/QC/hash provenance.
2. `benchmarks/ocr_arabic/results.csv` (8 rows) — inference runs + model
   records; run ids are deterministic; `manifest_sha256` links the canonical
   `2a499ed0…` manifest hash.
3. Foundation evaluation JSONL (`reference_text`/`prediction_text` rows) —
   page + GT + prediction + metrics in one step.
4. Training Experiment Framework run directories — optional model lineage.

## Clouda Lab integration points

- **Error analysis engine**: `get_page` + `get_ground_truth` +
  `get_predictions` per page.
- **Failure analysis**: `list_predictions(model/run)` + `get_worst_pages`.
- **Dataset selection**: `list_pages(...)` with `is_training_eligible`.
- **Training orchestrator**: `TrainingLineage` on models/runs.
- **Distortion engine**: `profile`, `distortions`, `distortion_seed` on pages.
- **Web UI**: `ResultsService` read-only instance + `ArtifactResolver`.

## Known limitations

- Metrics currently reuse CER/WER/exact-match only; additional foundation
  metrics can be added without schema changes (`metric_name` is open).
- The ingested benchmark run is metadata-only (raw prediction evidence is
  private); attaching it later is an append, not a migration.
- Indexes are per-run and rebuildable; a cross-run index is future work when
  query volume requires it.
