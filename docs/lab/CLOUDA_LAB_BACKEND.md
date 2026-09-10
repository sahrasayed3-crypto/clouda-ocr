# Clouda Lab — Backend Architecture

Status: **backend foundation only**. No web UI, no Streamlit/React, no HTTP
API yet. Everything in `clouda_lab/` is a domain service intended to be
consumed later by an API/UI layer.

> **REAL TRAINING HARDWARE VALIDATION IS STILL DEFERRED.** All training flows
> in this document execute through the deterministic `MockTrainer` /
> dry-run path of the Training Experiment Framework. No model download,
> inference, or gradient update happens. Nothing here validates GPU training.

## Module map

| Module | Purpose |
|---|---|
| `models.py` | Typed frozen contracts (`OCRSample`, `ErrorRecord`, `ErrorAnalysis`, `BatchReport`, `FailureComparison`, `FailureBucket`, `HardExampleScore`, `TrainingBatchRecommendation`, `SelectionResult`, `DistortionExperiment`, `RunAnalysis`, …) — every model has a JSON-safe `to_dict()`. |
| `error_taxonomy.py` | Deterministic Arabic-aware error classification rules (char + word level). |
| `error_analysis.py` | OCR Error Analysis Engine: char/word alignment, CER/WER/N-CER, per-error records with positions and context. |
| `batch_analysis.py` | Aggregation across samples by model/run/dataset/document-type/profile/distortion/source/split; percentiles; worst/best pages; CSV/JSON export. |
| `holdout_guard.py` | Fail-closed holdout protection for every selection. |
| `dataset_selection.py` | Dataset Selection Engine: criteria filters, deterministic sampling, top-N, percentile windows, derived manifests with provenance. |
| `selection_history.py` | Lightweight JSONL usage registry (training/evaluation/hard-example/active-learning batches). |
| `failure_analysis.py` | Before/after comparison of two models/runs/checkpoints per sample. |
| `failure_buckets.py` | Structured failure buckets from measured metrics + present metadata only. |
| `hard_examples.py` | Transparent weighted hard-example scoring (all signals and weights recorded in output). |
| `active_learning.py` | Deterministic next-batch recommender with rationale per sample. |
| `distortion_experiments.py` | Tiny distortion experiments **over** the canonical Data Factory engine + sensitivity summaries. |
| `evaluation_service.py` | Canonical backend entry point composing all engines. |
| `training_orchestrator.py` | Thin facade over the Training Experiment Framework + selection→experiment pipeline. |
| `run_pipeline.py` | Post-run analysis chain (run → evaluation → failure → hard → next batch). |
| `io.py` | Loaders for benchmark evidence records; JSON/JSONL/CSV exporters. |
| `cli.py` | `clouda-lab` CLI to exercise the backend. |

## Architectural rule: compose, never duplicate

The lab reuses the existing infrastructure and adds no parallel framework:

| Concern | Reused from |
|---|---|
| CER / WER | `clouda_data.evaluation.cer` / `.wer` |
| N-CER normalization | `clouda_data.ground_truth.normalization.normalize_for_comparison` (same policy as `normalize_ocr_text`, digits folded) |
| Manifest schema / writer | `clouda_data.pretraining.manifest` (`clouda.pretraining.manifest.v1`) |
| Dataset / sample identity | canonical manifest `sample_id` values — the lab never mints second identities |
| Hashing / provenance | `clouda_contracts.checksums.sha256_file`, `clouda_data.factory.provenance` |
| Holdout protection | `clouda_training.experiments.dataset` constants + strict superset guard (see below) |
| Distortion engine | `clouda_data.factory.distort.atomic.apply_distortion` |
| Distortion profiles | `clouda_data.factory.profiles.load_profile_book` |
| Seed derivation | `clouda_data.factory.seed.derive.derive_seed` |
| Experiment framework | `clouda_training.experiments` (configs, runs, registry, checkpoints, resume, comparison, MockTrainer) |
| CLI conventions | argparse subcommands + `--output`/`Path` args as in `clouda_training.cli` |

## Error Analysis Engine

`analyze_sample(OCRSample) -> ErrorAnalysis`

- Metrics: CER, WER, N-CER (canonical normalization policy), exact match,
  character/word counts.
- Alignment: character-level and word-level Levenshtein backtrace with a
  deterministic tie-break (substitution preferred, then deletion, then
  insertion) — identical input always yields identical records.
- Records: operation (`match/substitution/insertion/deletion`), gt/predicted
  token, gt position (0-based char index; `-1` for insertions), predicted
  position, word index, ±12-char context window.
- Normalized second pass: when texts differ only after normalization
  (diacritics, alef/ya/digit folding), those records are emitted separately
  with `normalized=True` so the future UI can show both layers.
- Summary: counts + rates by category, most common substitution pairs.

Arabic-aware categories (deterministic, rule-based, no linguistic
overclaiming): `whitespace`, `punctuation`, `arabic_digit`, `latin_digit`,
`digit_system_confusion`, `diacritic`, `tatweel`, `hamza_alef_variant`,
`ya_alef_maqsura`, `ta_marbuta_ha`, `hamza`, `arabic_latin_script`,
`character_substitution`, `missing_*`/`extra_*` (arabic/latin/generic),
word-level `missing_word`, `extra_word`, `digit_word`, `punctuation`,
`mixed_script_word`, `word_substitution`, `unknown`.

## Batch Analysis

`analyze_batch(samples, dimensions=…)` → `BatchReport` with:

- per-dimension summaries (mean/p50/p90/p95/max CER, WER, N-CER, exact match
  rate) for model, run, dataset, document_type, profile, distortion, source,
  split;
- worst/best page tables; global error-type distribution; common
  substitutions; JSON + CSV export.

## Dataset Selection Engine

`select_samples(manifest_path, SelectionCriteria, seed=…, scores=…)` →
`SelectionResult`

Criteria: explicit ids, dataset, split, document type, profile (row or
provenance), distortion, error type, CER/WER ranges, model, run, failure
bucket, source, tags/metadata, deterministic random N, top-N hardest/easiest,
percentile windows, limit.

Guarantees:

- deterministic for identical (manifest, criteria, seed) — `selection_id` is
  a hash of exactly those inputs;
- protected rows are filtered **before** any sampling and counted in
  `excluded_protected`;
- `write_selection_manifest` produces a canonical
  `clouda.pretraining.manifest.v1` file whose header records:
  `selection_id`, `selection_criteria`, `selection_seed`,
  `selection_created_utc`, `source_manifest`, `source_manifest_sha256`,
  source sample count, and a `selection_lineage` list.

## Holdout guarantees

`clouda_lab.holdout_guard` applies a **fail-closed** check to every row
(and its nested `provenance`/`metadata` blocks) and to derived-manifest
headers. A row is protected when any of:

- `target_split`/`split`/`source_split` ∈ {`holdout`, `protected_holdout`,
  `benchmark_holdout`, `private_holdout`} or contains `holdout`
  (case-insensitive, whitespace-tolerant) — aliases like `eval_holdout`
  are caught;
- `dataset_role`/`role`/`purpose` ∈ {`holdout`, `protected_holdout`,
  `benchmark`, `evaluation_only`} or contains any of those markers
  (`train+holdout`, `benchmark_holdout_v2` are caught);
- `protected` is `True` or the string `true`/`yes`/`1`/`protected`
  (case-insensitive);
- any protection-relevant field has a **malformed type** (number, list,
  dict, …) — treated as protected, never bypassed;
- a nested `provenance`/`metadata` block is not a mapping — treated as
  protected.

Additionally `validate_derived_manifest_for_training` re-checks a derived
manifest before experiment creation, and the Training Experiment Framework's
own `validate_training_dataset` runs again at `run_experiment` time
(defense in depth). Real protected data was never inspected; all holdout
tests use synthetic fixtures.

## Failure Analysis & Buckets

`compare_failure(baseline, candidate, thresholds=…)` classifies each common
sample as `improved`, `regressed`, `unchanged`, `newly_failed` (crossed the
failure CER upward), or `recovered` (crossed downward); thresholds
(`improve`, `regress`, `failure_cer`) are configurable and recorded in the
report. Deltas for CER/WER/N-CER and per-category error deltas are tracked;
worst regressions, best improvements, and persistent failures are listed.

Buckets (`failure_buckets.py`) are assigned only from measured metrics and
metadata that is actually present: high CER/WER, whitespace/digit/
punctuation/diacritics-heavy, deletion/insertion/substitution-heavy,
blur/skew/compression distortion, mixed-script, small-text, table/form,
`unknown` fallback. Nothing visual is inferred.

## Hard Examples & Active Learning

`rank_hard_examples` scores each sample as a weighted sum of normalized
signals (`cer`, `wer`, `ncer`, `regression_magnitude`, `error_diversity`,
`persistent_failure_count`, `distortion_severity`, `error_type_rarity`,
`cross_model_failure`). Default weights are documented constants; callers
may override any subset; unknown signals are rejected. Every result row
carries its raw signals, the weights applied, and the final score.

`recommend_next_batch` strategies: `hardest_only`, `balanced_hard`
(round-robin quotas over a group dimension), `diversity_first` (error-
category coverage first), `mixed_curriculum` (hard/mid/easy thirds),
`deterministic_random` (seed-hashed baseline). Previously used samples
(from `SelectionHistory` or passed directly) are excluded; each selection
records a human-readable rationale; balance counters are returned.

## Distortion Experiments

`plan_distortion_experiment` builds a deterministic plan (profile steps or
explicit atomic distortions × variants × per-variant canonical seeds).
`run_distortion_experiment` executes through
`clouda_data.factory.distort.atomic.apply_distortion` with per-stage seeds
from `clouda_data.factory.seed.derive.derive_seed` — the exact canonical
engine, never a reimplementation — and writes `experiment.json` +
`variants.jsonl` with full provenance.

`sensitivity_by_distortion` / `model_resilience_comparison` aggregate only
actually available evaluation rows into CER/WER-by-distortion/profile and
per-model sensitivity views.

## Evaluation Service

`EvaluationService` is the intended single entry point for the future
API/UI: `evaluate_sample`, `evaluate_batch`/`evaluate_file`,
`compare_models`, `failure_buckets`, `hard_examples`, `recommend_next`,
plus `export_*` helpers (JSON/JSONL/CSV). It owns no metric logic — it
composes the engines above.

## Training Orchestrator (facade)

`TrainingOrchestrator` wraps `clouda_training.experiments` and adds nothing
parallel:

- reads: `list_runs`, `inspect_run`, `run_status`, `get_metrics`,
  `get_checkpoints`, `compare`;
- lifecycle: `start_dry_run(config)` (refuses non-dry-run configs —
  `ConfigError`), `resume(run_id)` (pure passthrough; framework rules apply
  unchanged), `validate_experiment`;
- selection pipeline:
  `create_training_experiment_from_selection(...)` → validates the
  selection (holdout guard, empty-selection refusal), writes the derived
  manifest with provenance header, records usage history, resolves a
  framework-shaped experiment config (mock adapter, `dry_run: true`),
  and can immediately run it in dry-run mode.

Real training remains disabled end-to-end: the facade refuses
non-dry-run configs before the framework's own fail-closed check
(`RuntimeError: Real training adapters are not enabled`) is ever reached.

## Post-run analysis pipeline

`run_pipeline.analyze_run(...)` chains: run status/metrics from the
orchestrator → failure comparison against caller-provided baseline metrics →
hard-example mining → next-batch recommendation, returning a single
`RunAnalysis`. Mock/dry-run metrics stay in run artifacts; per-sample OCR
truth must be supplied by the caller (in tests: synthetic fixtures).
MockTrainer output is never presented as real OCR performance.

## CLI

```
clouda-lab analysis-page        --input samples.jsonl --sample-id ID [--output out.json]
clouda-lab analysis-batch       --input samples.jsonl [--output out.json] [--csv out.csv]
clouda-lab analysis-compare     --baseline base.json --candidate cand.json
clouda-lab dataset-select       --manifest m.jsonl [--output derived.jsonl] [filters...]
clouda-lab dataset-validate     --manifest derived.jsonl
clouda-lab dataset-hard-examples--input rows.json [--top-n N]
clouda-lab dataset-recommend-next --input rows.json [--strategy balanced_hard]
clouda-lab distortion-experiment --manifest m.jsonl --samples s1,s2 --profile P
```

## Storage / artifact contract

The lab writes only machine-readable artifacts and reuses project
conventions:

- derived selection manifests: canonical `.manifest.jsonl`
  (`clouda.pretraining.manifest.v1`, header + sorted rows);
- selection history: `selections/history.jsonl` (append-only usage events;
  the manifest remains the only dataset truth);
- distortion experiments: `experiment.json` + `variants.jsonl` + PNGs;
- analysis/evaluation outputs: JSON (default), JSONL for row sets, CSV where
  tabular. No derived UI state is ever stored.

## Future UI integration

The UI layer should depend only on:

- `EvaluationService` for evaluation/analysis requests;
- `TrainingOrchestrator` for experiment lifecycle + selection→training flow;
- `SelectionHistory` for usage queries;
- `clouda_lab.models` `to_dict()` payloads for serialization.

All are stateless or file-backed, JSON-serializable, and free of any
presentation logic. **Real training hardware validation remains deferred.**
