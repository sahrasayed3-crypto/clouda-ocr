# Clouda Dataset Quality Gate

Operator documentation for `clouda_data/quality/` — the dataset-level quality
gate for Clouda pre-training manifests. Branch
`feature/dataset-quality-dedup`; spec of record:
`DESIGN_DECISIONS.md` (Wave-2 contract) and `LEAD_SYNTHESIS.md`.

Companion documents:

- [Pre-Training Dataset Infrastructure](../pretraining_dataset_infrastructure.md) —
  the manifest producer this gate consumes.
- [Results Store](data_foundation/RESULTS_STORE.md) — optional summary sink.
- [VALIDATION_PIPELINE.md](data_foundation/VALIDATION_PIPELINE.md) — the
  sample-level validation the gate extends.

## Table of contents

1. [Purpose](#1-purpose)
2. [Architecture and pipeline](#2-architecture-and-pipeline)
3. [Relationship to other subsystems](#3-relationship-to-other-subsystems)
4. [Exact duplicate rules](#4-exact-duplicate-rules)
5. [Image near-duplicate algorithm](#5-image-near-duplicate-algorithm)
6. [Text fingerprints and Arabic normalization](#6-text-fingerprints-and-arabic-normalization)
7. [Candidate generation (LSH)](#7-candidate-generation-lsh)
8. [Confirmation and clustering](#8-confirmation-and-clustering)
9. [Leakage detection (L0–L6)](#9-leakage-detection-l0l6)
10. [Holdout protection](#10-holdout-protection)
11. [Artifact integrity catalog](#11-artifact-integrity-catalog)
12. [Heuristics defaults](#12-heuristics-defaults)
13. [Quality severity policy](#13-quality-severity-policy)
14. [Keep/exclude policy](#14-keepexclude-policy)
15. [Derived manifest and lineage fields](#15-derived-manifest-and-lineage-fields)
16. [Run identity](#16-run-identity)
17. [Resume behavior](#17-resume-behavior)
18. [CLI reference](#18-cli-reference)
19. [JSON report schema](#19-json-report-schema)
20. [Performance behavior](#20-performance-behavior)
21. [Limitations](#21-limitations)
22. [False-positive / false-negative tradeoffs](#22-false-positive--false-negative-tradeoffs)
23. [Future large-scale validation](#23-future-large-scale-validation)

---

## 1. Purpose

The quality gate answers one question before any dataset is trained on:
**"Is this manifest clean, deduplicated, and leakage-free?"** It scans a
canonical pre-training manifest (`clouda.pretraining.manifest.v1`), checks
every sample's artifact, finds exact and near duplicates (images and Arabic
ground-truth text), detects cross-partition leakage, and either certifies the
dataset or emits a deterministic exclusion plan that produces a clean derived
manifest.

It is a **cross-cutting dataset-level subsystem**, deliberately separate from:

- `clouda_data/validation/*` — generated-output validators (per-page OCR
  output checks), and
- `clouda_data/pretraining/validation.py` — sample-level validation run
  during preparation.

The quality gate never modifies the original dataset, never contains
protected content in its reports, and never re-implements protection rules.

## 2. Architecture and pipeline

All logic lives in the new subpackage `clouda_data/quality/` (not inside
`pretraining/`, which is a complete self-contained pipeline; the gate consumes
its manifests). One module per stage:

```
canonical manifest (JSONL, clouda.pretraining.manifest.v1)
        |
        v
manifest_adapter.py    load + path resolution + run identity
        |
        v
artifacts.py           integrity catalog (single decode per sample)
        |     \
        v      v
image_fp.py       text_dup.py     fingerprints (image / text)
        \              /
         v            v
       exact_dup.py   exact signals (file hash, sample id, source record, raw text)
         |
         v
near_index.py     LSH candidate generation (images + text)
         |
         v
near_index.py     candidate confirmation
exact_dup.py      deterministic duplicate clustering
         |
         v
leakage.py        L0–L6 cross-partition checks (clouda_contracts.protection)
         |
         v
health.py         DatasetHealthSummary (counts only)
         |
         v
policy.py + derived.py   keep/exclude decisions -> derived + quarantine manifests
         |
         v
report.py         clouda.quality.run.v1 report (JSON / text)
        |
        +-- results_bridge.py (optional: ResultsStore.save_summary)
        +-- run_state.py      (resume / stale-resume rejection)
        +-- cli.py            clouda-quality entry point
```

Data flow invariants: canonical JSON everywhere (`sort_keys=True`,
`separators=(",", ":")`, `ensure_ascii=False`), sorted iteration, stable IDs,
no `hash()` (Python's salted hash is never used), single decode discipline per
image (verify-then-load shared between `artifacts.py` and `image_fp.py`),
symlink refusal **before** any `open`.

## 3. Relationship to other subsystems

| Subsystem | Relationship |
|---|---|
| **Data Factory QC** (`clouda_data.factory`) | Upstream producer. Factory manifests (`clouda_data/factory/manifest`) arrive as canonical pre-training manifests; the gate validates them post-generation. Readability/QC inside the Factory is per-page render QC; the quality gate is dataset-level and independent of it. |
| **Results Store** (`clouda_data.results`, schema `clouda.ocr.results.v1`) | Optional downstream sink. `results_bridge.persist_quality_summary(store, run_id, summary)` calls `ResultsStore.save_summary` with metadata mark `kind="dataset_quality_run"`. Guarded import; **no schema change** to the store. |
| **Dataset Selection** (`clouda_lab.dataset_selection`, `clouda.lab.selection.v1`) | Selection reads the same manifests and filters through the holdout guard before writing derived manifests. The gate's exclusion plan and derived manifests follow the same contract: deterministic ordering, lineage in the header, and post-write re-validation via `validate_derived_manifest_for_training`. |
| **Training Data Loader** | **Deferred — not on main.** `manifest_adapter.py` exposes a documented stub `training_stream_contract()` describing the adapter contract only (a stream of `DatasetSample` rows with resolved artifact paths and protection status). No loader code is imported or built in this wave. |
| **Environment Doctor** | **Deferred hook — not on main.** A future doctor pass (environment audit, `CLOUDA_DOCTOR_WT`) may consume the quality report as an input; only the hook point is documented here. No code exists. |
| **`clouda_contracts.protection`** | Sole protection authority. The gate consumes `record_is_protected`, `protection_metadata_is_malformed`, `is_training_split_eligible`, `string_marks_protected`, `normalize_marker`, `PROTECTED_SPLIT_NAMES`, `PROTECTED_ROLES`. Never re-implemented. |
| **`clouda_lab.holdout_guard`** | Consumed as-is (facade over protection) for row/header checks. |

## 4. Exact duplicate rules

`exact_dup.py` wraps `pretraining.dedupe.classify_duplicates` **as-is** for
exact signals, then adds one extra evidence group:

- Duplicate signals (union-find over signal ownership): same `sample_id`,
  same `source_record_id`, same `file_sha256` (`duplicate_file_hash`,
  `duplicate_sample_id`, `duplicate_source_record`).
- **Raw text group**: `raw_text_sha256 = sha256("clouda.text.raw.v1\x00" + text)`
  (domain-separated). This is **evidence-level only**: identical raw text over
  *different* file hashes stays `CONFLICTING_DUPLICATE` and is **never**
  classified `DUPLICATE`. Never treats "same text alone" as "same page".
- Near-signal plumbing: confirmed image near-duplicate pairs
  (`NearDuplicateCandidate` at level `CONFIRMED_NEAR_DUPLICATE`) are unioned
  with the same union-find semantics, reason `"near_duplicate_image"`.
- Report schema: `clouda.quality.dedupe.v2` (adds near-duplicate counts over
  the pretraining `clouda.pretraining.dedupe.v1` shape).
- **Never deletes data.** Exclusion is a later, recorded decision (§14).

## 5. Image near-duplicate algorithm

`image_fp.py` computes three 64-bit fingerprints per sample, Pillow-only
(no numpy), deterministic integer-only arithmetic:

| FP | Algorithm |
|---|---|
| `ahash` | 8×8 mean hash; mean via integer `//64`; ties → bit 0. |
| `dhash` | 9×8 horizontal gradient hash, lifted from `pdfword/teacher_pipeline/manifests.py::perceptual_dhash` (verified against the installed Pillow 11 API). |
| `phash` | **Fixed-point DCT**: precomputed integer cosine table `T = round(2048·c(u)·cos(...))` committed as literals; 32×32 input, top-left 8×8 coefficients, DC dropped, threshold at the lower-median index 31; ties → bit 0. |

Preprocessing before fingerprinting: grayscale, content crop of white margins
(256-wide downsample, `margin_value − 12` threshold, 2% padding, fallback to
full page), then aspect-bucket key over anchors `0.707, 0.773, 1.0, 1.294,
1.414` at 2% tolerance (else `round(r·100)`). Blankish detection from the
32×32 grid: `std ≤ 3` or ≥ 99.5% of pixels within ±6 of the modal value.

Safety: `DecompressionBombWarning` escalated to error, `Image.MAX_IMAGE_PIXELS`
respected (env `CLOUDA_MAX_IMAGE_PIXELS`, default `40_000_000`), verify-then-
thumbnail, lazy decode only. Fingerprint version string:
`clouda.quality.imgfp.v1:pillow==<exact installed version>`.

**Thresholds.** Candidate generation requires same aspect bucket. Confirmation
is a **triple conjunction**: `d_p ≤ 8 AND d_d ≤ 10 AND d_a ≤ 10` (pHash,
dHash, aHash Hamming distances). The pipeline stage that forms candidates
before confirmation uses `hamming_candidate = 12` (on the chosen tier-1
fingerprint). So the three headline numbers are:

- **8** — pHash Hamming distance for CONFIRMED (see §8 for the full ladder),
- **10** — dHash / aHash Hamming distance for CONFIRMED,
- **10+ MAD** — pairs failing the triple conjunction fall to the 64×64
  integer **MAD** test (`MAD = Σ|a−b| // 4096`): `MAD ≤ 6` upgrades to
  CONFIRMED, `MAD > 24` downgrades to CANDIDATE, between stays LIKELY.

## 6. Text fingerprints and Arabic normalization

`text_dup.py` runs three tiers:

- **Tier 0** — raw `sha256` (domain-separated, same construction as §4).
- **Tier 1** — normalized-text hash via `pretraining.normalize.normalize_text`
  under `DEDUPE_TEXT_POLICY`.
- **Tier 2** — MinHash over character 4-grams (`shingle_k=4`, `blake2b`
  `digest_size=8` per shingle; 128 permutations, `h = (a·x + b) mod 2^61−1`
  with `a, b` derived from `blake2b` of a **persisted** seed). Skipped for
  text `< 40` chars (`min_text_chars=40`), which emits flag `near_dup_skipped`.
  Exact Jaccard verification: `≥ 0.85` (`jaccard_near`) joins the near-text
  family (splitter-union via group); `0.70–0.85` (`jaccard_review`) emits the
  quality flag `near_text_review` only. Band owner cap 20.

### DEDUPE_TEXT_POLICY

`DEDUPE_TEXT_POLICY = NormalizationPolicy(...)` (from
`clouda_data/pretraining/normalize.py`) with exactly:

```python
NormalizationPolicy(
    unicode_form="NFKC",
    presentation_forms="compose",     # required by NFKC
    strip_bom=True,
    normalize_line_endings=True,
    preserve_line_breaks=True,
    collapse_whitespace=True,
    remove_zero_width=True,
    remove_control_characters=True,
    remove_tatweel=True,
    remove_diacritics=True,           # harakat stripped
    fold_alef=True,                   # أ/إ/آ/ٱ -> ا
    fold_ya=True,                     # ى -> ي
    fold_digits=False,                # ARABIC-INDIC digits are NOT folded
)
```

Arabic normalization behavior, precisely:

- **NFKC** with `presentation_forms="compose"` — presentation-form ligatures
  (U+FB50–U+FDFF, U+FE70–U+FEFF) are composed back to base letters.
- **Diacritics removed** (tashkeel/harakat), plus tatweel (kashida) removed.
- **Alef folding** (أ إ آ → ا) and **ya folding** (ى → ي) — so the same word
  spelled with or without hamza seats hashes identically.
- **Digits are NOT folded**: Eastern Arabic-Indic digits (٠–٩) and
  Extended Arabic-Indic (۰–۹) remain distinct from ASCII 0–9. A page number
  written `٣` never matches `3`. This is intentional: digit rewrites are a
  content difference, not a spelling variant.

The policy's `version()` string (`NORMALIZATION_VERSION + policy fingerprint`)
is persisted with **every** text fingerprint; on mismatch the fingerprint is
recomputed, never trusted.

## 7. Candidate generation (LSH)

`near_index.py` performs LSH banding, never pairwise comparison:

- **Images**: 192-bit concatenation `phash || dhash || ahash`, banded into
  **12 bands × 16 bits** (`lsh_bands=12, lsh_rows=16` for the 192-bit image
  concat; the text config independently uses `lsh_bands=16, lsh_rows=8` over
  the 128-perm MinHash signature). Same aspect-bucket requirement applies.
- Bucket overflow guard: `MAX_BUCKET = 4096` — oversized buckets are
  **skipped and flagged**, never exploded into pairwise comparisons.
- Intra-document adjacent-page pass: candidates from consecutive
  `document_id + page_index` neighbors (catches same-document near-dupes
  LSH can miss).
- Optional SQLite (stdlib) index at `<index_dir>/quality.v1.db` (WAL mode,
  batched transactions): tables `sample(sample_pk, source_id, source_path,
  sample_id, file_sha256, normalized_text_sha256, fingerprint columns,
  UNIQUE…)`, `lsh_band(band_id, band_key, sample_pk)`, and `run_state(
  manifest_hash, config_hash, algorithm_versions, stage, stage_cursor,
  processed_count)`. Streaming inserts; queries are O(N) via index-ordered
  scans. If SQLite is unused in v1 (in-memory is acceptable for ≤ 10k
  samples), the schema constants plus a `run_state` JSON fallback remain.
- Text side reuses the same banding machinery over the 128-perm MinHash
  signature at 16 bands × 8 rows.

## 8. Confirmation and clustering

`near_index.py` assigns candidate levels and `exact_dup.py` clusters
confirmed pairs deterministically:

**Level ladder** (per pair, distances `d_p`/`d_d`/`d_a` = pHash/dHash/aHash
Hamming, `mad` = 64×64 integer mean absolute difference):

| Condition | Level |
|---|---|
| `d_p ≤ 8 AND d_d ≤ 10 AND d_a ≤ 10` | `CONFIRMED_NEAR_DUPLICATE` (triple conjunction) |
| `d_p ≤ 8 AND d_d ≤ 10 AND d_a ≤ 10` fails, then `MAD ≤ 6` at 64×64 | `CONFIRMED_NEAR_DUPLICATE` |
| `d_p ≤ 12` (otherwise) | `LIKELY_DUPLICATE` |
| `8 < d_p` and `MAD > 24` | `CANDIDATE` |
| between (MAD 7–24 with `d_p ≤ 12`) | stays `LIKELY_DUPLICATE` |
| anything else reaching this stage | `CANDIDATE` |

Union-find is applied **only** for CONFIRMED pairs (the pretraining
`canonical_key` is reused to pick the representative).

**Determinism** (byte-identical re-runs): buckets iterated in
`(band_idx, key)` sorted order; members sorted by `sample_id`; pair order
`(level_rank, d_p, d_d, d_a, mad, min_id, max_id)`. Cluster ids are
deterministic: `"CLU_" + sha256(...)[:12]`; leakage finding ids
`"LKG_" + sha256(...)[:12]`.

## 9. Leakage detection (L0–L6)

`leakage.py` consumes `clouda_contracts.protection` +
`clouda_lab.holdout_guard` **only** — no new protection logic. It computes
`effective_partition(sample) → TRAIN | EVAL | PROTECTED | UNPARTITIONED`
fail-closed, with precedence:

1. `row_is_protected` first → `PROTECTED` (quarantined; IDs + codes only,
   never content);
2. then `target_split` via `is_training_split_eligible` /
   `PROTECTED_SPLIT_NAMES`;
3. else `source_split` / `source_role` via `string_marks_protected`
   (`normalize_marker`).

A malformed sha256 format, or protection-relevant metadata whose value is
neither a string nor a boolean, **fails closed** (treated as protected /
malformed → CRITICAL).

Severity table (per Wave1-B7):

| Level | Check | Severity |
|---|---|---|
| **L0** | Malformed protection metadata / malformed sha256 | `CRITICAL` (`LEAK_MALFORMED_PROTECTION`) |
| **L1** | Exact file-hash duplicate across train ↔ eval | `CRITICAL` (`LEAK_EXACT_HASH`) |
| **L2** | Page-identity duplicate across train ↔ eval (`PageIdentity`) | `CRITICAL` (`LEAK_PAGE_IDENTITY`) |
| **L3** | Confirmed near-image duplicate across train ↔ eval, **corroborated by ≥ 1 independent signal** | `CRITICAL` (`LEAK_NEAR_IMAGE`) |
| **L4** | Derived/distorted page from the same source page across the train/eval boundary, corroborated | `CRITICAL` (`LEAK_DERIVED_PAGE`) |
| **L5** | Ground-truth text duplication alone across partitions | `WARN` (`LEAK_GT_TEXT`) — never CRITICAL on text alone |
| **L6** | Group/document straddle via `resolve_group_key` (train ↔ eval) | `CRITICAL` (`LEAK_GROUP_STRADDLE`) |
| — | eval ↔ eval duplicates | `WARN` (`LeakagePolicy.eval_eval_warn = True`) |

Finding shapes: `clouda.quality.leakage_finding.v1` /
`clouda.quality.leakage_report.v1` (`LeakageFinding(finding_id, kind, severity,
partitions frozenset, sample_ids, canonical_key, corroborating_signals,
raw_split_values, detail)`). Protected rows contribute IDs + issue codes only.

## 10. Holdout protection

Holdout safety is **fail-closed via `clouda_contracts.protection`**:

- Protection is never hand-rolled. `record_is_protected` inspects the row and
  its nested `provenance` / `metadata` / `protection` blocks; any value
  containing a protection marker substring (`holdout`, `protected`,
  `evaluation_only`, `benchmark`) or a non-string/non-boolean value marks the
  row protected.
- `PROTECTED_SPLIT_NAMES = {holdout, protected_holdout, benchmark_holdout,
  private_holdout}`; `PROTECTED_ROLES = {holdout, protected_holdout, benchmark,
  evaluation_only, protected}`.
- Gate verdicts never broaden `EXPORTABLE_SPLITS`; training eligibility is
  derived at read time (`is_training_split_eligible` — only the literal
  normalized split `train` is training-eligible).
- In any duplicate cluster, a training row can never override a protected/eval
  row: protected rows are always **kept and quarantined** (§14).
- Derived manifests are re-validated after write
  (`validate_derived_manifest_for_training`), the same post-write hook
  `clouda_lab.training_orchestrator` uses; re-checking protection fail-closed.
- `include_holdout` remains an explicit opt-in and is part of the config
  fingerprint (§16).
- `report.py` applies the protection filter **before any write**:
  `record_is_protected` rows emit `id + codes` only, and `redact_mapping`
  (from `clouda_contracts.security`) runs over all echoed metadata.

## 11. Artifact integrity catalog

`artifacts.py` validates each sample's artifact against its manifest row,
extending pretraining validation semantics and reusing its path/exists/decode/
dimension/text codes. Catalog (issue code → check → default severity):

| Code | Check | Default |
|---|---|---|
| `PATH_SAFE` | Relative path resolves under the source root (`validate_relative_components` + containment via `clouda_contracts.storage`); **symlink refused before any open** | error |
| `MISSING_IMAGE` / `NON_EMPTY_FILE` | File exists; zero-byte file rejected | error |
| `HASH_MISMATCH` | `sha256_file` recomputed (streaming) vs `file_sha256`; missing declared hash → `info`, not mismatch | error |
| `IMAGE_DECODE` | verify-then-load decode failure | error |
| `IMAGE_DIMENSIONS` | dimension mismatch vs manifest | error |
| `IMAGE_MODE` | suspicious Pillow modes `{P, 1, I, I;16, F, CMYK, YCbCr}` | warn |
| `METADATA_DIMENSION_MISMATCH` | declared vs decoded dimensions | warn |
| `GT_MISSING` / `GT_EMPTY` | ground truth present and non-empty | error |
| `VERY_SHORT_GT` | `0 < len(gt) < 3` | warn |
| `BLANK_PAGE` / `NEAR_BLANK_PAGE` / `EXTREME_DIMENSIONS` / `EXTREME_ASPECT_RATIO` / `SUSPICIOUSLY_SMALL_IMAGE` / `VERY_LARGE_ARTIFACT` | heuristics — see §12 | warn |

Single-decode discipline: exactly one verify-then-load `open` per sample,
shared with image fingerprinting. Hashing is streamed. Per-code counts appear
in the report.

## 12. Heuristics defaults

All heuristics default to **WARN-only** (unusual sizes never FAIL) and are
config-overridable via `SeverityPolicy(default="warn", overrides=…,
strict_escalates_warn=True)`; `HeuristicsPolicy` defaults:

| Heuristic | Default threshold |
|---|---|
| `BLANK_PAGE` | pixel stddev `< 4.0` on a 256-wide thumbnail |
| `NEAR_BLANK_PAGE` | stddev `< 12.0` or ink ratio `< 0.001` |
| `EXTREME_DIMENSIONS` | any side `> 40000` px or `> 400_000_000` px total |
| pixel ceiling | `CLOUDA_MAX_IMAGE_PIXELS` env, default `40_000_000` |
| `EXTREME_ASPECT_RATIO` | ratio `> 50.0` |
| `SUSPICIOUSLY_SMALL_IMAGE` | file `< 1024` bytes, or tiny with dims `< 32` |
| `VERY_LARGE_ARTIFACT` | stat-only size `> 2_000_000_000` bytes (no decode) |
| `VERY_SHORT_GT` | `min_gt_warn_chars = 3` |

Fingerprint-stage blankish parameters (`std ≤ 3`, modal ±6 at ≥ 99.5%) live in
`image_fp.py` and gate similarity participation, not severity.

## 13. Quality severity policy

`IssueSeverity ∈ {info, warning, error, critical}`; config severities validate
against `{info, warn, fail}`. Gate verdicts (`GateVerdict`):

| Verdict | Rule |
|---|---|
| `FAIL` | any `critical` or `error` issue **or** `split_report.passed` is `False` |
| `PASS_WITH_WARNINGS` | warnings only |
| `PASS` | everything else |

`strict_escalates_warn=True`: in strict mode, `warn`-severity items escalate
for gate purposes. Per-severity counts, reason-code tallies, and per-code
artifact counts are carried in the report (§19).

## 14. Keep/exclude policy

`policy.py` produces a deterministic `ExclusionDecision(sample_id, reason_code,
reason_source, evidence)` per sample. Never lets a training row override a
protected/eval row in any cluster: **protected is always kept and
quarantined**. Keep preference order (config `KeepExcludePolicy`):

```
protected > canonical_valid (lower canonical_key) > clean_over_distorted > stable_sample_id
```

Flags: `exclude_duplicate=True`, `exclude_holdout=True`, `exclude_error=True`,
`exclude_conflicting_duplicate=False` (conflicting duplicates are kept; their
exclusion would force the leakage merge semantics documented in the config).

Exclusion report schema: `clouda.pretraining.exclusion.v1` with a **closed**
reason vocabulary — validation codes plus `duplicate`,
`near_duplicate_image`, `holdout`, `unassigned_split`, `quality_gate:<id>`;
`reason_source ∈ {validation, dedupe, split, quality_gate}`. Fixed precedence
keeps counts additive. **Nothing is ever deleted** — exclusion is a manifest
operation (§15).

## 15. Derived manifest and lineage fields

`derived.py` writes two manifests with the canonical
`pretraining.manifest.write_manifest` writer (canonical v1 rows in its own
sorted order, atomic temp-file + `os.replace`):

- **Clean manifest** — kept samples only.
- **Quarantine manifest** — excluded samples; disjoint-ID membership is
  asserted before write.

Header lineage metadata:

| Field | Meaning |
|---|---|
| `source_manifest_sha256` | sha256 of the scanned original |
| `quality_run_id` | the gate run that produced it (§16) |
| `config_identity` | quality-gate config fingerprint |
| `derived_dataset_version` | `derived-1.0.0+<source_hash[:12]>` |
| `exclusion_report_sha256` | integrity of the exclusion report |
| gate verdict | `PASS` / `PASS_WITH_WARNINGS` / `FAIL` at write time |

**The original manifest is never modified** — its sha256 is verified before
and after the write. Post-write, the derived manifest is re-read and protection
is re-checked fail-closed (§10).

## 16. Run identity

A run is identified by the tuple (implemented in `manifest_adapter.run_identity`
→ `QualityRunIdentity`):

1. `manifest_sha256` — `hashing.sha256_file` of the manifest;
2. `config_identity` — canonical-JSON sha256 fingerprint of
   `QualityGateConfig` (version `clouda.quality.config.v1`), including the
   `DEDUPE_TEXT_POLICY` version, bucket-edge definitions hashed into identity,
   and the `include_holdout` opt-in;
3. `algorithm_versions` — dict of algorithm version strings, including
   `clouda.quality.imgfp.v1:pillow==<version>` and the normalization policy
   version.

`QualityRun` carries `run_id`, `manifest_sha256`, `config_identity`,
UTC ISO `started`/`finished`, `verdict`, `severity_counts`, `issues`.

## 17. Resume behavior

`run_state.py` persists stage checkpoints (index_dir/run_state JSON, or the
SQLite `run_state` row when the index owns it) containing: `manifest_sha256`,
`manifest_row_count`, `config_identity`, `algorithm_versions`, `stage`,
`stage_cursor`, `processed_count`. Writes are atomic JSON.

- `start_or_resume(run_dir, identity)` returns the prior state or raises
  **`StaleResumeError`** with a precise mismatch message.
- A changed manifest, config, or algorithm version ⇒ **refuse resume**
  (CLI exit-code path 2, §18).
- Stages checkpoint after each stage completes.

## 18. CLI reference

Entry point `clouda-quality` (registered in `pyproject.toml`
`[project.scripts]: clouda-quality = "clouda_data.quality.cli:main"`).
Argparse, prog `clouda-quality`, pipeline-CLI conventions (UTF-8 reconfigure,
`_cmd_*(args) -> int`, JSON output via
`json.dumps(..., ensure_ascii=False, indent=2)`).

| Subcommand | Arguments | Purpose |
|---|---|---|
| `scan` | `manifest` (positional), `--config`, `--output`, `--fresh` | Full quality scan; writes the report |
| `verify` | `--report`, `--config` | Re-verify a dataset against an existing report/config |
| `report` | `--input`, `--output`, `--format text\|json` | Re-render a stored report |
| `clean-manifest` | `manifest` (positional), `--output` (required), `--dry-run`, `--config` | Apply the exclusion policy; emit derived (+ quarantine) manifests; `--dry-run` prints decisions only |
| `inspect-cluster` | `--report`, `--cluster-id` (required) | Show one duplicate cluster |

Shared flags: `--json`, `--strict`, `--resume`, `--no-near-duplicates`,
`--cross-split-only`, `--max-samples N`.

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | gate `PASS` (and command success) |
| `1` | gate `FAIL` |
| `2` | config/usage error (`QualityGateConfigError` caught), including stale-resume rejection |

## 19. JSON report schema

`report.render_report(result, fmt in {text, json})` emits
**`clouda.quality.run.v1`** — the `QualityGateResult` canonical JSON:
`run_id`, `verdict`, `issues`, `severity_counts`, reason-code tallies,
`clusters` (summaries), `leakage_findings`, `health`
(`clouda.dataset.health.v1` — counts-only across 11 dimensions with small
closed cross-tabs; bucket edges pinned in `bucket_definitions`), `exclusions`
(`clouda.pretraining.exclusion.v1`), artifact check results with thresholds,
and artifact paths + sha256s. Every embedded artifact carries its own schema
string (`clouda.quality.<artifact>.v1`); all serialization is canonical JSON,
version-checked like the pretraining schemas. The human text format uses the
Dataset Health Report layout with the `GATE` line last.

## 20. Performance behavior

Constraints: **no O(N²) in the normal path** (LSH banding + bounded buckets +
union-find), bounded memory, streaming hashing. `benchmarks.py` builds
synthetic suites (`make_manifest(root, n, dup_rate, seed)`, tiny 64×48 PNGs)
at tiers **100 / 1k / 10k** and asserts these complexity budgets:

| Metric | Assertion |
|---|---|
| LSH candidate growth | `candidates(10k) / candidates(1k) < 15` |
| Scan time growth | `scan_time(10k) / scan_time(1k) < 20` |
| Decode count | `≤ 2×` records at 10k (single-decode discipline, PIL-open counter) |
| Memory (`tracemalloc` peak, measured in a separate pass from timing) | `peak(10k) / peak(1k) < 15` |
| Correctness invariants | duplicate counts match configured rates even in perf runs |

Default suite runs 100 + 1k; the 10k tier is behind `pytest.mark.slow`
(marker registered in `tests/quality/conftest.py`).

## 21. Limitations

- **Fingerprints, not understanding**: perceptual hashes detect rendering
  variants of the same page; they do not detect semantically different pages
  that render similarly (§22).
- **Text near-dup quality flag only at 0.70–0.85 Jaccard** — review signal,
  not a cluster.
- **10k-scale in-memory ceiling** for v1: SQLite index is optional; beyond
  that, the index path (§23) is the migration point.
- **Config-scoped**: the gate sees one manifest per run; cross-manifest
  duplication is out of scope in this wave.
- **No network, no model inference, no new dependencies** (Pillow core only;
  numpy/SQLite stdlib only where specified).
- Original datasets are never modified; all outputs are derived artifacts.
- Training Data Loader and Environment Doctor integrations are **deferred**
  (documented contracts only, §3).

## 22. False-positive / false-negative tradeoffs

- **Tight vs loose thresholds**: the CONFIRMED triple conjunction
  (8/10/10) is strict, so confirmed clusters are high-precision but the
  recall of *subtle* re-renders depends on the LIKELY band (`d_p ≤ 12`) and
  the MAD ladder (6/24). Raising thresholds increases false merges; lowering
  them increases missed duplicates. The pipeline's default ladder favors
  precision on CONFIRMED (which drives exclusion) and routes uncertainty to
  LIKELY/CANDIDATE (which drive review signals, not exclusions).
- **pHash blindness on sparse text pages**: pHash is computed on low-frequency
  DCT coefficients after margin cropping. On sparse pages — a few lines of
  Arabic on a mostly white A4 — the low-frequency energy is dominated by page
  layout, so *distinct* pages with the same layout can fall inside the
  CONFIRMED thresholds (false positive), and the mitigation is the aspect
  bucket + the requirement of corroboration (exact signals or an independent
  signal for leakage CRITICALs) before any exclusion-grade action. Conversely,
  text-dense pages where only a few words differ produce small dHash/aHash
  deltas, so the text tiers (§6) carry the discriminative weight there.
- **Text tier asymmetry**: exact-normalized equality is high-precision;
  MinHash at Jaccard ≥ 0.85 trades a little precision for recall on OCR-noisy
  duplicates. Digits are deliberately not folded (§6) — a deliberate
  false-negative on digit-variant duplicates in exchange for not merging
  genuinely different numeric content.
- **Blankish pages are excluded from similarity participation**, so two blank
  scans never form a cluster (removes a whole class of false positives at the
  cost of missing "duplicates" that are both effectively blank).
- **Malformed protection metadata fails closed** — a deliberate
  false-positive bias: a row is treated as protected (and quarantined) rather
  than risking a holdout leak.

## 23. Future large-scale validation

Validated in this wave on synthetic fixtures (CPU, offline, deterministic) at
the 100/1k tiers, with the 10k tier behind `pytest.mark.slow` and its
complexity assertions (§20). Future work:

- Run the full suite at ≥ 100k samples with the SQLite index enabled
  (`<index_dir>/quality.v1.db`) to validate O(N) query behavior and
  WAL/batched-write throughput.
- Validate on real mixed-source corpora (Factory output + external licensed
  sources) to calibrate the heuristics defaults (§12) against real blank-page
  and extreme-dimension distributions.
- Wire the deferred **Training Data Loader** adapter contract
  (`training_stream_contract()`) and re-run the gate end-to-end on loader
  streams.
- Add the deferred **Environment Doctor** hook consuming quality reports.
- Recalibrate LSH band parameters (12×16 image, 16×8 text) and the
  0.70/0.85 Jaccard cut points from measured precision/recall on real data.
