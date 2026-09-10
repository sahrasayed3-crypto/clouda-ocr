# DESIGN_DECISIONS.md — Quality Gate implementation spec (Wave-2 contract)
Branch feature/dataset-quality-dedup | Worktree F:\PROJECT\CLOUDA_QUALITY_WT | Base 12e9200
Read LEAD_SYNTHESIS.md first. Every module: Python 3.11, ruff+black (88) clean, mypy
clean (follow existing pretraining module style, full type hints), frozen dataclasses,
schema strings "clouda.quality.<artifact>.v1". NO git commits (Lead commits).
Do NOT edit files outside your ownership. Verify with:
python -m pytest tests/quality -q && python -m ruff check clouda_data/quality && python -m black --check clouda_data/quality && python -m mypy clouda_data/quality --ignore-missing-imports

## Package: clouda_data/quality/

### models.py [Agent A]
IssueSeverity enum info/warning/error/critical; GateVerdict enum PASS/PASS_WITH_WARNINGS/FAIL.
Frozen dataclasses: QualityIssue(code,severity,sample_ids tuple sorted,canonical_key,message,
evidence dict), IssueCode constants (module-level str constants; codes: MISSING_IMAGE,
NON_EMPTY_FILE, HASH_MISMATCH, IMAGE_DECODE, IMAGE_DIMENSIONS, IMAGE_MODE, GT_MISSING,
GT_EMPTY, PATH_SAFE, BLANK_PAGE, NEAR_BLANK_PAGE, EXTREME_DIMENSIONS, EXTREME_ASPECT_RATIO,
SUSPICIOUSLY_SMALL_IMAGE, VERY_LARGE_ARTIFACT, VERY_SHORT_GT, METADATA_DIMENSION_MISMATCH,
DUP_EXACT, DUP_NEAR_IMAGE, DUP_TEXT_NEAR, LEAK_MALFORMED_PROTECTION, LEAK_EXACT_HASH,
LEAK_PAGE_IDENTITY, LEAK_NEAR_IMAGE, LEAK_DERIVED_PAGE, LEAK_GT_TEXT, LEAK_GROUP_STRADDLE),
SampleFingerprint(sample_id, source_id, ahash/dhash/phash hex, aspect_bucket, blankish bool,
blank_stats dict, fingerprint_version), ExactDuplicateGroup, NearDuplicateCandidate(pair ids,
level CANDIDATE/LIKELY_DUPLICATE/CONFIRMED_NEAR_DUPLICATE, distances dict), DuplicateCluster(
cluster_id deterministic "CLU_"+sha256(...)[:12], member_ids sorted, level, evidence),
LeakageFinding(finding_id "LKG_"+sha256[:12], kind, severity, partitions frozenset,
sample_ids, canonical_key, corroborating_signals, raw_split_values, detail),
ArtifactCheckResult, DatasetHealthSummary, ExclusionDecision(sample_id, reason_code,
reason_source, evidence), QualityRun(run_id, manifest_sha256, config_identity, started/finished
UTC iso, verdict, severity_counts, issues), QualityGateResult(run_id, verdict, issues,
clusters, leakage_findings, health, exclusions, reason_codes, schema_version="clouda.quality.run.v1").
All to_dict()/from_dict() with canonical JSON (sort_keys=True, separators=(",",":"),
ensure_ascii=False), version-checked like pretraining schema.

### config.py [Agent A]
QUALITY_GATE_CONFIG_VERSION="clouda.quality.config.v1". Frozen dataclasses per B12 report:
ExactDuplicatePolicy, ImageFingerprintPolicy(algorithm="dhash"|None, hash_size=8,
hamming_candidate=12, hamming_confirmed requires triple-conjunction, max_bucket=4096),
TextNearDuplicatePolicy(normalization: NormalizationPolicy from pretraining.normalize,
shingle_k=4, minhash_perms=128, lsh_bands=16, lsh_rows=8, jaccard_near=0.85, jaccard_review=0.70,
min_text_chars=40), HeuristicsPolicy(blank_std=4.0, near_blank_std=12.0, extreme_max_side=40000,
extreme_max_pixels=400_000_000, max_aspect_ratio=50.0, small_image_min_bytes=1024,
max_artifact_bytes=2_000_000_000, min_gt_warn_chars=3, max_pixels from env
CLOUDA_MAX_IMAGE_PIXELS default 40_000_000), SeverityPolicy(default "warn", overrides dict,
strict_escalates_warn=True), LeakagePolicy(critical codes list, eval_eval_warn=True,
protected via clouda_contracts only), KeepExcludePolicy(exclude_duplicate=True,
exclude_holdout=True, exclude_error=True, exclude_conflicting_duplicate=False,
keep_preference=["protected","canonical_valid","clean_over_distorted","stable_sample_id"]),
ResourceLimits(workers=1, max_samples=0, decode_budget), QualityPaths(index_dir="", resume="").
Methods to_dict()/from_mapping(strict)/fingerprint()/identity() copying PreparationConfig
pattern (clouda_data/pretraining/config.py). Validate: severity values in {info,warn,fail};
conflicting_duplicate exclusion forces leakage merge semantics documented.

### manifest_adapter.py [Agent B]
load_manifest(path)->(header, list[DatasetSample]) via pretraining manifest.read_manifest +
DatasetSample.from_dict (strict). manifest_sha256 via hashing.sha256_file.
resolve_artifact_path(sample, root) -> Path: schema.canonical_relative_path check +
validate_relative_components (clouda_contracts.storage) + resolved.containment under root;
symlink check Path.is_symlink() BEFORE any open (refuse, finding PATH_SAFE error).
run_identity(manifest_sha, config_identity, algo_versions) -> QualityRunIdentity.
DatasetIdentity/PageIdentity from clouda_contracts where needed. Loader adapter boundary:
stub module docstring + function `training_stream_contract()` documenting deferred
Training Data Loader connection (NOT on main).

### image_fp.py [Agent D]
Lift 64-bit dHash from pdfword/teacher_pipeline/manifests.py (verify against installed
Pillow 11 API; add unit tests). Pillow-only (NO numpy in this module). Deterministic
integer-only: aHash 8x8 (mean via //64, ties->0), dHash 9x8 horizontal gradient,
pHash fixed-point DCT (precomputed integer cosine table T=round(2048*c(u)*cos(...))
committed as literals, 32x32 input, top-left 8x8, drop DC, lower-median threshold index 31,
ties->0). Preprocess: grayscale, content-crop white margins (256-wide downsample,
margin_value-12 threshold, pad 2%, fallback full page), aspect bucket key
(anchors 0.707/0.773/1.0/1.294/1.414 at 2% tolerance else round(r*100)).
Blankish: std_i<=3 or >=0.995 pixels within +-6 of modal (from 32x32, integer math).
DecompressionBombWarning->error, Image.MAX_IMAGE_PIXELS respected, verify-then-thumbnail,
lazy decode only. fingerprint_version="clouda.quality.imgfp.v1:pillow==<exact version>".
Safe load helper shared with artifacts.py (single decode discipline).

### exact_dup.py [Agent C]
Wrap pretraining dedupe.classify_duplicates as-is for exact signals; add raw_text_sha256
group (domain-separated sha256("clouda.text.raw.v1\x00"+text)) — evidence-level only:
same raw text over DIFFERENT file hashes stays CONFLICTING_DUPLICATE (never DUPLICATE).
Add near-signal plumbing: accept precomputed NearDuplicateCandidate CONFIRMED pairs and
union them via same union-find semantics (reason "near_duplicate_image"), bump report
schema to clouda.quality.dedupe.v2 with near_duplicate counts. NEVER delete data.

### near_index.py [Agent E]
LSH banding over 192-bit concat (phash||dhash||ahash), 12 bands x 16 bits, same
aspect-bucket requirement, MAX_BUCKET=4096 overflow guard (skip+flag, never pairwise).
Intra-document adjacent-page pass (document_id+page_index). Optional SQLite stdlib index
in <index_dir>/quality.v1.db (WAL, batched tx): tables sample(sample_pk, source_id,
source_path, sample_id, file_sha256, normalized_text_sha256, fingerprint cols, UNIQUE...),
lsh_band(band_id, band_key, sample_pk), run_state(manifest_hash, config_hash,
algorithm_versions, stage, stage_cursor, processed_count) per B8 schema. Streaming inserts,
O(N) queries via index-ordered scans. If SQLite unused in v1 (in-memory acceptable for
<=10k), keep schema constants + run_state JSON fallback; tests at 1k/10k must not go O(N^2).

### confirm.py [Agent F]
Levels: CONFIRMED iff d_p<=8 AND d_d<=10 AND d_a<=10 (triple conjunction).
LIKELY iff d_p<=12 (else CANDIDATE). LIKELY pairs get 64x64 integer MAD
(MAD=sum|a-b|//4096): MAD<=6 -> CONFIRMED; MAD>24 -> CANDIDATE; between stays LIKELY.
Union-find only for CONFIRMED (reuse dedupe canonical_key for representative).
Deterministic: buckets iterated (band_idx, key) sorted, members by sample_id, pair order
(level_rank, d_p, d_d, d_a, mad, min_id, max_id). Cluster ids deterministic.

### text_dup.py [Agent G]
Tier0 raw sha256 (domain-separated). Tier1 normalized via normalize_text with
DEDUPE_TEXT_POLICY = NormalizationPolicy(unicode_form="NFKC", presentation_forms="compose",
strip_bom, normalize_line_endings, preserve_line_breaks, collapse_whitespace, remove_zero_width,
remove_control_characters, remove_tatweel, remove_diacritics=True, fold_alef=True, fold_ya=True,
fold_digits=False) — persist policy.version() with every fingerprint; recompute on mismatch.
Tier2 MinHash char-4-grams (blake2b digest_size=8 per shingle; 128 perms
h=(a*x+b) mod 2^61-1, a,b derived from blake2b of persisted seed); LSH 16x8; verify exact
Jaccard: >=0.85 near_text family (splitter-union via group), 0.70..0.85 quality flag
near_text_review only. Skip tier2 for <40 chars (flag near_dup_skipped). Band owner cap 20.
Never python hash(). Never same-text-alone => same page.

### leakage.py [Agent H]
Consume clouda_contracts.protection + clouda_lab.holdout_guard ONLY (no new protection
logic). effective_partition(sample)->TRAIN|EVAL|PROTECTED|UNPARTITIONED with fail-closed
precedence: row_is_protected first (quarantine PROTECTED), then target_split via
is_training_split_eligible / PROTECTED_SPLIT_NAMES, else source_split/role via
string_marks_protected (normalize_marker). Checks L0..L6 per Wave1-B7 report
(malformed->CRITICAL; file hash train<->eval CRITICAL; page identity train<->eval CRITICAL;
near-image confirmed train<->eval CRITICAL when corroborated by >=1 independent signal;
derived/distorted same source page across train/eval boundary CRITICAL when corroborated;
GT-text alone WARN; group/document straddle via resolve_group_key CRITICAL; eval<->eval
duplicates WARN). LeakageFinding/LeakageReport per B7 shapes
(schema clouda.quality.leakage_finding.v1 / leakage_report.v1).
Protected rows: IDs + codes only, never content. malformed sha256 format -> fail-closed.

### artifacts.py [Agent I]
validate_artifact(sample, root, thresholds)->list[QualityIssue] extending pretraining
validation semantics: reuse path/exists/decode/dims/text codes; NEW: NON_EMPTY_FILE
(zero-byte, error), HASH_MATCH (sha256_file vs file_sha256; None->info not mismatch),
IMAGE_MODE warn {P,1,I,I;16,F,CMYK,YCbCr}, METADATA_DIMENSION_MISMATCH warn, heuristics
WARN-only defaults: BLANK_PAGE (stddev<4.0 on 256-wide thumbnail), NEAR_BLANK_PAGE
(<12.0 or ink<0.001), EXTREME_DIMENSIONS (>40000/side or >400MP; unusual sizes never FAIL),
EXTREME_ASPECT_RATIO (>50), SUSPICIOUSLY_SMALL_IMAGE (<1KB or tiny+dims<32), VERY_LARGE_ARTIFACT
(>2GB stat-only), VERY_SHORT_GT (0<len<3). Single-decode discipline: one verify-then-load
open shared with image fingerprinting; symlink refusal before open; streaming hash.
Per-code counts in output.

### health.py [Agent J]
DatasetHealthSummary schema clouda.dataset.health.v1: counts-only across 11 dimensions
(source, document_type, language, script, split, clean_distorted via
len(transformations)>0, quality_flags, validation_status, duplicate_state,
resolution_bucket pinned edges, gt_length_bucket pinned edges) + small closed cross-tabs;
bucket edges in bucket_definitions dict (hashed into config identity). No invented labels.

### policy.py + derived.py [Agent K]
policy.py: deterministic keep/exclude per sample: never let training row override
protected/eval row in any cluster (protected always kept+quarantined); preference order
from config (protected > canonical-valid (lower canonical_key) > clean-over-distorted >
stable_sample_id). Produce ExclusionDecision list + exclusion report schema
clouda.pretraining.exclusion.v1 (closed reason vocabulary: validation codes + duplicate/
near_duplicate_image/holdout/unassigned_split/quality_gate:<id>, reason_source in
{validation,dedupe,split,quality_gate}, fixed precedence so counts additive).
derived.py: clean manifest via pretraining manifest.write_manifest (canonical v1 rows
sorted by its own ordering, atomic) with header metadata lineage: source_manifest_sha256,
quality_run_id, config_identity, derived_dataset_version="derived-1.0.0+<srchash[:12]>",
exclusion_report_sha256, gate verdict. Quarantine manifest (same writer, disjoint ids
asserted). ORIGINAL NEVER MODIFIED (verify sha before/after write). Post-write
re-validation hook: re-read derived manifest and re-check protection (fail-closed)
like training_orchestrator does.

### run_state.py [Agent L]
Resume: index_dir/run_state JSON (or SQLite row if E built it): manifest_sha256,
manifest_row_count, config_identity, algorithm_versions dict, stage, stage_cursor,
processed_count. start_or_resume(run_dir, identity) -> state or raises StaleResumeError
with precise mismatch message. Stage checkpoints after each stage; atomic JSON writes.
Changed manifest/config/algorithm => refuse resume (exit code path 2).

### report.py [Agent M]
render_report(result, fmt in {text,json}) -> str. JSON: clouda.quality.run.v1 full
schema incl severity_counts, reason code tallies, checks with thresholds, clusters
summaries, leakage findings, health, exclusions, artifact paths+sha256s. Gate rule:
FAIL iff any critical/error issue or split_report.passed False; PASS_WITH_WARNINGS iff
warnings only; else PASS. Protection filter before ANY write: record_is_protected rows
emit id+codes only; redact_mapping (clouda_contracts.security) over echoed metadata.
Human text: Dataset Health Report layout per prompt with GATE line last.

### cli.py [Agent N]
clouda_data/quality/cli.py, argparse prog="clouda-quality", build_parser()+main(argv)
copying pipeline CLI conventions (UTF-8 reconfigure, _cmd_*(args)->int, JSON default
print(json.dumps(..., ensure_ascii=False, indent=2))). Subcommands: scan(manifest
positional, --config --output --fresh), verify(--report --config), report(--input
--output --format text|json), clean-manifest(manifest positional, --output required,
--dry-run, --config), inspect-cluster(--report --cluster-id required). Shared flags
--json --strict --resume --no-near-duplicates --cross-split-only --max-samples N.
Exit codes: 0 PASS, 1 gate FAIL, 2 config/usage error (QualityGateConfigError caught).
Register in pyproject [project.scripts]: clouda-quality = "clouda_data.quality.cli:main"
(you own the pyproject edit; nothing else in pyproject).

### results_bridge.py [Agent O]
Optional module: persist_quality_summary(store: ResultsStore, run_id, summary)->None via
store.save_summary; metadata marks kind="dataset_quality_run". Guarded import; no schema
change to results store. Unit test with fake store object.

### benchmarks.py [Agent P]
Synthetic suite: make_manifest(root, n, dup_rate, seed) tiny 64x48 PNGs; tiers 100/1k/10k;
metrics: scan_time, fingerprint rate, candidate count, decode count (monkeypatch
PIL.Image.open counter), tracemalloc peak (separate pass from timing). Assertions:
candidates(10k)/candidates(1k) < 15; scan_time ratio < 20; decodes <= 2x records at 10k;
peak ratio < 15; correctness invariants (dup counts match rates) even in perf runs.
100+1k default suite; 10k behind pytest.mark.slow. Add markers config to pyproject? NO —
put marker registration in tests/quality/conftest.py instead (pyproject owned by N).

### tests/quality/ [Agent Q]
conftest.py (fixtures: render_arabic_page(seed,layout,text,size) pure Pillow bars + optional
font probe; mutation helpers recompress/resize/brightness/noise/blank_like; make_manifest;
register 'slow' marker via pytest_configure). Files per B15/16 matrix: test_canonical_input,
test_exact_duplicates, test_image_near_duplicates, test_arabic_text, test_leakage (incl.
adversarial metadata T1..T13 from B7), test_artifact_corruption, test_clustering_determinism,
test_derived_manifest, test_resume, test_cli, test_scale_tiers. Reuse
tests/pretraining_fixture.build_tiny_dataset. All CPU/offline/deterministic/fast.
E2E test: synthetic dataset with exact dup + recompressed near-dup + distinct similar
layout + cross-split leak -> scan -> FAIL verdict -> apply exclusion policy -> derived
clean manifest -> verify (leak absent, dups absent, deterministic re-run byte-identical).

### docs/data_quality/CLOUDA_DATA_QUALITY.md [Agent R]
All 28 topics from the task brief (purpose, architecture, relations to Factory QC/
Results Store/Selection/Loader, dedup rules, algorithms+thresholds, Arabic normalization,
clustering, leakage+holdout protection, artifact checks, heuristics, severity policy,
keep/exclude, derived manifest, run identity, resume, CLI, JSON schema, performance,
limitations, FP/FN tradeoffs, future large-scale validation).

## Global invariants (every agent)
- Consume pretraining/contracts/holdout_guard APIs; never re-implement protection.
- No O(N^2) normal path; no python hash(); no new dependencies (Pillow core only;
  numpy/SQLite stdlib allowed only where specified).
- Determinism: canonical JSON everywhere; sorted iteration; stable ids.
- Original dataset never modified; reports never contain protected content.
- mypy strict-ish compliance matching repo gate (mypy clouda_data --ignore-missing-imports).
