# Lead Synthesis — Wave 1A + Lead Inspection (architecture decisions)
Branch: feature/dataset-quality-dedup | Base: origin/main 12e9200 | Worktree: F:\PROJECT\CLOUDA_QUALITY_WT

## DECIDED ARCHITECTURE (reconciled from A1/A2/A3/A4 + lead inspection)

### Package placement
NEW subpackage `clouda_data/quality/` — NOT inside pretraining/ (pretraining is a
complete self-contained pipeline; quality gate is a cross-cutting dataset-level
subsystem consuming its manifests). Docstring must distinguish it from
clouda_data/validation/* (generated-output validators) and pretraining/validation.py
(sample-level).

### Module map (Wave-2 file ownership)
clouda_data/quality/
  __init__.py            (public surface, schema version constants)
  models.py        [A]   frozen dataclasses: QualityGateConfig, QualityRun, QualityIssue,
                         IssueSeverity(info/warning/error/critical), IssueCode,
                         SampleFingerprint, ExactDuplicateGroup, NearDuplicateCandidate,
                         DuplicateCluster, LeakageFinding, ArtifactCheckResult,
                         DatasetHealthSummary, ExclusionDecision, QualityGateResult.
                         Verdicts PASS/PASS_WITH_WARNINGS/FAIL. Schema strings
                         clouda.quality.<artifact>.v1.
  config.py        [A]   QualityGateConfig + fingerprint() (canonical JSON -> sha256),
                         references pretraining NormalizationPolicy; pixel limit from
                         env CLOUDA_MAX_IMAGE_PIXELS default 40_000_000.
  manifest_adapter.py [B] canonical input: manifest.read_manifest/read_samples,
                         DatasetSample.from_dict, StorageRoots-safe path resolution,
                         run identity (manifest sha256 + config fingerprint +
                         algorithm versions). Adapter boundary stub for
                         Training Data Loader (NOT on main -> deferred doc only).
  exact_dup.py     [C]   wraps dedupe.classify_duplicates; adds raw-GT-hash and
                         normalized-GT-hash groups; NEVER treats same text alone as
                         same page (conflicting_duplicate semantics preserved).
  image_fp.py      [D]   dHash 64-bit lifted from pdfword/teacher_pipeline/manifests.py
                         (Pillow-only tier 1; verify getdata vs flattened API on
                         installed Pillow); blank-page exclusion from fp similarity;
                         aspect-ratio bucket key.
  near_index.py    [E]   LSH banding over 64-bit fingerprints; SQLite stdlib index
                         (tables: samples, fp_bands, exact_buckets, runs, stage_state);
                         resumable stage cursor keyed by (manifest_sha, config_fp,
                         algo versions); bounded memory.
  confirm.py       [F]   candidate confirmation (Hamming <= CONFIRMED threshold,
                         optional structural confirmation); confidence levels
                         CANDIDATE / LIKELY_DUPLICATE / CONFIRMED_NEAR_DUPLICATE;
                         deterministic clustering (union-find, stable member order,
                         stable cluster ids).
  text_dup.py      [G]   raw sha256 + normalized sha256 (normalize_text +
                         policy fingerprint pinned) + near-text MinHash/SimHash on
                         char n-grams [final choice per Wave1-B6 report].
  leakage.py       [H]   cross-split engine consuming clouda_contracts.protection
                         (record_is_protected, protection_metadata_is_malformed,
                         is_training_split_eligible) + holdout_guard facade;
                         CRITICAL findings for: protected row training-eligible,
                         exact cross-split dup (artifact hash / page identity),
                         confirmed near-dup across train vs eval boundary,
                         source/document group straddling train vs eval.
                         Fail-closed on malformed metadata. Protected rows: emit
                         IDs + codes only, never content.
  artifacts.py     [I]   integrity catalog: reuses validate_sample findings + adds
                         symlink check BEFORE open (closes main's gap), HASH_MATCH
                         recompute via hashing.sha256_file, IMAGE_DECODE/MODE,
                         heuristics BLANK_PAGE (reuse validation.image_quality.
                         detect_blank_from_pixels), NEAR_BLANK, EXTREME_DIMENSIONS,
                         EXTREME_ASPECT_RATIO, SUSPICIOUSLY_SMALL, VERY_LARGE,
                         VERY_SHORT_GT. Heuristics default WARN, config-severity.
  health.py        [J]   DatasetHealthSummary from EXISTING DatasetSample fields only.
  policy.py        [K]   keep/exclude: never let training sample override protected;
                         determinstic hierarchy; exclusions recorded with reason.
  derived.py       [K]   clean derived manifest via manifest.write_manifest
                         (canonical v1 rows, sorted, atomic), quarantine manifest,
                         exclusion report; lineage: source manifest sha256, run id,
                         config fingerprint; original untouched.
  run_state.py     [L]   resume: stage_state table; reject stale resume on
                         manifest/config/algorithm-version mismatch.
  report.py        [M]   human + JSON report; gate verdict + reason codes;
                         protection filter before write (record_is_protected ->
                         ID+code only); redact_mapping on echoed metadata.
  cli.py           [N]   NOT a new entry point: quality-* subcommands registered in
                         clouda_data/pipeline/cli.py via build_parser extension,
                         handler pattern <name>_cli(args)->int, JSON via
                         json.dumps(ensure_ascii=False, indent=2), exit 1 on gate
                         FAIL, exit 2 on config error.
  results_bridge.py [O]  optional: persist run summary via ResultsStore.save_summary
                         (no schema change, metadata marks quality run).
  benchmarks.py    [P]   synthetic perf suite 100/1k/10k; complexity assertion.

tests/quality/{unit,integration}/  [Q]  + reuse tests/pretraining_fixture.
docs/data_quality/                [R]  CLOUDA_DATA_QUALITY.md.

### Non-negotiable invariants (from A3)
- Protection = clouda_contracts.protection ONLY. Never hand-roll split checks.
- Gate verdicts never broaden EXPORTABLE_SPLITS; eligibility derived at read time.
- Training paths raise (assert_selection_safe semantics); analysis paths filter+tally.
- Derived manifests re-validated after write (validate_derived_manifest_for_training).
- include_holdout stays explicit opt-in, config-fingerprinted.
- No O(N^2) in normal path; no python hash(); no new deps beyond Pillow tier.

### Deferred integrations (documented, not built)
- Training Data Loader: NOT on main -> adapter contract + deferred E2E doc.
- Environment Doctor: NOT on main -> future hook doc only.
- Results Store: IS on main -> optional summary bridge only (O).

## WAVE-2 AGENT OWNERSHIP (letters = module owners above)
One agent per owned file group; no overlapping writes. Lead owns cli.py wiring in
clouda_data/pipeline/cli.py (single-writer rule for that file) and integration.
