# Data Pipeline Audit — Current Session (2026-09-21)

Scope: `clouda_data/` end-to-end (ingestion → pretraining build → factory
render/distort → quality gate → evaluation → results store → training
handoff). Verified from code with file:line evidence.

## Verdict

The core integrity machinery is unusually rigorous: seeded hash-bucket
splitting with union-find grouping, fail-closed holdout protection, atomic
fsync'd manifest writes, content-derived deterministic seeds at every stage,
and a training-side loader that re-verifies manifest digests. The material
risks concentrate in the quality gate's **text** half and the FAIL→clean
manifest path.

## Invariants verified OK (evidence)

1. **Split leakage / holdout.** One split per merged group by construction
   (`pretraining/splitting.py:86-171`); hash-bucket assignment
   `sha256(f"{seed}:split:{key}")` (`:96-102`); report re-verifies
   file/text/document/group/duplicate-cluster disjointness and
   `holdout_disjoint_from_training` (`:251-323`). Holdout excluded from
   export by default (`config.py:44`); export raises if holdout included
   without the explicit flag (`export.py:45-46`); CLI requires explicit
   `--include-holdout`; handoff candidates hardcode
   `include_holdout=False` (`handoff.py:71-73`); training input contract
   fails closed on protected rows (`training_data/input_contract.py:107-219`).
   No RNG shuffle in splitting at all.
2. **Determinism.** Default seed `20260722` recorded in manifest headers;
   distortion randomness derived per (base_seed, page_id, op, index) via
   SHA-256 (`distortion/randomness.py:7-13`); factory seeds are BLAKE2b over
   content+coordinates (`factory/seed/derive.py:23-47`); all discovery/sort
   orders are explicit sorts, no dict/set iteration order in artifacts.
3. **Dedup hashing.** Streamed SHA-256 with symlink refusal and
   stat-before/after rejection (`pretraining/hashing.py:22-36`);
   MinHash permutations from blake2b over a persisted seed, never `hash()`
   (`quality/text_dup.py:88-98`).
4. **Manifest integrity.** Atomic + fsync'd writes with retrying
   `os.replace` (`pretraining/manifest.py:24-41,109-137`); row-count
   self-check on read; schema version + unknown-field rejection per row;
   the gate verifies source-manifest SHA before and after derived writes
   (`quality/derived.py:88-96,163-168`).
5. **Training handoff.** `StreamingTrainingDataLoader.open` recomputes the
   manifest digest and compares to `source_manifest_sha256`
   (`training_data/loader.py:105-122`); shard ids derive from the manifest
   digest and are re-verified (`sharding.py:48-65,136-147`); resume cursors
   fail closed on identity change (`loader.py:357-385`).
6. **Distortion checkpoints.** SQLite WAL store with atomic claims, stale
   recovery, byte-comparison of existing outputs, DB↔JSONL drift
   reconciliation (`distortion/checkpoints.py`, `workflow.py:511-555`).
7. **Results store.** Domain-tagged sha256 identity, immutable run identity
   fields, reject-on-conflict idempotent ingestion, per-run cross-process
   locks, full `verify_bundle` re-hashing (`results/store.py`).

## Confirmed issues (not fixed — need owner decisions; see CODE_AUDIT_CURRENT.md U1–U5)

| ID | Sev | Issue | Evidence |
|---|---|---|---|
| ~~D1~~ | P2 | **RESOLVED 2026-09-22** — wired into the gate; see `RELEASE_READINESS_V0.2.1.md`. | **Text near-dup tier is dead code.** `classify_text_pairs` (MinHash/LSH + Jaccard) has no caller; the gate imports only its version constant. Near-duplicate text leakage across splits is undetected — diacritic/tatweel-only variants have different `normalized_text_sha256` under the dataset normalization, so exact-hash L5 passes them | `quality/gate.py:50-54,161-179`, `text_dup.py:184`, `leakage.py:451-481`, `pretraining/normalize.py:56-68` |
| D2 | P2 | **FAIL verdict still writes a "clean" manifest**, and ERROR-severity artifact findings (`HASH_MISMATCH`, `IMAGE_DECODE`, `NON_EMPTY_FILE`) exclude nothing — the FAIL "clean" manifest can contain the mismatching row. Header does record `verdict: "fail"` and CLI exits 1 | `quality/cli.py:187-217`, `derived.py:72-196`, `policy.py:151-164`, `artifacts.py:140-159,295-304` |
| D3 | P2 | **Download authenticity conditional**: sha256 enforced only when the registry asset carries one; self-computed digests record integrity, not authenticity | `datasets/downloader.py:259,334` |
| D4 | P2 | **Evaluation views diverge**: `evaluation/execution.py:37-38` scores CER/WER on raw strings; `results/metrics.py:44-50` scores on normalized (fold_digits) text. Same records → different canonical numbers | `evaluation/execution.py`, `results/metrics.py` |
| D5 | P2 | **Three Arabic normalizations** (dataset default: NFC only; text-dup policy: NFKC+diacritics/tatweel/alef/ya folding; scoring: NFC+full folding+digit folding) — equivalence classes differ per subsystem, so leakage/dedup classes ≠ scoring classes | `pretraining/normalize.py:56-68`, `quality/text_dup.py:34-48`, `ground_truth/normalization.py:24-51` |

Session fix in this area: `revalidate_derived` is now actually invoked by
`clouda-quality clean-manifest` after every write (the documented fail-closed
backstop was never called before) — `clouda_data/quality/cli.py`.

## Minor (P3)

- `pretraining/workflow.py:269-280,398-410` index/report writes lack fsync
  before replace (recoverable by re-scan).
- `HashCache` keyed on (path, size, mtime_ns) can mask a content change that
  preserves both (suspected, adversarial-only).
- Dedupe canonical selection is path-ordered, not GT-quality-ordered; a
  corrected GT sidecar can be the member dropped (preserved via
  `duplicate_of` linkage).
- `results/ingest.py:264-271` run_id "unknown-evidence" collisions silently
  overwrite mutable run metadata; `evaluation/execution.py:97` persists
  `time.monotonic()` into reports.
- `split_workspace` honors sticky prior splits unless `--seed` is passed
  explicitly (documented, but no warning when ratios change without a seed).

## Leak-glass note for the benchmark cohort

Published benchmark pages carry `source_split: "train"` from external HF
datasets — "held out" relative to Clouda's own training corpora, not to the
source datasets' own partitions. See BENCHMARK_STATE_CURRENT.md.
