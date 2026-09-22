# Benchmark State — Current Session (2026-09-21)

Evidence state only. **No results were invented, no rankings created, no
interim metrics published by this session.** The separate benchmark
repository (`clouda-ocr-benchmark`) was not read, touched, or modified; it is
not present on this machine.

## Published benchmark v0.1.0 (read-only evidence)

- Scope: **177 pages**, 6 models with full coverage ranked; 1 PARTIAL
  (`dots.mocr`, 30/177) and 1 FAILED_SMOKE (PaddleOCR-VL-1.6) excluded from
  ranking (`benchmarks/ocr_arabic/results.csv`, `release.json`).
- Release identity pinned by the SHA-256 of `benchmark_manifest.jsonl`
  (`release.json:6`); `validate_release.py` enforces canonical leaderboard
  order, sorted N-CER, and completion-based rankability; wrapped by
  `tests/benchmarks/test_ocr_arabic_release.py`.
- Raw evidence declared `retained_privately` (`release.json:13`); pages
  carry per-row provenance (source dataset, split, source/clean/GT sha256,
  seeds, repository revision).
- Docs (`README.md:208-211`, `ROADMAP.md`, `docs/ROADMAP.md`,
  `docs/MODEL_INTEGRATION.md`) and results are mutually consistent; no
  contradictions found.

## Current expanded evaluation (NOT published)

- Documented framing: **462 held-out pages, paused pending compute**.
  Confirmed consistent across all four prose locations.
- **No 462-page manifest, run bundles, or per-model results exist anywhere in
  this repository.** The string "10-model" appears nowhere in the repo; the
  only model inventory is `models.csv` (8 records). The expanded-evaluation
  plan lives outside this repo and is untracked here — recommend recording
  its plan/status in-repo before resuming so the evidence chain is visible.
- Local state directories (`runs/_backend_probe`, `outputs/demo`,
  `_state/datasets` incl. a rasam_first_batch ingestion of 88 pages + 16
  distortion runs) are unrelated to the expansion.

## Infrastructure findings

| ID | Sev | Finding |
|---|---|---|
| B1 | P2 | **Cohort provenance caveat**: manifest rows record `source_split: "train"` from external HF datasets — the 177-page cohort is "held out" relative to Clouda's own corpora, not relative to the source datasets' own partitions. Third parties trained on those datasets' train splits could overlap. The in-repo leakage machinery (`clouda_lab/holdout_guard.py`, `quality/leakage.py`) is fail-closed, but **no benchmark-specific artifact proves a dedup-vs-training-corpora run** for either cohort. Record one when the expansion resumes. |
| B2 | P2 | **Three CER/WER implementations** with divergent policies (foundation engine: caller-normalized, empty-ref→1.0, unclamped; local benchmark `calculate_metrics.py`: NFC+newline only, `max(1,ref_len)` denominator → unbounded CER; error-analysis N-CER: fold_digits normalization). Only the synthetic local benchmark uses the second; the published release uses the restored private implementation. Unify or document the mapping. |
| B3 | P3 | Published N-CER methodology (`methodology.md:24-36`) does not mention digit folding, while the in-repo canonical policy (`normalize_ocr_text`) folds digits; the published implementation is not in this repo to verify. Clarify in methodology text. |
| B4 | P3 | `tools/run_full_benchmark.py` writes wall-clock-stamped report filenames (content deterministic); Results Store `verify_bundle` validates JSONL records but not artifact blobs. |
| B5 | P3 | `evaluation/execution.py` labels its report "normalized CER" while scoring raw strings (see DATA_PIPELINE_AUDIT D4) — naming hazard only. |

## Reproducibility mechanisms (verified)

Deterministic category/page ordering, failure accounting with non-zero exit
on any failure (`tools/run_full_benchmark.py:577-632`), edit-count DP with
deterministic tie-breaking, idempotent reject-on-conflict results ingestion
with immutable run identity, and per-page checksummed provenance.
