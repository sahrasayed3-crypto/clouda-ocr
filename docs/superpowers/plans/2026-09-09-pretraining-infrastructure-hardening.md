# Pre-Training Dataset Infrastructure Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit commit `c92c9be`, reproduce invariant failures, harden only the pre-training dataset infrastructure, and leave a verified regression suite and commit.

**Architecture:** Preserve the existing `clouda_data.pretraining` package and JSONL workspace boundary. Harden provenance paths at discovery time, make cached/indexed hashes depend on stable file identity metadata, recompute dataset-wide derived state after incremental source changes, and validate serialized/configured state strictly at boundaries.

**Tech Stack:** Python 3.11, pathlib, dataclasses, JSON/JSONL, Pillow, pytest, Hypothesis where property coverage materially helps, ruff, black, mypy.

**Spec:** `C:/Users/Ahmed/.codex/attachments/a4cf5563-69d4-45eb-80fa-a3b84e2d392a/pasted-text.txt`

## Global Constraints

- Review only pre-training dataset infrastructure and directly interacting existing code.
- Do not download datasets or model weights, run training, add serving/billing/cloud infrastructure, or modify `sahrasayed3-crypto/clouda-data-factory`.
- Every production fix starts from a reproducing test that fails for the intended invariant.
- Raw OCR ground truth remains untouched; holdout remains excluded from training paths by default.
- Preserve deterministic byte output and Windows/POSIX-safe behavior.

---

### Task 1: Strict schema, source, and configuration boundaries

**Files:**
- Modify: `clouda_data/pretraining/schema.py`
- Modify: `clouda_data/pretraining/sources.py`
- Modify: `clouda_data/pretraining/config.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: JSON-compatible mappings and source definitions.
- Produces: strict version/unknown-field/type/ratio validation and a documented, normalized sample identity contract.

- [ ] Add failing tests for schema version mismatch, unknown sample fields, non-finite/negative split ratios, non-boolean holdout flags, unsafe source roots, and cross-platform semantic path identity.
- [ ] Run each focused test and confirm it fails for the boundary bug.
- [ ] Add minimal validation and canonical provenance-path normalization without changing unrelated metadata semantics.
- [ ] Run focused and existing unit tests until green.

### Task 2: Root-contained streaming discovery

**Files:**
- Modify: `clouda_data/pretraining/discovery.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: local source root plus image/sidecar/JSONL/CSV/TSV references.
- Produces: deterministic `SampleDraft` values whose file references are normalized POSIX-relative paths contained by the source root, or stable malformed findings.

- [ ] Add table-driven failing tests for `./`, `../`, absolute POSIX, Windows drive, UNC, symlink escape, null/control characters, malformed JSON, invalid UTF-8, Unicode names, and equivalent path spellings.
- [ ] Confirm hostile records fail without escaping or aborting valid sibling records.
- [ ] Implement one root-containment helper and stream JSONL line-by-line; preserve malformed records as findings.
- [ ] Run discovery and workflow tests until green.

### Task 3: Correct hash caching and mutation detection

**Files:**
- Modify: `clouda_data/pretraining/hashing.py`
- Modify: `clouda_data/pretraining/discovery.py`
- Modify: `clouda_data/pretraining/workflow.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: source id, normalized relative path, size, `mtime_ns`, and file path.
- Produces: SHA-256 cache/index entries reused only when source/path/size/mtime metadata agrees and hashing observes a stable file.

- [ ] Add a failing test that rewrites a file with same-size content and proves resume returns a stale hash.
- [ ] Add failing tests for same relative path in two sources, corrupted cache lines, and mutation during hashing where practical.
- [ ] Extend discovered/index/cache metadata and verify file stat before/after streaming; namespace cache keys by source.
- [ ] Run focused index/resume tests until green.

### Task 4: Deterministic duplicate and leakage grouping

**Files:**
- Modify: `clouda_data/pretraining/dedupe.py`
- Modify: `clouda_data/pretraining/splitting.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: complete semantic sample set.
- Produces: one deterministic canonical per connected duplicate family and integer-domain deterministic split assignments for connected leakage groups.

- [ ] Add permutation tests for tied canonical candidates, overlapping duplicate criteria, repeated ids, and transitive document/text/file chains.
- [ ] Add failing ratio tests for negative, greater-than-one, NaN/infinity, and zero-ratio boundaries.
- [ ] Replace order-sensitive canonical selection with connected-component classification and make split thresholds integer-domain.
- [ ] Run focused tests under multiple input permutations and seeds.

### Task 5: Validation safety and holdout-safe exports

**Files:**
- Modify: `clouda_data/pretraining/validation.py`
- Modify: `clouda_data/pretraining/export.py`
- Modify: `clouda_data/pretraining/handoff.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: samples plus explicit validation/export policy.
- Produces: per-record findings without global mutable thresholds, bounded image decoding, deterministic export replacement, and handoff candidates that never include holdout/training-restricted data.

- [ ] Add failing tests for decompression-bomb dimensions, per-call threshold isolation, stale split export files, malformed holdout config, and restricted handoff candidates.
- [ ] Pass thresholds explicitly, reject unsafe image loads before decoding, validate export options, and remove stale outputs atomically.
- [ ] Run validation/export/handoff tests until green.

### Task 6: Compatible dataset-wide resume and incremental recomputation

**Files:**
- Modify: `clouda_data/pretraining/workflow.py`
- Modify: `clouda_data/pretraining/manifest.py`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: registered sources, per-source indexes, prior manifest metadata, and preparation configuration fingerprint.
- Produces: a complete dataset-wide manifest whose validation/dedupe/split/export/report state matches a clean run, or a clear refusal when stored state is incompatible/corrupt.

- [ ] Add failing fresh-versus-incremental byte comparisons for adding source B, cross-source duplicates, source removal/disable, changed files/text, normalization policy, seed, ratios, and corrupt manifests.
- [ ] Store deterministic preparation metadata and rebuild all enabled indexed sources before dataset-wide derived stages.
- [ ] Reject malformed/incompatible serialized state where safe recomputation is impossible; never silently retain stale derived fields.
- [ ] Run focused resume, determinism, and end-to-end tests until green.

### Task 7: CLI, documentation, scale review, and final verification

**Files:**
- Modify: `clouda_data/pipeline/cli.py` only for confirmed command-contract bugs.
- Modify: `docs/pretraining_dataset_infrastructure.md`
- Test: `tests/test_pretraining_hardening.py`

**Interfaces:**
- Consumes: all hardened package APIs.
- Produces: accurate CLI exits/help, honest limitations, complete verification evidence, and one scoped commit.

- [ ] Exercise every dataset command for help, malformed ratios/config, dry-run, fresh/resume, seed, handoff, and machine-readable success/error behavior.
- [ ] Run a generated workload to identify quadratic paths and document remaining in-memory/JSONL limits without adding distributed infrastructure.
- [ ] Correct claims about bounded memory, atomicity, cache keys, resumability, supported adapters, identity, holdout, and Data Factory scope.
- [ ] Run all pretraining tests, full pytest, ruff, black check, mypy on modified modules, and a tiny end-to-end workflow.
- [ ] Inspect diff/status for secrets, generated data, and unrelated changes; commit as `fix: harden pre-training dataset infrastructure`.
