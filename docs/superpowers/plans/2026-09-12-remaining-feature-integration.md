# Remaining Feature Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one coherent, tested Clouda OCR canonical system from the six remaining feature branches and publish it to `main` with a normal fast-forward push.

**Architecture:** Preserve each named branch with a merge commit in dependency order, then reconcile imported capabilities at canonical manifest, protection, loader, planner, preflight, adapter, runtime, checkpoint, Doctor, and CLI boundaries. Use offline synthetic tests and the existing experiment checkpoint envelope as the integration spine.

**Tech Stack:** Python 3.11, dataclasses, JSON/JSONL, YAML, pathlib, hashlib, PyTorch as an optional lazy dependency, pytest, Ruff, Black, MyPy, setuptools build.

**Spec:** `docs/superpowers/specs/2026-09-12-remaining-feature-integration-design.md`

## Global Constraints

- Start from fetched `origin/main` at `14e9600eafece7be929c89552564ffa47d1b2696` unless a later fetch proves it advanced.
- Preserve all six branch histories with normal merge commits; do not squash, rewrite public commits, force-push, or delete remote branches.
- Do not download models or datasets, start real training, require GPU/CUDA/NCCL, or commit large artifacts.
- Reuse canonical manifests, protection policy, loader, orchestrator, experiment framework, checkpoint envelope, Doctor, and CLI.
- Keep benchmark manifest SHA-256 `2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893` unchanged.
- Add a failing regression test before every semantic production-code fix made during integration.

---

### Task 1: Preserve the audited integration design

**Files:**
- Create: `docs/superpowers/specs/2026-09-12-remaining-feature-integration-design.md`
- Create: `docs/superpowers/plans/2026-09-12-remaining-feature-integration.md`

**Interfaces:**
- Consumes: verified remote tips, merge bases, ancestry counts, changed-file audit, and baseline `pytest` result.
- Produces: a committed integration contract and executable gate sequence.

- [ ] Record the six exact tips, verified `origin/main`, branch topology, canonical ownership, safety constraints, and validation gates in the design.
- [ ] Check both documents for placeholder markers, incomplete tasks, contradictory order, and omitted branch names.
- [ ] Commit both documents with `docs: plan remaining feature integration`.

### Task 2: Merge and validate Dataset Quality/Dedup

**Files:**
- Merge: `origin/feature/dataset-quality-dedup`
- Inspect: `clouda_data/quality/**`
- Inspect: `clouda_data/pipeline/cli.py`
- Inspect: `clouda_lab/dataset_selection.py`
- Inspect: `clouda_data/training_data/input_contract.py`
- Test: `tests/quality/**`

**Interfaces:**
- Consumes: canonical manifests, stable sample identities, Results/Lab selections, and shared protection semantics.
- Produces: deterministic quality reports and lineage-bearing derived manifests accepted by the canonical loader.

- [ ] Merge the branch with `git merge --no-ff origin/feature/dataset-quality-dedup`.
- [ ] Run `python -m pytest tests/quality -q` and record exact pass/skip/warning counts.
- [ ] Inspect identity, protection, path, streaming, report atomicity, Results/Lab bridge, derived-manifest, loader, orchestration, and CLI JSON boundaries.
- [ ] For each defect, add a focused failing test, verify the expected failure, implement the smallest canonical fix, and rerun Quality plus affected Results/Lab/loader suites.

### Task 3: Merge and validate Runtime and Hunyuan bridge

**Files:**
- Merge: `origin/feature/real-training-runtime`
- Merge: `origin/feature/hunyuanocr15-sft-bridge`
- Reconcile: `clouda_training/experiments/runs.py`
- Reconcile: `pyproject.toml`
- Inspect: `clouda_training/runtime/**`
- Inspect: `clouda_training/hunyuan/**`
- Test: `tests/runtime/**`
- Test: `tests/hunyuan/**`

**Interfaces:**
- Consumes: canonical experiment runs/checkpoints, loader batches, explicit adapter identity, and synthetic model components.
- Produces: deterministic single-process runtime execution and an adapter-compatible Hunyuan conversion/training bridge.

- [ ] Merge Runtime with `--no-ff`, preserving current loader-aware run/resume metadata when resolving `runs.py` and retaining every canonical optional-dependency group in `pyproject.toml`.
- [ ] Run `python -m pytest tests/runtime tests/training tests/data_foundation/unit/test_training_data_loader.py -q`.
- [ ] Merge Hunyuan with `--no-ff`, then run `python -m pytest tests/hunyuan tests/runtime -q`.
- [ ] Inspect seed/rank/worker identity, failure cleanup, exact resume, checkpoint envelope reuse, portable lineage, adapter batch handling, lazy imports, revision metadata, protection, and deterministic preprocessing.
- [ ] Add red-green regression tests for semantic gaps before changing production code.

### Task 4: Merge and validate canonical adapters

**Files:**
- Merge: `origin/feature/multimodel-training-adapters`
- Inspect: `clouda_training/adapters/**`
- Inspect: `clouda_training/hunyuan/**`
- Inspect: `clouda_training/qwen/**`
- Reconcile: `clouda_training/cli.py`
- Test: `tests/multimodel/**`
- Test: `tests/hunyuan/**`
- Test: `tests/runtime/**`

**Interfaces:**
- Consumes: adapter descriptors, deterministic configs, loader rows/batches, and runtime backend contract.
- Produces: one explicit registry and model-family adapters with truthful capabilities and no silent semantic fallback.

- [ ] Merge with `git merge --no-ff origin/feature/multimodel-training-adapters`.
- [ ] Run `python -m pytest tests/multimodel tests/hunyuan tests/runtime -q`.
- [ ] Inspect registration idempotence, lazy optional dependencies, capability detection, config serialization, data conversion, runtime selection, checkpoint identity, CLI list/inspect/preflight, and no-download test isolation.
- [ ] Add red-green regression tests for any canonical-interface or safety defect before fixes.

### Task 5: Merge and validate Preflight and Planner

**Files:**
- Merge: `origin/feature/training-preflight-validator`
- Merge: `origin/feature/training-experiment-planner`
- Inspect: `clouda_training/preflight/**`
- Inspect: `clouda_training/planner/**`
- Reconcile: `clouda_training/cli.py`
- Test: `tests/preflight/**`
- Test: `tests/planner/**`

**Interfaces:**
- Consumes: canonical dataset/derived lineage, loader/shard identities, adapter descriptors, runtime/checkpoint metadata, dependencies, storage, and hardware facts.
- Produces: deterministic portable plans and fail-closed run-specific readiness reports for the canonical orchestrator.

- [ ] Merge Preflight with `--no-ff` and run `python -m pytest tests/preflight tests/multimodel tests/hunyuan tests/runtime -q`.
- [ ] Confirm Doctor remains project/environment diagnosis while Preflight validates a specific planned run, with ERROR/WARN/informational hardware distinctions.
- [ ] Merge Planner with `--no-ff` and run `python -m pytest tests/planner tests/preflight tests/multimodel tests/hunyuan tests/runtime -q`.
- [ ] Inspect deterministic plan identity, resource representation, portability, validation-before-run, orchestrator/runtime handoff, and resume compatibility.
- [ ] Add red-green regression tests before any semantic fixes.

### Task 6: Prove the complete offline canonical path

**Files:**
- Modify: `tests/integration/test_clouda_backend_e2e.py`
- Create only if needed: `tests/integration/test_remaining_feature_e2e.py`
- Modify only when a failing test proves a gap: canonical implementation files named by the failure.

**Interfaces:**
- Consumes: Factory/canonical data, Quality, Results, Lab selection, Training Data Loader, Planner, Preflight, adapter registry, Hunyuan mock bridge, runtime, and checkpoint envelope.
- Produces: portable artifacts and an exact interruption/resume completion trace with no protected-sample leakage.

- [ ] Run all imported feature E2E and integration tests to identify existing coverage of the requested full chain.
- [ ] If no test crosses every required boundary, write one synthetic CPU/offline test that fails at the first missing boundary.
- [ ] Verify interruption state, checkpoint identity/integrity, exact resumed sequence, final completion, lineage portability, protected exclusion, and pure JSON CLI output.
- [ ] Implement only integration hooks proven necessary by failing tests and rerun all affected suites after every red-green cycle.

### Task 7: Review, quality gates, package audit, and publication

**Files:**
- Modify only files required by Critical or Important review findings.
- Inspect: built `dist/*.whl` and `dist/*.tar.gz` contents.

**Interfaces:**
- Consumes: the full diff from starting `origin/main` through all integration commits.
- Produces: reviewed, buildable, clean canonical `main` synchronized 0/0 with `origin/main`.

- [ ] Review for duplicate architecture, stale assumptions, identity conflicts, holdout leakage, nondeterminism, resume/checkpoint fragmentation, broken CLI composition, path/secret leakage, unsafe config execution, packaging omissions, and tests that bypass real boundaries.
- [ ] Add a failing regression test and fix every valid Critical or Important finding.
- [ ] Run full `python -m pytest -q`, `python -m ruff check .`, `python -m black --check .`, `python -m mypy`, benchmark release validation, Doctor normal/deep modes, CLI human/JSON smokes, security/redaction tests, and repository hygiene scans.
- [ ] Build wheel and sdist with `python -m build`, inspect archives for every new package, and verify no model/data/secret/oversized artifact entered Git.
- [ ] Fetch `origin`, compare its `main` with the recorded start, and integrate legitimate advancement without rewriting history.
- [ ] Fast-forward local `main` to the verified integration commit, rerun the full suite on that exact tree, push with plain `git push origin main`, fetch, and verify `main == origin/main`, ahead/behind `0 0`, clean status, and unchanged benchmark hash.
- [ ] Remove only `.worktrees/integrate-remaining-features-20260912` and its local integration branch after the push and synchronization checks succeed.
