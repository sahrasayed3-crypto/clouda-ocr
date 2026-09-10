# Training Loader and Environment Doctor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the scalable Training Data Loader and modernized Environment Doctor into the current canonical Clouda OCR backend without weakening safety, identity, determinism, or offline operation.

**Architecture:** Preserve both feature histories, make the Data Loader a consumer of Lab's canonical derived manifest and a participant in the existing experiment checkpoint lifecycle, then make Environment Doctor diagnose the resulting stack read-only. Validate each boundary with synthetic CPU-only tests before fast-forwarding and pushing `main`.

**Tech Stack:** Python 3.11, dataclasses, JSON/JSONL, pathlib, hashlib, bounded queues, pytest, Ruff, Black, MyPy, `python -m build`.

**Spec:** `docs/superpowers/specs/2026-09-11-training-loader-doctor-integration-design.md`

## Global Constraints

- Integrate Training Data Loader before Environment Doctor.
- Do not download models or datasets, access protected holdout content, run real training/GPU work, build UI, force-push, or delete remote feature branches.
- Reuse `clouda_contracts.protection`, Lab selection/orchestration, and Training Experiment Framework checkpoints; do not create competing sources of truth.
- Persist portable relative paths and cryptographic identities; never use Python `hash()` for durable ordering.
- Default and deep Doctor modes are read-only, secret-safe, bounded, offline, and clean temporary artifacts.

---

### Task 1: Establish baseline and preserve feature history

**Files:**
- Create: `docs/superpowers/specs/2026-09-11-training-loader-doctor-integration-design.md`
- Create: `docs/superpowers/plans/2026-09-11-training-loader-doctor-integration.md`
- Merge: `origin/feature/training-data-loader`

**Interfaces:**
- Consumes: fetched `origin/main` and verified remote feature heads.
- Produces: a clean integration tree containing the loader branch history.

- [ ] Record branch heads, merge bases, worktrees, and the complete green baseline.
- [ ] Commit the approved design and implementation plan.
- [ ] Merge `origin/feature/training-data-loader` with `--no-ff` and resolve only genuine textual conflicts.
- [ ] Run the loader branch's focused tests unchanged to characterize its imported behavior.

### Task 2: Reconcile canonical manifest and protection policy

**Files:**
- Modify: `clouda_data/training_data/input_contract.py`
- Modify: `clouda_data/training_data/models.py`
- Modify: `clouda_data/training_data/sharding.py`
- Test: `tests/data_foundation/unit/test_training_data_manifest_sharding.py`
- Test: `tests/contracts/test_protection_policy.py`

**Interfaces:**
- Consumes: Lab-derived JSONL header/rows and `clouda_contracts.protection.evaluate_training_eligibility(payload)`.
- Produces: `ManifestIdentity`, validated lazy row iteration, and portable `ShardIndex` records.

- [ ] Add failing tests that feed current Lab manifests and synthetic nested, malformed, aliased, mixed-role, and case-varied protection markers to loader validation.
- [ ] Run the new tests and confirm failures arise from the stale branch-local policy/contract.
- [ ] Replace branch-local protection interpretation with the shared fail-closed policy and map current header/row fields without inventing another manifest format.
- [ ] Run manifest, protection, Results, and Lab selection tests to green.
- [ ] Commit the canonical contract reconciliation.

### Task 3: Harden deterministic sharding, streaming, topology, and artifacts

**Files:**
- Modify: `clouda_data/training_data/sharding.py`
- Modify: `clouda_data/training_data/loader.py`
- Modify: `clouda_data/training_data/ordering.py`
- Modify: `clouda_data/training_data/prefetch.py`
- Modify: `clouda_data/training_data/artifacts.py`
- Test: `tests/data_foundation/unit/test_training_data_loader.py`
- Test: `tests/data_foundation/integration/test_training_data_e2e.py`

**Interfaces:**
- Consumes: validated manifest rows, `TrainingDataConfig`, artifact root, epoch/rank/worker identity.
- Produces: deterministic `SampleReference` and batch iterators, bounded prefetch, portable shards, and `ResumeCursor`.

- [ ] Add failing tests for no-loss/no-duplication sharding, stable hashes, lazy iteration, bounded shuffle/prefetch, exception propagation, path traversal, uneven topology, final batches, and epoch reshuffle.
- [ ] Run each focused test and confirm the intended behavioral failure.
- [ ] Implement only the portability, boundedness, lifecycle, and determinism fixes required by those failures.
- [ ] Run loader unit, CLI, and E2E tests to green.
- [ ] Commit runtime delivery hardening.

### Task 4: Join loader cursor to the canonical experiment lifecycle

**Files:**
- Modify: `clouda_data/training_data/checkpoint_bridge.py`
- Modify: `clouda_lab/training_orchestrator.py`
- Modify: `clouda_training/experiments/checkpoints.py`
- Modify: `clouda_training/experiments/trainer.py`
- Test: `tests/lab/test_training_orchestrator.py`
- Test: `tests/training/test_experiment_framework.py`
- Test: `tests/integration/test_clouda_backend_e2e.py`

**Interfaces:**
- Consumes: `SelectionResult`, derived manifest, `StreamingTrainingDataLoader.get_cursor()`, and existing checkpoint metadata.
- Produces: loader construction from a prepared selection and checkpoint metadata containing one validated `data_cursor` payload.

- [ ] Add failing tests for Selection-to-loader orchestration, checkpoint cursor attachment, exact resume continuation, and rejection of changed dataset, manifest, loader config, seed, or topology.
- [ ] Run the tests and confirm missing integration or unsafe resume causes each failure.
- [ ] Add the smallest orchestrator/experiment hooks needed to create the loader and carry its cursor in canonical checkpoint metadata.
- [ ] Run loader, Lab, and Training Framework suites to green.
- [ ] Commit orchestration and checkpoint integration.

### Task 5: Merge and modernize Environment Doctor

**Files:**
- Merge: `origin/feature/environment-doctor`
- Modify: `clouda_data/doctor/environment.py`
- Modify: `clouda_data/doctor/factory.py`
- Modify: `clouda_data/doctor/training.py`
- Create or modify: `clouda_data/doctor/backend.py`
- Modify: `clouda_data/doctor/report.py`
- Modify: `clouda_data/pipeline/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/doctor/test_environment.py`
- Test: `tests/doctor/test_subsystems.py`
- Test: `tests/doctor/test_system_cli.py`

**Interfaces:**
- Consumes: current package metadata and import roots plus Results, Lab, loader, renderer, RAQM, storage, and Training APIs.
- Produces: secret-safe readiness sections and coherent `doctor` human/JSON/deep CLI output.

- [ ] Merge Doctor history with `--no-ff`, resolve CLI conflicts around current factory/results/lab/training-data/training commands, and run original Doctor tests.
- [ ] Add failing tests for Results/Lab/loader presence and absence, mixed import roots, stale editable installs, current extras, unwritable storage, native-RAQM false readiness, and secret-safe exception output.
- [ ] Run the tests and confirm the older Doctor lacks or misclassifies the current subsystems.
- [ ] Add read-only Results Store, Lab Backend, and Training Data Engineering checks; retain separate GPU readiness and native RAQM verification.
- [ ] Reconcile the `doctor` command and package metadata with current CLI conventions.
- [ ] Run Doctor, CLI, package-import, and security tests to green.
- [ ] Commit Doctor modernization.

### Task 6: Deep mode and complete offline backend E2E

**Files:**
- Modify: `clouda_data/doctor/report.py`
- Test: `tests/doctor/test_subsystems.py`
- Modify: `tests/integration/test_clouda_backend_e2e.py`

**Interfaces:**
- Consumes: temporary synthetic Results Store, Lab services, loader/shards, Training Orchestrator, and mock trainer.
- Produces: bounded deep-mode findings plus a verified checkpoint/resume flow without duplicate or missing samples.

- [ ] Add failing tests for deep-mode subsystem failure isolation/cleanup and the full source-to-resume chain.
- [ ] Run them and verify the failures identify missing chain integration.
- [ ] Implement bounded temporary smokes and complete the canonical offline E2E.
- [ ] Run all focused subsystem and cross-system tests to green.
- [ ] Commit deep diagnostics and E2E coverage.

### Task 7: Review, quality gates, and canonical integration

**Files:**
- Modify only files required by Critical or Important review findings.

**Interfaces:**
- Consumes: final integration diff and every requirement in the approved specification.
- Produces: reviewed, reproducible, buildable `main` synchronized with `origin/main`.

- [ ] Review the complete diff for duplicate ownership, holdout leakage, identity/hash defects, path traversal, nondeterminism, resume duplication, false READY states, and secret exposure.
- [ ] For every valid blocker, add a failing regression test, implement the minimal fix, and rerun affected suites.
- [ ] Verify benchmark SHA-256 and release tests, `git diff --check`, repository secret scan, and forbidden large-file scan.
- [ ] Run the full suite, Ruff, Black, MyPy, wheel/sdist build, package import audit, and all CLI help smokes.
- [ ] Fetch again, ensure `origin/main` has not moved incompatibly, fast-forward clean canonical `main`, and rerun the full suite on that exact tree.
- [ ] Push normally, fetch, verify `main == origin/main`, confirm 0/0 ahead/behind and clean status, then remove only the owned temporary integration worktree/branch.
