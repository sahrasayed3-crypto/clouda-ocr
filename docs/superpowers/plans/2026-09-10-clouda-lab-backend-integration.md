# Clouda Lab Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and push one validated canonical backend stack containing the Results Store and reconciled Clouda Lab services.

**Architecture:** Preserve both feature histories by ordered merges onto verified `origin/main`. Make `clouda_data.results` the persistence/query boundary, adapt `clouda_lab` services to it, and keep training behind the existing dry-run experiment framework.

**Tech Stack:** Python 3.11, dataclasses, JSON/JSONL/CSV, pytest, Ruff, Black, MyPy, setuptools.

**Spec:** `docs/superpowers/specs/2026-09-10-clouda-lab-backend-integration-design.md`

## Global Constraints

- Results Store merges before Lab Backend.
- No web UI, real training, GPU use, network access, model downloads, or dataset downloads.
- Holdout checks fail closed and tests use synthetic protected fixtures only.
- Persistent identities use deterministic cryptographic hashes and portable logical fields.
- No force push and no unrelated worktree changes.

---

### Task 1: Integrate and validate Results Store

**Files:**
- Merge: `origin/feature/lab-results-store`
- Review/modify: `clouda_data/results/*.py`, `clouda_data/pipeline/cli.py`
- Test: `tests/results/*.py`, existing Data Factory/benchmark/contract tests

**Interfaces:**
- Consumes: current Data Factory manifests, evaluation functions, checksum and normalization contracts.
- Produces: `ResultsStore`, `ResultsService`, canonical result records, ingestion adapters, integrity and export APIs.

- [ ] Merge the verified Results Store head with a non-fast-forward merge and inspect every conflict against current main.
- [ ] Run `python -m pytest tests/results -q` and record the initial result.
- [ ] For each compatibility defect, write a focused failing test naming the broken current-main contract and verify the expected failure.
- [ ] Apply the smallest Results Store/CLI fix, rerun the focused test, then rerun Results Store plus affected Data Factory/benchmark/contract tests.
- [ ] Verify portable paths, integrity checks, benchmark newline handling, holdout derivation, metric imports, and package resources.
- [ ] Commit only reconciliation changes after fresh focused verification.

### Task 2: Integrate Lab Backend and audit overlap

**Files:**
- Merge: `origin/feature/clouda-lab-backend-foundation`
- Review/modify: `clouda_lab/*.py`, `pyproject.toml`, `.gitattributes`
- Test: `tests/lab/*.py`

**Interfaces:**
- Consumes: Task 1 Results Store contracts and current factory/training APIs.
- Produces: analysis, comparison, selection, recommendation, distortion, evaluation, and training-facade services.

- [ ] Merge the verified Lab Backend head after Task 1 and resolve conflicts without replacing newer main or Results Store contracts blindly.
- [ ] Run `python -m pytest tests/lab -q` and record the initial result.
- [ ] Inventory duplicate persistence, identity, metrics, protection, hashing, provenance, and selection-history responsibilities.
- [ ] Preserve analysis DTOs but remove or redirect any competing persistence/query source of truth.
- [ ] Verify package discovery, script entry points, optional imports, and grouped CLI behavior.
- [ ] Commit the ordered merge and any mechanical packaging reconciliation separately.

### Task 3: Add Results Store to Lab service adapters

**Files:**
- Create or modify: `clouda_lab/results_service.py`
- Modify: `clouda_lab/evaluation_service.py`, `clouda_lab/failure_analysis.py`, `clouda_lab/dataset_selection.py`, `clouda_lab/active_learning.py`
- Test: `tests/lab/test_results_store_integration.py`

**Interfaces:**
- Consumes: `ResultsService` page, ground-truth, prediction, model, run, and metric queries.
- Produces: canonical `OCRSample` conversion, stored-run batch evaluation/comparison, and selection/recommendation rows retaining Results Store identity and eligibility.

- [ ] Write a failing synthetic test proving a stored page plus GT and prediction can be analyzed without direct file parsing.
- [ ] Implement the minimal adapter using public Results Store queries and verify the test passes.
- [ ] Write failing tests for run/model comparison and batch analysis over two stored fake runs.
- [ ] Implement comparison/batch methods with explicit missing-data and conflicting-identity errors, then verify.
- [ ] Write failing tests proving selection and recommendations retain dataset version, split, source identity, and training eligibility.
- [ ] Implement only the required metadata mapping, refactor after green, and rerun Results Store plus Lab suites.

### Task 4: Reconcile metrics, holdout, and lineage

**Files:**
- Modify as required: `clouda_data/results/models.py`, `clouda_data/results/ingest.py`, `clouda_lab/error_analysis.py`, `clouda_lab/holdout_guard.py`, `clouda_lab/training_orchestrator.py`
- Test: `tests/lab/test_results_store_integration.py`, `tests/lab/test_selection_holdout.py`, `tests/training/*`

**Interfaces:**
- Consumes: canonical CER/WER/normalization, `ProtectionInfo`, derived manifests, `TrainingLineage`.
- Produces: consistent metrics, defense-in-depth holdout rejection, and stored inference/training lineage.

- [ ] Add table-driven failing tests for holdout casing/whitespace, aliases, booleans, nested markers, protected roles, mixed rows, malformed types, and absent/ambiguous role metadata.
- [ ] Tighten shared boundary conversion so every case fails closed; verify red then green without weakening existing guards.
- [ ] Add failing golden cases showing Results Store and Lab CER/WER/N-CER agree for literal Arabic fixtures.
- [ ] Reuse canonical metric implementations and make computed-versus-stored boundaries explicit; verify focused suites.
- [ ] Add failing tests for model revision, run id, training run/checkpoint, manifest/source/output hashes, selection seed, and derived lineage preservation.
- [ ] Implement the minimal lineage bridge and verify training remains dry-run-only.

### Task 5: Complete offline backend E2E and portability checks

**Files:**
- Create: `tests/integration/test_clouda_backend_e2e.py`
- Modify as required: service adapters and documentation
- Test: new E2E plus benchmark/factory tests

**Interfaces:**
- Consumes: Data Factory, Results Store, Lab services, and Training Orchestrator.
- Produces: one synthetic proof of the complete canonical backend flow.

- [ ] Write the E2E with a tiny Arabic source, two fake models/runs, deterministic selection, derived manifest, dry-run experiment, and post-run linkage; verify it fails at the first missing integration boundary.
- [ ] Implement each missing boundary through red/green cycles until the E2E passes without network/GPU/training.
- [ ] Verify benchmark manifest SHA against release metadata and inspect working-tree attributes for byte stability.
- [ ] Run render-dependent tests and classify code defects separately from missing optional/native rendering dependencies.
- [ ] Verify all imported modules resolve from this integration worktree and no machine path affects stored identity.

### Task 6: Quality gates and independent review

**Files:**
- Review: complete `origin/main..HEAD` diff and repository metadata
- Modify: only files needed to fix verified findings

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: merge-ready reviewed integration with recorded evidence.

- [ ] Run all requested focused suites, then full `python -m pytest`.
- [ ] Run Ruff, Black check, configured MyPy, `git diff --check`, and `python -m build` because packaging changes are expected.
- [ ] Run existing repository security tests plus secret and large-file scans without printing secret contents.
- [ ] Request a read-only independent review of the full integration diff for leakage, corruption, traversal, data loss, duplication, nondeterminism, overwrite, identity, import, and packaging risks.
- [ ] Reproduce every valid Critical/Important finding with a failing test, fix it, and rerun the affected and full gates.
- [ ] Commit the final reviewed state only after fresh verification.

### Task 7: Merge, push, and verify canonical main

**Files:**
- Git refs only: `integration/clouda-lab-backend`, `main`, `origin/main`

**Interfaces:**
- Consumes: verified integration branch.
- Produces: clean canonical `main` equal to `origin/main`.

- [ ] Fetch and ensure remote main has not advanced; if it has, integrate the new head and repeat affected gates.
- [ ] Merge or fast-forward the validated integration branch into local main without rewriting public history.
- [ ] Run the full tests on the exact main tree to be pushed.
- [ ] Push normally to `origin/main`, fetch again, and compare exact SHAs.
- [ ] Verify both canonical and integration worktrees contain no uncommitted files; preserve user-owned feature worktrees.
- [ ] Produce the requested 35-point final report with exact commands, counts, SHAs, commits, limitations, and deferred hardware validation.

