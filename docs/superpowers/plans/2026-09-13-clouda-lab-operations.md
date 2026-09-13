# Clouda Lab Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Clouda Lab into a persistent, safe operations control center over every currently executable canonical Clouda OCR workflow.

**Architecture:** Add focused task, acquisition, model-catalog, benchmark-plan, and storage services behind the existing loopback/action-token boundary. Every operation resolves canonical IDs to server-managed roots and delegates to repository domain modules; unsupported model download, dependency installation, weighted dataset mixing, cooperative training stop, and published-model benchmark execution remain explicit blocked capabilities.

**Tech Stack:** Python 3.11, FastAPI, dataclasses, canonical Clouda data/training/results services, plain JavaScript/CSS, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-clouda-lab-operations-design.md`

## Global Constraints

- Bind to `127.0.0.1` and require loopback clients.
- Preserve action tokens and forbid extra request fields.
- Never accept arbitrary URLs, package strings, shell commands, or filesystem paths.
- Automatic and background downloads remain blocked.
- Only an explicit, confirmed canonical dataset sample plan may use the network.
- No large datasets/models, remote benchmark assets, real training, or large inference during validation.
- Preserve protected-holdout enforcement, deterministic identities, lineage, checkpoint integrity, Results Store semantics, and redaction.

---

### Task 1: Persistent operation task service

**Files:**
- Create: `clouda_lab/dashboard/tasks.py`
- Test: `tests/dashboard/test_tasks.py`

**Interfaces:**
- Produces: `TaskStatus`, `TaskContext`, and `OperationTaskService.enqueue(kind, target_id, worker)`, `list_tasks()`, `get_task()`, `cancel()`.

- [ ] Write tests proving deterministic JSON shape, persistence across service instances, cooperative cancellation, rejection of unknown kinds, sanitized errors, and restart recovery.
- [ ] Run `pytest tests/dashboard/test_tasks.py -q` and confirm failures because the module is absent.
- [ ] Implement atomic per-task JSON records, bounded threads, closed operation kinds, progress callbacks, and restart recovery.
- [ ] Run the task tests and dashboard suite; confirm all pass.
- [ ] Commit with `feat(lab): add persistent operation tasks`.

### Task 2: Canonical downloader progress and cancellation

**Files:**
- Modify: `clouda_data/datasets/downloader.py`
- Test: `tests/datasets/test_downloader.py`

**Interfaces:**
- Produces backward-compatible optional `progress_callback` and `cancellation_check` parameters on `download_http()` and `download_dataset_sample()`.

- [ ] Add tests using a local fake response to prove progress totals, cancellation preserves `.part`, resumed ranges, checksums, and unchanged legacy calls.
- [ ] Run the focused downloader tests and confirm the new assertions fail.
- [ ] Add `DownloadCancelled`, callback propagation, byte counters, cancellation checks between chunks, and preservation of resumable partials.
- [ ] Run downloader and registry tests; confirm all pass.
- [ ] Commit with `feat(data): expose safe download progress`.

### Task 3: Dataset lifecycle and explicit download plans

**Files:**
- Create: `clouda_lab/dashboard/datasets.py`
- Modify: `clouda_lab/dashboard/settings.py`
- Test: `tests/dashboard/test_dataset_operations.py`

**Interfaces:**
- Consumes: `OperationTaskService`, canonical registry/license/downloader/ingestion functions.
- Produces: `DatasetOperationsService.list_sources()`, `source_detail()`, `create_download_plan()`, `start_download()`, `verify_download()`, `list_imports()`, `validate_import()`, `register_import()`, and managed removal preview/execute.

- [ ] Add tests for all thirteen sources, separate license/purpose decisions, download authorization, token expiry/single-use, disk state, root-confined imports, canonical ingestion delegation, recoverable removal, and path/URL rejection.
- [ ] Run the tests and confirm failure due to the missing service.
- [ ] Add managed dataset/import/trash roots to `LabSettings` and implement the service using canonical functions only.
- [ ] Run dataset operations plus canonical dataset/ingestion tests.
- [ ] Commit with `feat(lab): add operational dataset lifecycle`.

### Task 4: Complete model catalog and managed asset configuration

**Files:**
- Create: `clouda_lab/dashboard/models.py`
- Test: `tests/dashboard/test_model_catalog.py`

**Interfaces:**
- Produces: `ModelCatalogService.list_models()`, `get_model()`, `configure_assets()`, `verify_assets()`, `removal_plan()`, and `remove_assets()`.

- [ ] Add tests proving all eight benchmark models appear, Hunyuan/Qwen training adapters merge correctly, benchmark-only models remain training-unsupported, model download is blocked without approved manifests, dependency commands come only from fixed optional groups, and asset IDs cannot escape `data/models`.
- [ ] Run the tests and confirm the module is absent.
- [ ] Parse benchmark CSV/results and adapter descriptors, persist managed asset selections, delegate verification to canonical preflight checks, and implement recoverable managed removal.
- [ ] Run model catalog, adapter, preflight, and security tests.
- [ ] Commit with `feat(lab): add canonical model catalog`.

### Task 5: Storage and system/network policy services

**Files:**
- Create: `clouda_lab/dashboard/storage.py`
- Modify: `clouda_lab/dashboard/observability.py`
- Test: `tests/dashboard/test_storage_and_policy.py`

**Interfaces:**
- Produces: `StorageService.status()` and expanded overview/offline policy summaries.

- [ ] Add tests for category byte totals, free-space state, symlink refusal, relative browser labels, pending task/download counts, and explicit-vs-automatic network policy.
- [ ] Run tests and confirm missing behavior.
- [ ] Implement bounded root traversal and policy aggregation.
- [ ] Run focused tests and Doctor tests.
- [ ] Commit with `feat(lab): add storage and network policy state`.

### Task 6: Planner, preflight, and guarded training execution

**Files:**
- Modify: `clouda_lab/dashboard/training.py`
- Modify: `clouda_lab/training_orchestrator.py`
- Test: `tests/dashboard/test_training_operations.py`
- Test: `tests/lab/test_training_orchestrator.py`

**Interfaces:**
- Consumes: managed model configuration and `OperationTaskService`.
- Produces: plan capability/remediation records, `start_training(plan_id)`, and `resume_training(run_id)` tasks.

- [ ] Add tests that missing assets/GPU/preflight block execution, browser model paths remain impossible, a fully passing injected preflight delegates exactly once to canonical `run_experiment`, resume delegates to canonical integrity checks, and stop stays disabled with a truthful reason.
- [ ] Run tests and confirm new behavior fails.
- [ ] Resolve server-owned model assets into plans, persist preflight reports, add a real orchestrator method that requires non-dry-run canonical configs, and enqueue guarded execution/resume.
- [ ] Run training/planner/preflight/runtime/checkpoint suites.
- [ ] Commit with `feat(lab): gate canonical training operations`.

### Task 7: Benchmark catalog, plans, and comparison

**Files:**
- Create: `clouda_lab/dashboard/benchmarks.py`
- Test: `tests/dashboard/test_benchmark_workspace.py`

**Interfaces:**
- Produces: complete benchmark inventory, deterministic `create_plan()`, eligibility, sorted actual results, and pairwise comparison.

- [ ] Add tests for the eight-model inventory, protected evaluation datasets, deterministic identities, incomplete-run ranking exclusion, actual metric sorting, and blocked execution without a registered canonical runner.
- [ ] Run tests and confirm missing service.
- [ ] Implement metadata-backed planning/comparison without inference or fabricated records.
- [ ] Run benchmark, evaluation, and Results Store tests.
- [ ] Commit with `feat(lab): add benchmark workspace services`.

### Task 8: Domain-specific API routes

**Files:**
- Modify: `clouda_lab/dashboard/app.py`
- Test: `tests/dashboard/test_operations_api.py`

**Interfaces:**
- Consumes services from Tasks 1-7.
- Produces strict routes for tasks/downloads, source detail/download/import/verify/removal, model detail/configure/verify/removal, storage, training start/resume, and benchmark plans/comparison.

- [ ] Add API tests proving loopback and action-token enforcement, two-phase confirmation, strict bodies, canonical-ID validation, no URL/path/package/command fields, safe errors, and real task records.
- [ ] Run tests and confirm 404/422 failures.
- [ ] Wire services into app state and add the closed routes.
- [ ] Run API and security suites.
- [ ] Commit with `feat(lab): expose controlled operations API`.

### Task 9: Operational dashboard UI

**Files:**
- Modify: `clouda_lab/dashboard/static/index.html`
- Modify: `clouda_lab/dashboard/static/app.js`
- Modify: `clouda_lab/dashboard/static/styles.css`
- Test: `tests/dashboard/test_ui_contract.py`
- Test: `tests/dashboard/test_network_safety.py`

**Interfaces:**
- Consumes the operations API; produces pages for Downloads, Tasks, Model Catalog/detail, Benchmark Workspace, Storage, and state-aware action bars on existing pages.

- [ ] Add static/UI tests for all pages, confirmation flow, polling, disabled reasons, explicit API error states, no fallback data, and no network-starting request during navigation.
- [ ] Run tests and confirm required UI/actions are absent.
- [ ] Implement accessible operational views and capability-driven buttons without browser-only success state.
- [ ] Run JavaScript syntax and dashboard tests.
- [ ] Commit with `feat(lab): build operational control center UI`.

### Task 10: Documentation, security audit, and complete validation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-13-clouda-lab-operations-design.md` only if implementation evidence requires a correction.

**Interfaces:**
- Produces operator documentation and final verified branch.

- [ ] Document startup, managed roots, explicit confirmation, supported dataset operations, blocked model/benchmark capabilities, dependency commands, and recovery semantics.
- [ ] Run focused dashboard, dataset, quality, adapter, planner/preflight, runtime/checkpoint, results, benchmark, and Doctor suites.
- [ ] Run the complete repository suite, Ruff, Black check, MyPy, JavaScript syntax, and `git diff --check`.
- [ ] Build the wheel with `python -m build --wheel --no-isolation --skip-dependency-check` and inspect packaged dashboard assets.
- [ ] Start on `127.0.0.1`, visit every page with browser automation, check console/page errors, and confirm navigation performs no download/task mutation.
- [ ] Review the final diff for secrets, private paths, unsafe filesystem access, arbitrary URLs/commands/packages, accidental network calls, fake state, and duplicated domain logic.
- [ ] Commit final documentation/fixes and report branch SHA and clean status without merging automatically.
