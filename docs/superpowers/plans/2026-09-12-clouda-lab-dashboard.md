# Clouda Lab Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete standalone, loopback-only Clouda Lab dashboard that exposes real canonical repository state and safe local operations under `/lab` and `/api/lab`.

**Architecture:** A dedicated FastAPI app factory owns transport concerns while focused dashboard services adapt canonical Clouda modules into redacted, stable view models. A packaged vanilla-JavaScript SPA consumes those APIs without external assets or duplicated business logic.

**Tech Stack:** Python 3.11, FastAPI/Starlette/Uvicorn, dataclasses/Pydantic request models, repository canonical services, HTML/CSS/vanilla JavaScript, pytest/FastAPI TestClient.

**Spec:** `docs/superpowers/specs/2026-09-12-clouda-lab-dashboard-design.md`

## Global Constraints

- Bind to `127.0.0.1` by default and reject non-loopback requests in local mode.
- Serve the dashboard under `/lab`; keep it isolated from `pdfword.worker_api`.
- Never download model weights, datasets, checkpoints, GPU runtimes, dependencies, or benchmark assets.
- Never make external provider calls or require network access for navigation.
- Never execute real training or large CPU inference.
- Never expose arbitrary commands, unrestricted paths, environment dumps, private absolute paths, or secrets.
- Reuse canonical protection, identity, planner, preflight, adapter, run, checkpoint, Results Store, benchmark, Doctor, and redaction semantics.
- Optional dependencies, model assets, and GPU absence are capability states, not startup errors.
- Use only repository fixtures and temporary directories in tests.

---

## File Structure

- `clouda_lab/dashboard/__init__.py`: public dashboard app/settings exports.
- `clouda_lab/dashboard/settings.py`: immutable server-owned roots and safe defaults.
- `clouda_lab/dashboard/security.py`: loopback guard, identifier/path containment, response redaction, action-token validation.
- `clouda_lab/dashboard/catalog.py`: canonical dataset discovery/detail/preview and quality delegation.
- `clouda_lab/dashboard/training.py`: adapter capabilities, canonical plan persistence/generation, preflight, runs, checkpoints, resume validation.
- `clouda_lab/dashboard/observability.py`: Results Store, benchmark manifests, Doctor snapshots, canonical hardware projection, overview aggregation.
- `clouda_lab/dashboard/app.py`: FastAPI app factory and domain routes.
- `clouda_lab/dashboard/static/index.html`: accessible dashboard shell and page containers.
- `clouda_lab/dashboard/static/styles.css`: dense responsive engineering-console design.
- `clouda_lab/dashboard/static/app.js`: routing, fetching, rendering, forms, status/error/empty states.
- `clouda_lab/cli.py`: add `serve` without changing existing commands.
- `pyproject.toml`: package static dashboard assets.
- `tests/dashboard/test_security_and_app.py`: binding/guard/CSP/routes/redaction/offline tests.
- `tests/dashboard/test_catalog.py`: temporary canonical fixtures plus dataset safety, preview, and quality delegation tests.
- `tests/dashboard/test_training.py`: adapters, plans, preflight, runs, checkpoints tests.
- `tests/dashboard/test_observability.py`: results, benchmarks, Doctor, hardware, overview tests.
- `tests/dashboard/test_ui_contract.py`: packaged SPA/navigation/disabled-action/empty/error contract tests.
- `README.md`: local launch instructions.

### Task 1: Secure standalone app skeleton

**Interfaces:**

- Produces `LabSettings.from_repo(repo_root: Path) -> LabSettings`.
- Produces `create_app(settings: LabSettings | None = None) -> FastAPI`.
- Produces `require_loopback(request: Request) -> None`, `safe_identifier(value: str) -> str`, `safe_relative_label(path: Path, roots: tuple[Path, ...]) -> str | None`, and `sanitize_payload(value: Any) -> Any`.

- [x] Write tests asserting `/lab` returns packaged HTML, `/api/lab/offline` is JSON, a non-loopback client receives 403, CSP forbids remote resources, traversal identifiers fail, and secret-shaped values are redacted.
- [x] Run `python -m pytest tests/dashboard/test_security_and_app.py -q` and confirm the imports/routes fail before implementation.
- [x] Implement immutable settings with explicit repository/run/results/plan/quality roots, bounded preview/scan limits, default loopback host, and local access mode.
- [x] Implement canonical redaction plus path-label containment and loopback/action-token dependencies.
- [x] Implement the FastAPI app factory, static asset routes, `/lab` SPA fallback, sanitized exception boundary, and `/api/lab/offline`.
- [x] Re-run the focused test and include the secure service in the final feature commit.

### Task 2: Canonical dataset catalog and safety views

**Interfaces:**

- Produces `DatasetCatalog.list_datasets() -> list[dict[str, Any]]`.
- Produces `DatasetCatalog.get_dataset(dataset_id: str) -> dict[str, Any]`.
- Produces `DatasetCatalog.preview(dataset_id: str, limit: int) -> dict[str, Any]`.
- Produces `DatasetCatalog.quality_summary(dataset_id: str) -> dict[str, Any]`.
- Produces `DatasetCatalog.run_quality(dataset_id: str, *, max_samples: int, duplicates_only: bool) -> dict[str, Any]` and `derive(dataset_id: str, output_label: str) -> dict[str, Any]`.

- [x] Build canonical manifest fixtures with safe and protected rows; write failing tests for identity/hash, lineage, evaluation-only/training flags, preview limits, loader compatibility, hidden absolute paths, and unknown IDs.
- [x] Run `python -m pytest tests/dashboard/test_catalog.py -q` and confirm failures.
- [x] Discover local datasets only from configured training configs/manifests, canonical dataset registry, and Results Store records; deduplicate by canonical identity and never recurse arbitrary roots.
- [x] Parse manifests through canonical manifest readers and training-data validation; expose header/row/page counts, protection, lineage, hash, timestamps, and compatibility.
- [x] Delegate explicit quality and duplicate analysis to `run_quality_gate`, build redacted canonical reports, and delegate derived-manifest creation to `write_clean_manifest` under the configured quality output root.
- [x] Add dataset/source/detail/preview/quality read and action routes using dataset IDs only.
- [x] Re-run the dataset suite and include the catalog in the final feature commit.

### Task 3: Adapter, plan, and preflight services

**Interfaces:**

- Produces `TrainingService.list_models() -> list[dict[str, Any]]`.
- Produces `TrainingService.planner_options() -> dict[str, Any]`.
- Produces `TrainingService.create_plan(payload: Mapping[str, Any]) -> dict[str, Any]`.
- Produces `TrainingService.list_plans()` and `get_plan(plan_id: str)`.
- Produces `TrainingService.run_preflight(plan_id: str, *, write_probe: bool = False) -> dict[str, Any]`.

- [x] Write failing tests that register canonical Hunyuan/Qwen descriptors, assert separate code/dependency/assets/GPU states, generate identical plan IDs for identical inputs, reject unregistered adapters/unknown datasets, persist inspectable canonical configs, and preserve preflight `PASS/WARN/FAIL/SKIP` semantics.
- [x] Run `python -m pytest tests/dashboard/test_training.py -q` and confirm failures.
- [x] Lazily import adapter registration modules, discover descriptors from `ModelAdapterRegistry`, and derive asset/dependency availability without instantiating or downloading models.
- [x] Build `ExperimentConfig` from only supported planner fields, force `runtime.offline=true`, prevent real execution, call `build_experiment_plan`, and atomically persist canonical config/plan records under the server-owned plan root.
- [x] Load plans by safe deterministic ID and call canonical `run_preflight`; redact paths and preserve logical validation separately from GPU capability.
- [x] Add model/planner option/plan/preflight routes and re-run tests.
- [x] Include adapter, planner, and preflight integration in the final feature commit.

### Task 4: Runs, checkpoints, Results Store, benchmarks, Doctor, and hardware

**Interfaces:**

- Extends `TrainingService` with `list_runs`, `get_run`, `list_checkpoints`, and `resume_check`.
- Produces `ObservabilityService.results(filters)`, `benchmarks(filters)`, `run_doctor(deep)`, `latest_doctor()`, `hardware()`, and `overview()`.

- [x] Add failing run/checkpoint tests using canonical temporary experiment artifacts, including failed status, sanitized failure metadata, integrity state, and resume compatibility.
- [x] Add failing Results Store, immutable local benchmark, Doctor, no-GPU hardware, and state-derived overview tests.
- [x] Run `python -m pytest tests/dashboard/test_training.py tests/dashboard/test_observability.py -q` and confirm failures.
- [x] Delegate run/checkpoint reads to `TrainingOrchestrator`, `ExperimentRegistry`, and canonical checkpoint functions; perform resume checks through canonical validation without executing resume.
- [x] Read Results Store through `ResultsService(read_only=True)` and local benchmark files through configured fixed paths; never recompute benchmarks on reads.
- [x] Execute `collect_report` only from explicit Doctor POST, cache the sanitized latest snapshot, and derive GPU capability from the canonical Doctor GPU check.
- [x] Aggregate overview counts strictly from the services above, returning unavailable values as null/status rather than invented numbers.
- [x] Add the run/results/diagnostic routes, re-run tests, and include them in the final feature commit.

### Task 5: Complete dashboard client

**Interfaces:**

- Consumes only `/api/lab/*` contracts from Tasks 1-4.
- Produces hash/history navigation for Overview, Datasets, Dataset Detail, Quality, Planner, Preflight, Models, Runs, Run Detail, Results, Benchmarks, Doctor, Hardware, and Offline Status.

- [x] Write UI contract tests asserting all navigation labels/routes, accessible landmarks, status vocabulary, dataset/run lineage elements, search controls, loading/empty/error regions, no-GPU copy, and disabled real-training controls.
- [x] Run `python -m pytest tests/dashboard/test_ui_contract.py -q` and confirm failures.
- [x] Build semantic HTML shell with local-only assets and no inline remote-capable dependencies.
- [x] Implement responsive CSS for fixed sidebar, compact cards, readable tables, badges, callouts, form grids, lineage flows, details panels, and mobile layout.
- [x] Implement client routing, cancellable JSON fetching, safe DOM text rendering, filters, page forms, explicit action invocation, action token use, and stable loading/empty/error states.
- [x] Render every required page using actual API fields; disable training/resume/unavailable operations with explicit reasons and never simulate success.
- [x] Re-run UI/app tests and include the UI in the final feature commit.

### Task 6: CLI, packaging, integration, and regression verification

**Interfaces:**

- Produces `clouda-lab serve [--host 127.0.0.1] [--port 8000] [--repo-root PATH]`.
- Packages `dashboard/static/*` in the existing wheel.

- [x] Write failing CLI tests proving the default host is `127.0.0.1`, public bindings are rejected, and the serving path keeps model/runtime imports lazy.
- [x] Add the CLI subcommand, Uvicorn lazy import, static package data, and README launch/security notes.
- [x] Run all dashboard tests and relevant canonical suites; the wider focused run passed 975 tests with 4 skips.
- [x] Run repository-wide Ruff, Black, MyPy, JavaScript syntax, and an offline package build; use a locally installed builder without dependency downloads.
- [x] Run the complete repository suite without downloads and preserve existing skip semantics.
- [x] Smoke-start on `127.0.0.1`, exercise the Lab/API surface, verify browser rendering/navigation, and terminate cleanly.
- [x] Inspect the diff for duplicated business rules, raw paths/secrets, shell/file endpoints, accidental network/model loads, fake actions, dead code, and misleading statuses; correct all findings.
- [x] Run final focused/static/full verification after corrections.
- [x] Update this checklist, commit all implementation/docs/tests, and confirm `git status --short --branch` is clean.
