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
- `tests/dashboard/conftest.py`: temporary canonical fixtures and app factory fixture.
- `tests/dashboard/test_security_and_app.py`: binding/guard/CSP/routes/redaction/offline tests.
- `tests/dashboard/test_catalog.py`: dataset safety, preview, quality delegation tests.
- `tests/dashboard/test_training.py`: adapters, plans, preflight, runs, checkpoints tests.
- `tests/dashboard/test_observability.py`: results, benchmarks, Doctor, hardware, overview tests.
- `tests/dashboard/test_ui_contract.py`: packaged SPA/navigation/disabled-action/empty/error contract tests.
- `README.md`: local launch instructions.

### Task 1: Secure standalone app skeleton

**Interfaces:**

- Produces `LabSettings.from_repo(repo_root: Path) -> LabSettings`.
- Produces `create_app(settings: LabSettings | None = None) -> FastAPI`.
- Produces `require_loopback(request: Request) -> None`, `safe_identifier(value: str) -> str`, `safe_relative_label(path: Path, roots: tuple[Path, ...]) -> str | None`, and `sanitize_payload(value: Any) -> Any`.

- [ ] Write tests asserting `/lab` returns packaged HTML, `/api/lab/offline` is JSON, a non-loopback client receives 403, CSP forbids remote resources, traversal identifiers fail, and secret-shaped values are redacted.
- [ ] Run `python -m pytest tests/dashboard/test_security_and_app.py -q` and confirm the imports/routes fail before implementation.
- [ ] Implement immutable settings with explicit repository/run/results/plan/quality roots, bounded preview/scan limits, default loopback host, and local access mode.
- [ ] Implement canonical redaction plus path-label containment and loopback/action-token dependencies.
- [ ] Implement the FastAPI app factory, static asset routes, `/lab` SPA fallback, sanitized exception boundary, and `/api/lab/offline`.
- [ ] Re-run the focused test and commit `feat(lab): add secure standalone dashboard service`.

### Task 2: Canonical dataset catalog and safety views

**Interfaces:**

- Produces `DatasetCatalog.list_datasets() -> list[dict[str, Any]]`.
- Produces `DatasetCatalog.get_dataset(dataset_id: str) -> dict[str, Any]`.
- Produces `DatasetCatalog.preview(dataset_id: str, limit: int) -> dict[str, Any]`.
- Produces `DatasetCatalog.quality_summary(dataset_id: str) -> dict[str, Any]`.
- Produces `DatasetCatalog.run_quality(dataset_id: str, *, max_samples: int, duplicates_only: bool) -> dict[str, Any]` and `derive(dataset_id: str, output_label: str) -> dict[str, Any]`.

- [ ] Build canonical manifest fixtures with safe and protected rows; write failing tests for identity/hash, lineage, evaluation-only/training flags, preview limits, loader compatibility, hidden absolute paths, and unknown IDs.
- [ ] Run `python -m pytest tests/dashboard/test_catalog.py -q` and confirm failures.
- [ ] Discover local datasets only from configured training configs/manifests, canonical dataset registry, and Results Store records; deduplicate by canonical ID/version and never recurse arbitrary roots.
- [ ] Parse manifests through `clouda_data.quality.manifest_adapter` and canonical training-data validation; expose header/row/page counts, protection, lineage, hash, timestamps, and compatibility.
- [ ] Delegate explicit quality and duplicate analysis to `run_quality_gate`, build redacted canonical reports, and delegate derived-manifest creation to `write_clean_manifest` under the configured quality output root.
- [ ] Add dataset/detail/preview/quality read and action routes using dataset IDs only.
- [ ] Re-run tests and commit `feat(lab): expose canonical dataset and quality views`.

### Task 3: Adapter, plan, and preflight services

**Interfaces:**

- Produces `TrainingService.list_models() -> list[dict[str, Any]]`.
- Produces `TrainingService.planner_options() -> dict[str, Any]`.
- Produces `TrainingService.create_plan(payload: Mapping[str, Any]) -> dict[str, Any]`.
- Produces `TrainingService.list_plans()` and `get_plan(plan_id: str)`.
- Produces `TrainingService.run_preflight(plan_id: str, *, write_probe: bool = False) -> dict[str, Any]`.

- [ ] Write failing tests that register canonical Hunyuan/Qwen descriptors, assert separate code/dependency/assets/GPU states, generate identical plan IDs for identical inputs, reject unregistered adapters/unknown datasets, persist inspectable canonical configs, and preserve preflight `PASS/WARN/FAIL/SKIP` semantics.
- [ ] Run `python -m pytest tests/dashboard/test_training.py -q` and confirm failures.
- [ ] Lazily import adapter registration modules, discover descriptors from `ModelAdapterRegistry`, and derive asset/dependency availability without instantiating or downloading models.
- [ ] Build `ExperimentConfig` from only supported planner fields, force `runtime.offline=true`, prevent real execution, call `build_experiment_plan`, and atomically persist canonical config/plan records under the server-owned plan root.
- [ ] Load plans by safe deterministic ID and call canonical `run_preflight`; redact paths and preserve logical validation separately from GPU capability.
- [ ] Add model/planner option/plan/preflight routes and re-run tests.
- [ ] Commit `feat(lab): integrate adapters planner and preflight`.

### Task 4: Runs, checkpoints, Results Store, benchmarks, Doctor, and hardware

**Interfaces:**

- Extends `TrainingService` with `list_runs`, `get_run`, `list_checkpoints`, and `resume_check`.
- Produces `ObservabilityService.results(filters)`, `benchmarks(filters)`, `run_doctor(deep)`, `latest_doctor()`, `hardware()`, and `overview()`.

- [ ] Add failing run/checkpoint tests using canonical temporary experiment artifacts, including interrupted/failed statuses, sanitized failure metadata, integrity state, and unsafe resume mismatch.
- [ ] Add failing Results Store, immutable local benchmark, Doctor, no-GPU hardware, and state-derived overview tests.
- [ ] Run `python -m pytest tests/dashboard/test_training.py tests/dashboard/test_observability.py -q` and confirm failures.
- [ ] Delegate run/checkpoint reads to `TrainingOrchestrator`, `ExperimentRegistry`, and canonical checkpoint functions; perform resume checks through canonical validation without executing resume.
- [ ] Read Results Store through `ResultsService(read_only=True)` and local benchmark files through configured fixed paths; never recompute benchmarks on reads.
- [ ] Execute `collect_report` only from explicit Doctor POST, cache the sanitized latest snapshot, and project hardware from its canonical sections without a parallel detector.
- [ ] Aggregate overview counts strictly from the services above, returning unavailable values as null/status rather than invented numbers.
- [ ] Add routes, re-run tests, and commit `feat(lab): add runs results and diagnostic APIs`.

### Task 5: Complete dashboard client

**Interfaces:**

- Consumes only `/api/lab/*` contracts from Tasks 1-4.
- Produces hash/history navigation for Overview, Datasets, Dataset Detail, Quality, Planner, Preflight, Models, Runs, Run Detail, Results, Benchmarks, Doctor, Hardware, and Offline Status.

- [ ] Write UI contract tests asserting all navigation labels/routes, accessible landmarks, status vocabulary, dataset/run lineage elements, search controls, loading/empty/error regions, no-GPU copy, and disabled real-training controls.
- [ ] Run `python -m pytest tests/dashboard/test_ui_contract.py -q` and confirm failures.
- [ ] Build semantic HTML shell with local-only assets and no inline remote-capable dependencies.
- [ ] Implement responsive CSS for fixed sidebar, compact cards, readable tables, badges, callouts, form grids, lineage flows, details panels, and mobile layout.
- [ ] Implement client routing, cancellable JSON fetching, safe DOM text rendering, filters, page forms, explicit action confirmations, action token use, and stable loading/empty/error states.
- [ ] Render every required page using actual API fields; disable training/resume/unavailable operations with the backend-supplied reason and never simulate success.
- [ ] Re-run UI/app tests and commit `feat(lab): build internal control center UI`.

### Task 6: CLI, packaging, integration, and regression verification

**Interfaces:**

- Produces `clouda-lab serve [--host 127.0.0.1] [--port 8000] [--repo-root PATH]`.
- Packages `dashboard/static/*` in the existing wheel.

- [ ] Write failing CLI tests proving the default host is `127.0.0.1`, `0.0.0.0` requires an explicit unsafe override or is rejected, and importing the service triggers no optional model imports/network calls.
- [ ] Add the CLI subcommand, Uvicorn lazy import, static package data, and README launch/security notes.
- [ ] Run all dashboard tests and the relevant canonical suites: `python -m pytest tests/dashboard tests/quality tests/planner tests/preflight tests/multimodel tests/runtime tests/results tests/doctor tests/lab -q`.
- [ ] Run `python -m ruff check .`, `python -m black --check .`, `python -m mypy .`, and `python -m build` with the repository interpreter.
- [ ] Run `python -m pytest -q` if it requires no prohibited downloads; preserve existing skip semantics.
- [ ] Smoke-start on `127.0.0.1`, request `/lab`, `/api/lab/overview`, `/api/lab/hardware`, and `/api/lab/offline`, then terminate cleanly.
- [ ] Inspect `git diff` for duplicated business rules, raw paths/secrets, shell/file endpoints, accidental network/model loads, fake actions, dead code, and misleading status colors; correct all findings.
- [ ] Run final focused/static/full verification after corrections.
- [ ] Update this checklist, commit all implementation/docs/tests, and confirm `git status --short --branch` is clean.

