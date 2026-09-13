# Clouda Lab Operations Control Center Design

## Purpose

Clouda Lab becomes a loopback-only operational control center over the existing
Clouda OCR domain systems. It remains a separate FastAPI application under
`/lab`; it does not join or weaken the authenticated worker API.

The service exposes only named, server-defined operations. A browser cannot
submit a shell command, package name, arbitrary URL, or unrestricted filesystem
path. Missing canonical support is rendered as a blocked capability rather than
simulated with frontend state.

## Repository audit and capability boundary

The implementation reuses these canonical systems:

- `clouda_data.datasets.registry`, `license_gate`, and `downloader` for the
  thirteen known dataset sources, license decisions, explicit sample downloads,
  checksum verification, archive checks, disk limits, safe redirects, and SSRF
  protection.
- `clouda_data.ingestion.workflow` for validation and registration of an
  existing local source manifest.
- `clouda_data.quality` for quality, duplicate analysis, quarantine policy, and
  immutable derived manifests.
- `clouda_training.adapters`, planner, preflight, experiment runtime, registry,
  and checkpoint manager for training operations.
- `clouda_data.results.ResultsService` and the immutable Arabic OCR benchmark
  release for results and benchmark history.
- `clouda_data.doctor` for system and hardware state.

The published benchmark tracks eight models, while only HunyuanOCR and Qwen3-VL
have training adapters. The repository has no approved model-download manifest
containing license, expected files, size, and checksums, and no canonical runner
for executing those eight published benchmark models. Therefore model download
and published-model benchmark execution remain blocked with explicit reasons.
The UI must not infer download permission from a Hugging Face repository name.

## Architecture

### Persistent operation tasks

`OperationTaskService` owns a bounded in-process executor and an append-safe JSON
record per operation under `runs/.lab-tasks`. Tasks accept only a closed enum of
operation kinds. Records contain status, phase, progress, sanitized result or
error, timestamps, and cancellation state. Browser reconnects read these files.
On service restart, unfinished records become `FAILED` with
`service_restarted`; a user may explicitly retry a resumable dataset download.

The task states are `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, and `CANCELLED`.
Download phases add `PENDING`, `CHECKING`, `DOWNLOADING`, `VERIFYING`, and
`VERIFIED`. Cancellation is cooperative. The canonical downloader is extended
with optional progress and cancellation callbacks while preserving its existing
API and `.part` resume behavior.

### Explicit confirmation

Every network or destructive action is two-phase. A preview endpoint builds a
server-owned plan containing the canonical target, provider, expected assets,
license state, estimated bytes, managed destination, disk availability, and a
random confirmation token. Execution accepts only the plan ID and exact token.
Plans expire and are single-use.

### Dataset operations

The source catalog combines canonical registry metadata with local download,
verification, registration, quality, and eligibility state. Training,
evaluation, redistribution, and download permissions remain separate.

Dataset downloads are limited to canonical `sample_assets` and the downloader's
existing 2 GiB hard limit. Local import accepts a canonical `import_id`, never a
path; it resolves only to `data/imports/<import_id>/source_manifest.json`, runs
canonical dry-run validation, then registration through the ingestion workflow.
Source data is copied into canonical managed roots.

Downloaded datasets and managed imports can be removed only after a preview and
confirmation, and only from their exact managed root. Removal uses recoverable
trash under `runs/.lab-trash`.

### Model catalog and assets

The model catalog is built from `benchmarks/ocr_arabic/models.csv`, benchmark
results, and registered training-adapter descriptors. Alias normalization joins
HunyuanOCR and Qwen3-VL records without hiding their source identities. Each
model reports benchmark evidence, training support, adapter registration,
dependencies, local asset state, precision/device capabilities, and validation.

Existing assets may be configured only by choosing an `asset_id` resolving to
`data/models/<asset_id>`. Verification delegates to canonical adapter preflight
and checks without instantiating a large model. Dependency remediation is a
display-only CLI command generated from fixed `pyproject.toml` optional groups;
there is no package-install endpoint.

### Plans, preflight, and training

The planner continues to accept canonical dataset and adapter IDs. When a model
has a verified managed asset configuration, the server resolves that asset path
into the experiment configuration; the browser never submits it. Preflight is
the authoritative readiness gate.

Starting training is available only for a stored plan whose most recent
preflight report passes, whose dataset is training-eligible, and whose adapter,
assets, dependencies, and GPU requirements are satisfied. Execution delegates
to the canonical experiment runtime through `TrainingOrchestrator`. Resume uses
the canonical checkpoint integrity and identity checks. Cooperative stop is
reported unavailable until the canonical training loop supports an external
cancellation callback; no fake stop button is enabled.

The canonical planner currently represents one dataset. Multi-dataset mixing is
not added in this change because no canonical weighted-composition contract or
loader exists; the UI reports this limitation and never simulates mixing.

### Benchmarks and results

The benchmark workspace exposes the complete eight-model inventory, immutable
release metadata, eligibility, sorting, and comparison of actual recorded
results. A deterministic benchmark plan may be stored from canonical dataset
and model IDs. Execution remains blocked unless a canonical inference runner is
registered for every selected model; partial published results remain unranked.
No model is instantiated by browsing or planning.

Results Store gains run-type/status/model/dataset/date filters and navigable
links to known dataset, model, training run, and benchmark identities. It never
constructs missing lineage.

### Storage and system policy

Storage reports managed roots, used bytes, free bytes, and category totals for
datasets, models, runs/checkpoints, benchmarks, tasks, and trash. Browser output
uses relative labels only. The System & Network Policy page distinguishes
blocked automatic/background network access from explicit confirmed dataset
sample downloads.

## API resources

New resources are domain-specific:

- `/api/lab/tasks` and `/api/lab/tasks/{id}` for persisted task state and
  cancellation.
- `/api/lab/downloads` for download tasks.
- `/api/lab/dataset-sources/{id}` plus download/verify/import/remove plan and
  execution routes.
- `/api/lab/imports` for managed local import candidates.
- `/api/lab/models/{id}` plus configure/verify/remove routes.
- `/api/lab/storage` for managed storage state.
- `/api/lab/training/plans/{id}/start` and run resume routes, gated by canonical
  preflight.
- `/api/lab/benchmark-plans` for deterministic, non-executing plans until a
  canonical runner is available.

All mutations keep the existing loopback and action-token protections. Request
models forbid extra fields.

## UI

The navigation adds Downloads, Tasks, Model Catalog, and Storage. Existing pages
gain state-aware action bars. Buttons are enabled only when the API supplies an
operation capability and a reason accompanies every disabled action. Confirmation
dialogs render the server plan rather than relying on browser-generated text.

Every page answers what exists, its state, and the next valid action. Technical
exception detail is placed in expandable sections; the primary status remains a
short stable code.

## Safety and validation

Tests must prove explicit confirmation, license gating, safe destinations,
SSRF rejection, disk checks, task persistence/cancellation/restart recovery,
download progress/checksums, import root confinement, complete model inventory,
asset confinement, holdout enforcement, preflight gating, training disabled on
this no-GPU/no-assets host, benchmark honesty, redaction, and zero network
activity while opening every page.

Validation includes focused and subsystem suites, the complete repository suite,
Ruff, Black, MyPy, JavaScript syntax, `git diff --check`, an offline wheel build,
and browser verification. Tests use isolated fixtures; no large model or dataset
is downloaded and no real training is executed.
