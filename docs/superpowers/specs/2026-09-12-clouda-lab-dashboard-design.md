# Clouda Lab Dashboard Design

Date: 2026-09-12  
Status: approved architecture, pending written-spec review  
Branch: `feature/clouda-lab-dashboard`

## Objective

Build Clouda Lab as a real, offline-first control plane over the existing
Clouda OCR dataset, quality, training, experiment, results, benchmark, and
diagnostic systems. The dashboard reports canonical repository state and
invokes canonical operations; it does not create parallel registries,
identities, loaders, checkpoint formats, or training orchestration.

The service is a standalone FastAPI application. It binds to `127.0.0.1` by
default, serves the user interface at `/lab`, exposes domain APIs under
`/api/lab`, and remains isolated from `pdfword.worker_api`.

## Verified Baseline

The source of truth is `F:\PROJECT\CLOUDA_UNIFIED_PROJECT`.

- Verified branch before changes: `main`
- Verified HEAD: `1fd5807fe01b8d8df5163a09633eaa36de6433e5`
- Verified tracking state: `main` matched `origin/main`
- Verified working tree before branch creation: clean
- Existing web infrastructure: FastAPI/Starlette/Uvicorn plus a separate
  Streamlit product UI
- Existing Lab: backend domain services and CLI only; no dashboard/web layer

## Architecture

### Process and network boundary

`clouda-lab serve` starts a dedicated Uvicorn process with host
`127.0.0.1` and port `8000` by default. Host and port may be changed only by
explicit CLI options. The default never binds to all interfaces.

The FastAPI app uses an app factory and injected settings/services. A
loopback guard protects both `/lab` and `/api/lab` when the app runs in its
default local mode. The guard is a transport policy, separate from the
domain service layer. A future authenticated host can mount the app with an
explicitly supplied access policy without changing canonical integrations.

The Clouda Lab app is not imported or mounted by `pdfword.worker_api`, and it
does not reuse the public worker API's authentication, queues, provider
routes, or secret-bearing configuration.

### Layers

1. **Canonical domain modules** remain the only source of business rules and
   persisted identities.
2. **Dashboard services** adapt canonical Python objects to small, stable,
   redacted view models. They accept repository-owned roots and canonical
   identifiers, never arbitrary client-supplied filesystem paths.
3. **FastAPI routes** validate requests, call dashboard services, and convert
   domain failures to sanitized API errors.
4. **Static dashboard client** is served from the package at `/lab`. It calls
   only `/api/lab` and contains presentation logic, not domain logic.

The initial client uses repository-packaged HTML, CSS, and vanilla JavaScript.
This avoids a new Node toolchain or runtime dependency, works offline, and can
be exercised with the existing Python test stack. The client is structured by
page/controller modules so a later frontend replacement does not affect the
service layer.

## Canonical Dependency Map

| Dashboard capability | Canonical source | Dashboard responsibility |
| --- | --- | --- |
| Dataset sources and licenses | `clouda_data.datasets.registry` | Redacted source summaries only; never initiate downloads |
| Approved training datasets | `clouda_training.datasets.approved` | Discover configured canonical manifests and expose identity/safety metadata |
| Dataset records and loader compatibility | `clouda_data.training_data` | Bounded preview and compatibility checks through the canonical input contract/loader |
| Holdout/evaluation protection | `clouda_contracts.protection`, `clouda_lab.holdout_guard`, canonical training-data validation | Render safety state and fail closed; no override path |
| Quality and dedup | `clouda_data.quality` | Serialize existing reports and invoke explicit supported operations |
| Derived datasets | `clouda_data.quality.derived`, `clouda_lab.dataset_selection` | Request canonical derivation and report resulting lineage |
| Model adapters | `clouda_training.adapters.registry`, Hunyuan/Qwen registration modules | Register/discover canonical descriptors and probe local capabilities without downloading assets |
| Experiment planning | `clouda_training.planner` | Validate supported inputs, call `build_experiment_plan`/canonical config generation, retain deterministic plan identity |
| Preflight | `clouda_training.preflight` | Call `run_preflight` and preserve its section, severity, blocker, warning, and final-status semantics |
| Runs and experiments | `clouda_training.experiments.registry`, `clouda_training.experiments.runs` | List/load canonical run metadata and redact failures |
| Checkpoints and resume | `clouda_training.experiments.checkpoints`, `clouda_lab.training_orchestrator` | List checkpoints and expose canonical resume validation; never implement resume logic |
| Results Store | `clouda_data.results.service` | Read/filter canonical datasets, models, runs, metrics, summaries, and bundle integrity |
| Lab evaluation/analysis | Existing `clouda_lab` services | Present existing local analysis/results without duplicating algorithms |
| Benchmarks | Canonical Results Store plus immutable local benchmark manifests | Read existing results only; never run/download benchmarks during navigation |
| Doctor | `clouda_data.doctor.collect_report` | Run normal/deep checks only on explicit user action and serialize canonical status |
| Hardware | `clouda_data.doctor` system/environment/hardware sections | Project the canonical Doctor checks into a capability view; do not create a second hardware detector |
| Redaction | `clouda_contracts.security`, `clouda_data.doctor.security` | Apply canonical recursive redaction and a route-level final safety pass |

## Configuration and Discovery

`LabSettings` is constructed at startup from explicit CLI arguments and safe
repository defaults. It contains the repository root, approved dataset
configuration, run root, Results Store root, benchmark manifest locations,
preview limit, and access policy. Browser requests cannot replace these roots.

Discovery uses canonical registries, manifests, indexes, and metadata files.
It does not recursively scan the whole repository on each request. Expensive
or mutable checks are initiated only through explicit POST actions. Small
summary responses may be cached in process with conservative invalidation
based on canonical metadata timestamps.

Offline mode is always active for this dashboard version:

- remote provider calls are disabled;
- automatic model downloads are disabled;
- automatic dataset downloads are disabled;
- navigation never invokes Hugging Face or another remote provider;
- missing optional dependencies/assets become `UNAVAILABLE` or `DEFERRED`
  capability states.

## API Design

All responses include a schema version where a durable contract is useful.
List endpoints support bounded limits and search/filter parameters. Identifiers
are validated through canonical lookup methods; no route accepts an arbitrary
path or command.

### Read routes

- `GET /api/lab/overview`
- `GET /api/lab/datasets`
- `GET /api/lab/datasets/{dataset_id}`
- `GET /api/lab/datasets/{dataset_id}/preview?limit=N`
- `GET /api/lab/quality?dataset_id={dataset_id}`
- `GET /api/lab/models`
- `GET /api/lab/planner/options`
- `GET /api/lab/plans`
- `GET /api/lab/plans/{plan_id}`
- `GET /api/lab/runs`
- `GET /api/lab/runs/{run_id}`
- `GET /api/lab/runs/{run_id}/checkpoints`
- `GET /api/lab/results`
- `GET /api/lab/benchmarks`
- `GET /api/lab/doctor/latest`
- `GET /api/lab/hardware`
- `GET /api/lab/offline`

### Explicit action routes

- `POST /api/lab/quality/check` (body contains a discovered dataset ID and bounded scan options)
- `POST /api/lab/quality/duplicates` (body contains a discovered dataset ID)
- `POST /api/lab/quality/derive` (body contains a discovered dataset ID and a safe output label)
- `POST /api/lab/plans`
- `POST /api/lab/preflight`
- `POST /api/lab/runs/{run_id}/resume-check`
- `POST /api/lab/doctor/run`

An action is enabled in the client only when its backing canonical operation
is available and its prerequisites are satisfied. Planning and preflight are
implemented. Real GPU training is not started in this phase. Resume execution
is not offered unless the canonical runtime declares a compatible checkpoint
and all required assets/hardware are available; the initial no-GPU environment
therefore presents resume validation and an honest blocked state.

There are no generic file, environment-dump, command, shell, module-import, or
provider proxy routes.

## Domain View Models and Status Semantics

Dashboard view models distinguish independently:

- code integration;
- dependency availability;
- local asset availability;
- hardware capability;
- logical configuration validity;
- runtime validation.

Canonical statuses are preserved when supplied. Dashboard-only capability
states use `PASS`, `READY`, `AVAILABLE`, `WARN`, `FAIL`, `UNAVAILABLE`,
`DEFERRED`, `RUNNING`, `INTERRUPTED`, and `COMPLETED`. Unknown data is `null`
or `NOT AVAILABLE`, never an invented zero or success.

Dataset detail includes canonical identity/version/hash, lineage, artifact
root represented as a safe repository-relative label, safety flags, integrity,
quality/dedup state, and loader compatibility. Preview is capped by settings
and reads only the requested first records through canonical loaders.

Plan records store the canonical generated configuration and deterministic
identity in a Lab-owned plan directory. The API returns the canonical Plan ID,
model/dataset/configuration identities, expected runtime backend, checkpoint
policy, and safe output label. A plan is inspectable before preflight.

Run detail and checkpoint responses are built from canonical registry and
checkpoint metadata. Failure messages and auxiliary metadata are recursively
redacted before serialization.

## User Interface

The UI is a dense, responsive engineering console with a persistent sidebar:

- Overview
- Datasets and Dataset Detail
- Quality & Dedup
- Experiment Planner
- Preflight
- Model Adapters
- Runs and Run Detail/Resume
- Results Store
- Benchmarks
- Doctor
- Hardware and Offline Status

Tables use horizontal containment on narrow screens, search where useful,
expandable JSON/configuration details, explicit loading/empty/error states,
and consistent status badges. Safety and blocked-action explanations remain
visible. Dataset lineage and run lineage use accessible CSS flow diagrams with
text equivalents. The client never simulates successful backend operations.

## Security and Privacy

- Default network binding and request guard are loopback-only.
- Settings roots are server-owned; the client sends identifiers, not paths.
- Canonical redaction is applied to data, exceptions, failure metadata, and
  logs before any browser response.
- Absolute private filesystem paths are converted to repository-relative or
  opaque display values; paths outside approved roots are not returned.
- Environment enumeration and secret-bearing configuration are never exposed.
- Error responses contain stable codes and sanitized summaries, with detailed
  server logs also redacted.
- Static assets use a restrictive Content Security Policy and no remote fonts,
  scripts, analytics, or CDN resources.
- State-changing routes validate content type and a same-origin request token
  issued by the local app to reduce accidental cross-origin invocation.

## Failure and Missing-Capability Behavior

Optional imports occur inside capability probes or operations. Missing Torch,
Transformers, CUDA, model-specific dependencies, model weights, tokenizer,
processor, RAQM, or WeasyPrint cannot prevent app startup.

Quality is always scoped to a discovered canonical dataset/manifest. There is
no fabricated global quality status and no browser-supplied manifest path.

Canonical not-found errors return `404`; invalid supported inputs return `422`;
protection or compatibility failures return `409`; unavailable optional
capabilities return `503` only for an explicitly invoked action. Read pages
return capability records and remain usable. Unexpected errors return a
redacted `500` payload without raw tracebacks or environment data.

## Testing Strategy

Use the existing pytest stack and FastAPI `TestClient`; do not add a frontend
test framework.

Tests will cover:

- app factory, loopback guard, `/lab` navigation, packaged assets, and CSP;
- overview aggregation without fabricated values;
- dataset list/detail/preview limits, lineage, integrity, and protected holdout;
- quality/dedup serialization and delegation to canonical operations;
- adapter discovery and distinct dependency/assets/GPU capability states;
- deterministic plan generation and canonical configuration inspection;
- preflight section/status/blocker serialization and no-GPU inspection;
- run list/detail, failure redaction, checkpoint metadata, and resume blocking;
- Results Store and immutable benchmark metadata reads;
- Doctor normal/deep explicit execution and latest-result behavior;
- offline state and proof that navigation performs no external calls/downloads;
- empty/loading/error/no-GPU/disabled-action states through static client
  contract tests and API integration tests;
- rejection of traversal-like identifiers, remote clients, unsafe roots,
  arbitrary file access, and secret-bearing values.

Canonical services will be injected in focused API tests. Additional
integration tests will use only repository fixtures and temporary directories.
No test will download model weights, datasets, checkpoints, GPU runtimes, or
remote benchmark data.

## Verification

Verification after implementation will run, in order:

1. dashboard-focused pytest tests;
2. relevant existing Lab/training/data/results/doctor tests;
3. Ruff;
4. Black check;
5. MyPy;
6. the full repository pytest suite when practical;
7. a local smoke launch on `127.0.0.1` and HTTP checks for `/lab`, overview,
   no-GPU behavior, and remote-client rejection.

The final report will include exact commands, results, deferred capabilities,
confirmation of no downloads/GPU training/secret exposure, final Git status,
and the final commit SHA.

## Deferred Capabilities

This phase does not install model assets, install optional model packages,
perform remote calls, execute real GPU training, or recompute expensive
benchmarks. The service/page boundaries include capability and action states
so these can be enabled later through canonical runtime operations without
redesigning the dashboard.
