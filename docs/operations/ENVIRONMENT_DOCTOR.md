# Clouda Environment Doctor (`doctor` command)

Diagnostic-only readiness inspection for the Clouda OCR environment. Before
running Data Factory rendering, benchmarks, the Training Experiment
Framework, OCR inference, or future GPU training, the Doctor answers one
question: **what is READY, PARTIALLY READY, NOT READY, or NOT PRESENT on
this machine right now — and what do I fix?**

## Usage

```powershell
python -m clouda_data.pipeline.cli doctor            # lightweight default
python -m clouda_data.pipeline.cli doctor --json     # machine-readable
python -m clouda_data.pipeline.cli doctor --verbose  # technical details per check
python -m clouda_data.pipeline.cli doctor --deep     # adds tiny offline smoke checks
python -m clouda_data.pipeline.cli doctor --no-render --no-training   # scope down
```

There is no separate CLI framework: `doctor` is a subcommand of the
canonical `clouda_data.pipeline.cli` (same parser family as
`factory-generate`, `dry-run`, `status`).

Options: `--json`, `--verbose`, `--deep`, `--no-factory`, `--no-render`,
`--no-training`, `--no-git`, `--no-storage`, `--min-free-gb N` (overrides the
repo-drive free-space WARN threshold).

## Modes

- **Default (lightweight):** interpreter facts, import/worktree resolution,
  dependency groups, Results Store, Lab and Training Data readiness, Data
  Factory imports + packaged resources, renderer
  availability (including native-library reality), Training Framework
  imports + output-root writability, GPU presence, storage, Git state.
  No renders, no training, no network, no large I/O.
- **Deep (`--deep`):** additionally runs tiny **offline** Results/Lab/Training
  Data and `MockTrainer` dry-runs in a temporary directory (forced
  `runtime.dry_run=true`, `runtime.offline=true`), then deletes the
  directory. No model downloads, no real training, CPU-only, seconds fast.

## Statuses and exit codes

Per-check / rollup statuses: `PASS`, `WARN`, `FAIL`, `SKIP` (subsystem not
present on this branch base), `INFO` (purely informational).

| Exit | Meaning |
|------|---------|
| 0 | no *required* check failed (WARN-only runs pass) |
| 1 | one or more required checks FAILED |
| 2 | the doctor itself could not execute (internal error) |

Only *required* checks can fail the run. Renderer/GPU checks are marked
optional: a machine without WeasyPrint, without native RAQM, or without a
GPU is a **clean diagnostic state**, not a doctor failure — it shows up as
`FAIL`/`INFO` on an optional check and `NOT READY` on the subsystem summary,
and the exit code stays 0 if everything *required* is fine.

## JSON contract

`doctor --json` emits one object, schema version
`clouda.ocr.doctor.v1` (see `clouda_data/doctor/models.py`
`DOCTOR_SCHEMA_VERSION`):

```json
{
  "schema_version": "clouda.ocr.doctor.v1",
  "timestamp": "2026-09-10T12:00:00.000000Z",
  "deep": false,
  "overall_status": "FAIL",
  "readiness": "NOT READY",
  "runtime": {"python": "...", "executable": "...", "platform": "...", "os": "...", "machine": "...", "in_venv": true},
  "repository": {"root": "...", "worktree_kind": "unknown", "env_vars": [{"name": "...", "set": true, "secret": false}]},
  "sections": [{"id": "...", "name": "...", "status": "...", "checks": [...]}]
}
```

Each check: `id`, `name`, `subsystem`, `status`, `message`, `required`,
`details` (structured technical metadata), `remediation` (fix hint or
null). Serialization is deterministic for a given report; the future
Clouda Lab UI/API should consume this directly. Paths inside are **local
diagnostic metadata** — treat them as private to the machine.

## Dependency groups

The Doctor does not maintain its own package list. It reads
`[project.optional-dependencies]` from `pyproject.toml` (with a hard-coded
fallback table verified at the current base) and checks each requirement by
distribution name — parsing version specs like `PyYAML==6.0.2` or
`WeasyPrint>=66,<70` — plus importability using a small name-mapping table
(`Pillow`→`PIL`, `opencv-python-headless`→`cv2`, `PyYAML`→`yaml`,
`WeasyPrint`→`weasyprint`, ...). Missing packages get the correct extras
hint, e.g. `pip install -e ".[factory]"` — the Doctor **never installs**.

## Import / worktree drift detection

For each canonical package (`clouda_data`, `clouda_training`,
`clouda_contracts`, `clouda_models`, `clouda_lab`) the Doctor resolves the real
import source (`module.__file__`) and compares the set of roots against the
active repository/worktree (`CLOUDA_PROJECT_ROOT` or the directory holding
`pyproject.toml`):

- **EXPECTED SOURCE ROOT** — the active worktree.
- **ACTUAL IMPORT ROOT** — where Python actually imports the packages from.

Outcomes: all roots match → `PASS`; a single different root (editable
install pointing at another worktree, or a stale site-packages copy) →
`WARN` with a `pip install -e .` hint; mixed roots across packages →
`FAIL`. The Doctor **diagnoses only** — it never rewrites environments.

## Renderer readiness

- **WeasyPrint**: package import is performed exactly like the vendored
  backend (stdout/stderr swallowed so native-library troubleshooting text
  cannot corrupt output). If importable, a tiny in-memory HTML→PDF smoke
  render proves native Pango/HarfBuzz actually work — import success alone
  is never reported as ready. Missing → FAIL with the `factory-render`
  extras hint.
- **RAQM**: the vendored backend's `_HAS_RAQM` flag is **known to be a
  false positive** (it is True whenever Pillow's `ImageFont.Layout.RAQM`
  enum exists, even without the native library). The Doctor instead reads
  `PIL._imagingft.HAVE_RAQM` and runs a tiny Arabic smoke draw with
  warning capture: if Pillow emits
  *"Raqm layout was requested, but Raqm is not available. Falling back to
  basic layout."* the renderer is reported NOT READY (no Arabic
  shaping/bidi). Tests pin this contract against regressions.

## GPU / CUDA interpretation

If torch is missing: `GPU TRAINING: NOT READY` (INFO, clean state). If
torch exists but `torch.cuda.is_available()` is False: same, with the torch
version and CUDA build recorded. If CUDA is available: device count, GPU
names, VRAM, compute capability, and BF16 support are reported. `nvidia-smi`
presence is informational. The Doctor does not invent driver/toolkit
compatibility rules; it reports observed versions only.

## Storage checks

Repo-drive free space with **configurable, project-local** thresholds
(default: WARN below 50 GB, FAIL below 10 GB — guidance for typical working
drives, *not* universal training requirements; override with
`--min-free-gb`). Also reports the configured `StorageRoots` state roots
(runtime/datasets/artifacts/models/cache, existence only), temp-directory
writability, and repository writability. No recursive scans of dataset
trees.

## Git / worktree checks

Read-only: branch, HEAD, dirty/clean, origin URL, git-dir vs common-dir
(worktree kind detection: main working tree vs secondary worktree),
ahead/behind only when an upstream is already configured. **No fetch, no
config changes, no state mutation.**

## Environment-variable privacy

The Doctor reports only *names* from a curated allowlist discovered from
the repository, as `SET` / `NOT SET` — **values are never emitted** in
human output, JSON, verbose output, or exception text. Credential-ish names
(keys, tokens, secrets, passwords, credentials, auth, service accounts)
are additionally flagged `secret: true` so downstream consumers know to
treat them carefully. Nothing is uploaded anywhere; all diagnostics stay
local.

## Remediation behavior

Every WARN/FAIL carries a `fix:` hint derived from the real project setup
(extras names verified against `pyproject.toml`). The Doctor never runs
installers, never modifies PATH, never edits venvs or Git config. A future
explicit `--fix` mode would be a separate, reviewed change.

## Limitations

- Renderer "READY" means the smoke checks passed; it does not guarantee
  byte-identical rendering to a reference machine.
- GPU "READY" does not verify specific kernel/CUDA-version compatibility.
- Storage thresholds are local guidance, not validated training minimums.
- Results Store, Clouda Lab, and Training Data readiness proves imports and
  local smoke behavior, not production-scale throughput.
- The doctor inspects the *active interpreter's* environment — run it with
  the same interpreter/venv you intend to use for the workload.
