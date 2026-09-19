# GitHub CI Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the existing cross-platform CI and release-hardening failures without adding features or weakening controls.

**Architecture:** Keep platform behavior in the workflow, expose model assets through a browser-safe projection while retaining private path resolution, and preserve existing optional-capability and security contracts.  Diagnose split and image behavior from canonical inputs before choosing minimal code or test changes.

**Tech Stack:** GitHub Actions YAML, Python 3.11, pytest, Pillow, Ruff, Black, MyPy.

**Spec:** `docs/superpowers/specs/2026-09-19-github-ci-repair-design.md`

## Global Constraints

- No product feature, model training, benchmark, GPU workload, model/data download, or dependency expansion.
- Do not weaken security, protected-holdout, near-duplicate thresholds, or storage-boundary behavior.
- Optional unavailable Doctor capabilities are non-required `SKIP`; installed broken capabilities remain failures.
- Browser/API payloads do not expose absolute server paths; trusted server-side resolution remains available.
- Final `main` CI must be green before completion.

## Review Focus

- PowerShell runner: both factory smoke commands execute and propagate a non-zero command status.
- A configured model asset: public response contains only its managed ID and a safe relative location while verifier receives the resolved root internally.
- Optional missing training dependency: Doctor reports unavailable `SKIP`, whereas an installed-but-broken subsystem fails.
- Canonically identical document identities differing only by platform representation obtain the same split.
- An escaping symlink always raises the storage-boundary exception.

---

### Task 1: Reproduce and repair workflow portability

**Files:**
- Modify: `.github/workflows/ci.yml:52-63`
- Test: `.github/workflows/ci.yml` smoke commands run locally through the runner shell's native output handling.

**Interfaces:**
- Consumes: `python -m clouda_data.pipeline.cli factory-profiles` and `factory-seeds`.
- Produces: shell-neutral smoke commands that retain exit status.

- [ ] **Step 1: Add a workflow assertion that avoids the Unix device path**

```yaml
python -m clouda_data.pipeline.cli factory-profiles
python -m clouda_data.pipeline.cli factory-seeds
```

- [ ] **Step 2: Validate both commands locally**

Run: `python -m clouda_data.pipeline.cli factory-profiles; python -m clouda_data.pipeline.cli factory-seeds`
Expected: both exit 0.

- [ ] **Step 3: Commit with the remaining repair work**

### Task 2: Preserve model asset privacy

**Files:**
- Modify: `clouda_lab/dashboard/models.py:125-225`
- Test: `tests/dashboard/test_model_catalog.py`

**Interfaces:**
- Consumes: `configure_assets(catalog_id, asset_id)` and trusted `_configured(catalog_id)`.
- Produces: browser-safe public mappings and a private `Path` returned only by `_configured`.

- [ ] **Step 1: Write a failing public-payload regression**

```python
assert str(tmp_path) not in repr(service.get_model("qwen3-vl-4b-instruct"))
```

- [ ] **Step 2: Run the targeted test and observe the absolute path leak on Linux**

Run: `python -m pytest tests/dashboard/test_model_catalog.py::test_model_assets_are_configured_only_by_managed_id_and_verified_canonically -q`
Expected: failure before the projection fix on the affected CI platform.

- [ ] **Step 3: Project asset locations through `safe_relative_label`/`browser_safe` while leaving `_configured` unchanged**

```python
"location": asset_root.relative_to(self.settings.repo_root).as_posix()
```

- [ ] **Step 4: Re-run dashboard catalog tests**

Run: `python -m pytest tests/dashboard/test_model_catalog.py -q`
Expected: PASS.

### Task 3: Align Doctor optional-capability tests

**Files:**
- Modify: `tests/doctor/test_subsystems.py:228-251`, `tests/doctor/test_system_cli.py:374-390`

**Interfaces:**
- Consumes: `check_training_framework(runs_root)` and `collect_report(deep=True)`.
- Produces: assertions for `DoctorStatus.SKIP`, `required is False`, and deep skip output.

- [ ] **Step 1: Write/adjust failing expectations for an unavailable optional training runtime**

```python
assert by_id["training.imports"].status is DoctorStatus.SKIP
assert by_id["training.imports"].required is False
```

- [ ] **Step 2: Run Doctor focused tests**

Run: `python -m pytest tests/doctor/test_subsystems.py tests/doctor/test_system_cli.py -q`
Expected: PASS with the existing categorical implementation.

### Task 4: Correct RAQM and storage-boundary regressions

**Files:**
- Modify: `clouda_data/factory/render/_raqm/corpus.py:35`, `clouda_data/factory/render/_raqm/layouts.py:16`, `tests/results/test_artifacts.py:73-82`
- Test: `tests/factory/test_render.py`, `tests/results/test_artifacts.py`

**Interfaces:**
- Consumes: factory provenance hashing and `ArtifactResolver.resolve_uri(uri)`.
- Produces: valid RAQM imports and an enforced escaping-symlink exception.

- [ ] **Step 1: Write failing assertions for import and symlink escape**

```python
with pytest.raises(ArtifactResolutionError, match="storage boundary"):
    resolver.resolve_uri("dataset://images/link.png")
```

- [ ] **Step 2: Run focused tests and observe the bad relative import/vacuous test**

Run: `python -m pytest tests/factory/test_render.py::test_raqm_backend_renders_deterministic_pages tests/results/test_artifacts.py::TestCrossPlatform::test_symlink_escape_rejected -q`
Expected: failure before correction where the optional renderer is available.

- [ ] **Step 3: Import from `...factory.provenance.hashing` and remove the vacuous assertion**

```python
from ...provenance.hashing import sha256_bytes, sha256_text
```

- [ ] **Step 4: Re-run render and artifact suites**

Run: `python -m pytest tests/factory/test_render.py tests/results/test_artifacts.py -q`
Expected: PASS or explicit platform skip only where symlinks/renderer are unavailable.

### Task 5: Audit and stabilize deterministic inputs

**Files:**
- Inspect: `clouda_training/sampling/splits.py`, `clouda_data/quality/image_fp.py`, `clouda_data/quality/near_index.py`
- Modify only if evidence requires: the canonicalization source or a platform-fragile test fixture.
- Test: `tests/integration/test_clouda_backend_e2e.py`, `tests/quality/test_near_index.py`

**Interfaces:**
- Consumes: canonical document ID/seed and decoded image fingerprint.
- Produces: platform-invariant split and semantically stable near-duplicate assertion without threshold changes.

- [ ] **Step 1: Add a failing regression only after identifying a non-canonical platform input**

```python
assert deterministic_document_split([canonical_id], seed=seed) == expected_split
```

- [ ] **Step 2: Run targeted integration and near-index mutation tests**

Run: `python -m pytest tests/integration/test_clouda_backend_e2e.py tests/quality/test_near_index.py::TestMutations -q`
Expected: evidence identifies implementation nondeterminism or fixture sensitivity.

- [ ] **Step 3: Apply the smallest evidence-backed normalization or fixture correction**

```python
canonical = canonical_identity.replace("\\\\", "/").casefold()
```

- [ ] **Step 4: Re-run focused suites**

Run: `python -m pytest tests/integration/test_clouda_backend_e2e.py tests/quality/test_near_index.py tests/quality/test_image_near_duplicates.py -q`
Expected: PASS.

### Task 6: Integrate and release

**Files:**
- Verify all changed files and generated wheel only.

- [ ] **Step 1: Run final validation**

Run: `python -m pytest -q; python -m ruff check .; python -m black --check .; python -m mypy .; node --check clouda_lab/dashboard/static/app.js; python -m build --wheel --no-isolation --skip-dependency-check; git diff --check`
Expected: every command exits 0.

- [ ] **Step 2: Commit, push, and wait for feature CI**

Run: `git add ...; git commit -m "fix: repair cross-platform CI failures"; git push -u origin feature/fix-github-ci-failures`
Expected: GitHub Actions feature run is green.

- [ ] **Step 3: Synchronize, merge normally, and verify final main CI**

Run: `git fetch origin; git switch main; git merge --no-ff feature/fix-github-ci-failures; git push origin main`
Expected: final `main` workflow is green with no history rewrite.
