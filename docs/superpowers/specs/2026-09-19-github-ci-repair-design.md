# GitHub CI Repair Design

## Goal

Repair the current GitHub Actions failures on `main` without adding product
features, weakening security, changing protected-holdout behavior, or running
models, training, benchmarks, data downloads, or GPU workloads.

## Scope and evidence

The repair starts from `origin/main` at `2bf809aefc1f723049767a4f8245e9e291b491c6`.
The checked-out workflow uses Unix `/dev/null` redirection in a PowerShell job.
The documented Linux failures are audited independently; local Windows-focused
tests that depend on unavailable optional capability correctly skip.

## Decisions

### Cross-platform CI smoke

The factory profile and seed commands remain real commands whose non-zero exit
status fails CI.  Their output is not redirected using a Unix device path; the
workflow must be valid under the shell selected by each runner.

### Browser-safe model catalog

`ModelCatalogService` continues to resolve a managed asset identifier to an
absolute server-side path solely inside trusted service operations.  Its public
catalog/configure/verify projections expose the managed identifier and a
repository-relative location only.  No public payload may contain an absolute
`Path` or storage-root string.

### Optional Doctor capabilities

Training/factory capabilities absent from the installed extras remain
non-required `SKIP` checks with an explicit unavailable reason.  Tests must
assert those states rather than demanding a synthetic PASS.  If the optional
dependency is present and the subsystem is broken, Doctor still reports a
required failure.

### RAQM provenance boundary

RAQM render modules import hashing helpers from
`clouda_data.factory.provenance.hashing`, their actual package boundary.  No
duplicate helper or compatibility shim is introduced.

### Determinism and image candidates

Dataset split inputs are traced through canonical identity and SHA-256 based
assignment.  Any platform-dependent input is normalized at the source; an
over-specific test is changed only after this audit proves the algorithm is
already canonical.  Near-duplicate thresholds are unchanged.  Tests use a
deterministic mutation fixture or normalized fingerprint invariant only when
the existing image fixture is dependency-sensitive; false-positive protection
remains covered.

### Storage boundary

An artifact URI that resolves through a symlink outside its configured root
must raise `ArtifactResolutionError`.  The test asserts that exception directly
and contains no vacuous success condition.

## Verification

Focused regressions cover each repaired boundary.  Final verification runs the
full test suite, Ruff, Black, MyPy, JavaScript syntax, wheel build, and
`git diff --check`.  The feature branch is pushed only after local validation,
then synchronized and merged normally; completion requires the final `main`
GitHub Actions run to be green.
