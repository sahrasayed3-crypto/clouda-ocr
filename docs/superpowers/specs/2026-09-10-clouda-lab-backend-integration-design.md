# Clouda Lab Backend Integration Design

## Objective

Integrate the Results Store and Clouda Lab backend branches onto the latest
canonical `main` while preserving branch history and making the Results Store
the only persistence/query source of truth. The Lab remains a stateless
analysis and orchestration layer over canonical Data Factory, Results Store,
and Training Experiment Framework contracts.

## Integration order and history

The integration branch starts at verified `origin/main`. Merge the verified
Results Store remote head first, resolve it against current Data Factory
contracts, and validate its focused tests. Merge the verified Lab Backend head
second and reconcile it against the already-integrated Results Store. Keep both
merge parents so the public feature histories remain visible; add focused fix
commits for cross-system reconciliation.

## Canonical boundaries

- `clouda_data.factory` owns generation, rendering, distortion transforms,
  deterministic seeds, manifests, hashes, and artifact provenance.
- `clouda_data.results` owns persistent datasets, pages, ground truth,
  predictions, model identities, inference-run identities, stored metrics,
  artifact resolution, integrity checks, and queries.
- `clouda_lab` owns error interpretation, batch aggregation, failure
  classification, dataset-selection policy, hard-example scoring,
  next-batch recommendation, distortion experiment coordination, and service
  facades. Lab DTOs may describe analysis results but must not become a second
  persistence schema or registry.
- `clouda_training.experiments` remains the sole training framework.
  `TrainingOrchestrator` is a fail-closed dry-run facade and lineage bridge.

## Service data flow

Add one explicit adapter/service path from a configured `ResultsService` to Lab
domain inputs. It must query canonical pages, raw ground truth, predictions,
and stored run/model metadata without ad-hoc parsing. Batch analysis and
run/model/checkpoint comparisons use these canonical queries. Selection and
recommendation outputs preserve the canonical page/sample id, dataset id and
version, split, protection/training eligibility, source metadata, and lineage.

Stored CER/WER values use `clouda_data.evaluation`. Lab error analysis imports
the same functions and may compute detailed alignments or normalized CER, but
must not introduce incompatible formulas. Any compatibility DTO clearly marks
whether a metric is computed or loaded.

## Holdout and provenance safety

Protection is fail-closed at every boundary. Results ingestion derives
eligibility from canonical protection metadata; Lab selection rejects
case/whitespace aliases, explicit and nested protection markers, protected
roles, malformed types, and missing or ambiguous role metadata; derived
manifests are validated again before the training facade and again by the
experiment framework. Tests use synthetic fixtures only.

Stable identities use deterministic cryptographic hashes over logical fields.
Absolute paths, drive letters, checkout location, Python `hash()`, and mutable
wall-clock fields do not influence persistent identity. Selection manifests
record source manifest hash, criteria, deterministic seed, and lineage.

## End-to-end verification

An offline CPU-only test creates a tiny Arabic canonical dataset, registers two
fake model/run results, queries page/GT/predictions through Results Store, runs
error and failure analysis, hard-example ranking, next-batch recommendation,
selection and derived-manifest creation, creates a dry-run experiment through
the facade, and verifies post-run lineage. It performs no network request,
model download, GPU operation, inference, or real training.

## Portability, CLI, and packaging

Retain the byte-stability rule for pinned benchmark assets without conflicting
`.gitattributes` behavior. Artifact URIs remain portable and traversal-safe.
Package discovery includes Lab modules and exposes a grouped `clouda-lab`
entry point while the existing Data Factory/results and training CLIs retain
their conventions. Imports have no optional-dependency side effects.

## Acceptance gates

Run focused Results Store, Lab, Data Factory, training, benchmark, contracts,
holdout, CLI, and cross-system tests, then the full suite. Run Ruff, Black
check, configured MyPy, `git diff --check`, package build when package metadata
changes, and repository secret/large-file checks. Classify baseline or optional
renderer failures separately, but do not merge with unresolved Critical or
Important integration findings. Push normally to `origin/main`, fetch again,
and verify clean `main == origin/main`.

