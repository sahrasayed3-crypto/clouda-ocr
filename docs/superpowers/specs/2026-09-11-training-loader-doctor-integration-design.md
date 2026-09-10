# Training Loader and Environment Doctor Integration Design

## Objective

Integrate `feature/training-data-loader` and `feature/environment-doctor` into the
canonical backend at `origin/main`, preserving their histories while reconciling
both older branches with the current Results Store, Lab Backend, shared protection
policy, Training Orchestrator, and Training Experiment Framework.

## Ownership boundaries

- Results Store owns OCR result persistence and queries.
- Lab Backend owns analysis, selection, active learning, and orchestration.
- Training Data Loader owns deterministic sharding and bounded runtime delivery of
  an already-selected canonical training manifest.
- Training Experiment Framework owns experiment, run, checkpoint, and resume
  lifecycle state.
- Environment Doctor performs read-only diagnostics and never mutates application
  state.

## Integration order and history

Create an isolated integration branch from the fetched `origin/main`. Merge the
Training Data Loader first with a history-preserving merge commit, reconcile and
test it, then merge Environment Doctor and update its stale subsystem knowledge.
After all gates pass, fast-forward the clean canonical `main`, test that exact tree,
and push normally without rewriting remote history.

## Training-data contract

The loader consumes the derived JSONL manifest written by Lab selection. The header
and every row retain dataset ID/version, manifest identity, split, sample/page ID,
artifact and ground-truth references, provenance, explicit training eligibility,
and protection metadata. The loader delegates policy decisions to
`clouda_contracts.protection`; malformed, ambiguous, nested, aliased, or
case-varied protected metadata fails closed.

Shards record portable relative references plus source-manifest and sharding-config
identities. Iteration remains lazy, shuffle and topology assignment are stable,
prefetch is bounded and exception-safe, and batching has explicit final-batch
behavior. Resume identity covers dataset, manifest, loader configuration, seed,
epoch, rank/world-size, worker topology, positions, and yielded count. Loader state
is embedded in existing experiment checkpoint metadata rather than persisted by a
second checkpoint system.

## Orchestration and lineage

The canonical flow is `SelectionResult` to derived manifest to shard index to
`StreamingTrainingDataLoader` to the existing Training Orchestrator and mock
Training Experiment Framework run. Lineage includes selection, source results,
dataset ID/version, shard index, loader configuration hash, and summary or full
sample trace according to the configured trace mode. PyTorch support stays optional
and lazily imported.

## Environment Doctor

Preserve the older branch's runtime, dependency, rendering, RAQM, system, Git,
privacy, and GPU diagnostics. Extend it with distinct Results Store, Lab Backend,
and Training Data Engineering readiness sections. Default checks are lightweight;
deep mode uses temporary synthetic artifacts for a small Results/Lab/loader/mock
training smoke and cleans them afterward. Training-data readiness remains separate
from real GPU readiness.

Import-drift checks compare the expected repository root with every relevant
package's resolved import root and detect mixed roots, editable installs elsewhere,
and stale site-packages shadowing. RAQM readiness requires native capability or a
tiny real-layout smoke and must not infer readiness from an enum alone. Dependency
groups are derived from the current package metadata. Secret values are never
rendered in human, JSON, verbose, or exception output.

## Verification

Use test-first fixes for every semantic reconciliation. Focused tests cover policy
bypasses, manifest compatibility, deterministic sharding/order/topology, bounded
streaming/prefetch, artifact safety, checkpoint resume identity, orchestration,
Doctor readiness and false-ready cases, privacy, CLI wiring, and the complete
offline CPU E2E. Then run the full suite, Ruff, Black, MyPy, package build, CLI
smokes, benchmark byte hash/release tests, secret scan, and large-file scan. No
network, datasets, models, GPU training, or web UI are in scope.
