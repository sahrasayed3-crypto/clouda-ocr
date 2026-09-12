# Remaining Feature Integration Design

## Objective

Integrate the six remaining Clouda OCR feature branches into the current
canonical backend at `origin/main`, preserve their public histories, and reconcile
their older assumptions with the already-integrated Results Store, Lab, Training
Data Loader, Training Experiment Framework, Training Orchestrator, checkpoint
integrity envelope, Environment Doctor, CLI, packaging, and tests.

The verified starting `origin/main` is
`14e9600eafece7be929c89552564ffa47d1b2696`. The six verified feature tips are:

- `feature/dataset-quality-dedup` at `e9ee58d5d51559d4cc977898efbeca87d921407c`;
- `feature/real-training-runtime` at `759f248842d99b0eef5c7d76cd22a71714996c51`;
- `feature/hunyuanocr15-sft-bridge` at `08d77c4d5fbe022eeffe3216650e470ec8e3fd61`;
- `feature/multimodel-training-adapters` at `e72b801b53dd912e3ac107d13bb16eefeca2c23b`;
- `feature/training-preflight-validator` at `b758b8e1b24e460d34a3a4bec914bccf8e9cd371`;
- `feature/training-experiment-planner` at `e25f1a93b9a61447f44cf6272ac37ab90d261af9`.

## Branch topology and merge order

The training branches are an intentional linear history:

`real-training-runtime` -> `hunyuanocr15-sft-bridge` ->
`multimodel-training-adapters` -> `training-preflight-validator` ->
`training-experiment-planner`.

Merge `dataset-quality-dedup` first because it is independent and establishes the
canonical derived-data gate used by the later end-to-end reconciliation. Then
merge every training branch in the topology order with `--no-ff`. This records a
separate integration decision for each named feature while avoiding duplicate
content. The runtime comes first because every later training branch already
contains it; this dependency evidence justifies the order.

The remote branch audit found no additional unique feature work. The Lab, Results
Store, Training Data Loader, and Environment Doctor feature tips are already
ancestors of `main`.

## Canonical ownership

- Data Factory and Lab selection own source and selected-manifest creation.
- Dataset Quality consumes canonical manifests, applies deterministic quality and
  leakage policy, and emits a portable derived manifest with lineage.
- Training Data Loader owns bounded, deterministic sharding and sample delivery.
- The planner produces a deterministic, identity-bound plan for the canonical
  orchestrator; it never executes a second workflow.
- Preflight validates one concrete planned run. Environment Doctor remains the
  broader environment/project readiness diagnostic.
- The adapter registry owns model-family selection and capability detection.
  HunyuanOCR-1.5 and Qwen behavior stay behind that interface with lazy optional
  imports.
- The runtime executes through the existing Training Experiment Framework and
  stores resume state inside its checkpoint integrity envelope.

## Safety and portability

Protection and holdout metadata fail closed. Persistent identity and ordering use
cryptographic canonical serialization, never Python `hash()`. Portable artifacts
must not contain machine-local absolute paths or configured secret values. Paths
and artifacts are treated as untrusted input. The integration remains offline and
CPU-safe: integration performs no downloads, no real training, no CUDA/NCCL
benchmark, and no model-weight access.

The canonical Arabic OCR benchmark manifest must remain byte-identical at SHA-256
`2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893`.

## Validation and completion

After every merge, run that feature's focused tests. Any semantic reconciliation
uses a failing regression test before production changes. The final gates cover
the complete pytest suite, the offline synthetic pipeline including interruption
and exact resume, Doctor normal/deep modes, CLI human/JSON behavior, redaction,
benchmark release, Ruff, Black, MyPy, wheel/sdist build and contents, repository
hygiene, review findings, and Git synchronization.

Fetch `origin` again before publishing. Only if all blocking gates are green,
fast-forward local `main` to the integration result and push normally. Never force
push, delete remote feature branches, or disturb unrelated worktrees.
