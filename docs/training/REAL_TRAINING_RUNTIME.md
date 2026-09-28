# Real training runtime — validated capability and safety model

Status date: 2026-09-22. This document states what the canonical training
runtime **actually does and has actually executed** — no aspirational
claims.

## What is validated today

The canonical experiment runtime (`clouda_training.experiments.runs`) can
execute a **genuine optimization loop** end to end:

- forward pass, backward pass, AdamW optimizer step, LR scheduler step,
- gradient accumulation (verified against the full-batch gradient),
- gradient clipping with non-finite-gradient abort,
- periodic evaluation hooks with metric persistence and best-checkpoint
  selection,
- atomic checkpoints carrying model/optimizer/scheduler/RNG state with
  sha256 integrity,
- interruption + resume with bit-identical continuation
  (`tests/runtime/test_resume_determinism.py`).

This is proven with the tiny deterministic `SyntheticLinearAdapter` on CPU
(`tests/runtime/test_torch_e2e.py`, `tests/runtime/test_runtime_hardening.py`,
`tests/runtime/test_evaluation_hook.py`). **That adapter is a synthetic
regression task used to validate infrastructure only — it is not an OCR
model, and no real OCR training has been run.**

## What a real run requires (all enforced fail-closed)

`run_experiment` / `resume_run` refuse to start a real registered-adapter
run unless **every** condition holds:

1. `runtime.dry_run=false` — real execution is explicit opt-in, never a
   fallback.
2. `runtime.offline=true` — no network paths in the canonical runtime.
3. PyTorch installed; configured device/precision supported by the
   adapter descriptor.
4. Dataset passes `validate_training_dataset`: split must be an explicit
   training split (protected holdout names are refused), no protected rows
   (`clouda_contracts.protection`, content-based, not path-based), and a
   recorded `quality_gate_verdict` of `FAIL` in the manifest header blocks
   the run. The verdict value is recorded in run metadata.
5. **Model training-use approval**: the model must have an explicit
   `approved: true` record in the approval catalog
   (`configs/models/model_training_approvals.v1.json`, override with
   `CLOUDA_MODEL_APPROVALS`). Missing catalog, malformed catalog, missing
   record, or explicit rejection all fail closed
   (`clouda_training/adapters/approval.py`). The approval record (license,
   approver, catalog path) is copied into run metadata. The same guard runs
   in preflight (`model.training_approval` blocker) and again on resume.
6. Local model assets exist; no implicit download, no implicit
   `trust_remote_code`, no silent fallback to dry-run.

The shipped approval catalog is **empty on purpose**: no base model is
approved until the benchmark winner is selected and its license/training
terms are reviewed. The final model remains TBD.

## Failure behaviour

Non-finite loss/gradient norm aborts the step before `optimizer.step()`
(run recorded FAILED, resumable from the latest checkpoint). CUDA OOM is
re-raised with actionable guidance. `training.mixed_precision=true` is
rejected until AMP is validated with a real model on real hardware.
Failures are recorded as FAILED with redacted messages — never converted
into successful artifacts.

## Still unvalidated (requires the selected model / real hardware)

- real model adapter paths (batch collation, processor, loss) for weights
  that have never been loaded,
- `StreamingTrainingDataLoader` → `TorchTrainerBackend` wiring with a real
  dataset,
- CUDA memory behaviour, BF16/FP16 stability, throughput,
- multi-GPU (DDP/FSDP/DeepSpeed) — deliberately unimplemented; `world_size`
  is a planning multiplier only,
- real OCR convergence.

See `docs/training/HARDWARE_VALIDATION_TODO.md` and
`docs/engineering/TRAINING_READINESS.md` for the full honest matrix.
