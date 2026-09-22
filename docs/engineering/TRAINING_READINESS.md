# Training Readiness — Current Session (2026-09-21, updated 2026-09-22)

No training was run. Verdict from code + tests: the **orchestration shell is
READY and heavily tested** (strict config validation, fail-closed preflight,
planner, checkpointing/resume, provenance); **real-model training is BLOCKED
by explicit, documented gaps**, and no final base model has been selected
(both real-adapter descriptors carry `real_weights_validated=False`,
`gpu_validated=False`). The canonical runtime itself executes a genuine
optimization loop validated with a tiny synthetic trainer — see
[`docs/training/REAL_TRAINING_RUNTIME.md`](../training/REAL_TRAINING_RUNTIME.md)
for the exact validated boundary.

## Capability matrix

| Capability | Status | Evidence |
|---|---|---|
| Config validation (`validate-config`) | READY | Unknown fields/sections rejected (`experiments/config.py:154-160,288-297`); semantic checks; portable config hash embedding manifest sha256 (`:127-138`); preflight config blockers (`preflight/checks_config.py:76-326`) incl. false-READY max_steps guard |
| Planner | READY (design-only) | Step math + VRAM + storage/cost + assumptions ledger (`planner/planner.py:54-230`); hardware is **declared, not detected** (operator flags); unknown parameter count → VRAM UNKNOWN, never fabricated; ~70 planner tests |
| Preflight | READY, fail-closed | Orchestrator aggregates blockers → NOT_READY (`preflight/models.py:233-238`); cuda-unavailable is a blocker; dependency probes never install; dataset protection via canonical guard; resume identity + torch-state integrity checks; output write-probe (creates missing output_root as documented side effect) |
| Hunyuan export/pack/validate | READY | Manifest-hash pre-check, per-row protection fail-closed, holdout refusal, zero-sample refusal, lineage sidecar + output sha256 (`hunyuan/exporter.py:129-201`); packing plan Mode A only, official packer never run (`hunyuan/packing.py:1-16`); ~45 tests |
| Qwen export | PARTIAL (library only) | `qwen/data_adapter.py` (official qwen-vl-finetune format, tested) but **no `qwen export` CLI subcommand** and no preflight hook (`cli_adapters.py:157-167`) |
| Torch training loop | PARTIAL (synthetic only) | Genuine optimization loop (grad accumulation, clipping, AdamW, scheduler) in `runtime/torch_backend.py:114-179`; the only fully-working adapter is `SyntheticLinearAdapter` |
| Real dataset → trainer wiring | **BLOCKED** | `TorchTrainerBackend` never receives `data_loader` (`experiments/runs.py:218-225` vs mock at `:227-234`); real adapters' batch path explicitly unimplemented pending real weights (`hunyuan/adapter.py:232-248`, `qwen/adapter.py:269-284`) |
| Real-model training | **BLOCKED** | Requires real weights; descriptors not validated; since 2026-09-22 also requires an explicit training-use approval record (`adapters/approval.py`, catalog `configs/models/model_training_approvals.v1.json`, shipped empty); legacy trainer intentionally disabled (`trainers/base.py:14-17`) |
| LoRA/QLoRA | NOT IMPLEMENTED | Zero peft references; `supports_lora=False` on both descriptors; `configs/training/first-full-run.json` "qlora" has no code path (legacy schema) |
| Multi-GPU (DDP/accelerate/deepspeed) | NOT IMPLEMENTED | Zero references; documented as verified fact (`planner/memory.py:43-44`); `world_size` is a planning multiplier only |
| Mixed precision | FAIL-CLOSED | Flag validated (`config.py:64`, `checks_config.py:229-240`); since 2026-09-22 the torch backend rejects `mixed_precision=true` until AMP is validated with a real model on real hardware (`runtime/torch_backend.py`) |
| Evaluation during training | READY (synthetic-validated) | Mock trainer computes CER/WER vs fixture (`experiments/trainer.py:84-104`); torch backend runs an optional adapter `evaluate()` hook on the eval cadence, persists metrics under the eval split, and feeds best-checkpoint selection (`runtime/torch_backend.py`, tested in `tests/runtime/test_evaluation_hook.py`); `clouda_training/evaluation/` is a one-line stub; protected-holdout eval forbidden at config level (`config.py:252-256`) |
| Image preprocessing/tokenization | NOT IMPLEMENTED | Exporters are text-level with fixed `<image>` placeholder; no resize/normalize/tokenize anywhere; qwen collator (label −100 masking) does not exist yet (`qwen/adapter.py:255-259`) |
| Checkpoints + resume | READY (mock/synthetic) | Atomic staging save, sha256 enforced on load, retention + best tracking (`experiments/checkpoints.py`); torch state fsync'd atomic save + digest on load (`runtime/checkpoint_torch.py`); resume restores model/optimizer/scheduler/RNG and rejects adapter-identity change (`torch_backend.py:236-269`); resume-determinism test proves interrupted == uninterrupted (`tests/runtime/test_resume_determinism.py:45`) |
| Seed handling | READY | `apply_seed` seeds python/numpy/torch/cuda + deterministic algorithms (`experiments/environment.py:10-49`); loader shuffle a pure function of (seed, epoch, topology, config_hash); caveat: `runtime.num_workers` is dead config (no torch DataLoader used) |
| Failure recovery | READY | Truthful FAILED/INTERRUPTED status with redacted error; resume only from INTERRUPTED/FAILED with matching data identity; corrupt checkpoint rejected by digest — all tested |
| Provenance | READY | Run metadata records config hash, git commit+dirty, dataset manifest hash, rows, source ids/licenses, seed, model id/revision, redacted command line (`experiments/runs.py:330-361`); artifact integrity manifest; loader lineage identity incl. shard index hash |

## Legacy config note

`configs/training/*.json` (smoke-100, pilot-1000, first-full-run) use the old
`TrainingConfig` schema with placeholder base models — **not consumable by
the experiment framework** (which requires YAML with experiment/model/dataset
sections). Do not treat them as runnable; they predate the framework.

## Recommended unblocking order (evidence-based, no code changed here)

1. Select and license a base model; validate real weights into the adapter
   (`real_weights_validated`); record the training-use approval in
   `configs/models/model_training_approvals.v1.json`.
2. Implement the real processor/collator path and pass `StreamingTrainingDataLoader`
   into `TorchTrainerBackend`.
3. ~~Add eval hooks to the torch backend~~ DONE (2026-09-22): optional
   adapter `evaluate()` hook, tested.
4. ~~Decide mixed-precision support~~ DECIDED (2026-09-22): fail-closed
   until validated on real hardware with the selected model.
