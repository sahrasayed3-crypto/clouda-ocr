# Multi-Model Adapters (hunyuan + qwen)

Local-only, no-training documentation for the Clouda model-adapter layer:
`clouda_training/adapters/` (Wave 1 framework core) and the concrete adapter
packages `clouda_training/hunyuan/` and `clouda_training/qwen/` (Wave 2).

## 1. Adapter architecture & registry

The adapter layer decouples the generic Clouda training runtime from any
specific upstream model family. Three pieces cooperate:

- **`ModelCapabilities`** (`clouda_training/adapters/capabilities.py`) — a
  frozen dataclass of 13 verified-support flags (all default `False`), e.g.
  `supports_full_finetune`, `supports_lora`, `supports_packed_sequences`,
  `supports_local_only_loading`, `supports_bf16`, `real_weights_validated`,
  `gpu_validated`. A flag asserts the capability was **observed working in the
  Clouda runtime**, not that the upstream model theoretically could do it.
- **`ModelAdapterDescriptor`** (`clouda_training/adapters/descriptor.py`) — a
  frozen dataclass: adapter identity (`adapter_type`, `adapter_version`,
  `model_family`, `task_family`), the capabilities record, dependency and
  precision/device/data-mode declarations, upstream pin
  (`upstream_repository` + `upstream_revision`), expected model/processor
  symbols, and `checkpoint_compatibility_id`. Methods:
  `require_capabilities(*flags)` (fail-closed `UnsupportedCapabilityError`)
  and `identity_dict()` (path-free metadata persisted into checkpoints).
- **`ModelAdapterRegistry`** (`clouda_training/adapters/registry.py`) — an
  explicitly-populated, thread-safe singleton (`get_default_registry()`).
  Registration is explicit and idempotent (`is_registered` guard inside each
  adapter package's `_register()`), refusal to overwrite duplicates
  (`DuplicateAdapterError`), unknown types fail with an actionable
  `UnknownAdapterError`, and iteration is sorted by `adapter_type` for
  determinism. Factories are **lazy callables**: creating an adapter instance
  never imports torch/transformers at registration time.

CLI surface (no command starts training):

```
python -m clouda_training.cli adapters list
python -m clouda_training.cli adapters inspect <adapter_type> [--json]
python -m clouda_training.cli adapters preflight <adapter_type> --model-path <local_dir> [--json]
```

- `list` imports the known adapter packages (which self-register), then prints
  every registered `adapter_type` with a capability summary.
- `inspect` prints the descriptor's `identity_dict()` plus the full
  capabilities record and expected model/processor symbols.
- `preflight` runs the adapter's local preflight hook when one exists
  (`hunyuanocr15_sft` → `clouda_training.hunyuan.preflight.run_preflight`);
  for an adapter without a preflight hook it exits non-zero with a clear
  message rather than guessing. `--model-path` must be an operator-supplied
  local directory; nothing is ever downloaded.

## 2. Capabilities honesty rules

1. **Every flag defaults to `False`.** An adapter that forgets to declare a
   capability is reported as NOT supporting it — fail closed, never open.
2. **Flags are verified-support claims**, not aspirational: set a flag only
   after the capability was exercised in the Clouda runtime (mock or real).
3. **Frozen at runtime.** `ModelCapabilities` and `ModelAdapterDescriptor`
   are frozen dataclasses; runtime code cannot flip a flag to make a failing
   path pass.
4. **`require_capabilities()` gates code paths.** Asking for an undeclared
   capability raises `UnsupportedCapabilityError` naming the missing flag.
5. **`real_weights_validated` / `gpu_validated` stay `False`** until real
   weights and a real GPU run has occurred. Mock-verified behaviour never
   upgrades these flags. Current status for both adapters: `False`.
6. **Identity is path-free.** `identity_dict()` excludes machine paths and
   capability claims so checkpoint resume checks a stable code identity.

## 3. Adapter status (verified upstream revisions)

### hunyuanocr15_sft (verified, mock-level)

- Upstream: `Tencent-Hunyuan/HunyuanOCR` @ `c55965d3da1e`
  (compat revision pinned in `clouda_training/hunyuan/models.py`).
- Adapter id `hunyuanocr15_sft`; task family Arabic-document-OCR SFT.
- Local bridge: exporter (canonical manifest → raw OCR JSONL, fail-closed
  holdout/protection), validators (raw + packed JSONL), packing plan builder,
  compatibility drift check, local-only preflight.
- Model load is **local-only** with `trust_remote_code=True` (upstream ships a
  custom `HunYuanVLForConditionalGeneration` via a transformers-style module).
- Capabilities: verified at mock level only — `real_weights_validated=False`,
  `gpu_validated=False`.
- See `UPSTREAM_COMPATIBILITY.md` (Hunyuan section) for the full upstream
  schema/packing/training-entry snapshot.

### qwen_vl_sft (verified upstream audit; adapter implementation landing in Wave 2A)

- Upstream: `QwenLM/Qwen3-VL` @ `96588727e44c78b25ba03ea03b8e12f7e64fd0da`
  (2026-01-30), `qwen-vl-finetune/`.
- Model classes: `Qwen3VLForConditionalGeneration` (dense) /
  `Qwen3VLMoeForConditionalGeneration` (MoE); native in
  `transformers>=4.57.0`, **no `trust_remote_code`**.
- Processor: `AutoProcessor` → `Qwen3VLProcessor`; training uses a separate
  `AutoTokenizer(use_fast=False, padding_side='right')`.
- Data format: JSON/JSONL `{"image": "path.jpg", "conversations": [...]}`
  with media **file paths**; the `<image>` tag appears only in the human turn
  (model-level `<|vision_start|>/<|image_pad|>/<|vision_end|>` are inserted by
  the processor).
- Loss: causal LM with labels `-100` masked except assistant spans
  (token ids 77091..151645); `IGNORE_INDEX=-100`.
- Selective tuning: `tune_mm_vision` (model.visual), `tune_mm_mlp`
  (model.visual.merger), `tune_mm_llm` (model.language_model + lm_head), with
  separate `vision_tower_lr` / `mm_projector_lr`.
- LoRA: official peft `LoraConfig` (targets q/k/v/o_proj); gradient
  checkpointing via `enable_input_require_grads`; packing via official
  `--data_flatten` (cu_seqlens-style attention) and `--data_packing`
  (`tools/pack_data.py`).
- SFT reference profile (`sft_qwen3_4b.sh`): lr 1e-6..2e-7 range per README,
  bf16, gradient checkpointing, `tune_mm_vision=False` / `tune_mm_mlp=True` /
  `tune_mm_llm=True`, `model_max_length=8192`.
- Recommended registration: `adapter_type="qwen_vl_sft"`,
  `model_family="qwen3_vl"`.

## 4. Local-only loading (hard rule)

Neither adapter ever downloads weights, tokenizers, processors, or code.

- `hunyuanocr15_sft` builds the model from an operator-supplied local
  directory (`local_model_path`) with `local_files_only=True`; a missing
  directory or `config.json` raises immediately.
- `qwen_vl_sft` follows the same contract: weights come from a local path
  (`transformers>=4.57` provides the model classes natively — still
  `local_files_only=True`).
- The CLI `preflight`/`inspect` commands accept `--model-path` only as a
  **local** directory and never trigger network access.

## 5. Data adapter contract

Wave 1 also defines the data-side mirror of the adapter registry
(`clouda_training/adapters/data_adapter.py`):

- `ModelTrainingDataAdapter` Protocol with three methods:
  - `export(manifest_rows, config)` — canonical Clouda manifest rows → the
    family's upstream training format (e.g. Hunyuan raw OCR JSONL with
    `image_path` lists; Qwen JSON/JSONL with `image` + `conversations`).
  - `validate(records)` — family-specific structural checks before anything
    reaches the model.
  - `describe_contract()` — a machine-readable description of the format,
    versioned via the descriptor's `data_contract_version`.
- `DataAdapterRegistry` + `get_default_data_adapter_registry()` with the same
  explicit-registration, duplicate-refusing, sorted-iteration semantics as the
  model registry (`UnknownDataAdapterError`, `DuplicateDataAdapterError`).

## 6. How to add a third family

1. Create `clouda_training/<family>/` with `adapter.py` (adapter class),
   `descriptor.py` (a `ModelAdapterDescriptor` with an **honest**
   `ModelCapabilities` record), and optionally `data_adapter.py` /
   `preflight.py`.
2. Self-register in the package `__init__.py` (or a `registration.py` it
   calls): guard with `if not reg.is_registered("<type>"):` then
   `reg.register(DESCRIPTOR, factory)`. The factory must be lazy — heavy
   imports (torch/transformers) belong inside the factory body.
3. Pin the upstream: set `upstream_repository` + `upstream_revision` to the
   audited commit and add a compatibility section to
   `UPSTREAM_COMPATIBILITY.md`.
4. Extend the CLI without touching this design: `_import_known_adapter_packages()`
   in `clouda_training/cli_adapters.py` imports known packages best-effort;
   add the new package's module name to that tuple. If the family has a
   preflight, wire it in `_run_adapter_preflight` (or promote a generic
   `run_preflight` convention).
5. Declare `checkpoint_compatibility_id` so old checkpoints fail loudly
   instead of silently resuming against a different adapter.
6. Add tests under `tests/multimodel/` following the existing mock pattern
   (`tests/hunyuan/mock_hunyuan.py`); keep `torch` behind
   `pytest.importorskip("torch")`.

## 7. Limitations — what needs real weights / GPU

Unvalidated until a real-weights GPU run happens (all capability honesty
flags stay conservative until then):

- Real tokenizer/processor outputs: token counts, packed boundaries, label
  masking spans (incl. Qwen assistant-span tokens 77091..151645).
- Actual packed-attention behaviour (FlashAttention varlen / cu_seqlens) for
  both families.
- Real model load, forward/loss values, memory footprint, bf16 numerics.
- Checkpoint round-trip compatibility with real HF save formats
  (`HunYuanVLForConditionalGeneration` weights; Qwen3-VL/MoE weights).
- Any end-to-end training quality claim; nothing in this repo has trained on
  real data.
- The qwen adapter ships without a preflight hook in this wave —
  `adapters preflight qwen_vl_sft` fails with an explicit message until one
  is implemented.

## 8. Compatibility matrix

The authoritative capability record is the registered descriptor — run
`adapters list --json` / `adapters inspect <type> --json`. The table below is
the **expected** record derived from the verified upstream audits (WAVE2
brief); it must be reconciled against the actual descriptors when Wave 2A's
registration lands. Both adapters are mock-verified only, so
`real_weights_validated` and `gpu_validated` must stay `False` in any
descriptor written today.

| Capability | hunyuanocr15_sft | qwen_vl_sft |
|---|---|---|
| supports_full_finetune | True (upstream sft_base.sh tunes vision+mlp+llm) | True (official SFT path; selective flags per-component) |
| supports_selective_finetune | True (per-component tune_mm_* gates) | True (tune_mm_vision/mlp/llm + separate LRs) |
| supports_lora | True (optional LoraConfig r/alpha/dropout) | True (official peft LoraConfig, q/k/v/o_proj) |
| supports_gradient_checkpointing | True (training_args flag) | True (official enable_input_require_grads) |
| supports_packed_sequences | True (FSD pack pipeline, cu_seqlens) | True (--data_flatten / --data_packing) |
| supports_multimodal_batches | True | True |
| supports_local_only_loading | True (enforced) | True (enforced) |
| supports_bf16 | True | True (sft_qwen3_4b.sh) |
| supports_fp16 | False (unverified) | False (unverified) |
| supports_cpu_smoke | False | False |
| supports_resume | False (unverified) | False (unverified) |
| real_weights_validated | False | False |
| gpu_validated | False | False |

The flag values marked "unverified" must remain `False` in the descriptors
until a real run proves them — see §2 honesty rules.

| Property | hunyuanocr15_sft | qwen_vl_sft |
|---|---|---|
| upstream_repository | Tencent-Hunyuan/HunyuanOCR | QwenLM/Qwen3-VL |
| upstream_revision | `c55965d3da1e` | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |
| model_family | hunyuanocr | qwen3_vl |
| trust_remote_code | True | not needed (transformers>=4.57) |
| heavy deps | transformers (trust_remote_code) | transformers>=4.57.0 |
| data modes | raw / packed JSONL | JSON / JSONL (media file paths) |
| preflight hook | run_preflight | none yet |

> The exact flag values are claims recorded in code (`capabilities.py`
> defaults overridden per descriptor). Treat this table as a snapshot;
> `adapters list --json` is authoritative.
