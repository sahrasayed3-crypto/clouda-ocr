# HunyuanOCR-1.5 Upstream Compatibility Snapshot

## Verified upstream (Phase 2)
- repository: Tencent-Hunyuan/HunyuanOCR (official)
- inspected revision: `c55965d3da1e` (latest commit at audit time: "Refactor: rename modules, unify scorer signatures, and clean lint warnings"; repo pushed_at 2026-07-29)
- default branch: main; license: NOASSERTION (custom Tencent license — model/data use must comply)
- files inspected (all HTTP 200 at this revision): README.md, docs/training.md, docs/data_format.md,
  train/train_hunyuan.py, train/argument.py, train/trainer.py, scripts/sft_base.sh, scripts/pack_data.sh,
  tools/pipeline_count_and_pack.py, tools/pack_from_counted.py

## Verified raw OCR JSONL schema (docs/data_format.md)
{"image_path": ["/absolute/path.png"],            # list[str], REQUIRED, absolute paths, usually 1 image
 "conversations": [                                # list[dict], REQUIRED, alternating human/gpt
   {"from": "human", "value": "<image>\nPROMPT"}, # <image> placeholder marks image insertion point
   {"from": "gpt",   "value": "GROUND_TRUTH"}]}

## Verified packing pipeline
- flow: raw JSONL → tokenize+count (parallel) → pack (First-Fit Decreasing, greedy) → packed JSONL
- driver: `scripts/pack_data.sh` → `tools/pipeline_count_and_pack.py`
- config: plain text list of raw JSONL paths (comments with # allowed)
- env knobs: MODEL_PATH (required, tokenizer+processor source), PACK_LEN (default 20480),
  NUM_PROCESSES=32, THREADS_PER_PROCESS=8, COUNT_OUTPUT_DIR, PACK_OUTPUT
- count phase tokenizes TEXT ONLY (images lazy at train time); count cache reused on re-run

## Verified packed JSONL schema
{"packed_samples": [ <raw samples as above> ],
 "cu_seqlens": [0, 4123, 8567, ..., 20351],   # cumulative token boundaries for FlashAttention varlen
 "total_tokens": 20351}                        # sum <= pack_length

## Verified SFT training entry (base model)
- entry: train/train_hunyuan.py via scripts/sft_base.sh
- model class: HunYuanVLForConditionalGeneration (imported from a local `transformers`-style module, trust_remote_code=True)
- tokenizer: AutoTokenizer.from_pretrained(..., trust_remote_code=True)
- processor: AutoProcessor.from_pretrained(...)
- dataset/collator: VLDataset, PackedVLDataCollator, VLDataCollator (train/data_processor.py)
- trainer: Hugging Face Trainer
- model args (train/argument.py): tune_mm_llm=True, tune_mm_mlp=True, tune_mm_vision=True (all default True),
  packed_max_length=2048 (script default differs — docs profile uses 20480), lora_enable=False, lora_r=64,
  lora_alpha=128, lora_dropout=0.0, from_scratch=False
- freezing: safe_save_model_for_hf_trainer + per-component tune_mm_vision/mlp/llm gates in train_hunyuan.py
- gradient checkpointing: supported (training_args.gradient_checkpointing)
- precision: bf16 supported (model cast to torch.bfloat16 when bf16 flag)
- LoRA: optional via LoraConfig (r/alpha/dropout)
- SFT baseline profile (scripts/docs): LR=2e-5, EPOCHS=5, BATCH_SIZE=1, GRAD_ACCUM=1, SAVE_STEPS=200, NPROC_PER_NODE=8

## Notes for the bridge
- official Chinese extraction prompt appears in docs examples but is NOT the only prompt — data is prompt-agnostic;
  conversation structure is what matters. Clouda prompt profiles are therefore compatible.
- absolute image paths required by upstream → Clouda must keep canonical manifests portable and absolutize only in
  generated export artifacts (Phase 4 rule).
- upstream code will NOT be vendored; compatibility adapters only.

## Upstream-compatible SFT baseline (NOT tuned for Clouda Arabic)
Recorded separately from any future Clouda-tuned profile. Values are the
verified upstream defaults, NOT recommendations for Arabic fine-tuning:
- learning_rate=2e-5, epochs=5, batch_size=1, grad_accum=1, save_steps=200
- precision: bf16 (GPU required)
- tuning: vision+projector+llm all enabled (full SFT)
- pack_length=20480

## License
Tencent-Hunyuan/HunyuanOCR ships a custom Tencent license (NOASSERTION on
GitHub). Any use of HunyuanOCR weights, data produced with them, or derived
artifacts MUST comply with that upstream license. Clouda does not vendor,
redistribute, or auto-download any upstream code, weights, or data.

## Exact steps for the FIRST real HunyuanOCR-1.5 training experiment
Prerequisites: local HunyuanOCR-1.5 weights (operator-supplied at
`<local-model-path>`), a CUDA GPU with enough VRAM, transformers with
trust_remote_code support for HunYuanVLForConditionalGeneration.

1. Export: `python -m clouda_training.cli hunyuan export <manifest.jsonl>
   --output exports/hunyuan/raw.jsonl --dataset-id <id> --dataset-version <v>
   --image-root <absolute dataset root>`
   (protection/holdout fail closed; lineage report written next to output).
2. Validate: `python -m clouda_training.cli hunyuan validate-raw
   exports/hunyuan/raw.jsonl --check-images`
3. Packing plan: `python -m clouda_training.cli hunyuan plan
   exports/hunyuan/raw.jsonl` -> data_list.txt + packing_plan.json.
4. Run the OFFICIAL upstream packer from the operator's local HunyuanOCR
   checkout (never auto-run by Clouda):
   `MODEL_PATH=<local model dir> INPUT_LIST=<data_list.txt>
   PACK_OUTPUT=exports/hunyuan/packed.jsonl PACK_LEN=20480 bash scripts/pack_data.sh`
5. Validate packed: `python -m clouda_training.cli hunyuan validate-packed
   exports/hunyuan/packed.jsonl`
6. Preflight: `python -m clouda_training.cli hunyuan preflight
   --model-path <local model dir> --packed-data exports/hunyuan/packed.jsonl`
7. Train via the Clouda runtime with adapter_type=hunyuanocr15_sft
   (local model path only, never downloads).

## Still requiring real weights / GPU (unvalidated)
- real tokenizer/processor outputs (counts, packed token boundaries)
- actual packed-attention behavior (FlashAttention varlen kernels)
- real model load, forward/loss values, memory footprint, bf16 behavior
- real checkpoint compatibility (HF save format for HunYuanVL weights)
- any end-to-end training quality claim

---

# Qwen3-VL Upstream Compatibility Snapshot

## Verified upstream (Wave 2, official-source audit, revision pinned)
- repository: QwenLM/Qwen3-VL (official), `qwen-vl-finetune/`
- inspected revision: `96588727e44c78b25ba03ea03b8e12f7e64fd0da` (2026-01-30)
- Clouda adapter: `adapter_type="qwen_vl_sft"`, `model_family="qwen3_vl"`

## Verified model / processor surface
- model classes: `Qwen3VLForConditionalGeneration` (dense) and
  `Qwen3VLMoeForConditionalGeneration` (MoE); **native in
  transformers >= 4.57.0 — NO `trust_remote_code`** (contrast with Hunyuan).
- processor: `AutoProcessor` -> `Qwen3VLProcessor`
- tokenizer: training uses a separate
  `AutoTokenizer(use_fast=False, padding_side='right')`

## Verified data format (JSON/JSONL)
```json
{"image": "path.jpg",
 "conversations": [
   {"from": "human", "value": "<image>\nquestion"},
   {"from": "gpt",   "value": "answer"}]}
```
- media are FILE PATHS (like Hunyuan raw OCR JSONL; never embedded data URIs)
- the `<image>` tag appears ONLY in the human turn; model-level
  `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` tokens are inserted
  by the processor, not authored in the data

## Verified loss + tuning surface
- loss: causal LM, labels `-100` masked except assistant spans
  (assistant token ids 77091..151645); `IGNORE_INDEX=-100`
- selective tuning: `tune_mm_vision` (model.visual), `tune_mm_mlp`
  (model.visual.merger), `tune_mm_llm` (model.language_model + lm_head);
  separate `vision_tower_lr` / `mm_projector_lr`
- LoRA: official peft `LoraConfig` (targets q/k/v/o_proj)
- gradient checkpointing: official (`enable_input_require_grads`)
- packing: official `--data_flatten` (cu_seqlens-style attention) and
  `--data_packing` (`tools/pack_data.py`) — same varlen concept as Hunyuan's
  packed JSONL, different file format

## Verified SFT reference profile (sft_qwen3_4b.sh — NOT a Clouda recommendation)
- learning rate: 1e-6..2e-7 range per README, bf16
- gradient checkpointing on; `tune_mm_vision=False`, `tune_mm_mlp=True`,
  `tune_mm_llm=True`; `model_max_length=8192`

## Notes for the bridge
- Clouda does NOT vendor upstream code; compatibility adapters only.
- Loading is local-only: weights come from an operator-supplied directory;
  transformers >= 4.57 provides the model classes natively, so no
  trust_remote_code and no auto-download are involved.
- Unlike Hunyuan, no custom-code license gate applies at the code level —
  but upstream model weights carry their own license terms, which any
  operator use must comply with. Clouda never redistributes weights.

## Still requiring real weights / GPU (unvalidated)
- real Qwen3VLProcessor outputs (image token expansion, label spans)
- actual cu_seqlens/flatten-attention behavior
- real model load, forward/loss values, MoE routing behavior, bf16 numerics
- checkpoint compatibility (HF save format for Qwen3-VL / MoE weights)
- any end-to-end training quality claim
