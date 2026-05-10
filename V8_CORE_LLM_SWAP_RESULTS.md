# V8 Core LLM Swap Results

**Status:** Complete. Qwen3 integration/training passed, but checkpoint-100 failed the viable decoder-swap eval gates.  
**Primary backbone:** `Qwen/Qwen3-8B`  
**Excluded route:** Llama 3.1 canary/fallback is intentionally not part of this execution.

## 1. Backbone Choice And Rationale

Selected `Qwen/Qwen3-8B` as the V8 default target.

- 8B-class dense causal LM, matching the intended cost envelope.
- Apache 2.0 model-card licensing.
- HF config reports `model_type=qwen3`, `hidden_size=4096`, `num_hidden_layers=36`, `num_attention_heads=32`, and `num_key_value_heads=8`.
- Qwen3 non-thinking/direct-answer prompt handling is implemented through `chat_template_family=qwen3_non_thinking`.

## 2. Implementation Summary

Files changed:

- `V8_experiment.md`
- `models/llm/backbone.py`
- `models/llm/llama_wrapper.py`
- `models/llm/__init__.py`
- `models/anymal_v2.py`
- `models/anymal_v3.py`
- `data/chat_templates.py`
- `data/data_utils.py`
- `data/instruction_dataset.py`
- `data/laion_dataset.py`
- `modal_train.py`
- `training/finetune.py`
- `evaluation/vqa_eval.py`
- `vqa_checkpoint_eval.py`
- `pope_checkpoint_eval.py`
- `gqa_checkpoint_eval.py`
- `scripts/v8_llm_swap_smoke.py`
- `scripts/launch_v8_eval_bundle.py`
- `modal_v8_llm_swap_smoke.py`
- `scripts/train_pretrain.py`
- `scripts/train_finetune.py`
- `configs/base.yaml`

Implemented:

- `--llm-backbone` support for Modal Stage 1 and Stage 2, including `Qwen/Qwen3-8B`.
- Modal cache path `/checkpoints/qwen3-8b` for Qwen3.
- Guardrails rejecting Llama 3.1 as a V8 option.
- Metadata guard so Qwen runs cannot auto-load legacy LLaMA V3 connectors.
- Per-backbone chat template handling: historical LLaMA 3 format and Qwen3 non-thinking format.
- Placeholder handling that can add/use a dedicated single-token `<image>` for Qwen while preserving LLaMA sentinel behavior.
- V3 exact 128-image-token enforcement for both encoded images and precomputed image-token paths.
- Generic stop-token handling for EOS, `<|eot_id|>`, and `<|im_end|>`.
- LoRA target-module validation before PEFT attaches adapters.
- HF `.generate(inputs_embeds=...)` path with custom greedy fallback if HF generation rejects first-step embeds.
- VQA, POPE, and GQA checkpoint evals can resolve Qwen from checkpoint metadata or `--llm-backbone`.
- Modal smoke/training images pin `transformers>=4.53.0,<5.0.0`; Qwen3 loading required a newer Transformers than the prior image had.

Stage 0 smoke script:

```bash
python3 scripts/v8_llm_swap_smoke.py --llm-backbone Qwen/Qwen3-8B
```

Modal smoke command that passed:

```bash
modal run modal_v8_llm_swap_smoke.py --llm-backbone Qwen/Qwen3-8B --max-new-tokens 4
```

Writes:

- `outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/tokenizer_report.json`
- `outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/prompt_contract_report.json`
- `outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/inputs_embeds_report.json`
- `outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/generation_report.json`

Stage 0 smoke artifacts:

- Local: `outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/`
- Remote: `/checkpoints/outputs/v8_llm_swap_smoke/Qwen__Qwen3-8B/`
- Modal app: `https://modal.com/apps/babakd/main/ap-zZ24FAyAMD2FNMpLQqetFw`

Stage 0 pass summary:

- Tokenizer/model: `Qwen/Qwen3-8B` cached at `/checkpoints/qwen3-8b`, `model_type=qwen3`.
- Config: hidden size 4096, 36 layers, 32 attention heads, 8 KV heads.
- Token IDs: pad 151643 (`<|endoftext|>`), EOS 151645 (`<|im_end|>`), BOS null.
- Image placeholder: added single special token `<|image|>` at id 151669.
- Prompt contract: 128 placeholders found, contiguous, positions 33-160, total input length 176.
- Inputs-embeds parity: `input_ids` vs equivalent `inputs_embeds` last-logit max absolute diff 0.0.
- Random visual replacement forward: passed with logits shape `[1, 176, 151670]`.
- Generation path: HF `.generate(inputs_embeds=...)`.
- Batched left-padded generation: passed with generated shape `[2, 11]`.
- Text-only direct-answer sanity: raw answer `Yes.`, no assistant or thinking markers in decoded answer.
- LoRA targets found: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

## 3. Training Summary

### Stage 1A smoke20

Passed.

Command:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset v3_caption_alignment \
  --pretrain-image-tokens 128 \
  --max-steps 20 \
  --batch-size 4 \
  --pretrain-save-steps 20 \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1a-smoke20
```

- Modal app: `https://modal.com/apps/babakd/main/ap-nVgpoJOAceekL1Lc9gEOsw`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/llo0xsfu`
- Checkpoint: `/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-smoke20/checkpoint-20`
- Dataset: `v3_caption_alignment`, LLaVA pretrain captions from `/checkpoints/llava_data/blip_laion_cc_sbu_558k.json`, images from `/checkpoints/llava_pretrain/images.zip`.
- Usable samples: 555,258; split 527,496 train / 27,762 val.
- Trainables: connector only, 1,616,384,000 trainable params; vision and base LLM frozen.
- Effective batch: 128 (`batch_size=4`, 4 GPUs, grad accumulation 8).
- Learning rate: 2e-4, warmup 2, max steps 20.
- Placeholder diagnostics: sample prompts contained exactly 128 image placeholder tokens.
- Health: initial loss warning only; no persistent alert or stop condition observed.
- Step 10: loss 6.2457, LR 1.26e-4, grad norm 3.2575.
- Step 20: loss 4.5188, LR 2.00e-5, grad norm 0.4594.
- Wall clock: 0.06h.

### Stage 1A fixed300

Passed.

- Modal app: `https://modal.com/apps/babakd/main/ap-91oIR0kjmcdE1RkQfx8oGD`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/b2e6xgpv`
- Checkpoint: `/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-fixed300/checkpoint-300`
- Command uses the same dataset and architecture as smoke20 with `--max-steps 300` and `--pretrain-save-steps 300`.
- Early health: warmup grad spike at step 10 (grad norm 58.2043) then norms settled. No stop condition observed.
- Step 50 eval: average loss 4.4870 over 200 valid batches.
- Step 150 eval: average loss 3.8719.
- Step 200 eval: average loss 3.6935.
- Step 250 eval: average loss 3.5731.
- Step 300 eval: average loss 3.5134.
- Health warning: train/val gap trend warning at steps 150 and 250; it did not become a stop condition.
- Final logged steps: step 260 loss 3.6954 grad norm 0.4947; step 270 loss 3.5028 grad norm 0.5005; step 280 loss 3.6625 grad norm 0.5227; step 290 loss 3.4830 grad norm 0.5125; step 300 loss 3.5421 grad norm 0.4783.
- Checkpoint saved and volume committed successfully.
- Wall clock: 0.74h.

### Stage 1B grounding fixed400

Passed.

Command:

```bash
modal run modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset v3_grounding \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-fixed300/checkpoint-300 \
  --pretrain-image-tokens 128 \
  --max-steps 400 \
  --batch-size 4 \
  --pretrain-save-steps 400 \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300
```

- Modal app: `https://modal.com/apps/babakd/main/ap-nDolaXUje5PT2zE8ZaI7zI`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/7xnrhz2k`
- Checkpoint: `/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300/checkpoint-400`
- Loaded connector weights from Stage 1A fixed300 successfully.
- Dataset: `v3_grounding`.
- Dataset sources: VQAv2 train direct, COCO object/count/color direct, and short LLaVA mix.
- Usable mixture length: 27,446,000; split 26,073,700 train / 1,372,300 val.
- Dataset diagnostics: sampled prompts have exactly 128 image placeholder tokens.
- Labels: short direct-answer spans only; sampled supervised-token rates roughly 1.4-1.7 percent for examples, 0.6 percent for the first batch.
- Trainables: connector only, 1,616,384,000 trainable params; vision and base Qwen frozen.
- Effective batch: 128 (`batch_size=4`, 4 GPUs, grad accumulation 8).
- Learning rate: 2e-4, warmup 40, max steps 400.
- Health: no startup contract failure or stop condition observed.
- Initial warning: initial loss 7.8886 was 32.9 percent below ln(vocab)=11.7618, expected given very short direct-answer labels.
- Step 50 eval: average loss 0.8177 over 200 valid batches.
- Step 60: loss 0.6949, LR 1.99e-4, grad norm 0.5231.
- Step 70: loss 0.8410, LR 1.97e-4, grad norm 0.5566.
- Step 80: loss 0.4832, LR 1.95e-4, grad norm 0.3057.
- Step 90: loss 0.8203, LR 1.92e-4, grad norm 0.3114.
- Step 100: loss 0.6568, LR 1.88e-4, grad norm 0.3694.
- Step 150 eval: average loss 0.7183 over 200 valid batches.
- Step 200 eval: average loss 0.6846 over 200 valid batches.
- Step 250 eval: average loss 0.6556 over 200 valid batches.
- Step 300 eval: average loss 0.6340 over 200 valid batches.
- Step 350 eval: average loss 0.5974 over 200 valid batches.
- Step 400 eval: average loss 0.5782 over 200 valid batches.
- Step 110: loss 0.4751, LR 1.84e-4, grad norm 0.3087.
- Step 120: loss 0.6506, LR 1.79e-4, grad norm 0.3431.
- Step 130: loss 0.5434, LR 1.74e-4, grad norm 0.3518.
- Step 140: loss 0.6373, LR 1.68e-4, grad norm 0.3449.
- Step 150: loss 0.6446, LR 1.62e-4, grad norm 0.3004.
- Step 160: loss 0.6982, LR 1.55e-4, grad norm 0.4007.
- Step 170: loss 0.7954, LR 1.48e-4, grad norm 1.3089.
- Step 180: loss 0.6873, LR 1.41e-4, grad norm 0.3111.
- Step 190: loss 0.5261, LR 1.33e-4, grad norm 0.4793.
- Step 200: loss 0.7343, LR 1.26e-4, grad norm 0.2730.
- Step 210: loss 0.3339, LR 1.18e-4, grad norm 0.2966.
- Step 220: loss 0.6239, LR 1.10e-4, grad norm 0.3589.
- Step 230: loss 0.5328, LR 1.02e-4, grad norm 0.2702.
- Step 240: loss 0.7787, LR 9.44e-5, grad norm 0.3585.
- Step 250: loss 0.7651, LR 8.67e-5, grad norm 0.2374.
- Step 260: loss 0.4202, LR 7.92e-5, grad norm 0.3881.
- Step 270: loss 0.5874, LR 7.20e-5, grad norm 0.2690.
- Step 280: loss 0.7207, LR 6.50e-5, grad norm 0.2749.
- Step 290: loss 0.7196, LR 5.84e-5, grad norm 0.2636.
- Step 300: loss 0.6050, LR 5.21e-5, grad norm 0.2307.
- Step 310: loss 0.7771, LR 4.64e-5, grad norm 0.3370.
- Step 320: loss 0.6174, LR 4.11e-5, grad norm 0.3229.
- Step 330: loss 0.6923, LR 3.63e-5, grad norm 0.3033.
- Step 340: loss 0.5241, LR 3.21e-5, grad norm 0.2813.
- Step 350: loss 0.4508, LR 2.84e-5, grad norm 0.2676.
- Step 360: loss 0.3972, LR 2.54e-5, grad norm 0.2021.
- Step 370: loss 0.6579, LR 2.30e-5, grad norm 0.2625.
- Step 380: loss 0.3952, LR 2.13e-5, grad norm 0.2304.
- Step 390: loss 0.5419, LR 2.03e-5, grad norm 0.2787.
- Step 400: loss 0.3832, LR 2.00e-5, grad norm 0.2263.
- Checkpoint saved and volume committed successfully.
- Wall clock: 2.03h.

## 3.5 V7 Report Gap Work

Completed in parallel:

- B1 checkpoint-100 VQA leakage audit: 865 eval image IDs, zero exact val2014 overlap, zero numeric-ID overlap, exact and numeric gates pass. Sources checked included the robust calibration sources plus `/checkpoints/vqa_data/vqa_train2014_counterfactual_grounding_30000.json` and `/checkpoints/llava_data/coco_pope_style_absence_train2017_10000.json`.
- A1 checkpoint-50 VQA leakage audit: 865 eval image IDs, zero exact val2014 overlap, zero numeric-ID overlap, exact and numeric gates pass against the default robust calibration sources.
- Metadata extracted from existing V7 eval artifacts:
  - V3 robust: `architecture=anymal_v3`, `connector_type=perceiver_resampler`, 128 image tokens, 6 layers, 16 heads, FF mult 4, direct projection to LLM hidden space.
  - B1 checkpoint-100: same V3 connector metadata as V3 robust, `question_conditioning=null`.
  - A1 checkpoint-50: `connector_type=question_conditioned_perceiver_resampler`, 128 image tokens, 6 layers, 16 heads, FF mult 4, direct projection, `question_conditioning=pooled_prompt_embedding_additive_latent_shift`.

Still outstanding from the V7 gap list:

- W&B health table for B1 and A1. Exact run IDs were not present in local V7 artifacts.
- A1 checkpoint-225 explanation beyond the existing training/eval metadata.

### Stage 2

Passed.

Command:

```bash
modal run modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300/checkpoint-400 \
  --freeze-connector \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003
```

- Modal app: `https://modal.com/apps/babakd/main/ap-jb5JC3IIuA4zfGxbZYRAXg`
- W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/b3s0vkcz`
- Checkpoint: `/checkpoints/finetune-output/v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100`
- Loaded connector weights from Stage 1B checkpoint-400 successfully.
- Dataset: `v5_semantic_calibration_robust`.
- Dataset sources: Mix-665K direct-answer subset, VQAv2 yes/no-number-other slices, and COCO object/count/color direct data.
- Usable mixture length: 27,446,000; split 26,073,700 train / 1,372,300 val.
- Dataset diagnostics: sampled prompts have exactly 128 image placeholder tokens.
- Labels: short direct-answer spans only; first-batch supervised-token rate about 0.2-0.3 percent.
- Trainables: decoder LoRA only, 174,587,904 trainable LoRA params; connector, vision tower, and base Qwen frozen.
- Effective batch: 64 (`batch_size=4`, 4 GPUs, grad accumulation 16).
- Learning rate: 1e-5, cosine decay to 1e-6 floor, warmup 10, max steps 100.
- Loss scale: 0.03.
- Health: no NaNs, placeholder failures, max-token failures, or stop condition observed.
- Initial warning: initial loss 0.5414 was far below ln(vocab)=11.7618, expected for very short direct-answer labels under loss scale 0.03.
- Low-gradient warning: step 13 reported gradients below 0.01 for 10 consecutive steps. This matched the established scaled LoRA-only regime; training stayed numerically stable.
- Step 50 eval: average loss 0.5361 over 200 valid batches.
- Step 100 eval: average loss 0.5346 over 200 valid batches.
- Step 90: loss 0.4633, LR 1.27e-6, grad norm 0.0032.
- Step 91: loss 0.4519, LR 1.22e-6, grad norm 0.0042.
- Step 92: loss 0.5365, LR 1.17e-6, grad norm 0.0044.
- Step 93: loss 0.5702, LR 1.13e-6, grad norm 0.0048.
- Step 94: loss 0.5355, LR 1.10e-6, grad norm 0.0046.
- Step 95: loss 0.4515, LR 1.07e-6, grad norm 0.0044.
- Step 96: loss 0.4553, LR 1.04e-6, grad norm 0.0051.
- Step 97: loss 0.6896, LR 1.02e-6, grad norm 0.0038.
- Step 98: loss 0.5477, LR 1.01e-6, grad norm 0.0062.
- Step 99: loss 0.4932, LR 1.00e-6, grad norm 0.0045.
- Step 100: loss 0.5391, LR 1.00e-6, grad norm 0.0037.
- Final metrics: train loss 0.5246.
- Checkpoint saved and volume committed successfully.
- Wall clock: 1.62h.

### V8 eval bundle

Launched after Stage 2 checkpoint-100:

```bash
python3 scripts/launch_v8_eval_bundle.py \
  --checkpoint /checkpoints/finetune-output/v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100 \
  --label "Qwen3 V8" \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --artifact-prefix v8_qwen3_v3_robustcal_ckpt100 \
  --remote-dir /checkpoints/v8_remote \
  --parallelism 5
```

The initial `--background` launch returned function-call IDs but did not leave active Modal tasks, so the final recorded evals used foreground Modal calls with local parallelism. Fetched artifacts:

- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_clean_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_blank_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_shuffled_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_wrongimage_sameanswertype_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_mildblur_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_centercrop90_currentcache.json`
- `/checkpoints/v8_remote/vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_translate5pct_currentcache.json`
- `/checkpoints/v8_remote/pope_eval_v8_qwen3_v3_robustcal_ckpt100_currentcache.json`
- `/checkpoints/v8_remote/gqa_eval_v8_qwen3_v3_robustcal_ckpt100_currentcache.json`

Local copies and derived analysis artifacts:

- `outputs/v8_remote/v8_remote/*.json`
- `outputs/v8_analysis/qwen3_v8_eval_summary.txt`
- `outputs/v8_analysis/qwen3_v8_clean_answer_analysis.txt`
- `outputs/v8_analysis/qwen3_correct_v3_wrong.json`
- `outputs/v8_analysis/v3_correct_qwen3_wrong.json`
- `outputs/v8_analysis/qwen3_correct_b1_wrong.json`
- `outputs/v8_analysis/b1_correct_qwen3_wrong.json`
- `outputs/v8_analysis/qwen3_v8_vqa_leakage_audit.json`
- `outputs/v8_analysis/qwen3_v8_pope_leakage_audit.json`
- `outputs/v8_analysis/qwen3_v8_gqa_leakage_audit.json`

## 4. Metadata Summary

Stage 2 checkpoint-100 `model_meta.json` key fields:

- `architecture`: `anymal_v3`
- `vision_tower`: `SigLIP2-So400m-384`
- `vision_encoder_type`: `siglip2`
- `connector_type`: `perceiver_resampler`
- `connector_layers`: 6
- `connector_heads`: 16
- `connector_ff_mult`: 4
- `project_directly_to_llm_dim`: true
- `question_conditioning`: null
- `image_tokens`: 128
- `min_image_tokens`: 128
- `max_image_tokens`: 128
- `llm_backbone`: `Qwen/Qwen3-8B`
- `llm_model_name`: `/checkpoints/qwen3-8b`
- `llm_model_type`: `qwen3`
- `llm_hidden_size`: 4096
- `llm_num_hidden_layers`: 36
- `llm_num_attention_heads`: 32
- `llm_num_key_value_heads`: 8
- `tokenizer_name`: `/checkpoints/qwen3-8b`
- `chat_template_family`: `qwen3_non_thinking`
- `pad_token_id`: 151643
- `bos_token_id`: null
- `eos_token_id`: 151645
- `image_placeholder_token`: `<|image|>`
- `image_placeholder_token_id`: 151669
- `image_placeholder_count`: 128
- `added_special_tokens`: [`<|image|>`]
- `padding_side_for_generation`: `left`
- `generation_path`: `hf_generate`
- `stage1_connector_init`: `fresh_random`
- `stage2_lora_target_modules`: [`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`]
- `llm_base_weights_saved`: false
- `llm_checkpoint_saved`: true

## 5. Evaluation Table

| Model | Clean | Blank | Shuffle | Wrong | Blank gap | Shuffle gap | Wrong gap | POPE | GQA | EOS | Max hit | Prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 robust aligned | 62.967 | 39.733 | 37.367 | 38.900 | 23.233 | 25.600 | 24.067 | 77.100 | 43.800 | 1.000 | 0.000 | 0.000 |
| B1 control | 63.267 | 39.433 | 36.633 | 37.833 | 23.833 | 26.633 | 25.433 | 75.000 | 44.600 | 1.000 | 0.000 | 0.000 |
| Qwen3 V8 | 55.800 | 38.933 | 39.333 | 39.633 | 16.867 | 16.467 | 16.167 | 74.500 | 35.200 | 1.000 | 0.000 | 0.000 |

Perturbation checks:

| Model | Mild blur | Center crop 90 | Translate 5 pct |
|---|---:|---:|---:|
| Qwen3 V8 | 55.733 | 55.333 | 55.733 |

Key deltas vs V3 robust:

- Clean: -7.167.
- Blank: -0.800.
- Shuffle: +1.966.
- Wrong-image same-answer-type: +0.733.
- Blank gap: -6.367.
- Shuffle gap: -9.133.
- Wrong gap: -7.900.
- POPE: -2.600.
- GQA: -8.600.

## 6. Grounding Interpretation

Qwen3 V8 is an integration success but not a quality success.

- Clean VQAv2 fell from 62.967 to 55.800.
- Blank-image accuracy fell slightly, but shuffled and wrong-image accuracies increased versus V3 robust. That means the lower clean score is not a cleaner grounding tradeoff.
- All three grounding gaps regressed materially: blank gap 16.867, shuffle gap 16.467, wrong-image gap 16.167.
- POPE fell to 74.500, below both V3 robust and the 76.0 viable-swap gate.
- GQA fell to 35.200, far below the 43.0 viable-swap gate.
- Hygiene is clean: strict-clean gap 0.000, EOS about 1.000, max-token-hit 0.000 except blank at 0.001, assistant-prefix 0.000.

Answer distribution on clean VQAv2:

- Top cleaned answers: `yes` 240, `no` 165, `1` 42, `white` 30, `2` 25, `3` 23, `blue` 15, `black` 15.
- Answer-kind rates match answer type almost perfectly: yes/no questions produce yes/no answers, number questions produce number answers, and other questions produce other answers.
- Pairwise clean deltas: Qwen3 correct / V3 wrong = 79 rows; V3 correct / Qwen3 wrong = 141 rows.
- Pairwise clean deltas vs B1: Qwen3 correct / B1 wrong = 79 rows; B1 correct / Qwen3 wrong = 144 rows.

Interpretation: the V8 Qwen path learned a mechanically healthy direct-answer interface, but the visual-to-decoder alignment is weaker than the incumbent. The failure pattern is closer to connector/decoder alignment loss than to prompt leakage or answer-format collapse.

## 7. Leakage Audit

Leakage audits passed against the robust calibration training-source set:

| Artifact group | Eval image IDs | Exact val2014 overlap | Numeric ID overlap | Gate |
|---|---:|---:|---:|---|
| VQAv2 V8 bundle | 865 | 0 | 0 | PASS |
| POPE | 455 | 0 | 0 | PASS |
| GQA | 263 | 0 | 0 | PASS |

Training sources checked:

- `/checkpoints/vqa_data/vqa_train2014_direct_yes_no_balanced_40000.json`
- `/checkpoints/vqa_data/vqa_train2014_direct_number_50000.json`
- `/checkpoints/vqa_data/vqa_train2014_direct_other_80000.json`
- `/checkpoints/llava_data/coco_object_direct_train2017.json`
- `/checkpoints/llava_data/mix665k_direct_answer_filtered.json`

## 8. Decision

- Integration pass/fail: pass.
- Viable decoder swap: no.
- Decoder improvement: no.
- Promotion candidate: no; larger confirmation is not warranted for this checkpoint.

Gate rationale:

- Tier 0 integration passes: smoke, training, generation hygiene, EOS, max-token-hit, and assistant-prefix gates passed.
- Tier 1 viable swap fails: clean 55.800 is below 61.5; POPE 74.500 is below 76.0; GQA 35.200 is below 43.0; all grounding gaps are below gates.
- Tier 2 improvement fails: clean, POPE, GQA, and grounding gaps all regress versus V3 robust.

Recommended next action: do not run larger n=3000 confirmation for this checkpoint. Diagnose Stage 1 alignment and Qwen-specific scaling first: connector output norms, hidden-state scale, Stage 1 data/learning-rate transfer, and whether Qwen needs longer or differently weighted connector alignment before robust Stage 2.

## 9. Local Verification

Passed:

```bash
python3 -m py_compile models/llm/backbone.py models/llm/llama_wrapper.py models/anymal_v2.py models/anymal_v3.py data/chat_templates.py data/data_utils.py data/instruction_dataset.py data/laion_dataset.py evaluation/vqa_eval.py modal_train.py modal_v8_llm_swap_smoke.py vqa_checkpoint_eval.py pope_checkpoint_eval.py gqa_checkpoint_eval.py scripts/v8_llm_swap_smoke.py scripts/launch_v8_eval_bundle.py scripts/train_pretrain.py scripts/train_finetune.py training/finetune.py
```

Blocked locally:

```bash
pytest tests/test_evaluation.py tests/test_training.py::TestDatasets::test_instruction_dataset tests/test_training.py::TestDatasets::test_instruction_dataset_uniform_token_policy -q
```

Reason: this local Python environment has no `torch` installed (`ModuleNotFoundError: No module named 'torch'`). Modal image dependencies include torch and the Stage 0/Stage 1A checks above ran there.

No local test suite was run because this Python environment lacks `torch`; all GPU-dependent smoke, training, and eval paths ran on Modal.
