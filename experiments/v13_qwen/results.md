# V13 Qwen3 Substrate-Break Results

Date: 2026-05-14

## Bottom Line

V13 did not produce a candidate that robustly beats the V11 Qwen frontier.
The strongest new substrate, repaired V3 visual cross-attention with V11
teacher KL, can imitate V11 on cheap GQA slices but does not improve matched
search/confirm evals. V11 remains the best overall Qwen checkpoint:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

## Baselines

| Model | Checkpoint | Key GQA Result | Notes |
| --- | --- | ---: | --- |
| LLaMA/V3 robust reference | `/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100` | 43.7 n1000 | From V13 plan baseline. |
| V9 Qwen scale-calibrated | `/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105` | 43.1 n1000 | Stable fallback. |
| V11 Qwen frontier | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` | 44.9 n1000 / 42.6 n3000 | Best known Qwen basin. |

Matched V11 reruns:

| Eval | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| GQA search | `/tmp/v11_frontier_gqa_search1000.json` | 44.900 | 449 / 1000 | 65.527 | 33.744 |
| GQA confirm | `/tmp/v11_frontier_gqa_confirm3000.json` | 42.600 | 1278 / 3000 | 60.758 | 32.751 |
| GQA n200 seed42 | `/tmp/gqa_v11_c1_n200_seed42.json` | 46.000 | 92 / 200 | 62.500 | 35.000 |

GQA split metadata uses the Hugging Face `Mineru/GQA` `testdev_balanced`
mirror with seeded windows:

```text
split_definition_version = gqa_hf_seeded_windows_v1
selection_seed = 42
search_probe = offset 0
```

The GQA checkpoint evaluator now records Wilson/binomial and bootstrap CIs.

## Full Run Ledger

This section records the execution ledger for V13. Entries marked
`pre-existing artifact inspected` were discovered in the workspace or Modal
volume during the campaign; the result artifact/checkpoint was available, but
the original Modal app URL, wall time, or exact local launch command was not
always recoverable from the persisted JSON. In those cases, only parameters
that are present in the checkpoint/result metadata or preserved execution notes
are listed.

### Training / Continuation Runs

| ID | Status | Run / Checkpoint | Duration | Key Parameters | Outcome |
| --- | --- | --- | --- | --- | --- |
| T0 | completed | `/checkpoints/pretrain-output/v13-qwen3-nobranch-c1diag-scale105-lr5e5-200/checkpoint-200` | not captured | Qwen3 V3, no learned 2D branch, C1 diagnostic, `max_steps=200`, `learning_rate=5e-5`, materialized/eval scale `1.125` | GQA n200 ladder peaked at 42.0; search 42.3, confirm 41.033. Negative. |
| T1 | pre-existing artifact inspected | spatial-grid 256/512 family | not captured | spatial-grid substrate, 256/512 visual tokens, teacher/contrastive variants | Mostly 25-34 GQA n200; best noted 512 spatial-contrastive step200 n1000 was 31.8. Negative. |
| T2 | pre-existing artifact inspected | AnyRes-lite 256 family | not captured | AnyRes / MLP pass-through, 256-token lite control/spatial-contrastive variants | Around 25-30 GQA n200. Negative. |
| T3 | completed | `/checkpoints/pretrain-output/v13-qwen3-tail128-v11warm-kl200-lr5e5/checkpoint-200` | not captured | V11-warm hybrid tail, 128 tail tokens, KL branch, `max_steps=200`, `learning_rate=5e-5` | n200 44.0; search 40.0. Negative. |
| T4 | completed | `/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-kl300-lr5e5-ga4` | not captured | repaired gated visual cross-attention, upper layers `18,22,26,30,34`, gate `1e-4`, KL weight `1.0`, `learning_rate=5e-5`, `gradient_accumulation=4`, `max_steps=300` | n200 ladder 46.0 -> 45.0; imitated V11 but no gain. |
| T5 | completed | `/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300` | not captured | repaired gated visual cross-attention, upper layers `18,22,26,30,34`, gate `1e-3`, KL weight `5.0`, `learning_rate=2e-5`, `gradient_accumulation=4`, `max_steps=300`, trainables `42,024,965` vxattn params | Best V13 substrate. Step50 n200 46.5; search 44.6; confirm 42.433. Did not beat V11. |
| T6 | completed after one failed launch and code fix | `/checkpoints/finetune-output/v13-qwen3-vxattn-step50-visualintake-qv-lora-r8-gqa-stage2-50/checkpoint-50` | not captured | Stage2 from T5 step50, q/v LoRA only, rank 8, alpha 16, dropout 0.05, `learning_rate=5e-6`, `lora_learning_rate=5e-6`, `loss_scale=0.03`, `gradient_accumulation=8`, `max_steps=50`, connector/vxattn frozen | n200 46.0. No gain. |
| T7 | smoke completed | `/checkpoints/pretrain-output/v13-qwen3-vxattn-step50-visionattnlast1-smoke1-lr1e6/checkpoint-1` | not captured | one-step Stage1 vision-prefix smoke from T5 step50, connector frozen, vxattn trainable, SigLIP layer 26 attention/norm trainable, `learning_rate=1e-6`, `max_steps=1` | Verified trainability/checkpoint contract: 12 SigLIP tensors, `5,317,632` vision params plus `42,024,965` vxattn params. |
| T8 | completed | `/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-siglip-last1-attnnorm-freezeconn-lr3e6-step50` | 0.31 hr training wall time | Stage1 from T5 step50, dataset `v12_qwen_gqa_spatial_contrastive_stage1b`, connector frozen, vxattn upper layers trainable, SigLIP layer 26 self-attn + ln1/ln2 trainable, KL weight `0.5`, KL direction `student_to_teacher`, teacher checkpoint V11, `learning_rate=3e-6`, `gradient_accumulation=32`, `max_steps=50`, save steps 25 | step25 n200 46.0; step50 n200 46.5. Tied T5 small slice, no expansion. |

Exact preserved launch commands for the main runs launched in this pass:

```bash
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --dataset v10_qwen_gqa_antishuffle_stage1b \
  --pretrain-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --pretrain-image-tokens 128 \
  --v3-visual-cross-attention-mode gated \
  --v3-visual-cross-attention-layers 18,22,26,30,34 \
  --v3-visual-cross-attention-gate-init 0.001 \
  --v3-visual-cross-attention-freeze-connector \
  --pretrain-teacher-kl-weight 5.0 \
  --pretrain-teacher-kl-image-tokens 128 \
  --max-steps 300 \
  --batch-size 1 \
  --pretrain-gradient-accumulation-steps 4 \
  --pretrain-save-steps 50 \
  --learning-rate 2e-5 \
  --run-name v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300
```

```bash
modal run modal_train.py \
  --stage finetune \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300/checkpoint-50 \
  --pretrain-image-tokens 128 \
  --dataset v9_qwen_gqa_preserving_stage2 \
  --max-steps 50 \
  --batch-size 1 \
  --freeze-connector \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --lora-learning-rate 5e-6 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --run-name v13-qwen3-vxattn-step50-visualintake-qv-lora-r8-gqa-stage2-50
```

```bash
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --dataset v12_qwen_gqa_spatial_contrastive_stage1b \
  --pretrain-checkpoint /checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300/checkpoint-50 \
  --pretrain-image-tokens 128 \
  --v3-visual-cross-attention-mode gated \
  --v3-visual-cross-attention-layers 18,22,26,30,34 \
  --v3-visual-cross-attention-gate-init 0.001 \
  --v3-visual-cross-attention-freeze-connector \
  --connector-trainable-prefixes freeze \
  --vision-trainable-prefixes model.encoder.layers.26.self_attn,model.encoder.layers.26.layer_norm1,model.encoder.layers.26.layer_norm2 \
  --pretrain-teacher-kl-weight 0.5 \
  --pretrain-teacher-kl-direction student_to_teacher \
  --pretrain-teacher-kl-image-tokens 128 \
  --pretrain-teacher-kl-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --max-steps 50 \
  --batch-size 1 \
  --pretrain-gradient-accumulation-steps 32 \
  --pretrain-save-steps 25 \
  --learning-rate 3e-6 \
  --run-name v13-qwen3-vxattn-upper-siglip-last1-attnnorm-freezeconn-lr3e6-step50
```

### Evaluation Runs

| ID | Command Shape / Parameters | Artifact | Result |
| --- | --- | --- | --- |
| E0 | V11 GQA search, `testdev_balanced`, seed 42, offset 0, n1000, batch 8 | `/tmp/v11_frontier_gqa_search1000.json` | 44.900, 449/1000, yes/no 65.527, other 33.744, CI 41.84-48.00 |
| E1 | V11 GQA confirm, `testdev_balanced`, seed 42, n3000, batch 8 | `/tmp/v11_frontier_gqa_confirm3000.json` | 42.600, 1278/3000, yes/no 60.758, other 32.751, CI 40.84-44.38 |
| E2 | V11 GQA n200 non-thinking, seed 42 | `/tmp/gqa_v11_c1_n200_seed42.json` | 46.000, 92/200, yes/no 62.5, other 35.0 |
| E3 | V11 GQA n200 thinking prefill, `qwen3_thinking`, max new tokens 128 | `/tmp/v11_frontier_gqa_thinking_n200.json` | 33.500, 67/200, thinking tag rate 0.075, negative |
| E4 | no-branch C1 ladder, checkpoints 50/100/150/200, n200 | `/tmp/nobranch_c1diag_*_scale1125_gqa_n200.json` | 42.0, 40.5, 40.5, 42.0 |
| E5 | no-branch C1 search, n1000 | `/tmp/nobranch_c1diag_200_scale1125_gqa_search1000.json` | 42.300, 423/1000 |
| E6 | no-branch C1 confirm, n3000 | `/tmp/nobranch_c1diag_200_scale1125_gqa_confirm3000.json` | 41.033, 1231/3000 |
| E7 | tail128 V11-warm GQA n200 | `/tmp/gqa_tail128_kl200_n200.json` | 44.000 |
| E8 | tail128 V11-warm GQA search, n1000 | `/tmp/tail128_v11warm_kl200_gqa_search1000.json` | 40.000, 400/1000 |
| E9 | first repaired vxattn ladder, checkpoints 50-300, n200 | `/tmp/v13_vxattn_upper_kl300_step*_gqa_n200.json` | 46.0, 45.0, 45.5, 45.0, 44.5, 45.0 |
| E10 | second repaired vxattn ladder, checkpoints 50-300, n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step*_gqa_n200.json` | 46.5, 45.5, 45.0, 45.5, 45.5, 45.0 |
| E11 | second repaired vxattn step50 search, n1000 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step50_gqa_search1000.json` | 44.600, 446/1000, yes/no 65.242, other 33.436, CI 41.546-47.696 |
| E12 | second repaired vxattn step50 confirm, n3000 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step50_gqa_confirm3000.json` | 42.433, 1273/3000, yes/no 60.758, other 32.494, CI 40.676-44.210 |
| E13 | Stage2 q/v LoRA n200 | `/tmp/v13_vxattn_step50_qv_lora_r8_stage2_step50_gqa_n200.json` | 46.000, 92/200 |
| E14 | SigLIP adapter step25 n200 | `/tmp/v13_vxattn_siglip_attnnorm_step25_gqa_n200.json` | 46.000, 92/200, yes/no 62.5, other 35.0 |
| E15 | SigLIP adapter step50 n200 | `/tmp/v13_vxattn_siglip_attnnorm_step50_gqa_n200.json` | 46.500, 93/200, yes/no 63.75, other 35.0 |
| E16 | ChartQA smoke, V11, train n1 | `/tmp/v11_chartqa_smoke1.json` | passed harness/model-load smoke |
| E17 | ChartQA val n200, V11 | `/tmp/v11_chartqa_val_n200.json` | 6.000 exact match, 12/200, EOS 0.94 |
| E18 | ChartQA val n200, V13 vxattn step50 | `/tmp/v13_vxattn_step50_chartqa_val_n200.json` | 5.500 exact match, 11/200, EOS 0.94 |
| E19 | GQA n200 with predictions, V11 | `/tmp/v11_gqa_n200_pred.json` | 46.000, 200 prediction rows |
| E20 | GQA n200 with predictions, V13 vxattn step50 | `/tmp/v13_vxattn_step50_gqa_n200_pred.json` | 46.500, 200 prediction rows |
| E21 | Pairwise taxonomy on E19/E20 | `/tmp/v13_gqa_pairwise_taxonomy_n200/` | V13 had one extra yes/no object-presence correct item; otherwise identical |
| E22 | Leakage audit on E19/E20 vs V9/V10/V12 GQA train sources | `/tmp/v13_gqa_n200_leakage_audit.json` | PASS, zero exact/numeric/raw overlaps |

Exact preserved evaluation commands for the final expanded/taxonomy/audit pass:

```bash
modal run evaluation/checkpoint_eval/chartqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --candidate-label v11_chartqa_val_n200 \
  --candidate-architecture v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --split val \
  --max-samples 200 \
  --batch-size 8 \
  --seed 42 \
  --prediction-samples 10 \
  --output /tmp/v11_chartqa_val_n200.json

modal run evaluation/checkpoint_eval/chartqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300/checkpoint-50 \
  --candidate-label v13_vxattn_step50_chartqa_val_n200 \
  --candidate-architecture v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --split val \
  --max-samples 200 \
  --batch-size 8 \
  --seed 42 \
  --prediction-samples 10 \
  --output /tmp/v13_vxattn_step50_chartqa_val_n200.json

modal run evaluation/checkpoint_eval/gqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --candidate-label v11_gqa_n200_pred \
  --candidate-architecture v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gqa-split testdev_balanced \
  --seed 42 \
  --sample-offset 0 \
  --max-samples 200 \
  --eval-slice-name search_probe \
  --batch-size 8 \
  --prediction-samples 200 \
  --output /tmp/v11_gqa_n200_pred.json

modal run evaluation/checkpoint_eval/gqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300/checkpoint-50 \
  --candidate-label v13_vxattn_step50_gqa_n200_pred \
  --candidate-architecture v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gqa-split testdev_balanced \
  --seed 42 \
  --sample-offset 0 \
  --max-samples 200 \
  --eval-slice-name search_probe \
  --batch-size 8 \
  --prediction-samples 200 \
  --output /tmp/v13_vxattn_step50_gqa_n200_pred.json

python3 scripts/analyze_gqa_pairwise.py \
  --artifact v11=/tmp/v11_gqa_n200_pred.json \
  --artifact v13_vxattn=/tmp/v13_vxattn_step50_gqa_n200_pred.json \
  --comparison v11:v13_vxattn \
  --comparison v13_vxattn:v11 \
  --output-dir /tmp/v13_gqa_pairwise_taxonomy_n200

modal run scripts/audit_v9_leakage.py \
  --eval-artifacts /tmp/v11_gqa_n200_pred.json,/tmp/v13_vxattn_step50_gqa_n200_pred.json \
  --train-sources /checkpoints/gqa_data/v9_qwen_gqa_train_balanced_10000.json,/checkpoints/gqa_data/v10_qwen_gqa_contrastive_train_balanced_10000.json,/checkpoints/gqa_data/v12_qwen_gqa_spatial_relation_train_balanced_10000.json,/checkpoints/gqa_data/v12_qwen_gqa_contrastive_spatial_relation_train_balanced_10000.json \
  --output-json /tmp/v13_gqa_n200_leakage_audit.json
```

## Phase 0 Diagnostics

### No-Branch C1

Checkpoint:

```text
/checkpoints/pretrain-output/v13-qwen3-nobranch-c1diag-scale105-lr5e5-200/checkpoint-200
```

| Eval | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| GQA search | `/tmp/nobranch_c1diag_200_scale1125_gqa_search1000.json` | 42.300 | 423 / 1000 | 58.689 | 33.436 |
| GQA confirm | `/tmp/nobranch_c1diag_200_scale1125_gqa_confirm3000.json` | 41.033 | 1231 / 3000 | 57.536 | 32.082 |

Interpretation: V11 was not reproduced by C1 data/objective without the branch.
The disabled-at-inference branch likely changed the optimization path.

### Qwen Thinking Mode

| Model / Prompt | Artifact | GQA n200 | Notes |
| --- | --- | ---: | --- |
| V11 non-thinking | `/tmp/gqa_v11_c1_n200_seed42.json` | 46.000 | Clean short answers. |
| V11 thinking prefill | `/tmp/v11_frontier_gqa_thinking_n200.json` | 33.500 | `thinking_tag_rate=0.075`; answer quality regressed. |

Interpretation: thinking-mode prompting is negative for this evaluator and is
not a promotion path.

## Substrate Tracks

### Track 1: Spatial-Preserving Grid

Existing Modal artifacts for 256/512-token spatial-grid variants were inspected.
They received real smoke/short-to-medium exposure but did not reach V11
imitation health. The best noted 512 spatial-contrastive run reached only
31.8 GQA n1000, with n200 probes mostly in the 25-34 range. This track is
completed-negative for the current implementation.

### Track 2: Hybrid V11 + Spatial Tail

Checkpoint:

```text
/checkpoints/pretrain-output/v13-qwen3-tail128-v11warm-kl200-lr5e5/checkpoint-200
```

| Eval | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| GQA n200 | earlier probe | 44.000 | - | - | - |
| GQA search | `/tmp/tail128_v11warm_kl200_gqa_search1000.json` | 40.000 | 400 / 1000 | 59.259 | 29.584 |

Interpretation: adding a tail to the V11 path did not preserve the V11 basin.

### Track 3: AnyRes / MLP Pass-Through

AnyRes-lite 256-token control and spatial-contrastive runs were found in the
existing V13 artifacts. They remained around 25-30 GQA n200 and did not pass
the imitation screen. A checkpoint metadata compatibility issue for AnyRes
`None` fields was fixed during this campaign so these checkpoints can be
loaded safely.

### Track 4: Repaired Visual Cross-Attention

The teacher-KL path for V3 visual cross-attention was fixed and then tested
with V11 as teacher.

First repaired run:

```text
/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-kl300-lr5e5-ga4
```

GQA n200 ladder: step50 46.0, step100 45.0, step150 45.5, step200 45.0,
step250 44.5, step300 45.0.

Second repaired run:

```text
/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-gate1e3-kl5-lr2e5-ga4-300
```

| Checkpoint | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| step50 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step50_gqa_n200.json` | 46.500 | 93 / 200 | 63.750 | 35.000 |
| step100 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step100_gqa_n200.json` | 45.500 | - | - | - |
| step150 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step150_gqa_n200.json` | 45.000 | - | - | - |
| step200 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step200_gqa_n200.json` | 45.500 | - | - | - |
| step250 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step250_gqa_n200.json` | 45.500 | - | - | - |
| step300 n200 | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step300_gqa_n200.json` | 45.000 | - | - | - |
| step50 search | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step50_gqa_search1000.json` | 44.600 | 446 / 1000 | 65.242 | 33.436 |
| step50 confirm | `/tmp/v13_vxattn_gate1e3_kl5_lr2e5_step50_gqa_confirm3000.json` | 42.433 | 1273 / 3000 | 60.758 | 32.494 |

Interpretation: visual cross-attention can imitate V11, but the apparent n200
high point does not beat V11 search/confirm.

## Adaptation Attempts

### Qwen Visual-Intake LoRA

Checkpoint:

```text
/checkpoints/finetune-output/v13-qwen3-vxattn-step50-visualintake-qv-lora-r8-gqa-stage2-50/checkpoint-50
```

| Eval | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| GQA n200 | `/tmp/v13_vxattn_step50_qv_lora_r8_stage2_step50_gqa_n200.json` | 46.000 | 92 / 200 | 62.500 | 35.000 |

Interpretation: q/v LoRA on the best visual-cross-attention substrate returned
to the V11 n200 baseline and did not show a hill.

### SigLIP Last-Block Attention/Norm Adapter

Checkpoint:

```text
/checkpoints/pretrain-output/v13-qwen3-vxattn-upper-siglip-last1-attnnorm-freezeconn-lr3e6-step50
```

| Checkpoint | Artifact | Accuracy | Correct / Total | Yes/No | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| step25 | `/tmp/v13_vxattn_siglip_attnnorm_step25_gqa_n200.json` | 46.000 | 92 / 200 | 62.500 | 35.000 |
| step50 | `/tmp/v13_vxattn_siglip_attnnorm_step50_gqa_n200.json` | 46.500 | 93 / 200 | 63.750 | 35.000 |

The evaluator loaded `vision_adapter.pt` with 12 SigLIP tensors at both
checkpoints. The result ties the best cross-attention substrate slice and does
not justify expansion.

## Expanded Benchmark

A lightweight ChartQA evaluator was added for V13 expanded-benchmark coverage.
It converts a deterministic Hugging Face `anhdang000/ChartQA-V2` slice into the
repo's VQA-style evaluation layout and reports exact match.

| Model | Artifact | Split | Accuracy | Correct / Total | Avg Tokens | EOS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V11 frontier | `/tmp/v11_chartqa_val_n200.json` | ChartQA val n200 seed42 | 6.000 | 12 / 200 | 6.195 | 0.940 |
| V13 vxattn step50 | `/tmp/v13_vxattn_step50_chartqa_val_n200.json` | ChartQA val n200 seed42 | 5.500 | 11 / 200 | 6.190 | 0.940 |

Interpretation: neither checkpoint has meaningful chart-reading ability, and
the best V13 substrate is slightly below V11 on this expanded probe.

## Pairwise Taxonomy And Leakage

Matched GQA n200 artifacts with prediction samples:

| Model | Artifact | Accuracy | Prediction Samples |
| --- | --- | ---: | ---: |
| V11 frontier | `/tmp/v11_gqa_n200_pred.json` | 46.000 | 200 |
| V13 vxattn step50 | `/tmp/v13_vxattn_step50_gqa_n200_pred.json` | 46.500 | 200 |

Heuristic taxonomy output:

```text
/tmp/v13_gqa_pairwise_taxonomy_n200/
```

The taxonomy deltas were effectively identical. V13 had one extra correct
yes/no object-presence item:

```text
v11_correct_v13_vxattn_wrong: 0 rows
v13_vxattn_correct_v11_wrong: 1 row, yes_no_object_presence
```

Leakage audit:

```text
/tmp/v13_gqa_n200_leakage_audit.json
```

The audit compared the matched GQA prediction artifacts against the V9/V10/V12
GQA train and contrastive train sources used by the V13 runs. It passed:

```text
exact_val2014_overlap = 0
numeric_id_overlap = 0
raw_ref_overlap = 0
missing_sources = 0
overall = PASS
```

## Implementation Changes

- Added Qwen3 thinking-template support and thinking-output scoring hygiene.
- Added GQA split metadata, confidence intervals, sample offsets, and
  `max_new_tokens` recording.
- Added a lightweight ChartQA checkpoint evaluator.
- Fixed V3 checkpoint metadata compatibility for `None` AnyRes fields.
- Fixed V3 visual-cross-attention teacher KL for self-teacher mode.
- Fixed Stage 2 visual-cross-attention adapter loading and freezing.
- Added Modal `vision_trainable_prefixes` wiring for Stage 1 vision adaptation.

Validation after the final code edits:

```text
python3 -m compileall -q models training evaluation data scripts
git diff --check
python3 scripts/repo_health_check.py
```

All passed.

## Blockers / Not Completed

- No full largest-slice GQA final was run because no V13 candidate beat the
  V11 search/confirm baseline.
- The full VQAv2 clean/control and POPE suite was not rerun for V13 candidates
  that failed GQA promotion screens.

## Recommendation

V11 remains the Qwen frontier. The V13 evidence points away from connector-only
geometry changes as the next hill: spatial grid, AnyRes/pass-through, V11 tail,
cross-attention, Qwen q/v LoRA, and a small SigLIP adapter all fail to beat V11
on matched GQA checks.

The next serious hill should be a capability-expansion path rather than another
V3 connector variant: build a reliable expanded benchmark harness first
especially TextVQA/ChartQA, then train on a targeted OCR/chart/compositional
mix with V11 replay KL and controls. If that does not move, the likely bottleneck
is frozen Qwen/SigLIP adaptation rather than visual-token topology.
