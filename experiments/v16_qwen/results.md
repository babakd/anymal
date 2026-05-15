# V16 Qwen3 Capability-Led Pareto Campaign

Date started: 2026-05-14/15

## Objective

V16 refines the V15 capability-data signal under the fixed V11 Qwen3
frontier architecture. The target is a Pareto improvement over V11: materially
better ChartQA/TextVQA, retained GQA, and retained VQA/POPE controls.

Frontier / retention teacher:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

## Phase 0A: V15 Checkpoint-500 Confirmation

The planned V15 checkpoint-500 confirmation is blocked by missing checkpoint
state on the Modal volume.

Expected path:

```text
/checkpoints/pretrain-output/v15-qwen3-balanced-v11-cachekl-lr2e6-3000/checkpoint-500
```

Observed available checkpoints under the V15 balanced run:

```text
checkpoint-1000
checkpoint-1500
checkpoint-1750
checkpoint-2000
checkpoint-2250
checkpoint-2500
checkpoint-2750
checkpoint-3000
```

Only n=200 checkpoint-500 result artifacts remain:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_textvqa_val_n200.json
```

Those artifacts do not include prediction samples, so they cannot support the
full Phase 0A confirmation or prediction-level leakage review. The V16
stop-point sweep must preserve checkpoint-500 and redo the larger-slice
confirmation from a reproducible saved checkpoint.

## Phase 0B: Retroactive V15 Leakage Audit

Artifact:

```text
/checkpoints/v16_qwen/phase0b_v15_metadata_leakage_audit_n200.json
```

Status: metadata-level audit passed for the available V15 n=200 artifacts, but
this is weaker than the full required slice audit because the checkpoint-500
eval artifacts do not contain prediction samples.

Split-aware exact overlaps:

| Source vs eval slice | Verdict | Image/ref overlap | Question-id overlap |
| --- | --- | ---: | ---: |
| ChartQA capability train vs val n200 | pass | 0 | 0 |
| TextVQA capability train vs validation n200 | pass | 0 | 0 |
| GQA spatial/relation train vs testdev n200 | pass | 0 | 0 |
| GQA replay train vs testdev n200 | pass | 0 | 0 |

Raw source-index collisions were observed across train/val splits, but those
indices are split-local and not treated as leakage without shared split-aware
IDs or image refs.

## Phase 0C: Connector Drift Probe

Artifacts:

```text
/checkpoints/v16_qwen/drift_probe_set_v1.json
/checkpoints/v16_qwen/drift_probe_set_v1.pt
/checkpoints/v16_qwen/drift_v11_selfcheck.json
```

Probe details:

| Field | Value |
| --- | --- |
| Source | VQAv2 val |
| Seed | 42 |
| Size | 64 |
| Teacher | V11 frontier checkpoint |
| Cached tensor shape | 64 x 128 x 4096 |
| Cached answer tokens | 98 |

V11 self-check:

| Metric | Result |
| --- | ---: |
| Connector MSE to V11 | 0.0 |
| Answer-token KL to V11 | 0.0 |
| Exact answer agreement | 64 / 64 |
| Strict answer agreement | 64 / 64 |
| Candidate / teacher probe accuracy | 62.5 / 62.5 |
| Mean cosine to V11 | 0.9999952912 |

## Phase 0D: C1 Auxiliary-Branch Diagnostic

Run:

```text
v16-qwen3-c1-diagnostic-v9scale105-antishuffle-lr5e5-200
```

Command:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105 \
  --dataset v10_qwen_gqa_antishuffle_stage1b \
  --run-name v16-qwen3-c1-diagnostic-v9scale105-antishuffle-lr5e5-200 \
  --max-steps 200 \
  --learning-rate 5e-5 \
  --batch-size 4 \
  --pretrain-gradient-accumulation-steps 8 \
  --pretrain-image-tokens 128 \
  --v3-connector-type perceiver_resampler \
  --v3-connector-output-scale 1.05 \
  --v3-use-2d-patch-position-features \
  --v3-patch-position-feature-type learned_table \
  --pretrain-save-steps 200 \
  --use-wandb
```

W&B run:

```text
https://wandb.ai/babakdam/anymal-pretrain/runs/spgveel9
```

Status: complete. The run loaded the V9 scale-1.05 checkpoint, initialized the
new `patch_position_embedding` parameter, froze Qwen and SigLIP, and trained
the V3 connector path as intended. Modal/W&B showed no NaNs, tracebacks,
metadata mismatches, or persistent health alerts. Final eval loss was `0.3724`;
W&B reported `placeholder_contract_valid=1.0`.

Checkpoint:

```text
/checkpoints/pretrain-output/v16-qwen3-c1-diagnostic-v9scale105-antishuffle-lr5e5-200/checkpoint-200
```

Evaluation matrix:

| Mode | Checkpoint |
| --- | --- |
| Branch on, scale 1.0 | `/checkpoints/pretrain-output/v16-qwen3-c1-diagnostic-v9scale105-antishuffle-lr5e5-200/checkpoint-200` |
| Branch off, scale 0.0 | `/checkpoints/v16_qwen/phase0d_c1/checkpoint-200-patchscale000` |
| Branch attenuated, scale 0.5 | `/checkpoints/v16_qwen/phase0d_c1/checkpoint-200-patchscale050` |

GQA `testdev_balanced` n=1000, seed 42:

| Mode | Artifact | Accuracy | Correct / total | Other | Yes/no | EOS | Max-hit | Prefix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale 1.0 | `/checkpoints/v16_qwen/phase0d_c1/gqa_checkpoint200_patchscale100_n1000.json` | 43.8 | 438 / 1000 | 32.512 | 64.672 | 1.0 | 0.0 | 0.0 |
| scale 0.5 | `/checkpoints/v16_qwen/phase0d_c1/gqa_checkpoint200_patchscale050_n1000.json` | 43.7 | 437 / 1000 | 32.666 | 64.103 | 1.0 | 0.0 | 0.0 |
| scale 0.0 | `/checkpoints/v16_qwen/phase0d_c1/gqa_checkpoint200_patchscale000_n1000.json` | 43.3 | 433 / 1000 | 32.357 | 63.533 | 1.0 | 0.0 | 0.0 |

Connector drift on the fixed V11 VQAv2-val probe:

| Mode | Artifact | MSE to V11 | Mean cosine | Candidate RMS | V11 RMS | Answer KL | V11 agreement | Probe accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale 1.0 | `/checkpoints/v16_qwen/phase0d_c1/drift_checkpoint200_patchscale100.json` | 5.3616e-05 | 0.976776 | 0.031270 | 0.033500 | 0.064065 | 54 / 64 | 58.333 |
| scale 0.5 | `/checkpoints/v16_qwen/phase0d_c1/drift_checkpoint200_patchscale050.json` | 5.3596e-05 | 0.976785 | 0.031270 | 0.033500 | 0.064780 | 53 / 64 | 58.333 |
| scale 0.0 | `/checkpoints/v16_qwen/phase0d_c1/drift_checkpoint200_patchscale000.json` | 5.3577e-05 | 0.976794 | 0.031270 | 0.033500 | 0.064541 | 54 / 64 | 58.333 |

Interpretation:

- Branch-off scale `0.0` did not reproduce V11's GQA search n1000 result
  (`44.9`). It landed at `43.3`.
- Branch-on scale `1.0` was best of the three at `43.8`, but this is still
  `-1.1` below V11 search n1000 and close to V9-style performance.
- The three branch scales have nearly identical connector drift and probe
  accuracy. This argues against the learned branch being an inference-time
  lever after the 200-step diagnostic; if it acted as an optimization scaffold,
  the effect did not recover the V11 basin from the V9 scale-1.05 anchor.
- Result classification: capability/optimization diagnostic negative for
  direct V11 reproduction, not an implementation failure.

## Phase 1A: Balanced Stop-Point Sweep

Run:

```text
v16-qwen3-balanced-v11-cachekl-lr2e6-stop-sweep-3000
```

Command:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --dataset v15_qwen_balanced_stage1b \
  --run-name v16-qwen3-balanced-v11-cachekl-lr2e6-stop-sweep-3000 \
  --max-steps 3000 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --pretrain-gradient-accumulation-steps 8 \
  --pretrain-image-tokens 128 \
  --v3-connector-type perceiver_resampler \
  --v3-connector-output-scale 1.125 \
  --pretrain-teacher-kl-weight 1.0 \
  --pretrain-teacher-kl-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --pretrain-teacher-kl-cache-path /checkpoints/v15_qwen/v15_v11_teacher_topk128_retention_all.pt \
  --pretrain-teacher-kl-cache-top-k 128 \
  --pretrain-save-steps 10000 \
  --pretrain-save-checkpoint-steps 100,250,400,500,600,800,1000,1500,2000,3000 \
  --pretrain-save-total-limit 5 \
  --pretrain-preserve-checkpoint-steps 100,250,400,500,600,800,1000,1500,2000,3000 \
  --use-wandb
```

W&B run:

```text
https://wandb.ai/babakdam/anymal-pretrain/runs/j11m2ppr
```

Status: complete. The run loaded V11 connector weights, froze Qwen/SigLIP,
loaded cached V11 answer-token KL retention targets, accepted the exact
checkpoint list, and reached `max_steps=3000`. The final checkpoint contains
`model_meta.json`, `projector.pt`, and `trainer_state.pt`. No tracebacks, NaNs,
or checkpoint-save failures were observed. Health monitoring produced plateau
or optimizer-gap warnings late in the run, but no hard technical failure.

Cheap-screen artifacts:

| Step | GQA n200 | ChartQA val n200 | TextVQA val n200 | VQAv2 clean n200 | Drift |
| ---: | --- | --- | --- | --- | --- |
| 100 | `/checkpoints/v16_qwen/phase1a/ckpt100_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt100_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt100_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt100_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt100_drift.json` |
| 250 | `/checkpoints/v16_qwen/phase1a/ckpt250_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt250_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt250_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt250_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt250_drift.json` |
| 400 | `/checkpoints/v16_qwen/phase1a/ckpt400_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt400_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt400_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt400_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt400_drift.json` |
| 500 | `/checkpoints/v16_qwen/phase1a/ckpt500_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt500_drift.json` |
| 600 | `/checkpoints/v16_qwen/phase1a/ckpt600_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt600_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt600_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt600_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt600_drift.json` |
| 800 | `/checkpoints/v16_qwen/phase1a/ckpt800_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt800_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt800_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt800_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt800_drift.json` |
| 1000 | `/checkpoints/v16_qwen/phase1a/ckpt1000_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt1000_drift.json` |
| 1500 | `/checkpoints/v16_qwen/phase1a/ckpt1500_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt1500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt1500_drift.json` |
| 2000 | `/checkpoints/v16_qwen/phase1a/ckpt2000_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt2000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt2000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt2000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt2000_drift.json` |
| 3000 | `/checkpoints/v16_qwen/phase1a/ckpt3000_gqa_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt3000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt3000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase1a/ckpt3000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase1a/ckpt3000_drift.json` |

Cheap-screen metrics:

| Step | GQA acc | ChartQA EM | TextVQA exact | TextVQA soft | VQAv2 clean | Drift MSE | Drift KL | V11 agree | Mean cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 45.5 | 4.5 | 34.0 | 32.333 | 65.167 | 1.0947e-06 | 0.013186 | 62 / 64 | 0.999508 |
| 250 | 41.5 | 8.5 | 36.0 | 34.167 | 62.833 | 3.4410e-06 | 0.041199 | 54 / 64 | 0.998462 |
| 400 | 41.0 | 7.5 | 36.5 | 34.667 | 61.000 | 5.0094e-06 | 0.041599 | 55 / 64 | 0.997764 |
| 500 | 42.5 | 7.0 | 37.0 | 35.500 | 62.000 | 6.3494e-06 | 0.036784 | 56 / 64 | 0.997167 |
| 600 | 42.0 | 7.0 | 38.5 | 37.000 | 62.500 | 6.5175e-06 | 0.036986 | 55 / 64 | 0.997092 |
| 800 | 41.5 | 8.0 | 39.0 | 37.500 | 62.000 | 7.6267e-06 | 0.038806 | 55 / 64 | 0.996598 |
| 1000 | 41.5 | 9.5 | 39.0 | 37.500 | 62.667 | 8.5901e-06 | 0.039590 | 54 / 64 | 0.996169 |
| 1500 | 42.0 | 10.5 | 38.5 | 37.000 | 60.833 | 1.0770e-05 | 0.043229 | 53 / 64 | 0.995198 |
| 2000 | 41.5 | 10.0 | 38.5 | 36.667 | 60.833 | 1.1962e-05 | 0.046332 | 53 / 64 | 0.994667 |
| 3000 | 42.0 | 9.5 | 39.5 | 37.667 | 62.333 | 1.3006e-05 | 0.049552 | 52 / 64 | 0.994202 |

Interpretation so far:

- Step `100` is mostly a warmup/retention anchor: GQA and VQAv2 are strong on
  the n=200 slices, drift is tiny, and ChartQA has not moved.
- Step `250` starts showing the intended ChartQA/TextVQA capability movement,
  but already trades off GQA/VQAv2 and answer agreement. It is not a promotion
  candidate without later recovery on GQA/retention.
- Step `400` preserves the TextVQA movement but does not improve the overall
  Pareto trade: ChartQA slips from step `250`, GQA/VQAv2 slip again, and drift
  continues to grow.
- Step `500` gives the best TextVQA exact score so far and partially recovers
  GQA/VQAv2 from step `400`, but ChartQA slips again and the n=200 slice is
  still far short of the intermediate promotion gate.
- Step `600` strengthens the TextVQA signal again, but GQA remains materially
  below the V11 retention target and ChartQA is flat versus step `500`; it is
  therefore also not an intermediate promotion candidate.
- Step `800` is the strongest capability-side stop point so far on n=200
  (`ChartQA=8.0`, `TextVQA exact=39.0`), but retention remains the blocker:
  GQA is `41.5`, VQAv2 clean is `62.0`, and drift continues upward.
- Step `1000` becomes the strongest ChartQA stop point so far (`9.5`) and
  holds the step-`800` TextVQA exact score (`39.0`), but the same retention
  blocker remains: GQA is `41.5`, VQAv2 clean is `62.667`, and agreement with
  V11 drops to `54 / 64`.
- Step `1500` extends the ChartQA movement to `10.5` and partially recovers GQA
  to `42.0`, but VQAv2 clean drops to `60.833` and drift continues upward. It
  is not an intermediate promotion candidate on n=200.
- Step `2000` does not improve the Pareto position over step `1500`: ChartQA
  slips to `10.0`, TextVQA exact stays at `38.5`, GQA returns to `41.5`, and
  VQAv2 clean remains low at `60.833`.
- Step `3000` gives the best TextVQA exact score in Phase 1A (`39.5`) and
  recovers VQAv2 from the step-`1500`/`2000` trough, but ChartQA is below the
  step-`1500` peak and GQA remains far below the retention gate.
- Phase 1A stop-point sweep is complete. The n=200 Pareto frontier is
  capability-positive but retention-negative: no checkpoint satisfies the
  intermediate promotion gate, so no Phase 1A checkpoint should be promoted
  from cheap screens alone.

## Phase 2A: Vision-Only Last-2 Full Exposure

Run:

```text
v16-qwen3-v11-visionlast2-only-connfrozen-cachekl-lr1e6-3000
```

Command:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --dataset v15_qwen_balanced_stage1b \
  --run-name v16-qwen3-v11-visionlast2-only-connfrozen-cachekl-lr1e6-3000 \
  --max-steps 3000 \
  --learning-rate 1e-6 \
  --batch-size 4 \
  --pretrain-gradient-accumulation-steps 8 \
  --pretrain-image-tokens 128 \
  --v3-connector-type perceiver_resampler \
  --v3-connector-output-scale 1.125 \
  --connector-trainable-prefixes freeze \
  --vision-trainable-prefixes model.encoder.layers.25.self_attn,model.encoder.layers.25.layer_norm1,model.encoder.layers.25.layer_norm2,model.encoder.layers.26.self_attn,model.encoder.layers.26.layer_norm1,model.encoder.layers.26.layer_norm2 \
  --pretrain-teacher-kl-weight 1.0 \
  --pretrain-teacher-kl-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --pretrain-teacher-kl-cache-path /checkpoints/v15_qwen/v15_v11_teacher_topk128_retention_all.pt \
  --pretrain-teacher-kl-cache-top-k 128 \
  --pretrain-save-steps 10000 \
  --pretrain-save-checkpoint-steps 250,500,1000,1500,2000,3000 \
  --pretrain-save-total-limit 5 \
  --pretrain-preserve-checkpoint-steps 250,500,1000,1500,2000,3000 \
  --use-wandb
```

Modal app:

```text
https://modal.com/apps/babakd/main/ap-0Ta1fIrx81RJFMwvrrcOAg
```

W&B run:

```text
https://wandb.ai/babakdam/anymal-pretrain/runs/55z4sq8d
```

Status: complete. The run used the mandatory Phase 2A scope: V11 connector
warm-started and then frozen, Qwen frozen, SigLIP layers 25 and 26
`self_attn/layer_norm1/layer_norm2` trainable, cached V11 retention-only KL,
and the full 3000-step checkpoint ladder. Startup confirmed the intended
vision-only training surface: connector trainable params `0`, image-encoder
trainable params `10,635,264`, and total trainable params `10,635,264`. The
trainer emitted the expected adapter-verification warning because this phase
intentionally trains selected vision-layer tensors rather than projector
adapters. The run reached `max_steps=3000` with finite losses and committed the
final checkpoint successfully.

Checkpoints through `3000` were confirmed complete with `model_meta.json`,
`projector.pt`, `vision_adapter.pt`, and `trainer_state.pt`; eval loading
confirmed the 24-tensor vision adapter was applied.

Cheap-screen artifacts:

| Step | GQA n200 | ChartQA val n200 | TextVQA val n200 | VQAv2 clean n200 | Drift |
| ---: | --- | --- | --- | --- | --- |
| 250 | `/checkpoints/v16_qwen/phase2a/ckpt250_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt250_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt250_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt250_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt250_drift.json` |
| 500 | `/checkpoints/v16_qwen/phase2a/ckpt500_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt500_drift.json` |
| 1000 | `/checkpoints/v16_qwen/phase2a/ckpt1000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt1000_drift.json` |
| 1500 | `/checkpoints/v16_qwen/phase2a/ckpt1500_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt1500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt1500_drift.json` |
| 2000 | `/checkpoints/v16_qwen/phase2a/ckpt2000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt2000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt2000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt2000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt2000_drift.json` |
| 3000 | `/checkpoints/v16_qwen/phase2a/ckpt3000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt3000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt3000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2a/ckpt3000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2a/ckpt3000_drift.json` |

Cheap-screen metrics:

| Step | GQA acc | ChartQA EM | TextVQA exact | TextVQA soft | VQAv2 clean | Drift MSE | Drift KL | V11 agree | Mean cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 46.0 | 5.5 | 32.5 | 31.167 | 65.833 | 4.8739e-07 | 0.001918 | 62 / 64 | 0.999778 |
| 500 | 45.0 | 6.0 | 32.0 | 30.167 | 65.833 | 1.5110e-06 | 0.006696 | 58 / 64 | 0.999322 |
| 1000 | 44.5 | 7.0 | 33.5 | 31.667 | 66.167 | 3.2046e-06 | 0.013022 | 58 / 64 | 0.998568 |
| 1500 | 43.5 | 6.5 | 34.0 | 32.500 | 67.000 | 4.1598e-06 | 0.019086 | 58 / 64 | 0.998142 |
| 2000 | 45.0 | 7.0 | 34.0 | 32.500 | 66.500 | 4.7561e-06 | 0.022769 | 58 / 64 | 0.997876 |
| 3000 | 44.5 | 6.5 | 34.0 | 32.500 | 67.000 | 5.1030e-06 | 0.024625 | 58 / 64 | 0.997722 |

Phase 2A gate artifacts:

| Slice | Artifact | Metric |
| --- | --- | --- |
| ChartQA val n1000 | `/checkpoints/v16_qwen/phase2a/ckpt3000_chartqa_val_n1000.json` | exact match `9.0` (`90 / 1000`) |
| TextVQA validation n2000 | `/checkpoints/v16_qwen/phase2a/ckpt3000_textvqa_val_n2000.json` | exact match `30.15` (`603 / 2000`), soft `27.733` |

Interpretation so far:

- Step `250` is strongly retention-preserving on the n=200 screens: GQA is
  above the V11 n=200 operating point used for intermediate gating, VQAv2 clean
  is high, and drift is tiny.
- Capability movement has not appeared yet: ChartQA is below the V11 n=200
  reference and TextVQA is below the Phase 1A connector-adaptation trajectory.
- Step `500` keeps the retention profile strong (`GQA=45.0`,
  `VQAv2=65.833`) with only modest drift growth, but capability still has not
  moved: ChartQA is back at the V11 n=200 reference and TextVQA exact is
  slightly lower than step `250`.
- Step `1000` shows the first small capability movement in this branch:
  ChartQA rises to `7.0` and TextVQA exact to `33.5`, while GQA remains near
  V11 (`44.5`) and VQAv2 improves to `66.167`. The movement is still much
  smaller than the Phase 1A connector branch, so this is a trajectory note, not
  a promotion result.
- Step `1500` continues the TextVQA drift upward (`34.0`) and improves VQAv2
  again (`67.0`), but ChartQA slips to `6.5` and GQA declines to `43.5`.
  Retention is still much healthier than Phase 1A, while capability movement
  remains modest.
- Step `2000` recovers GQA to `45.0` and keeps VQAv2 high (`66.5`) while
  holding TextVQA exact at `34.0`; ChartQA returns to `7.0`. This is the best
  retention/capability compromise in Phase 2A so far, but capability movement
  remains below the Phase 2B gate and below the Phase 1A connector branch.
- Step `3000` holds the retention profile (`GQA=44.5`, `VQAv2=67.0`) but does
  not improve the n=200 capability probes beyond the earlier Phase 2A points.
- The required n>=1000 gate evidence does move: ChartQA val n1000 is `9.0`,
  which clears the V11 `7.8` reference by `+1.2`, and TextVQA validation n2000
  exact is `30.15`, clearing the V11 `28.5` reference plus `1.5` by `+0.15`.
  Phase 2B is therefore triggered.

## Phase 2B: vision-only last-4

Run:

```text
v16-qwen3-v11-visionlast4-only-connfrozen-cachekl-lr1e6-3000
```

Command:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --dataset v15_qwen_balanced_stage1b \
  --run-name v16-qwen3-v11-visionlast4-only-connfrozen-cachekl-lr1e6-3000 \
  --max-steps 3000 \
  --learning-rate 1e-6 \
  --batch-size 4 \
  --pretrain-gradient-accumulation-steps 8 \
  --pretrain-image-tokens 128 \
  --v3-connector-type perceiver_resampler \
  --v3-connector-output-scale 1.125 \
  --connector-trainable-prefixes freeze \
  --vision-trainable-prefixes model.encoder.layers.23.self_attn,model.encoder.layers.23.layer_norm1,model.encoder.layers.23.layer_norm2,model.encoder.layers.24.self_attn,model.encoder.layers.24.layer_norm1,model.encoder.layers.24.layer_norm2,model.encoder.layers.25.self_attn,model.encoder.layers.25.layer_norm1,model.encoder.layers.25.layer_norm2,model.encoder.layers.26.self_attn,model.encoder.layers.26.layer_norm1,model.encoder.layers.26.layer_norm2 \
  --pretrain-teacher-kl-weight 1.0 \
  --pretrain-teacher-kl-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --pretrain-teacher-kl-cache-path /checkpoints/v15_qwen/v15_v11_teacher_topk128_retention_all.pt \
  --pretrain-teacher-kl-cache-top-k 128 \
  --pretrain-save-steps 10000 \
  --pretrain-save-checkpoint-steps 250,500,1000,1500,2000,3000 \
  --pretrain-save-total-limit 5 \
  --pretrain-preserve-checkpoint-steps 250,500,1000,1500,2000,3000 \
  --use-wandb
```

Modal app:

```text
https://modal.com/apps/babakd/main/ap-9iZTbX19wHx43wmkNEDzLL
```

W&B run:

```text
https://wandb.ai/babakdam/anymal-pretrain/runs/kpdf7saf
```

Status: complete. This branch was gated by the Phase 2A step-`3000` n>=1000
evidence above and followed the hard-exposure rule through `3000` steps.
Startup confirmed connector frozen with `0` trainable connector parameters,
Qwen frozen, and SigLIP layers 23-26 active with `48` trainable tensors /
`21,270,528` trainable image-encoder parameters. Cached V11 retention-only KL
loaded successfully. The trainer emitted the same expected non-adapter warning
as Phase 2A because this branch intentionally trains selected vision-layer
tensors. Training completed cleanly in `3.33` hours and saved checkpoint
`3000`; no NaNs, OOMs, tracebacks, broken generation, or checkpoint failures
were observed.

Checkpoints through `3000` were confirmed complete with `model_meta.json`,
`projector.pt`, `vision_adapter.pt`, and `trainer_state.pt`; eval loading
confirmed the 48-tensor vision adapter was applied.

Cheap-screen artifacts:

| Step | GQA n200 | ChartQA val n200 | TextVQA val n200 | VQAv2 clean n200 | Drift |
| ---: | --- | --- | --- | --- | --- |
| 250 | `/checkpoints/v16_qwen/phase2b/ckpt250_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt250_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt250_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt250_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt250_drift.json` |
| 500 | `/checkpoints/v16_qwen/phase2b/ckpt500_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt500_drift.json` |
| 1000 | `/checkpoints/v16_qwen/phase2b/ckpt1000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt1000_drift.json` |
| 1500 | `/checkpoints/v16_qwen/phase2b/ckpt1500_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1500_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1500_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt1500_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt1500_drift.json` |
| 2000 | `/checkpoints/v16_qwen/phase2b/ckpt2000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt2000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt2000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt2000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt2000_drift.json` |
| 3000 | `/checkpoints/v16_qwen/phase2b/ckpt3000_gqa_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt3000_chartqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt3000_textvqa_val_n200.json` | `/checkpoints/v16_qwen/phase2b/ckpt3000_vqa_clean_n200_seed42.json` | `/checkpoints/v16_qwen/phase2b/ckpt3000_drift.json` |

Cheap-screen metrics:

| Step | GQA acc | ChartQA EM | TextVQA exact | TextVQA soft | VQAv2 clean | Drift MSE | Drift KL | V11 agree | Mean cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 45.0 | 6.0 | 31.5 | 30.000 | 66.333 | 7.9898e-07 | 0.003504 | 62 / 64 | 0.999639 |
| 500 | 44.5 | 6.5 | 32.5 | 31.000 | 66.833 | 2.3303e-06 | 0.009979 | 59 / 64 | 0.998957 |
| 1000 | 43.5 | 7.5 | 35.0 | 33.000 | 67.167 | 4.1551e-06 | 0.020098 | 57 / 64 | 0.998144 |
| 1500 | 43.0 | 6.5 | 34.5 | 32.500 | 67.500 | 4.9233e-06 | 0.025461 | 58 / 64 | 0.997802 |
| 2000 | 43.5 | 8.0 | 35.5 | 33.167 | 67.500 | 5.4474e-06 | 0.028416 | 56 / 64 | 0.997568 |
| 3000 | 44.5 | 7.0 | 35.0 | 33.000 | 68.000 | 5.8025e-06 | 0.031054 | 55 / 64 | 0.997410 |

Interpretation so far:

- Step `250` preserves retention (`GQA=45.0`, `VQAv2=66.333`) and keeps drift
  low, but it does not match the Phase 2A capability trajectory. ChartQA is at
  the V11 n=200 reference and TextVQA is below Phase 2A step `250`.
- Step `500` improves slightly over step `250` on ChartQA, TextVQA, and VQAv2,
  while retaining GQA at `44.5`. It remains below Phase 2A's late capability
  movement and below any promotion threshold.
- Step `1000` is the first clearly capability-positive point in Phase 2B:
  ChartQA reaches `7.5` and TextVQA exact reaches `35.0`, ahead of Phase 2A's
  n=200 trajectory, while VQAv2 remains high at `67.167`. GQA drops to `43.5`
  and drift rises, so this is a trajectory signal rather than a promotion
  result.
- Step `1500` does not improve the Pareto position over step `1000`: ChartQA
  falls to `6.5`, TextVQA exact to `34.5`, and GQA to `43.0`, although VQAv2
  remains strong at `67.5`.
- Step `2000` becomes the best Phase 2B cheap-screen capability point so far:
  ChartQA reaches `8.0` and TextVQA exact reaches `35.5`, with VQAv2 still at
  `67.5`. The tradeoff is continued drift growth and GQA still only `43.5`.
- Step `3000` recovers GQA to `44.5` and improves VQAv2 to `68.0`, but the
  capability probes slip from step `2000` (`ChartQA=7.0`,
  `TextVQA exact=35.0`). It is the best retention endpoint in the branch, but
  step `2000` is the best cheap-screen capability point.
- Phase 2B completed the hard exposure without a hard technical failure. It
  improves over Phase 2A on the n=200 capability probes, but does not establish
  a promotion candidate by itself because the best capability point still has
  GQA below the intermediate gate and no n>=1000 confirmation.

## Phase 3A: capability-data inventory

Status: local/Modal inventory complete. The current volume has materialized
ChartQA and TextVQA train/eval caches, but no additional scaled capability
sources were found for OCR-VQA, DocVQA, AI2D, ScienceQA image QA, Visual
Genome, or RefCOCO.

| Source | Availability | Local/Modal evidence | Size observed | Notes |
| --- | --- | --- | ---: | --- |
| ChartQA train | available | `/checkpoints/chartqa_data/v15_chartqa_train_seed1501_n20000.json`; `/checkpoints/chartqa_images_hf`; generator uses `anhdang000/ChartQA-V2` train | 20,000 instruction rows; 20,979 image files including eval caches | Split-aware n200 leakage audit passed. The active HF repack's README is empty/no license declared; upstream `vis-nlp/ChartQA` is GPL-3.0. Treat as license-sensitive and verify whether the repack inherits upstream/source-image terms before scaled pretraining use. |
| ChartQA augmented | not found | no augmented ChartQA cache or top-level volume directory found | 0 | Requires acquisition or generation before Phase 3B scaled use. |
| TextVQA train | available | `/checkpoints/textvqa_data/v15_lmms-lab_textvqa_train_seed1502_n20000.json`; `/checkpoints/textvqa_images_hf`; generator uses `lmms-lab/textvqa` train | 20,000 instruction rows; 22,000 image files including eval caches | Split-aware n200 leakage audit passed. `lmms-lab/textvqa` is a formatted copy of TextVQA; upstream `facebook/textvqa` lists `cc-by-4.0` and notes OpenImages-source/PII considerations, so attribution and source-image terms still need to be tracked before scaled use. |
| OCR-VQA train | not found | no top-level `ocr`/`ocrvqa` volume directory or repo cache found | 0 | Requires acquisition and leakage audit. |
| DocVQA train | not found | no top-level `docvqa` volume directory or repo cache found | 0 | Requires acquisition and leakage audit. |
| AI2D train | not found | no top-level `ai2d` volume directory or repo cache found | 0 | Requires acquisition and leakage audit. |
| ScienceQA image QA | not found | no top-level `scienceqa` volume directory or repo cache found | 0 | Requires acquisition and image-only subset audit. |
| Visual Genome | not found | no top-level `visual_genome`/`vg` volume directory or repo cache found | 0 | Requires acquisition and leakage audit. |
| RefCOCO | not found | no top-level `refcoco` volume directory or repo cache found | 0 | Requires acquisition and leakage audit. |

## Phase 2C / 3B / 4 decisions

| Phase | Status | Decision |
| --- | --- | --- |
| Phase 2C joint vision+connector | not run | Gate blocked. Phase 2A showed n>=1000 movement, but Phase 1 did not produce a Pareto-promotable checkpoint confirmed at n=3000. Running joint adaptation would violate the Phase 2C gate. |
| Phase 3B scaled capability mix | not run | Blocked by data availability and recipe quality. Phase 3A found only 20k ChartQA and 20k TextVQA materialized training caches, no augmented ChartQA, and none of OCR-VQA/DocVQA/AI2D/ScienceQA/VG/RefCOCO. Phase 1 also did not produce a promotable recipe worth scaling. |
| Phase 4 Qwen q/v LoRA | not run | Gate blocked. Phase 4 is allowed only after Phase 1-3 plateau with a clear winning non-LoRA candidate; V16 produced no promotion candidate. Qwen therefore stayed frozen throughout executed runs. |

## Final report answers

1. Phase 0A V15 checkpoint-500 confirmation: blocked. The Modal volume no
   longer contains V15 checkpoint-500 state, only later checkpoints and n=200
   artifacts without prediction samples. V16 therefore fell back to V11 as the
   frontier/teacher.
2. Phase 0B leakage audit: metadata-level split-aware audit passed for the
   available V15 n=200 artifacts: ChartQA, TextVQA, GQA spatial/relation, and
   GQA replay exact image/question overlaps were all `0`. This is weaker than a
   full prediction-level audit because V15 prediction samples were absent.
3. Phase 0C drift probe: built and used. The fixed VQAv2-val seed-42 n64 probe
   lives at `/checkpoints/v16_qwen/drift_probe_set_v1.json` with tensor sidecar
   `/checkpoints/v16_qwen/drift_probe_set_v1.pt`.
4. Phase 0D C1 diagnostic: negative for V11 reproduction. GQA n1000 landed at
   `43.8` branch-on, `43.7` scale-0.5, and `43.3` branch-off, all below V11
   search `44.9`; drift was nearly identical across modes.
5. Phase 1A stop-point curve: capability-positive but retention-negative. Best
   ChartQA was `10.5` at step `1500`; best TextVQA exact was `39.5` at step
   `3000`; GQA stayed `41.0-42.5` after step `250`, so no checkpoint met the
   intermediate promotion gate.
6. Phase 1B retention sweep: not run. Phase 1A produced no n=200 intermediate
   promotion candidate or n=3000-confirmed candidate to recover.
7. Phase 1C contrastive counterfactual ablation: not run. The main stop-point
   sweep already failed retention strongly enough that optional branches were
   deprioritized behind the must-run Phase 2A/2B exposure.
8. Phase 2A last-2 vision: capability moved modestly with GQA/VQA preserved.
   Step `3000` confirmed ChartQA n1000 `9.0` and TextVQA n2000 exact `30.15`,
   triggering Phase 2B, while n200 GQA remained `44.5` and VQAv2 `67.0`.
9. Phase 2B/2C: last-4 vision helped versus Phase 2A on n=200 capability and
   VQAv2, but not enough for promotion. Best 2B capability point was step
   `2000` (`GQA=43.5`, `ChartQA=8.0`, `TextVQA=35.5`, `VQAv2=67.5`);
   endpoint step `3000` recovered GQA to `44.5` and VQAv2 to `68.0` but slipped
   to `ChartQA=7.0`, `TextVQA=35.0`. 2C was gate-blocked.
10. Phase 3A data availability: only ChartQA and TextVQA 20k train caches were
    available. No augmented ChartQA, OCR-VQA, DocVQA, AI2D, ScienceQA image QA,
    Visual Genome, or RefCOCO caches were found.
11. Phase 3B scaled data: not run because the scaled sources were not
    available/materialized and no Phase 1 recipe was promotable.
12. Phase 4 q/v LoRA: not run because there was no clear winning non-LoRA
    candidate after Phase 1-3.
13. Connector drift summary: Phase 1A drift grew fastest (`MSE` from
    `1.09e-06` to `1.30e-05`, KL to `0.0496`). Phase 2A stayed lower
    (`4.87e-07` to `5.10e-06`, KL to `0.0246`). Phase 2B ended in between on
    MSE but with growing KL (`7.99e-07` to `5.80e-06`, KL to `0.0311`).
14. Leakage audits: no exact split-aware overlaps found for audited V15
    sources; new missing Phase 3 sources still require acquisition-specific
    leakage audits before use.
15. Pairwise taxonomy: not run because no promotion candidate exists.
16. Final candidate metadata/table: no V16 promotion candidate exists. The
    strongest cheap-screen candidates were Phase 1A step `1500` for raw
    capability and Phase 2B step `3000` for retention, but neither satisfies
    Pareto promotion criteria.
17. Paired bootstrap CIs: not run because no checkpoint passed the promotion
    gates requiring full matched-slice confirmation.
18. Failure classification: Phase 0A is a provenance/storage blocker; Phase 0D
    is an imitation/optimization negative; Phase 1A is capability improvement
    with control-recovery failure; Phase 2A is control recovery with modest
    capability movement; Phase 2B is partial capability improvement with
    retention tradeoff; Phase 2C/3B/4 are gate/data blocked, not implementation
    failures.
19. V17 recommendation: continue Pareto refinement around vision-side
    adaptation plus capability data only after fixing data availability. The
    next useful work is to acquire/audit scaled OCR/document/chart/text sources
    and test whether the Phase 2B step-2000/3000 behavior survives larger
    slices, not to change connector architecture or unfreeze Qwen broadly.

## Tooling Changes

- Added `evaluation/checkpoint_eval/connector_drift_eval.py` for fixed-probe
  connector drift caches and candidate drift reports.
- Extended Stage 1 pretraining checkpoint controls with
  `pretrain_save_checkpoint_steps`, `pretrain_save_total_limit`, and
  `pretrain_preserve_checkpoint_steps` so V16 stop-point sweeps can save and
  preserve exact milestones without retaining every intermediate checkpoint.
