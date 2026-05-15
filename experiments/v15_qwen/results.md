# V15 Qwen3 Vision-Adaptation Campaign

Date: 2026-05-14

## Objective

V15 closes the final connector-isolation question and then pivots to
capability-led adaptation with Qwen3 frozen. V11 remains the retention teacher:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

## Setup And Harness

- Added retention-only cached-KL weighting through dataset metadata and the
  collator. Capability rows can now opt out of V11 KL with
  `teacher_kl_weight=0.0`.
- Added weighted/hash instruction-mixture indexing with fixed epoch length for
  large V15 mixtures.
- Added V15 balanced, ChartQA, TextVQA, retention-replay, and counterfactual
  stage-1 datasets in the Modal trainer.
- Added a TextVQA checkpoint evaluator backed by `lmms-lab/textvqa`.
- Extended ChartQA/VQA eval dataloaders to honor checkpoint/eval image size.
- Added cached answer-token KL mode for V3 Perceiver warm-starts when the
  student token count matches the V11 cache.

Local checks passed after these changes:

```text
python3 -m compileall -q data training evaluation scripts textvqa_checkpoint_eval.py
python3 scripts/repo_health_check.py
```

## Baseline Rechecks

Same n=200 smoke slices:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| V11 frontier | 46.0 | 6.0 | exact 30.5 / soft 29.5 |

Artifacts:

```text
/checkpoints/v15_qwen/v11_gqa_n200.json
/checkpoints/v15_qwen/v11_chartqa_val_n200.json
/checkpoints/v15_qwen/v11_textvqa_val_n200.json
```

## Teacher Cache

Full V15 retention cache:

```text
/checkpoints/v15_qwen/v15_v11_teacher_topk128_retention_all.pt
entries: 7000
answer tokens: 25029
top-k: 128
skipped_kl_disabled: 0
```

Capability data caches:

```text
/checkpoints/chartqa_data/v15_chartqa_train_seed1501_n20000.json
/checkpoints/textvqa_data/v15_lmms-lab_textvqa_train_seed1502_n20000.json
```

## Part A: Final Connector-Isolation Ablation

Run:

```text
v15-qwen3-spatialgrid128-nopos-cachekl-topk128-3000
```

Configuration:

```text
spatial_grid_projector
128 image tokens
patch_position_feature_scale = 0.0
connector_output_scale = 1.125
cached V11 top-k=128 answer-token KL
frozen SigLIP2, frozen Qwen3
```

Observed n=200 results:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| 250 | 29.0 | - | - |
| 500 | 25.0 | - | - |
| 750 | 32.5 | - | - |
| 1000 | 35.0 | - | - |
| 3000 | 33.5 | 5.0 | exact 5.5 / soft 5.17 |

Decision: negative. The controlled 128-token spatial-grid connector did not
imitate V11. Pure connector replacement should not remain the main hill.

## Part B: Balanced Capability And Retention Run

Main run:

```text
v15-qwen3-balanced-v11-cachekl-lr2e6-3000
```

Configuration:

```text
V11 warm start
Qwen/Qwen3-8B frozen
SigLIP2 frozen
128-token V3 Perceiver connector
cached_answer_tokens KL on retention rows only
balanced V15 mixture with ChartQA/TextVQA/GQA/counterfactual rows
lr=2e-6, max_steps=3000, save_steps=250
```

n=200 checkpoint sweep:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| 250 | 43.5 | 6.0 | exact 33.0 / soft 31.5 |
| 500 | 44.0 | 8.0 | exact 36.0 / soft 34.67 |
| 1000 | 44.0 | 6.0 | exact 35.5 / soft 34.0 |
| 1500 | 44.0 | 7.0 | exact 37.0 / soft 34.83 |
| 2000 | 42.5 | 7.0 | exact 38.0 / soft 35.83 |
| 3000 | 42.5 | 7.0 | exact 38.0 / soft 35.83 |

Checkpoint-500 artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt500_textvqa_val_n200.json
```

Checkpoint-1000 artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1000_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1000_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1000_textvqa_val_n200.json
```

Checkpoint-1500 artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1500_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1500_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt1500_textvqa_val_n200.json
```

Checkpoint-2000 artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt2000_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt2000_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt2000_textvqa_val_n200.json
```

Checkpoint-3000 artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_gqa_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_chartqa_val_n200.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_textvqa_val_n200.json
```

n=1000 confirmation:

| Model / checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| V11 frontier | 44.9 prior trusted n1000 | 7.8 | exact 28.5 / soft 26.47 |
| balanced checkpoint-3000 | 42.2 | 9.7 | exact 32.3 / soft 29.93 |

n=1000 artifacts:

```text
/checkpoints/v15_qwen/v11_chartqa_val_n1000.json
/checkpoints/v15_qwen/v11_textvqa_val_n1000.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_gqa_n1000.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_chartqa_val_n1000.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_textvqa_val_n1000.json
```

Interpretation: the balanced data/objective branch produced real OCR/text
capability movement under a larger slice: TextVQA improves by +3.8 exact / +3.47
soft over the V11 n=1000 recheck, and ChartQA improves by +1.9. This is a
capability-gaining but control-leaky model, not a clean replacement candidate:
GQA falls to 42.2 versus V11's prior 44.9 n=1000 search result.

Current status: stopped after the planned 3000 steps. Retention VQA/POPE screens
for checkpoint-3000 are complete.

Checkpoint-3000 retention screens:

| Eval | Result | V11 reference |
| --- | ---: | ---: |
| VQAv2 clean n1000 seed42 | 63.967 | V11 clean VQA n3000 mean 65.922 |
| VQAv2 blank n1000 seed42 | 39.300 | V11 blank n3000 seed42 39.078 |
| VQAv2 shuffled n1000 seed42 | 37.233 | V11 shuffled n3000 seed42 36.767 |
| VQAv2 wrong-image n1000 seed42 | 37.867 | V11 wrong-image n3000 seed42 37.178 |
| POPE adversarial n1000 seed42 | 80.000 | V11 POPE adversarial n1000 80.100 |
| POPE popular n1000 seed42 | 82.400 | - |

Retention artifacts:

```text
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_vqa_none_n1000_seed42.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_vqa_blank_image_n1000_seed42.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_vqa_shuffled_image_n1000_seed42.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_vqa_wrong_image_same_answer_type_n1000_seed42.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_pope_adversarial_n1000_seed42.json
/checkpoints/v15_qwen/balanced_lr2e6_ckpt3000_pope_popular_n1000_seed42.json
```

Interpretation: retention did not broadly collapse. Clean VQA is down versus
the V11 n3000 mean, corrupted-image controls are slightly worse but close to
V11 references, and POPE adversarial effectively ties V11. The main retention
cost is still GQA: 42.2 n1000 versus V11's 44.9 prior trusted n1000.

## Part B Controls And Branches

Vision-side co-adaptation branch:

```text
v15-qwen3-balancedckpt250-visionlast2-connlastblock-cachekl-lr1e6-3000
```

Configuration:

```text
init: balanced checkpoint-250
trainable connector prefixes: projector.layers.5, projector.norm
trainable vision prefixes:
  model.encoder.layers.25.self_attn
  model.encoder.layers.25.layer_norm1
  model.encoder.layers.25.layer_norm2
  model.encoder.layers.26.self_attn
  model.encoder.layers.26.layer_norm1
  model.encoder.layers.26.layer_norm2
Qwen frozen
cached_answer_tokens KL
lr=1e-6, grad_accum=8
```

Status: stopped after checkpoint-500 because checkpoint-250 and checkpoint-500
were flat/negative against the cheaper connector-only balanced run.

Checkpoint-250 n=200 results:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| vision-last2 + connector-last checkpoint-250 | 43.0 | 6.0 | exact 35.0 / soft 33.5 |
| vision-last2 + connector-last checkpoint-500 | 43.0 | 6.0 | exact 35.0 / soft 33.5 |

Artifacts:

```text
/checkpoints/v15_qwen/visionlast2_connlast_ckpt250_gqa_n200.json
/checkpoints/v15_qwen/visionlast2_connlast_ckpt250_chartqa_val_n200.json
/checkpoints/v15_qwen/visionlast2_connlast_ckpt250_textvqa_val_n200.json
/checkpoints/v15_qwen/visionlast2_connlast_ckpt500_gqa_n200.json
/checkpoints/v15_qwen/visionlast2_connlast_ckpt500_chartqa_val_n200.json
/checkpoints/v15_qwen/visionlast2_connlast_ckpt500_textvqa_val_n200.json
```

Interpretation: early vision-side co-adaptation is not yet better than the
connector-only balanced run. It is behind balanced checkpoint-500 on GQA,
ChartQA, and TextVQA. Checkpoint-500 has not recovered on any probe.

High-resolution frozen-SigLIP pilot:

```text
v15-qwen3-balancedckpt250-hires512-frozensiglip-connlastblock-cachekl-lr1e6-3000
```

Requested configuration:

```text
init: balanced checkpoint-250
vision_image_size=512
trainable connector prefixes: projector.layers.5, projector.norm
SigLIP frozen
Qwen frozen
cached_answer_tokens KL
```

Status: initial detached launch and wrapper foreground launch stopped or stuck
before creating a remote training task. The direct H100 implementation launch
also stalled at zero remote tasks and was stopped.

Follow-up launch:

```text
v15-qwen3-balancedckpt500-hires512-frozensiglip-connlastblock-cachekl-lr1e6-a100-3000
```

Status: stopped after checkpoint-250. It ran on A100-80GB:4 after H100 launches
repeatedly stalled before remote task creation. Confirmed
`vision_image_size=512`, frozen SigLIP/Qwen, trainable `projector.layers.5` and
`projector.norm`, cached answer-token KL loaded, and real training progress.

Checkpoint-250 n=200 results:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| hires512 frozen-SigLIP checkpoint-250 | 43.0 | 7.0 | exact 34.5 / soft 33.0 |

Artifacts:

```text
/checkpoints/v15_qwen/hires512_a100_ckpt250_gqa_n200.json
/checkpoints/v15_qwen/hires512_a100_ckpt250_chartqa_val_n200.json
/checkpoints/v15_qwen/hires512_a100_ckpt250_textvqa_val_n200.json
```

Interpretation: negative against the cheaper connector-only balanced branch.
It trails balanced checkpoint-500 on all three n=200 probes and trails balanced
checkpoint-2000 on ChartQA/TextVQA while also losing GQA. Higher pixel density
with frozen SigLIP and connector-last-block training is not the next hill.

Vision-only diagnostic:

```text
v15-qwen3-v11-visionlast2-only-connfrozen-cachekl-lr1e6-3000
```

Configuration:

```text
init: V11 frontier
connector frozen via connector_trainable_prefixes=freeze
trainable vision prefixes:
  model.encoder.layers.25.self_attn
  model.encoder.layers.25.layer_norm1
  model.encoder.layers.25.layer_norm2
  model.encoder.layers.26.self_attn
  model.encoder.layers.26.layer_norm1
  model.encoder.layers.26.layer_norm2
Qwen frozen
cached_answer_tokens KL on retention rows only
lr=1e-6, grad_accum=8
```

Status: stopped after checkpoint-500. This is the missing attribution branch
for testing whether vision-side updates can move capability with the V11
connector held fixed.

Checkpoint-250 n=200 results:

| Checkpoint | GQA | ChartQA val | TextVQA val |
| --- | ---: | ---: | ---: |
| vision-only connector-frozen checkpoint-250 | 45.5 | 6.0 | exact 31.5 / soft 30.17 |
| vision-only connector-frozen checkpoint-500 | 45.0 | 5.0 | exact 32.0 / soft 30.33 |

Artifacts:

```text
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt250_gqa_n200.json
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt250_chartqa_val_n200.json
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt250_textvqa_val_n200.json
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt500_gqa_n200.json
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt500_chartqa_val_n200.json
/checkpoints/v15_qwen/visiononly_connfrozen_ckpt500_textvqa_val_n200.json
```

Interpretation: useful attribution signal but not a capability win yet. With
the connector frozen, late-SigLIP updates preserve GQA much better than the
connector-only balanced run, nearly matching the V11 n=200 baseline, but they
do not produce the TextVQA/ChartQA improvement seen in the connector-only
balanced branch. Checkpoint-500 keeps that shape: GQA remains close to V11,
TextVQA is only slightly above V11, and ChartQA falls below V11.

## Final V15 Report

### Required Report Checklist

1. Final connector-isolation ablation: completed negative. The 128-token
   spatial-grid/no-position branch reached only 33.5 GQA n200 and 5.5 TextVQA
   exact at checkpoint-3000.
2. C1 auxiliary-branch diagnostic: no new V15 training-only scaffold was run.
   The V13/V14 historical diagnostic remains the current evidence:
   no-branch C1 reached 42.3 GQA n1000 / 41.033 n3000, while V11's
   branch-present disabled-at-eval path reached 44.9 n1000 / 42.6 n3000.
3. V11 baseline reconfirmation: V11 rechecked at n200 for GQA/ChartQA/TextVQA
   and n1000 for ChartQA/TextVQA; V11 GQA n1000 uses the prior trusted search
   result from `docs/STATUS.md`.
4. Data mix and loss assignment:
   - Retention replay: weight 30, CE plus V11 cached answer-token KL
     (`teacher_kl_weight=1.0`).
   - ChartQA: weight 25, CE only, KL disabled.
   - TextVQA: weight 20, CE only, KL disabled.
   - GQA spatial/relation: weight 15, CE only, KL disabled.
   - VQA counterfactual controls: weight 10, CE only, KL disabled.
   - The separate contrastive-answer-suppression dataset path exists, but the
     main balanced run used CE counterfactual rows rather than an activated
     contrastive loss.
5. High-resolution frozen-SigLIP pilot: completed negative at checkpoint-250
   on A100; 43.0 GQA, 7.0 ChartQA, exact 34.5 / soft 33.0 TextVQA.
6. Connector-only same-data control: completed to 3000; this is the main
   positive capability signal.
7. Vision-only diagnostic: completed to checkpoint-500 and stopped; it preserved
   GQA better but did not produce the TextVQA/ChartQA gains.
8. Joint vision + connector result: stopped after checkpoint-500; flat/negative
   against connector-only balanced.
9. Vision adapter scope/LR: last two SigLIP encoder blocks, self-attention and
   both layer norms, `lr=1e-6`, grad accumulation 8.
10. Connector trainable scope/LR:
    - Connector-only: full V3 connector, `lr=2e-6`, grad accumulation 4.
    - Joint branch: `projector.layers.5` and `projector.norm`, `lr=1e-6`,
      grad accumulation 8.
    - High-res branch: `projector.layers.5` and `projector.norm`, `lr=1e-6`.
    - Vision-only: connector frozen with the `freeze` sentinel.
11. Connector drift metrics: direct fixed-image connector-output cosine/MSE was
    not instrumented in this pass. Available drift proxies are clean VQA
    63.967, blank 39.300, shuffled 37.233, wrong-image 37.867, POPE
    adversarial 80.000, and GQA n1000 42.2 versus V11 references. Add a proper
    connector-output drift probe before treating any continuation as final.
12. ChartQA/TextVQA: checkpoint-3000 improved ChartQA n1000 from 7.8 to 9.7 and
    TextVQA n1000 from exact 28.5 / soft 26.47 to exact 32.3 / soft 29.93.
13. GQA search/confirm: checkpoint-3000 scored 42.2 on GQA n1000. It was not
    promoted to GQA confirm n3000 because it is already below V11's trusted
    44.9 n1000 search result.
14. VQAv2 clean/control and POPE retention: no broad collapse; clean VQA is
    lower than V11, controls are slightly worse, POPE adversarial ties V11.
15. Leakage audit: no new formal leakage audit was run for V15. Eval artifacts
    preserve train-source metadata and should be audited before any promotion.
16. Clear attribution: the gain is data/objective and connector adaptation, not
    vision adaptation. Connector-only gained TextVQA/ChartQA; vision-only did
    not; joint vision+connector and high-res frozen-SigLIP were weaker than the
    connector-only branch.
17. Recommendation: do not promote V15 checkpoint-3000 over V11. Continue the
    data/objective route only if the next run protects GQA more aggressively:
    raise retention replay/KL, lower connector LR, stop near checkpoint-500 for
    better GQA/ChartQA balance, or freeze more of the connector. Do not add Qwen
    visual-intake LoRA yet.

### Bottom Line

V15 changes the map. Pure connector replacement is now closed as the main hill,
and early vision adaptation did not beat the cheaper connector-only control.
The real signal is that retention-only KL plus capability CE can move TextVQA
and a little ChartQA without wrecking VQA/POPE, but GQA retention is not
protected enough. The next useful experiment is not broader architecture search;
it is a tighter version of the balanced connector/data recipe with stronger GQA
retention constraints and explicit connector-output drift logging.
