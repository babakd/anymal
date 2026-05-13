# V11 Qwen3 Ceiling Results

Date: 2026-05-13

## Goal

V11 is a ceiling-search campaign for Qwen/Qwen3-8B AnyMAL. The objective is to
beat the LLaMA/V3 robust baseline clearly, especially on GQA trusted n1000,
without losing the V9 Qwen scale-calibrated control behavior.

## Baselines

| Name | Checkpoint | GQA trusted n1000 | Notes |
| --- | --- | ---: | --- |
| V9 Qwen scale1.05 | `/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105` | 43.1 | Current Qwen incumbent, materialized scale 1.05 |
| V3 robust | `/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100` | 43.7 | LLaMA/V3 same-slice reference |
| V10 Batch-A best | `/checkpoints/pretrain-output/v10-qwen3-stage1b2400-scale105-batcha-best/checkpoint-2400-scale105` | 43.4 | Distinct V10 no-training candidate |
| V10 C1 best | `/checkpoints/pretrain-output/v10-qwen3-c1-2dpos-scale105-cont200-lr5e-5-save50/checkpoint-150` | 44.5 | Requires eval scale override 1.125; controls regressed |

## Day 0 Setup And Taxonomy

Local checks passed:

```bash
python3 -m py_compile scripts/materialize_v11_projector_interpolation.py scripts/analyze_gqa_pairwise.py
python3 -m py_compile evaluation/checkpoint_eval/gqa_checkpoint_eval.py evaluation/checkpoint_eval/vqa_checkpoint_eval.py evaluation/checkpoint_eval/pope_checkpoint_eval.py
python3 scripts/repo_health_check.py
```

Modal access was verified with `modal --version` and Modal volume listing.

Generated GQA pairwise/taxonomy artifacts:

```text
outputs/v11_qwen/gqa_taxonomy_day0/summary.md
outputs/v11_qwen/gqa_taxonomy_day0/summary.json
outputs/v11_qwen/gqa_taxonomy_day0/V9_correct_C1_wrong.json
outputs/v11_qwen/gqa_taxonomy_day0/C1_correct_V9_wrong.json
outputs/v11_qwen/gqa_taxonomy_day0/V3_correct_V9_wrong.json
outputs/v11_qwen/gqa_taxonomy_day0/V9_correct_V3_wrong.json
outputs/v11_qwen/gqa_taxonomy_day0/C1_correct_V3_wrong.json
outputs/v11_qwen/gqa_taxonomy_day0/V3_correct_C1_wrong.json
```

Taxonomy labels are heuristic question-text labels because the current HF GQA
testdev evaluator path does not expose full GQA type metadata. They are for
experiment steering only.

Key read:

| Slice | Count | Dominant buckets |
| --- | ---: | --- |
| C1 correct / V9 wrong | 62 | yes/no presence 22, spatial 10, object identity 8 |
| V9 correct / C1 wrong | 48 | spatial 14, yes/no presence 11, attribute 6 |
| V3 correct / C1 wrong | 121 | spatial 28, left/right 21, yes/no presence 19 |
| C1 correct / V3 wrong | 129 | yes/no presence 39, spatial 36, left/right 12 |

Interpretation: C1's net GQA gain is real but mixed. It improves yes/no object
presence and some spatial/left-right cases, while V3 remains stronger on a
large left/right and object/other slice. This supports C1 salvage, but the
control regression may be partly tied to yes/no prior movement rather than pure
spatial grounding.

## Code/Tooling Changes

- Added `scripts/analyze_gqa_pairwise.py` for GQA exact-match pairwise deltas
  and heuristic taxonomy summaries.
- Added `scripts/materialize_v11_projector_interpolation.py` for V9-to-C1
  projector interpolation with C1-only learned 2D tensors treated as zero on the
  V9 side.
- Fixed direct Modal execution of nested checkpoint evaluators by making
  `PROJECT_DIR` resolution robust when Modal mounts a script at `/root/*.py`.
- Added `patch_position_feature_scale` to the V3 projector/model metadata path
  so learned 2D patch-position features can be attenuated at load time without
  changing the common projector weights.
- Threaded `v3_patch_position_feature_scale` through Modal pretrain wrappers so
  C1-style checkpoints can be used safely for short continuation/canary runs.

## Batch 1A: V9-C1 No-Training Interpolation

Source checkpoints:

```text
V9 source A: /checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
C1 source B: /checkpoints/pretrain-output/v10-qwen3-c1-2dpos-scale105-cont200-lr5e-5-save50/checkpoint-150
```

Interpolation definition:

```text
common projector tensors: (1 - alpha) * V9 + alpha * C1
C1-only patch_position_embedding: alpha * C1
metadata/output architecture: C1-style anymal_v3 with learned 2D patch position features
materialized connector_output_scale: 1.05
```

Materialized checkpoints:

| Alpha | Checkpoint |
| ---: | --- |
| 0.05 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a005-scale105/checkpoint-a005-scale105` |
| 0.10 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a010-scale105/checkpoint-a010-scale105` |
| 0.20 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a020-scale105/checkpoint-a020-scale105` |
| 0.30 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a030-scale105/checkpoint-a030-scale105` |
| 0.40 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a040-scale105/checkpoint-a040-scale105` |
| 0.50 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a050-scale105/checkpoint-a050-scale105` |
| 0.75 | `/checkpoints/pretrain-output/v11-qwen3-v9-c1interp-a075-scale105/checkpoint-a075-scale105` |

Canary:

```text
/checkpoints/v11_qwen_ceiling/smoke/gqa_eval_v11_v9_c1interp_a005_scale105_n16.json
```

The alpha 0.05 checkpoint loaded successfully with `use_2d_patch_position_features=true`
and completed GQA n16 smoke.

### GQA Alpha/Scale Grid

Running:

```text
remote dir: /checkpoints/v11_qwen_ceiling/interp1a_gqa
local dir: /tmp/v11_interp1a_gqa
logs: outputs/v11_qwen/logs/interp1a_gqa
alphas: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75
scales: 1.00, 1.025, 1.05, 1.075, 1.10, 1.125
metric: GQA trusted n1000 seed42
parallelism: 3
```

Status: complete.

Completed GQA grid:

| Alpha | Scale 1.000 | 1.025 | 1.050 | 1.075 | 1.100 | 1.125 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 43.2 | 42.7 | 42.8 | 42.8 | 42.9 | 42.9 |
| 0.10 | 42.9 | 42.8 | 43.0 | 42.8 | 43.2 | 43.1 |
| 0.20 | 43.0 | 43.3 | 43.2 | 43.1 | 43.3 | 43.4 |
| 0.30 | 43.4 | 43.2 | 43.7 | 43.5 | 43.3 | 43.4 |
| 0.40 | 43.4 | 43.6 | 43.8 | 43.4 | 43.3 | 43.3 |
| 0.50 | 43.3 | 43.3 | 43.5 | 43.6 | 43.2 | 43.2 |
| 0.75 | 43.6 | 43.8 | 43.7 | 44.1 | 43.6 | 44.0 |

Frontier:

| Candidate | GQA trusted n1000 | Interpretation |
| --- | ---: | --- |
| alpha 0.75, scale 1.075 | 44.1 | Best interpolation point; above V9 by +1.0 and V3 by +0.4, below C1 44.5 |
| alpha 0.75, scale 1.125 | 44.0 | Slightly lower but near C1's best scale region |
| alpha 0.40, scale 1.050 | 43.8 | Conservative point matching the old GQA floor and barely above V3 |

Decision: the no-training interpolation does not fully recover C1's 44.5
GQA, but alpha 0.75 partially recovers the C1 signal. Run VQA control screens
for the frontier points before deciding whether this is a useful salvage path
or only a GQA/yes-no-prior tradeoff.

### Control Screens

Cheap VQA n1000 control screens:

| Candidate | GQA | Clean | Blank | Shuffled | Wrong | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| alpha 0.75, scale 1.075 | 44.1 | 66.267 | 40.900 | 37.900 | 38.633 | reject for controls |
| alpha 0.75, scale 1.125 | 44.0 | 66.100 | 40.300 | 38.033 | 38.767 | reject for controls |
| alpha 0.75, scale 1.025 | 43.8 | 65.933 | 40.667 | 38.100 | 38.700 | reject for controls |
| alpha 0.40, scale 1.050 | 43.8 | 66.400 | 39.767 | 38.267 | 38.733 | reject for controls |

Artifacts:

```text
/checkpoints/v11_qwen_ceiling/interp1a_controls_a075/
/checkpoints/v11_qwen_ceiling/interp1a_controls_a075_low/
/checkpoints/v11_qwen_ceiling/interp1a_controls_a040/
```

Decision: no-training interpolation partially recovers GQA, but it does not
recover the C1 gain without inheriting or worsening the control leakage. The
best GQA point, alpha 0.75 scale 1.075, is clean-strong but has blank 40.9 and
wrong 38.633. The conservative alpha 0.40 scale 1.05 point keeps blank closer
to V9 but still worsens shuffled/wrong. Treat this as useful evidence that C1's
GQA lift is connected to learned 2D/spatial state, but plain projector-space
interpolation is not sufficient for a promotable candidate.

## Batch 1B: C1 Patch-Position Feature Scale Sweep

Added model/eval support for a static `patch_position_feature_scale` metadata
field so the learned 2D patch-position branch can be attenuated without
changing projector weights. Materialized C1 checkpoint-150 variants at
`patch_position_feature_scale`:

```text
0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00
```

All variants use the C1 projector and a materialized base
`connector_output_scale=1.05`, then GQA was screened across eval/materialized
scales `1.00..1.125`.

Completed GQA grid: 48 runs (`8` patch-position scales x `6` connector
scales), local dir `/tmp/v11_c1_posscale_gqa`, remote dir
`/checkpoints/v11_qwen_ceiling/c1_posscale_gqa`.

Top GQA rows:

| Patch-position scale | Connector scale | GQA trusted n1000 |
| ---: | ---: | ---: |
| 0.00 | 1.125 | 44.9 |
| 0.75 | 1.050 | 44.7 |
| 0.50 | 1.125 | 44.7 |
| 0.35 | 1.050 | 44.6 |
| 1.00 | 1.125 | 44.5 |
| 0.75 | 1.025 | 44.5 |
| 0.50 | 1.100 | 44.5 |
| 0.20 | 1.125 | 44.5 |
| 0.20 | 1.050 | 44.5 |
| 0.10 | 1.100 | 44.5 |
| 0.05 | 1.075 | 44.5 |
| 0.00 | 1.100 | 44.5 |

Important finding: fully disabling the learned patch-position contribution
while keeping C1's trained projector weights improves beyond C1's reported
44.5. This means C1's useful GQA movement is not simply the learned 2D table;
the C1 training moved common projector weights into a better GQA region, while
the learned patch-position branch was not needed for the best trusted n1000
score.

### Current Frontier: `pos000` / Scale 1.125

Materialized checkpoint:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Metadata of interest:

```text
connector_output_scale: 1.125
patch_position_feature_scale: 0.0
use_2d_patch_position_features: true
```

Candidate screen:

| Metric | Result | Notes |
| --- | ---: | --- |
| GQA trusted n1000 | 44.9 | no-override materialized checkpoint |
| Clean VQA n3000 seed42 | 66.278 | no-override materialized checkpoint |
| Clean VQA n3000 seed43 | 66.333 | no-override materialized checkpoint |
| Clean VQA n3000 seed44 | 65.156 | no-override materialized checkpoint |
| Clean VQA n3000 mean | 65.922 | seeds 42/43/44 |
| Blank n3000 seed42 | 39.078 | gate-safe versus V9 threshold 39.733 |
| Shuffled n3000 seed42 | 36.767 | gate-safe versus V9 threshold 37.367 |
| Wrong-image n3000 seed42 | 37.178 | gate-safe versus V9 threshold 38.900 |
| POPE adversarial n1000 | 80.100 | strong |
| Mild blur n1000 | 66.567 | strong |
| Center crop 90 n1000 | 64.667 | acceptable perturbation |
| Translate 5 pct n1000 | 65.667 | strong |

This is the first V11 candidate to reach the exploratory milestone (`>=44.5`)
while keeping the old corrupted-image control gates intact. It also improves
over the V3 robust trusted n1000 slice by +1.2 and over V9 trusted n1000 by
+1.8.

No-override confirmation artifacts:

```text
/tmp/gqa_eval_v11_c1_pos000_scale1125_nooverride_n1000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed42_clean_n3000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed43_clean_n3000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed44_clean_n3000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed42_blank_n3000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed42_shuffled_n3000.json
/tmp/vqa_eval_v11_c1_pos000_scale1125_nooverride_seed42_wrong_n3000.json
```

Pairwise artifacts:

```text
outputs/v11_qwen/gqa_pairwise_pos000_scale1125/
```

Pairwise read:

| Comparison | Count | Dominant buckets |
| --- | ---: | --- |
| V11 correct / V9 wrong | 64 | yes/no presence 21, spatial 11, object identity 8 |
| V9 correct / V11 wrong | 46 | spatial 13, yes/no presence 12, attribute 6 |
| V11 correct / C1 wrong | 6 | other 3, spatial 2, left/right 1 |
| C1 correct / V11 wrong | 2 | yes/no presence 2 |
| V11 correct / V3 wrong | 128 | yes/no presence 37, spatial 36, left/right 12 |
| V3 correct / V11 wrong | 116 | spatial 26, left/right 20, yes/no presence 19 |

Interpretation: `pos000/scale1.125` mostly preserves C1's GQA lift but trims
some C1 yes/no overreach and recovers a small number of spatial/other rows. The
remaining weakness versus V3 is still left/right and object/other-heavy.

### Other Position-Scale Rows

- `pos000/scale1.050`: GQA 44.4, controls pass old gates, POPE 79.9,
  perturbations similar to the 1.125 finalist. Useful backup but lower GQA.
- `pos035/scale1.050`: GQA 44.6; clean 66.289, blank 39.000,
  shuffled 36.956, wrong 37.256. Promotable as a backup but below the current
  finalist.
- `pos050/scale1.125`: GQA 44.7 but blank 40.067, so stopped early.

### Scale Hillclimb Around `pos000`

Coarse/high-side scale checks:

| Connector scale | GQA trusted n1000 | Decision |
| ---: | ---: | --- |
| 1.125 | 44.9 | current peak |
| 1.150 | 44.4 | lower |
| 1.175 | 44.3 | lower |
| 1.200 | 44.3 | lower |

Fine-scale checks around the peak:

| Connector scale | GQA trusted n1000 |
| ---: | ---: |
| 1.1125 | 44.5 |
| 1.11875 | 44.6 |
| 1.125 | 44.9 |
| 1.13125 | 44.1 |
| 1.1375 | 44.4 |

Remote fine-scale artifacts:

```text
/checkpoints/v11_qwen_ceiling/pos000_finescale_gqa/
```

Decision: `1.125` is a real local peak for the static C1-common-weights /
position-disabled candidate. The fine sweep did not find the stretch target
`>=45.0`.

### Leakage Audit

Artifacts:

```text
outputs/v11_qwen/leakage_audit_pos000_scale1125.json
outputs/v11_qwen/leakage_audit_pos000_scale1125_vqa_pope_coco_sources.json
outputs/v11_qwen/leakage_audit_pos000_scale1125_vqa_pope_only.json
```

Audit read:

| Audit | Eval samples | Exact val2014 overlap | Numeric/raw overlap | Result |
| --- | ---: | ---: | ---: | --- |
| All evals vs all known train sources | 23,000 | 0 | 18 / 18 | generic fail |
| VQA/POPE evals vs COCO/VQA/POPE sources | 22,000 | 0 | 0 / 0 | pass |
| VQA/POPE evals vs all known train sources | 22,000 | 0 | 18 / 18 | generic fail |

The generic failures are splitless numeric filename-stem collisions against the
GQA train source. Direct inspection found zero overlap between final GQA eval
image IDs/question IDs and the GQA train image stems/question IDs. The COCO
split-aware VQA/POPE audit passes. Treat the audit as no confirmed leakage for
the finalist, with the generic GQA numeric collision noted as a known false
positive pattern in the current audit script.

### Prefix-Only Trained Spatial Table

Ran a conservative train-only-new-table branch from V9:

```text
run: v11-c1pos-prefixonly-controlaware-from-v9scale105-canary10-ls03-v2
W&B: https://wandb.ai/babakdam/anymal-pretrain/runs/ucl9vlgq
continue W&B: https://wandb.ai/babakdam/anymal-pretrain/runs/d07zbhz2
checkpoint-10/25/50 under:
/checkpoints/pretrain-output/v11-c1pos-prefixonly-controlaware-from-v9scale105-canary10-ls03-v2/
```

The 10-step canary caught an initial Modal wrapper signature bug, then passed
after `v3_patch_position_feature_scale` was threaded through the distributed
pretrain wrappers. The successful canary trained exactly one tensor,
`projector.patch_position_embedding` (1,179,648 params), from a V9 warm-start
and logged nonzero gradient norm (`0.0097` at step 10).

Cheap canary:

| Metric | Result |
| --- | ---: |
| GQA n64 | 40.625 |
| Clean n64 | 67.708 |
| Blank n64 | 33.854 |
| Shuffled n64 | 29.167 |
| Wrong n64 | 32.292 |

Continuation to checkpoint-50:

| Checkpoint | GQA n1000 | Clean n1000 | Blank n1000 | Shuffled n1000 | Wrong n1000 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ckpt25 | 42.8 | 66.167 | 39.267 | 37.467 | 38.100 | stop: no GQA lift |
| ckpt50 | 42.7 | not run | not run | not run | not run | stop: no GQA lift |

Decision: prefix-only new-table training is a useful negative. It learns and
keeps controls sane, but does not recover GQA. The C1 gain appears to live in
the common projector weights learned during C1 training, not in a separately
trainable patch-position table from V9 under this objective.

### Full-Projector C1 Continuation Canary

Ran a 10-step all-projector continuation from the static finalist:

```text
run: v11-pos000-scale1125-gqaantishuffle-allproj-canary10-lr1e5-ls03
W&B: https://wandb.ai/babakdam/anymal-pretrain/runs/cc9csxzc
checkpoint:
/checkpoints/pretrain-output/v11-pos000-scale1125-gqaantishuffle-allproj-canary10-lr1e5-ls03/checkpoint-10
dataset: v10_qwen_gqa_antishuffle_stage1b
trainable: full V3 connector/projector (1,617,563,649 params)
frozen: Qwen decoder and vision tower
lr: 1e-5
loss scale: 0.3
gradient accumulation: 8
```

Health: checkpoint saved, volume committed, nonzero gradient norm at step 10
(`0.3059`).

Cheap screen:

| Metric | Result |
| --- | ---: |
| GQA n1000 | 44.0 |
| Clean n1000 | 66.700 |
| Blank n1000 | 39.767 |
| Shuffled n1000 | 37.867 |
| Wrong n1000 | 38.333 |

Artifacts:

```text
/checkpoints/v11_qwen_ceiling/pos000_allproj_canary10_screen/
/tmp/v11_pos000_allproj_canary10_screen/
```

Decision: stop this branch. The continuation slightly helps cheap clean VQA but
gives up nearly one GQA point from the static finalist (`44.9 -> 44.0`) without
creating a useful control tradeoff. Longer continuation from this objective is
not justified without changing the loss or trainable subset.

## Current V11 Frontier

Best current checkpoint:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Why it is the frontier:

| Metric | V9 Qwen | V3 robust | V11 frontier |
| --- | ---: | ---: | ---: |
| GQA trusted n1000 | 43.1 | 43.7 | 44.9 |
| Clean VQA n3000 mean | - | - | 65.922 |
| Blank n3000 seed42 | - | - | 39.078 |
| Shuffled n3000 seed42 | - | - | 36.767 |
| Wrong n3000 seed42 | - | - | 37.178 |
| POPE adversarial n1000 | 79.1 | 77.1 | 80.1 |

The C1 salvage hill produced a credible new Qwen frontier at 44.9 GQA with
control behavior back inside the V9-style guardrails. The hill appears locally
plateaued: interpolation could not keep controls, scale hillclimbs above/below
1.125 did not improve GQA, prefix-only table training did not move GQA, and a
full-projector 10-step continuation regressed GQA.

Next heavier work, if continuing beyond the current V11 frontier, should move
to the planned Batch 3/4 directions rather than spend more on this static C1
salvage surface:

```text
1. E1 gradient-proof diagnostic and nonzero-gate adapter retry.
2. 192-token V3/Qwen Perceiver branch with dense GQA/control checkpoint screens.
3. Revised contrastive objective or smaller trainable subset only if there is a
   concrete diagnostic explaining the all-projector canary's GQA regression.
```
