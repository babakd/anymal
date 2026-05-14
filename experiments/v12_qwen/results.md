# V12 Qwen3 Aggressive Ceiling Search Results

Date: 2026-05-13

## Objective

V12 is an aggressive Qwen/Qwen3-8B ceiling search. The campaign starts from the
V11 frontier and prioritizes mechanisms that can move compositional GQA before
requiring final promotion controls.

Default base checkpoint named by the V12 plan:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Operational nuance: the historical `44.9` GQA result was reproduced from the
scale-1.05 V11 checkpoint with eval-time connector scale `1.125` and the trusted
answer-only system prompt. V12 bridge materializations therefore use:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale105/checkpoint-posscale000-scale105
connector_output_scale_override = 1.125
```

## Setup Checks

Local checks passed:

```bash
python3 -m compileall -q models training evaluation data scripts
python3 scripts/repo_health_check.py
python3 -m py_compile models/anymal_v3.py models/anymal_v4.py scripts/modal/train.py tests/test_model.py scripts/materialize_v12_token_budget_checkpoint.py scripts/v12_vxattn_gradient_proof.py
python3 -m py_compile training/trainer.py evaluation/checkpoint_eval/gqa_checkpoint_eval.py scripts/materialize_v12_gate_checkpoint.py
python3 -m py_compile models/anymal_v4.py scripts/modal/train.py evaluation/checkpoint_eval/gqa_checkpoint_eval.py scripts/modal_repo_smoke.py
python3 -m py_compile models/anymal_v4.py evaluation/checkpoint_eval/gqa_checkpoint_eval.py
```

Local `pytest` is blocked in the current local environment because `torch` is
not installed.

## Frontier Leaderboard

| Candidate | Checkpoint | Connector scale | Patch-pos scale | Image tokens | GQA n1000 | Clean VQA | Blank | Shuffled | Wrong | POPE | Perturb mean | Generation hygiene | Leakage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V3 robust | `/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100` | - | - | 128 | 43.7 | 62.967 n1000 | 39.733 | 37.367 | 38.900 | 77.100 | 60.189 | pass | historical pass |
| V9 Qwen scale1.05 | `/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105` | 1.05 | none | 128 | 43.1 trusted | 66.133 n3000 seed42 | 38.811 n3000 | 36.900 n3000 | 37.967 n1000 | 79.100 | 66.856 | pass | pass |
| V10 Batch-A best | `/checkpoints/pretrain-output/v10-qwen3-stage1b2400-scale105-batcha-best/checkpoint-2400-scale105` | 1.05 | none | 128 | 43.4 | - | - | - | - | - | - | pass in V10 GQA screen | not audited |
| V10 C1 best | `/checkpoints/pretrain-output/v10-qwen3-c1-2dpos-scale105-cont200-lr5e-5-save50/checkpoint-150` | 1.05 materialized, 1.125 eval best | 1.0 | 128 | 44.5 | - | 40.0 | 37.8 | 38.5 | - | ~65.6 | pass in V10 GQA screen | not promoted |
| V11 frontier | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` | 1.125 | 0.0 | 128 | 44.9 | 65.922 n3000 mean | 39.078 n3000 | 36.767 n3000 | 37.178 n3000 | 80.100 | 65.633 | pass | no confirmed leakage |

Notes:

- V10 C1 is listed as an informative high-GQA, control-leaky branch rather than
  a promotion candidate.
- V11 remains the starting frontier, not a stopping point.

## Launch Ledger

### Implemented Fixes And Infrastructure

| Area | Status |
| --- | --- |
| Token-budget init | Added materialization script for expanded image-token checkpoints. |
| Visual cross-attn proof | Added gradient-proof script and A0 diagnostics path. |
| Visual cross-attn dtype | Fixed dtype cast handling. |
| Patch-position metadata | `scripts/modal/train.py` preserves metadata by not normalizing `None` to `none`. |
| Frozen connector + vxattn | Disabled gradient checkpointing for frozen-connector visual cross-attn runs. |
| Vxattn + connector-prefix DDP | Disabled decoder gradient checkpointing for all V3 visual-cross-attention Stage 1 runs after a connector-prefix smoke exposed DDP unused-parameter failures. |
| Track D spatial residual | Added spatial residual branch plus `train.py` flags and tests. |
| AnyMALv4 compatibility | Patched compatibility defaults and `encode_images` question-summary acceptance/diagnostics after the first E2 failure. |
| Higher-resolution eval/train | Added `vision_image_size` metadata, training, and GQA eval plumbing for V3/V4. |
| Checkpoint resume robustness | Trainer now tolerates scheduler-state restore mismatch and resumes model/optimizer state from earlier checkpoints. |
| GQA eval diagnostics | Added model-load progress prints and moved V3 visual cross-attention adapters onto the eval device. |
| Gate materialization | Added script for V12 visual cross-attn and spatial-residual gate overrides/materializations. |
| Query-conditioned patch selector | Added question-summary conditioning path and residual query selector screens. |
| Vision-side adapter | Added prefix-only SigLIP training, `vision_adapter.pt` save/load, and checkpoint/eval restore plumbing. |
| V3-to-V4 DeepStack bridge | Added bridge materializer that copies V3 projector weights into V4 DeepStack checkpoints, maps V3 latents into global/local latents, and rejects incompatible V4 positional-feature settings. |
| Trusted generation prompt | GQA checkpoint eval now defaults to the historical answer-only prompt used by the V11 trusted result. |
| Modal mount stability | Training, repo smoke, and GQA eval mounts now ignore `.git` to avoid build failures from changing git objects. |
| V4 checkpoint metadata | V4 `save_pretrained()` now writes decoder metadata such as `llm_backbone`, while old metadata-less V4 checkpoints load with an explicit warning. |
| V4 eval scale override | V4 `from_pretrained()` now permits an explicit eval-time `connector_output_scale` override without weakening shape/architecture metadata guards. |
| Continuation semantics | Confirmed `--resume-checkpoint` restores optimizer/scheduler/global step; reset-optimizer hillclimbs must use `--pretrain-checkpoint`. |
| Spatial-focus GQA data | Added V12 GQA spatial/relation focused dataset builders and direct/contrastive Stage 1 dataset options. |

### A0 Proof

| Artifact | Layers | Adapter dim | Gate | Result |
| --- | --- | ---: | ---: | --- |
| `/checkpoints/v12_qwen/vxattn_gradient_proof_gate1e3_midupper_v3_nockpt.json` | 12,18,24,30 | 512 | `1e-3` | Passed: nonzero grads/deltas and logits changed. |

### Materialized Initializations

| Branch | Artifact |
| --- | --- |
| B0/B1 192 tokens | `/checkpoints/pretrain-output/v12-qwen3-v11-192tok-copy-noise/checkpoint-init` |
| B0/B2 256 tokens | `/checkpoints/pretrain-output/v12-qwen3-v11-256tok-copy-noise/checkpoint-init` |
| V4 DeepStack one-level bridge `[-1]` | `/checkpoints/pretrain-output/v12-v4deepbridge1-v11scale105ov1125-split64-d4096/checkpoint-init` |
| V4 DeepStack three-level bridge `[-3,-2,-1]` | `/checkpoints/pretrain-output/v12-v4deepbridge3-v11scale105ov1125-split64-d4096/checkpoint-init` |
| V4 DeepStack two-level bridge `[-2,-1]` | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m2m1-d4096/checkpoint-init` |
| V4 DeepStack two-level bridge `[-3,-1]` | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m3m1-d4096/checkpoint-init` |
| V4 DeepStack two-level bridge `[-4,-1]` | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m4m1-d4096/checkpoint-init` |

## Completed GQA Screens

All completed screens below are trusted GQA n1000 at connector scale `1.125`.
Generation hygiene passed for each: EOS rate `1.0`, max-token hit `0`, assistant
prefix rate `0`.

| Branch | GQA n1000 | Note |
| --- | ---: | --- |
| V11 frontier | 44.9 | Baseline to beat. |
| A1 vxattn mid10 | 44.4 | Visual cross-attn mid-layer screen. |
| A1 vxattn upper10 | 44.5 | Tied best V12 screen so far. |
| A1 vxattn mid gate1e-2 | 42.8 | Hard negative; higher mid-layer gate regressed. |
| A1 vxattn upper gate1e-4 | 44.6 | Best V12 screen so far, still below V11. |
| A1 vxattn upper gate1e-4 eval calibration | 45.0 n1000 / 43.2 n3000 | Best cell was gate multiplier `0.5` at connector scale `1.125`; tied V11 on matched n3000, so not a robust ceiling break. |
| A1 vxattn upper gate1e-4 cont25 | 43.8 | Continuation regressed. |
| A1 vxattn upper gate1e-4 cont100 | 42.6 | Longer continuation regressed hard. |
| A1 vxattn upper gate1e-4 d1024 | 43.1 | Wider adapter changed answers but regressed vs d512. |
| A1 vxattn d512 gate sweeps | smoke only | Constant gates and learned-pattern multipliers did not beat V11 on n128; no n1000 promotion. |
| C2 last2_10 | 44.3 | Last-2 branch. |
| C3 scaleglobal10 | 44.2 | Scale-global branch. |
| C1 latents10 | 44.5 | Tied best V12 screen so far. |
| C1 latents_cont25 | 44.4 | Continuation did not improve. |
| C1 cont50 | 44.5 | Tied best V12 screen so far. |
| C1 cont75 | 44.2 | Regressed. |
| C1 cont100 | 44.4 | Did not recover ceiling. |
| C4 input_proj | 44.3 | Input-projection-only canary did not move ceiling. |
| C5 cross-attn | 44.5 | Perceiver cross-attn-only canary tied prior V12 cluster. |
| C5 cross-attn cont25 | 44.3 | Continuation regressed. |
| C5 cross-attn cont100 | 44.0 | Longer continuation regressed. |
| C6 last2+input_proj | 44.3 | Combined subset did not move ceiling. |
| D1 spatial10 | 44.5 | Tied best V12 screen so far. |
| D1 spatial_cont75 | 44.5 | Continuation tied prior D1 result. |
| D1 spatial_cont100 | 44.3 | Longer continuation regressed. |
| D2 spatial gate0.005 | 42.5 | Gate increase hurt full-slice GQA, especially left/right/spatial. |
| D2 spatial gate0.05 | 0.0 smoke | Broke generation on n128; EOS `0.055`, max-token-hit `0.945`. |
| E1 V11 eval-time highres448 | 42.7 | Higher-res eval without training regressed. |
| E1 V3 highres448 trained10 | 42.7 | Training at 448 did not recover the eval-time loss. |
| E2 V4 deepstack128 | 23.5 | Hard negative. |
| E2 V4 deepstack192 | 0.0 | Fresh V4/Qwen DeepStack, not V11-warm-start comparable; generation broke. |
| E3 vision last1 step25 | 42.9 | Technically clean but regressed; adapter save/load verified with 16 tensors. |
| E3 vision last2 step25 | 43.0 | Technically clean but regressed; adapter save/load verified with 32 tensors. |
| B0 192init | 39.9 | Token-budget init regressed. |
| B1 192tok10 | 38.0 | Blunt full-connector token-budget update regressed badly. |
| B0 256init | 42.1 | Token-budget init below V11. |
| B2 256tok10 | 39.4 | Blunt full-connector token-budget update regressed badly. |
| Q-selector residual0.25 step25 | 42.7 | Question-conditioned token selection damaged V11. |
| Q-selector residual0.25 step50 | 43.0 | Continuation did not recover. |
| Q-selector residual0.10 step25 | 43.1 | Attenuation still below V11. |
| Q-selector residual0.05 step25 | 43.2 | Best attenuated query selector, still a hard negative. |
| F1 q/v LoRA GQA-preserving | 44.3 | Decoder-side q/v LoRA did not move ceiling. |
| F2 q/v LoRA contrastive | 43.8 | Contrastive q/v LoRA regressed. |
| F3 q/k/v/o LoRA GQA-preserving | 44.0 | Wider LoRA target set regressed. |
| F4 q/k/v/o LoRA contrastive | 42.8 | Harder decoder-side contrastive regression. |
| V4 DeepStack bridge `[-3,-1]` init | 41.4 | Clean true bridge; generation break from fresh E2 does not reproduce. |
| V4 DeepStack bridge `[-3,-1]` step25 lr5e-5 | 42.5 | n128 rose to `44.531`, but n1000 stayed below V11. |
| V4 DeepStack bridge `[-3,-1]` plus25 reset lr5e-5 | 43.0 | Reset-optimizer continuation improved n1000 by +0.5 over step25, still below V11. |
| V4 DeepStack bridge `[-3,-1]` plus25 scale sweep | 42.7-43.0 | Eval scales `1.05/1.10/1.15/1.175`; best was `1.15` at `43.0`, all clean. |
| V4 DeepStack bridge `[-3,-1]` changed continuation | 42.7 | Late-layer/level/norm contrastive continuation from plus25 reset; clean but below the scale sweep. |
| V4 DeepStack bridge `[-3,-1]` plus50 reset lr5e-5 | 42.1 | Same n128 as plus25 but worse n1000; stop extending this branch for now. |
| V4 DeepStack bridge `[-4,-1]` step25 lr5e-5 | 40.7 | Adjacent bridge tied n128 but failed n1000; do not continue. |
| Spatial-focus full connector step25 lr1e-5 | 39.844 n128 | Clean generation but hard negative. |
| Spatial-focus last2+out step25 lr2e-5 | 40.625 n128 | Clean generation but hard negative. |
| RMS-preserving V11 last1 step25 lr5e-6 | 42.188 n128 | Clean but flat; identical first-slice score to several dead branches. |
| V4 one-level global/local bridge step25 lr2e-5 | 42.188 n128 | Clean global/local geometry test; no evidence for n1000 promotion. |
| Coord-MLP patch-position only step25 lr1e-4 | 42.188 n128 | Smooth coordinate path trained only 149k params; low-gradient warning and no gain. |
| Vxattn upper + last1/norm step25 lr1e-5 | 42.188 n128 | Mechanically viable after checkpointing/batch fixes; clean generation but no first-slice gain. |
| 256-token l5norm continuation step25 lr5e-6 | 42.1 | Larger image-token budget with normalized last-layer update regressed cleanly. |
| Per-token output-scale continuation | 44.1 | Learned nonzero per-token scale and generated cleanly, but stayed below V11. |
| C1 mining step150 scale1125 | 44.9 n1000 / 43.2 n3000 | Rediscovered the V11 basin; matched V11 exactly on the same n3000 check. |
| Spatial-contrastive last1+out | 43.7 | Best of lr/step grid; clean but below V11 and worsening with continuation. |
| V11 eval-time highres512 | 43.3 | Higher-resolution eval without promotion training regressed. |
| Token-budget clean-init static grid | 40.1 | 192/256-token clean inits all far below V11. |
| Joint SigLIP last1 + projector tail | 41.406 n128 | Full final SigLIP block plus projector tail trained cleanly but failed the cheap gate. |
| V11/C1 step150 projector soup | 42.969 n128 | Local projector interpolation did not reveal an intermediate hill. |
| Attention-only SigLIP last1 + projector tail | 42.969 n128 | Lighter final-block attention/norm unfreeze trained cleanly but failed the cheap gate. |

## DeepStack Bridge Screens

The early E2 DeepStack failures should no longer be read as evidence that
multi-level visual features inherently break generation. They were effectively
fresh V4/Qwen visual-conditioning runs. True V3-to-V4 bridge controls generated
cleanly and isolated feature-level changes from architecture/metadata drift.

Trusted n128 bridge screens:

| Branch | Checkpoint | GQA n128 | Note |
| --- | --- | ---: | --- |
| One-level `[-1]` init | `/checkpoints/pretrain-output/v12-v4deepbridge1-v11scale105ov1125-split64-d4096/checkpoint-init` | 42.188 | Matches old first-128 behavior; clean generation. |
| Three-level `[-3,-2,-1]` init | `/checkpoints/pretrain-output/v12-v4deepbridge3-v11scale105ov1125-split64-d4096/checkpoint-init` | 40.625 | Earlier feature injection is not free at init. |
| Three-level `[-3,-2,-1]` step25 lr5e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge3-split64-step25-lr5e5-ls03-v1/checkpoint-25` | 41.406 | Small recovery, not promotable. |
| Three-level `[-3,-2,-1]` step25 lr1e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge3-split64-step25-lr1e5-ls03-v1/checkpoint-25` | 39.844 | Lower LR underfit/regressed. |
| Two-level `[-2,-1]` init | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m2m1-d4096/checkpoint-init` | 40.625 | Same first-screen score as three-level init. |
| Two-level `[-3,-1]` init | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m3m1-d4096/checkpoint-init` | 42.188 | Best init variant; promoted to n1000 and training. |
| Two-level `[-3,-1]` step25 lr5e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge2-m3m1-split64-step25-lr5e5-ls03-v1/checkpoint-25` | 44.531 | First strong n128 bridge movement. |
| Two-level `[-3,-1]` step25 lr2e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge2-m3m1-split64-step25-lr2e5-ls03-v1/checkpoint-25` | 40.625 | Hard negative. |
| Two-level `[-3,-1]` plus25 reset lr5e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge2-m3m1-split64-plus25reset-lr5e5-ls03-v2/checkpoint-25` | 45.312 | Best bridge n128 so far; n1000 improved to `43.0`. |
| Two-level `[-3,-1]` plus50 reset lr5e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge2-m3m1-split64-plus50reset-lr5e5-ls03-v1/checkpoint-50` | 45.312 | Held plus25 n128 exactly, but n1000 fell to `42.1`. |
| Two-level `[-4,-1]` init | `/checkpoints/pretrain-output/v12-v4deepbridge2-v11scale105ov1125-split64-m4m1-d4096/checkpoint-init` | 42.188 | Similar to `[-3,-1]` init. |
| Two-level `[-4,-1]` step25 lr5e-5 | `/checkpoints/pretrain-output/v12-v4deepbridge2-m4m1-split64-step25-lr5e5-ls03-v1/checkpoint-25` | 44.531 | Tied `[-3,-1]` step25 n128, but n1000 was only `40.7`. |

Trusted n1000 scale sweep for the best `[-3,-1]` plus25 reset checkpoint:

| Eval connector scale | GQA n1000 | Yes/no | Other | Hygiene | Artifact |
| ---: | ---: | ---: | ---: | --- | --- |
| 1.05 | 42.8 | 58.120 | 34.515 | EOS 1.0, role-prefix 0.0 | `/checkpoints/v12_qwen_ceiling/gqa_deepbridge_m3m1_plus25_scale105_n1000_rerun.json` |
| 1.10 | 42.7 | 58.120 | 34.361 | EOS 1.0, role-prefix 0.0 | `/checkpoints/v12_qwen_ceiling/gqa_deepbridge_m3m1_plus25_scale110_n1000_rerun.json` |
| 1.15 | 43.0 | 58.689 | 34.515 | EOS 1.0, role-prefix 0.0 | `/checkpoints/v12_qwen_ceiling/gqa_deepbridge_m3m1_plus25_scale115_n1000_rerun.json` |
| 1.175 | 42.8 | 57.835 | 34.669 | EOS 1.0, role-prefix 0.0 | `/checkpoints/v12_qwen_ceiling/gqa_deepbridge_m3m1_plus25_scale1175_n1000_rerun.json` |

Trusted n1000 changed-continuation followup:

| Branch | Checkpoint | GQA n1000 | Yes/no | Other | Hygiene | Artifact |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `[-3,-1]` plus25, train levels/last1/norm | `/checkpoints/pretrain-output/v12-v4deepbridge2-m3m1-levels-last1-contrastive-plus25-lr1e5-ls01-bs1acc32-step25-v1/checkpoint-25` | 42.7 | 57.265 | 34.823 | EOS 1.0, role-prefix 0.0, max-token 0.0 | `/checkpoints/v12_qwen_ceiling/gqa_deepbridge_m3m1_changed_plus25_step25_n1000.json` |

Operational notes:

- A global128/local0 bridge hit DDP unused-parameter failure because the local
  latent path was zero-length. Split64 global/local latents fixed training.
- `v12-v4deepbridge2-m3m1-split64-cont50-lr5e5-ls03-v1` stopped before training
  because `--dataset` was omitted and defaulted to unsupported `v3_grounded`.
- `v12-v4deepbridge2-m3m1-split64-cont50-lr5e5-ls03-v2` stopped before optimizer
  updates after we caught that `--resume-checkpoint` restored old scheduler
  state. Reset-optimizer continuations use `--pretrain-checkpoint` instead.
- The first plus25 scale sweep launch failed before generation because V4 did
  not yet allow explicit eval-time connector scale overrides through metadata
  validation. The loader/eval path now mirrors V3: an override intentionally
  skips only the `connector_output_scale` equality check while preserving all
  architecture and shape guards.
- `ap-BMvCBxe6Ball6Ps7llspdT` smoke-tested a changed bridge continuation from
  the plus25 checkpoint with only `projector.level_embeddings`,
  `projector.layers.5`, and `projector.norm` trainable. It passed and saved
  checkpoint-1. A batch-size-4 smoke (`ap-OQ7h2gXGGg7pmvcSizvII2`) failed at
  distributed NCCL setup before training, so the real run uses the proven
  batch-size-1 path with gradient accumulation.
- The changed continuation finished at checkpoint-25 and scored `42.7` n1000
  with clean generation. This closes the DeepStack branch unless a new mechanism
  changes the problem; longer repeats are not justified by the current evidence.

## Pairwise And Taxonomy Notes

Core pairwise outputs:

```text
/tmp/v12_gqa_pairwise_core
/tmp/v12_gqa_pairwise_a1_d1024
/tmp/v12_gqa_pairwise_d1_gate005
```

Summary:

- A1 d512 has almost no useful trade versus V11: it gains 1 V11-wrong row and
  loses 4 V11-correct rows.
- C5 and D1 are similarly local: C5 gains 3 / loses 7 versus V11; D1 gains 0
  / loses 4; D1 cont75 gains 1 / loses 5.
- A1 d1024 changes the answer distribution and trades more rows, but loses the
  trade: 44 gains versus V11 and 62 losses, with drops in spatial, left/right,
  logical, and yes/no buckets.
- D1 gate0.005 loses 63 V11-correct rows and gains 39, with the largest damage
  in spatial/left-right/yes-no buckets. Gate0.05 is a generation-breaker.
- Q-selector residual0.25 step25 loses the trade by 44 gains / 66 losses versus
  V11. The damage concentrates in yes/no, spatial, and left/right buckets and
  shifts the answer prior toward `yes`/`left`.
- E3 vision-side adapter screens generated cleanly but scored only `42.9` and
  `43.0`, so current full-block SigLIP unfreezing is not the next hill.

## C1 Rerun From Scale1.05 Anchors

The C1 rerun branch did not reproduce the V11 hill. We trained from the
materialized scale1.05 `2350` and `2400` anchors with learned 2D patch-position
features, then materialized eval checkpoints to V11-style `connector_output_scale
= 1.125` and `patch_position_feature_scale = 0.0`.

| Anchor | Checkpoint | Trusted GQA n128 | Hygiene | Note |
| --- | --- | ---: | --- | --- |
| 2350 scale1.05 | step25 | 37.500 | EOS 1.0, role-prefix 0.0 | Hard drop. |
| 2350 scale1.05 | step50 | 40.625 | EOS 1.0, role-prefix 0.0 | Recovered only to low-40s. |
| 2350 scale1.05 | step75 | 41.406 | EOS 1.0, role-prefix 0.0 | Flat. |
| 2350 scale1.05 | step100 | 41.406 | EOS 1.0, role-prefix 0.0 | Flat; stopped remaining training. |
| 2400 scale1.05 | step25 | 41.406 | EOS 1.0, role-prefix 0.0 | Below V11. |
| 2400 scale1.05 | step50 | 40.625 | EOS 1.0, role-prefix 0.0 | Worse. |
| 2400 scale1.05 | step75 | 42.188 | EOS 1.0, role-prefix 0.0 | Best early rebound. |
| 2400 scale1.05 | step100 | 40.625 | EOS 1.0, role-prefix 0.0 | No trend. |
| 2400 scale1.05 | step125 | 41.406 | EOS 1.0, role-prefix 0.0 | No trend. |
| 2400 scale1.05 | step150 | 42.188 | EOS 1.0, role-prefix 0.0 | Ties step75, still far below V11. |
| 2400 scale1.05 | step200 | 41.406 | EOS 1.0, role-prefix 0.0 | Final negative. |

Interpretation: C1 rerun is completed-negative. The answer prior remains
heavily `no`/`yes`, and n128 never gets close to the V11 `44.9` target. The
new materializer path correctly records both `connector_output_scale` and
`patch_position_feature_scale`, which is now the required way to salvage C1-ish
checkpoints into the V11 eval shape.

## W&B Health Notes

| Run | Status | Note |
| --- | --- | --- |
| `k1osv2ox` | Finished | C1 continuation; recent loss/grad spike alerts. |
| `ykidfm0r` | Failed | E2 V4 first attempt failed on missing AnyMALv4 `query_conditioned_visual_scale_mode`. |
| `ykefi5xm` | Finished | E2 relaunch finished; had `high_grad_clip_fraction` and grad norm `24.13`; GQA eval was a hard negative at `23.5`. |
| `85oj948f` | Finished | E1 highres448 proof step succeeded with real `[3,448,448]` images. |
| `z4d3vpmx` | Finished | E1 highres448 resumed from checkpoint-1 after scheduler restore patch; GQA stayed `42.7`. |
| `m5rn22wl` | Finished | E2 deepstack192 canary trained from effectively fresh V4; loss/grad unhealthy and GQA `0.0`. |
| `whk9k6vc` | Finished | A1 d1024 trained cleanly; GQA `43.1`, below d512 and V11. |
| `do502o1h` | Stopped | Q-selector residual0.25 reached checkpoints 25/50; stopped before 100 after poor GQA. |
| `45o2wce1` | Finished | Q-selector residual0.10 step25 clean but GQA `43.1`. |
| `6yjk4v8d` | Finished | Q-selector residual0.05 step25 clean but GQA `43.2`. |
| `hiv0qtyb` | Finished | E3 last-layer proof passed after `encode_images` no-grad fix; `vision_adapter.pt` present. |
| `rb133msw` | Finished | E3 last-layer step25 clean; GQA `42.9`, EOS `1.0`, max-token-hit `0.0`. |
| `ii2reg9c` | Finished | E3 last-two-layer step25 clean; GQA `43.0`, EOS `1.0`, max-token-hit `0.0`. |
| `jz608bkd` | Finished | Three-level DeepStack bridge step25 lr5e-5; n128 `41.406`. |
| `uguknopf` | Finished | Three-level DeepStack bridge step25 lr1e-5; n128 `39.844`. |
| `pzpe3x2n` | Finished | Two-level `[-3,-1]` DeepStack bridge step25 lr5e-5; n1000 `42.5`. |
| `90lra8dx` | Finished | Two-level `[-3,-1]` DeepStack bridge step25 lr2e-5; n128 `40.625`. |
| `pzn9k5qy` | Finished | Two-level `[-3,-1]` plus25 reset lr5e-5; n128 `45.312`, n1000 `43.0`. |
| `1f3g51cb` | Finished | Two-level `[-4,-1]` step25 lr5e-5; n128 `44.531`. |
| `gki4p4h0` | Finished | Two-level `[-3,-1]` plus50 reset lr5e-5; n128 `45.312`, n1000 `42.1`. |
| `ap-dRyE45z38QgRwYa4VVjVjK` | Finished | Eval app for two-level `[-4,-1]` step25 lr5e-5; n1000 `40.7`. |
| `ap-BKrs7DWIWNihWQTsqopbuz` | Finished | Spatial-focus GQA dataset one-step smoke; built 15k focused examples and saved checkpoint-1. |
| `ap-HGIwTlHLOtvVdEuKlaraQB` | Finished | Spatial-focus full connector step25; n128 `39.844`. |
| `ap-LjaVoRCFWztJkJM3otE3t4` | Finished | Spatial-focus last2+out step25; n128 `40.625`. |
| `ap-VF9O3AF4wltr0NtYdfBNgX` | Finished | Query-conditioned scalar visual-scale step50; gradients were near-zero and both step25/step50 scored n128 `41.406`. |
| `ap-LS4oes97V2UjqboMnYO42T` | Failed early | Raw V8 anchor C1 smoke correctly hit metadata guard: checkpoint scale `1.0` cannot be loaded as training scale `1.05` without materialization. |
| `ap-q00fSamwtzv1upbtWIQeLo` | Finished | C1 rerun smoke from materialized V9 scale1.05 anchor; initialized zero `patch_position_embedding` and saved checkpoint-1. |
| `ap-bQAoq1A1EJpmSbfslD8E1L` | Finished | C1 rerun from anchor2400 scale1.05 completed to step200; best materialized n128 was `42.188`. |
| `ap-9YOqfWH14jfk6WPaVxSGrx` | Stopped | C1 rerun from anchor2350 scale1.05 stopped after step100 eval stayed `41.406`. |
| `ap-5MtCPkT3EN0jHpmIYtp5N1` | Stopped | First V3-error-slice smoke was stopped during a quiet model-load window; relaunch showed it was slow construction, not a fatal data error. |
| `ap-Lye9prYnIELjjJVfy6E2x9` | Finished | V3-error-slice last2+norm smoke passed; built 20k/12k/10k focused train-balanced GQA proxy slices plus replay/POPE controls. |
| `ap-ii0SrMNrbqi5m4FZCvNqlO` | Finished | V3-error-slice last2+norm step25 trained cleanly but scored GQA n128 `40.625`; branch completed-negative. |
| `ap-jmLb6wFQgpu6nZVafjImX6` | Finished | Query-conditioned per-token visual-scale smoke from V11 frontier passed. |
| `ap-5i8I3yT789djqKzQLaaQP1` | Finished | Query-conditioned per-token visual-scale step50 trained cleanly; step25/step50 both scored GQA n128 `42.188`. |
| `ap-SQ93U6nqHK6Cbrmc2DxH2I` | Finished | RMS-preserving V11 last1 smoke passed and saved checkpoint-1. |
| `ap-dGOpjkiJG2AMoz9EzD67b7` | Finished | V4 one-level global/local bridge smoke passed and saved checkpoint-1. |
| `ap-K5cZPJZzLNyBiGQE5h3xK4` | Finished | RMS-preserving V11 last1 step25 trained cleanly; GQA n128 `42.188`. |
| `ap-AwaPGoafDnBR5cCIYEK7QZ` | Finished | V4 one-level global/local bridge step25 trained cleanly; GQA n128 `42.188`. |
| `ap-Gqd73802tvgqzVoxYdLWlG` | Finished | GQA n128 eval for RMS-preserving last1 step25. |
| `ap-BOha9tNNDy3vQBBIy0a5Ef` | Finished | GQA n128 eval for V4 one-level global/local bridge step25. |
| `ap-kXSldkILBYvvHH6b4Uz4YX` | Finished | Materialized V11 coord-MLP patch-position init; removed stale learned-table tensor and set scale `0.05`. |
| `ap-XGVr2A7h0Aj7VcWhGu3JVv` | Finished | Coord-MLP patch-position-only smoke passed; 149,376 trainable params. |
| `ap-NO2Bj6rRuo2zIJrA6GNWW7` | Finished | Coord-MLP patch-position-only step25 trained cleanly but had low gradient norm `~0.0011`; GQA n128 `42.188`. |
| `ap-GYtBPkBaLz9dSNOFhO8eMs` | Finished | GQA n128 eval for coord-MLP patch-position-only step25. |
| `ap-CJZ15fxWtCQCdS2ywbtnI3` | Failed early | Vxattn upper + last1/norm smoke exposed DDP unused-parameter failure with decoder gradient checkpointing. |
| `ap-zMZJsSpmnbEEtVC3oHnFkf` | Failed early | Same smoke after checkpointing fix reached forward/backward but OOMed at batch size 4 during contrastive negative forward. |
| `ap-e7kLoiQRVzUtVhULUH9YXT` | Finished | Batch-size-1 smoke passed and saved checkpoint-1 for vxattn upper + last1/norm. |
| `ap-wZaFOqzVvhGrXqN8lGw2MJ` | Finished | Vxattn upper + last1/norm step25 trained cleanly at batch1/acc32; grad norms `0.0487` and `0.0542`. |
| `ap-vKMfYemquLovGiTMInLCb7` | Finished | GQA n128 eval for vxattn upper + last1/norm step25; n128 `42.188`. |
| `ap-Xffm6I1KRdhVCA2mJNJOLb` / `ap-4HGD5dHbbiGLaysgHPuhhH` / `ap-ZNynfVsDYwHLAzCHiBbGdV` / `ap-9eAhW4KvkR3LRG5xPyvDjn` | Failed early | First DeepStack plus25 scale sweep hit V4 metadata mismatch before generation; no metrics emitted. |
| `ap-KgKgbmduMo2zNWsVDz7YYb` / `ap-GW8Xybf3rUW1js0jTMXpzf` / `ap-H17UE31I6zEFROpM5l2GH7` / `ap-ybSBpNipG2kIFsWrMfx8f1` | Finished | Corrected DeepStack plus25 scale sweep: best scale `1.15` tied `43.0` n1000, all clean. |
| `ap-BMvCBxe6Ball6Ps7llspdT` | Finished | Changed DeepStack continuation smoke passed from plus25 reset checkpoint; trained `projector.level_embeddings`, `projector.layers.5`, and `projector.norm`, saved checkpoint-1. |
| `ap-OQ7h2gXGGg7pmvcSizvII2` | Failed early | Batch-size-4 smoke failed at distributed NCCL setup (`device not ready`) before model/training; batch-size-1 path remains proven. |
| `ap-nPuutsAUe6nhGpKcmMtazs` / W&B `v4o6jegh` | Finished | Changed DeepStack plus25 continuation, batch1/acc32, LR `1e-5`, loss scale `0.1`; saved checkpoint-25. |
| `ap-ocn079zbXIJyBvyzU0iyvn` | Finished | Trusted GQA n1000 eval for changed DeepStack continuation; scored `42.7`, clean hygiene. |

## Active And Pending Runs

| Track | Run | Status |
| --- | --- | --- |
| V3-to-V4 DeepStack bridge | Current bridge variants below V11 on n1000 | completed-negative |
| Spatial-focus V3 objective | Direct-answer mixture regressed on n128 | completed-negative |
| Query-conditioned scalar visual scale | Step25/step50 both n128 `41.406` | completed-negative |
| C1-family rerun | Anchor2350/2400 scale1.05 reruns stayed in the low 40s on n128 | completed-negative |
| V3-error-slice diagnostic | Last2+norm step25 scored n128 `40.625` despite clean hygiene | completed-negative |
| Query-conditioned per-token visual scale | Step25/step50 both n128 `42.188` | completed-negative |
| RMS-preserving V11 last1 continuation | Step25 n128 `42.188` with clean generation | completed-negative |
| V4 one-level global/local bridge | Step25 n128 `42.188` with clean generation | completed-negative |
| Coord-MLP patch-position only | Step25 n128 `42.188`; low gradients | completed-negative |
| Vxattn upper + last1/norm co-adaptation | Step25 n128 `42.188`; clean after DDP/OOM fixes | completed-negative |
| V4 DeepStack plus25 scale sweep | Best eval scale `1.15` tied `43.0` n1000; no scale rescue | completed-negative |
| V4 DeepStack changed continuation | Step25 scored `42.7` n1000 with clean hygiene; no objective/slice rescue | completed-negative |
| A1 vxattn eval calibration | Best n1000 cell `45.0`, but matched V11 at `43.2` on n3000 | completed-neutral |
| 256-token l5norm continuation | Step25 scored `42.1` n1000 with clean hygiene | completed-negative |
| Per-token output-scale continuation | Best eval scale `1.05` scored `44.1` n1000; below V11 | completed-negative |
| C1 mining grid | Best n1000 tied V11 at `44.9`; matched V11 exactly at `43.2` on n3000 | completed-neutral |
| Spatial-contrastive last1+out | Best n1000 `43.7`; continuations regressed | completed-negative |
| Highres512 eval/train smoke | Eval-time highres512 scored `43.3`; smoke saved but not promoted | completed-negative |
| Token-budget clean-init static grid | Best n1000 `40.1` across 192/256-token inits | completed-negative |
| Joint vision-last1 + projector-tail | Best n128 `41.406`; no n1000 promotion | completed-negative |
| V11/C1 projector soup | Best n128 `42.969`; no n1000 promotion | completed-negative |
| Attention-only vision-last1 + projector-tail | Best n128 `42.969`; no n1000 promotion | completed-negative |

## Current Interpretation

No V12 branch has beaten the V11 frontier (`44.9`) on a robust matched check.
The two apparent first-1000 ceiling moves both collapsed under stricter
reading: A1 eval calibration reached `45.0` n1000 but tied V11 exactly at
`43.2` on matched n3000, while C1 mining tied V11 at `44.9` n1000 and also
matched V11 exactly at `43.2` n3000. Treat both as rediscoveries or local
perturbations of the V11 basin, not new hills. Larger token budgets, higher
resolution, wider visual cross-attention, D2 gate amplification,
decoder-side LoRA, query-conditioned patch selection, token/output scaling,
spatial-contrastive objectives, and vision-side SigLIP adaptation all regressed
or failed promotion screens despite clean generation/checkpoint hygiene.

The visual-cross-attention plus light connector-unfreeze followup is now also
closed negative. The idea was sound enough to test because A1 upper gate `1e-4`
was the best V12 n1000 perturbation (`44.6`), but adapter-only continuation had
regressed. A smoke revealed two mechanical constraints: V3 vxattn Stage 1 runs
must disable decoder gradient checkpointing so hook adapters participate in DDP,
and the combined vxattn + contrastive negative forward needs batch size 1 on
H100 when checkpointing is off. With those fixes, step25 trained cleanly from
V11 using `projector.layers.5` + `projector.norm` plus upper vxattn layers
`18,22,26,30,34`, but trusted GQA n128 was only `42.188` with clean hygiene.
Do not promote this exact co-adaptation recipe to n1000.

A cheap eval-time calibration around the original A1 upper gate `1e-4` checkpoint
briefly found a first-1000 bump: gate multiplier `0.5` and connector scale
`1.125` scored `45.0` n1000 with clean generation. The surrounding cells were
lower (`44.1`-`44.9`) and a tighter local sweep did not find a better point. On
a matched n3000 trusted GQA evaluation, the same calibrated checkpoint tied V11
exactly at `43.2`, with clean hygiene for both runs. Treat this as a useful
local perturbation and possible pairwise-analysis target, not a frontier break.

The DeepStack story changed after the bridge controls. The generation break was
not expected as a direct consequence of DeepStack, and the true bridge controls
show it was mostly a fresh-V4/Qwen out-of-distribution failure. The branch did
not beat V11 on n1000. `[-3,-1]` plus25 reset improved from `42.5` to `43.0`
n1000 and reached the best bridge n128 (`45.312`), but plus50 reset held the
same n128 while falling to `42.1` n1000. A connector scale sweep around plus25
also failed to rescue the branch: scales `1.05/1.10/1.15/1.175` stayed in the
`42.7`-`43.0` band with clean generation. The changed-objective/slice followup
from plus25 reset trained only `projector.level_embeddings`,
`projector.layers.5`, and `projector.norm`, but scored `42.7` n1000. Treat
n128 as a cheap generation/sanity screen only. The adjacent `[-4,-1]` selector
also failed n1000 (`40.7`). Stop DeepStack repeats unless a new mechanism changes
the branch; the evidence says this hill is below V11.

The first spatial-focus V3 objective also failed. The focused data builder is
technically usable, and the run built `15,000` spatial/relation GQA examples,
but both full-connector and last2+output-projection 25-step screens dropped to
roughly `40` on n128. The direct spatial/relation mixture is not a promising
hill in its current form.

The query-conditioned scalar visual-scale branch also failed. It was a tiny,
bounded adapter around the V11 scale (`0.95..1.25`, init `1.125`) and generated
cleanly, but training reported essentially zero gradient norm and both
checkpoint-25 and checkpoint-50 scored `41.406` on trusted GQA n128. Do not
promote this branch without a new gradient/logit diagnostic.

The C1-family rerun is now closed negative. A smoke run from the raw V8
checkpoint caught the expected metadata mismatch when trying to train at
connector scale `1.05` from a checkpoint whose metadata records scale `1.0`;
using the materialized scale1.05 checkpoints passed, but both independent reruns
stayed in the low 40s on trusted GQA n128 after V11-style materialization.

The narrow V3-error-slice diagnostic is also closed negative. It used the
existing V3-correct/V11-wrong taxonomy only as an error profile, built
train-balanced GQA proxy slices for non-yes/no spatial/object/logical-color
rows, and trained only the final V3 projector blocks plus norm from the current
V11 frontier. The smoke was technically good, but step25 fell to `40.625` n128
with clean generation, so this objective is not the next hill.

## Query-Conditioned Per-Token Visual Scale

This was a more expressive D2 variant of the failed scalar visual-scale branch:
each visual latent received a bounded query-conditioned scale initialized at the
V11 value (`1.125`) and constrained to `0.9..1.35`. It trained only the new
`projector.query_visual_scale` adapter (`532,608` parameters) from the V11
frontier on the antishuffle stage1b mixture.

| Checkpoint | Trusted GQA n128 | Yes/no | Other | Hygiene | Note |
| --- | ---: | ---: | ---: | --- | --- |
| step25 | 42.188 | 60.000 | 28.767 | EOS 1.0, role-prefix 0.0 | Clean but below V11. |
| step50 | 42.188 | 60.000 | 28.767 | EOS 1.0, role-prefix 0.0 | Identical to step25. |

Interpretation: completed-negative. The branch generated cleanly, but the
identical step25/step50 answer distribution and low-40s GQA score do not justify
a longer continuation without a new mechanism or diagnostic.

## RMS-Preserve And One-Level Bridge Followups

Two independent followups were smoke-tested and then run to step25:

| Branch | Checkpoint | Trusted GQA n128 | Yes/no | Other | Hygiene | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| RMS-preserving V11 last1 | `/checkpoints/pretrain-output/v12-qwen-rmspreserve-last1-step25-lr5e6-ls01-v1/checkpoint-25` | 42.188 | 60.000 | 28.767 | EOS 1.0, role-prefix 0.0 | Low-grad but nonzero; no first-slice gain. |
| V4 one-level global/local bridge | `/checkpoints/pretrain-output/v12-v4bridge1-globalloc-step25-lr2e5-ls03-v1/checkpoint-25` | 42.188 | 60.000 | 28.767 | EOS 1.0, role-prefix 0.0 | Valid bridge geometry test, but below V11. |

Interpretation: both are completed-negative. The RMS branch tested whether a
last-block update could stay closer to the Qwen embedding manifold using
supervised-token-target normalization and a prompt-text RMS regularizer; it
still collapsed to the same low-40s first-slice plateau. The one-level bridge
isolated V4 global/local token geometry without earlier SigLIP feature levels;
it generated cleanly, but did not justify n1000 promotion.

## Coord-MLP Patch-Position Branch

The V11 learned-table patch-position checkpoint was converted into a true
coord-MLP init rather than a metadata-only scale edit:

```text
/checkpoints/pretrain-output/v12-qwen3-v11-coordmlp-pos005-init/checkpoint-init
```

The materializer changed `patch_position_feature_type` to `coord_mlp`, set
`patch_position_feature_scale = 0.05`, and removed the stale
`patch_position_embedding` tensor so the training loader initialized only the
new MLP weights.

| Checkpoint | Trusted GQA n128 | Yes/no | Other | Hygiene | Note |
| --- | ---: | ---: | ---: | --- | --- |
| step25 | 42.188 | 60.000 | 28.767 | EOS 1.0, role-prefix 0.0 | Low-gradient warning, grad norm `~0.0011`. |

Interpretation: completed-negative. The conversion path is now technically
usable, but this tiny smooth-coordinate branch was effectively inert and does
not deserve a longer run in its current form.

## Late V12 Ceiling Probes

These probes were run after the DeepStack and vxattn followups to make sure the
campaign did not stop at a local negative. All listed GQA runs generated cleanly
unless noted otherwise: EOS `1.0`, no observed assistant-prefix failure, and no
max-token pathology in the trusted screens.

Trusted n1000/n3000 checks:

| Branch | Best checkpoint / artifact | GQA | Yes/no | Other | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| 256-token l5norm continuation | `/checkpoints/pretrain-output/v12-qwen3-256tok-latents-l5norm-smoke/checkpoint-1` smoke, step25 eval artifact `gqa_v12_256tok_l5norm_lr5e6_step25_s1125_n1000.json` | 42.1 n1000 | 60.969 | 31.895 | Clean but well below V11. |
| Per-token output scale | `/checkpoints/pretrain-output/v12-qwen3-v11-per-token-scale-cont25-lr1e2-ls1-v1/checkpoint-25` at eval scale `1.05` | 44.1 n1000 | 64.957 | 32.820 | Real learned-scale movement, but not a frontier break. |
| C1 mining step150 | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125`-family step150 screen, scale `1.125` | 44.9 n1000 / 43.2 n3000 | 65.527 / 62.582 | 33.744 / 32.594 | Ties V11 n1000 and matches V11 exactly on n3000. |
| V11 frontier rerun | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` | 43.2 n3000 | 62.582 | 32.594 | Matched comparison for C1 mining. |
| Spatial-contrastive last1+out | best step25 among lr `5e-6` and `1e-5` | 43.7 n1000 | 63.248-63.533 | 32.974-33.128 | Step50/75 regressed; stop. |
| Highres512 eval | V11 frontier evaluated at image size 512 | 43.3 n1000 | 63.533 | 32.357 | Higher resolution alone does not rescue GQA. |
| Token-budget clean-init grid | 192 copy-noise / 256 mean init static evals | 40.1 n1000 | 56.695-59.544 | 29.584-31.125 | Hard negative across ten scale/init cells. |

Cheap n128 gates that were not promoted:

| Branch | Checkpoint / artifact | Best GQA n128 | Yes/no | Other | Note |
| --- | --- | ---: | ---: | ---: | --- |
| Joint SigLIP last1 + projector tail | `/checkpoints/pretrain-output/v12-qwen-v11-joint-visionlast1-projlast1outnorm-antishuffle-step25-lr5e6/checkpoint-25` | 41.406 | 60.000 | 27.397 | Full final SigLIP block plus projector tail; technically clean but far below gate. |
| V11/C1 step150 projector soup | materialized alpha `0.25/0.50/0.75` checkpoints | 42.969 | 61.818 | 28.767 | No interpolation alpha exposed an intermediate win. |
| Attention-only SigLIP last1 + projector tail | `/checkpoints/pretrain-output/v12-qwen-v11-joint-visionattnlast1-projlast1outnorm-antishuffle-step25-lr3e6/checkpoint-25` | 42.969 | 61.818 | 28.767 | Lighter attention/norm-only unfreeze improved over full-block joint but still missed promotion. |

Operational notes:

- The 256-token l5norm smoke passed, but the step25 n1000 score was only
  `42.1`; this closes the trained larger-token branch unless paired with a new
  mechanism.
- Per-token output-scale continuations learned nonzero scale deltas and had
  strong training loss (`~0.299`), but the best trusted n1000 was `44.1` at
  eval scale `1.05`. Do not confuse this with the query-conditioned per-token
  visual-scale adapter above; that earlier D2-style adapter stayed flat at
  `42.188` n128.
- C1 mining found the same basin as V11. Step150 at scale `1.125` is the best
  n1000 cell (`44.9`), but the exact n3000 match against the V11 rerun makes it
  a neutral result rather than a new ceiling.
- The full-block joint SigLIP run initially failed a smoke because the prefix
  used `vision_model.encoder.layers.26`; the correct SigLIP prefix is
  `model.encoder.layers.26`, with `--v3-patch-position-feature-type
  learned_table` for V11-compatible loading.
- The corrected full-block joint run trained 16 final-block SigLIP tensors plus
  the projector tail, but n128 peaked at `41.406`. The attention-only variant
  trained 12 final-block attention/norm tensors plus the same projector tail
  and peaked at `42.969`; both are completed-negative.
- The highres512 training smoke saved
  `/checkpoints/pretrain-output/v12-highres512-spatialcontrast-last1out-smoke/checkpoint-1`,
  but the cheaper V11 highres512 eval was already negative enough (`43.3`) that
  no longer highres run was justified.

Campaign interpretation after these late probes: V12 has now pushed the major
required mechanisms from the plan: repaired vxattn, larger token budgets, V11
continuations/objectives/trainable subsets, DeepStack/multi-level features,
higher resolution, and vision-side adaptation. The robust frontier remains V11,
not because we stopped early, but because every apparent hill either regressed
on n1000/n3000 or landed back in the same V11 basin.
