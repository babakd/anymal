# V5 Progress Handoff

Date: 2026-05-09

## Current job status

All V5 jobs started in this thread have been stopped.

- R1 robust accumulation-8 training app: `ap-vYcwU9fpufZN5UXNBncP3L`, stopped, 0 tasks.
- R1b robust accumulation-16 training app: `ap-TvaVjB2cLqR8upiUwcb0XC`, stopped, 0 tasks.
- Local process sweep after stopping found no `modal run`, `modal_train`, `vqa_checkpoint_eval`, or `inspect_wandb_run` processes.
- Monitoring subagents `Nash` and `Raman` were closed.

R1b was interrupted by user request while the step-50 validation loop was running. W&B had synced train step 50, but no eval loss was logged yet and the Modal volume check reported the output directory still empty, with no visible `checkpoint-50`.

## Operating principles from V4 critique

The critique changed how V5 was run:

- Raw and strict generation are co-primary metrics. A postprocessed score cannot promote a recipe by itself.
- W&B alerts are not cleared by console optimism. A run is only healthy when the inspector reports no active alerts.
- Do cheap causal and robustness tests before spending architecture budget.
- Stop source-recipe work when the recipe is weak; do not launch architecture experiments on shaky evidence.
- Use more than seed 42, and check leakage/robustness/prompt isolation before promotion.
- Subagents are useful only for crisp monitoring tasks. The parent agent stays in the driver seat.

## Key discovery

The original V4 raw-output critique was partly confounded by an evaluation batching bug.

`evaluation/vqa_eval.py` was right-padding decoder-only prompts in VQA eval. Batched rows that were shorter than the longest prompt generated from pad positions, which caused misleading raw outputs such as `assistant\n\n`. A batch-size-1 preflight showed clean raw outputs, and changing VQA eval to left-pad fixed the issue.

After the left-padding fix, both corrected V4 and V5-R0 are raw-clean:

- `assistant_role_prefix_rate`: 0
- strict accuracy equals clean accuracy in the corrected runs

This means V5 should not claim a role-prefix breakthrough over V4. The real question became whether V5 can improve accuracy without losing robustness.

## Code and config changes made

Main changes:

- `evaluation/vqa_eval.py`
  - Added strict/raw diagnostics.
  - Added `strict_accuracy`, per-answer-type strict accuracy, assistant-prefix rates, top raw answers, answer-kind rates, and `image_id` in predictions.
  - Fixed VQA eval batching by left-padding `input_ids` and `attention_mask`.
  - Treats `anymal_v4` as a SigLIP architecture for VQA eval.
- `scripts/analyze_vqa_predictions.py`
  - Helper for inspecting VQA prediction JSONs.
- `scripts/audit_vqa_leakage.py`
  - Compares eval `image_id` values against training sources.
- `scripts/check_v5_promotion.py`
  - Promotion checker for complete metrics, strict parity, low prefix rate, EOS/max-hit hygiene, no yes/no collapse, and prediction sample count.
  - Tightened to `--allow-clean-drop=0.0` after the mild-blur reversal.
- `scripts/check_vlm_promotion.py`
  - Prints strict/prefix diagnostics.
- `modal_train.py`
  - Added `v5_semantic_calibration` and `v5_semantic_calibration_robust`.
  - Added `--finetune-gradient-accumulation-steps`.
  - Added augmentation plumbing and auto-freeze connector for V5 semantic datasets.
- `data/data_utils.py`
  - Added `image_augmentation_mode`.
  - Added `vqa_light` augmentation: mild blur plus constrained random resized crop.
- `data/instruction_dataset.py`
  - Mixture entries can pass augmentation options.
- `configs/finetune_v5_semantic_calibration.yaml`
- `configs/finetune_v5_semantic_calibration_robust.yaml`
- `tests/test_evaluation.py`
  - Added coverage for strict metrics, `image_id`, and left padding.
- `V5_RESEARCH_PLAN_20260509.md`
  - Longer research-plan log with the detailed rationale and decisions.

Note: `modal_viewer.py` and `arch_sxs_inference.py` were already dirty/untracked in the worktree and were not part of the V5 edits.

## Completed V5-R0 training

V5-R0 used the same architecture family as V4 (`anymal_v4`) and changed the Stage 2 recipe.

Run:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 --stage finetune --dataset v5_semantic_calibration \
  --max-steps 100 --batch-size 4 --learning-rate 1e-5 --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-bs4-lossscale003-from-stage1b248-20260509-codex
```

Identifiers:

- Modal app: `ap-r4xWCKv5hEMULoUHbPiLvv`
- W&B: `babakdam/anymal-finetune/jpqogd86`
- W&B URL: `https://wandb.ai/babakdam/anymal-finetune/runs/jpqogd86`
- Output: `/checkpoints/finetune-output/v5-stage2a-roleclean-semanticcal-bs4-lossscale003-from-stage1b248-20260509-codex`

Health:

- W&B finished green.
- Alerts: none.
- Gradient clipping: 0.
- Accumulation: 8.
- Eval losses: `0.84634`, `0.84325`.
- `checkpoint-100` completed.

## Corrected clean VQA results

Corrected left-padding, 1000 samples, `training_chat` prompt:

| Seed | V5-R0 strict/clean | V4 strict/clean | Margin |
| --- | ---: | ---: | ---: |
| 42 | 51.933333 | 51.266667 | +0.666667 |
| 43 | 52.466667 | 52.400000 | +0.066667 |
| 44 | 51.033333 | 51.000000 | +0.033333 |

Mean:

- V5-R0: `51.811111`
- V4: `51.555556`
- Margin: `+0.255556`
- Margin SD: about `0.356`

Interpretation: V5-R0 is a small clean-set recipe win across three seeds, but the margins are thin enough that robustness checks matter more than architecture spend.

## Leakage audit

Artifact audited:

- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`

Result:

- 751 unique eval image IDs.
- Exact/numeric overlap with VQA train yes/no/number/other: 0.
- COCO object direct overlap: 0.
- Mix665k direct filtered overlap: 0.

Interpretation: no obvious image-ID leakage was found for the seed-42 corrected eval sample.

## Robustness and prompt isolation

Mild-blur robustness on seed 42 reversed the clean-set result:

| Run | Strict/clean |
| --- | ---: |
| V4, V4 prompt, mild blur | 51.600000 |
| V5-R0, V5 prompt, mild blur | 51.466667 |

Prompt isolation on mild blur:

| Checkpoint | Prompt | Strict/clean |
| --- | --- | ---: |
| V5-R0 | V4 prompt | 51.300000 |
| V4 | V5 prompt | 51.600000 |

Interpretation: the mild-blur regression appears to be from the checkpoint/training mix, not just stricter V5 prompt wording.

Decision made: do not launch V5 architecture A1 yet. Fix recipe robustness first.

## Robust training attempts

### R1: robust augmentation, accumulation 8

Run:

```bash
modal run modal_train.py --architecture anymal_v4 --stage finetune \
  --dataset v5_semantic_calibration_robust --max-steps 100 --batch-size 4 \
  --learning-rate 1e-5 --lora-learning-rate 1e-5 --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-robust-bs4-lossscale003-from-stage1b248-20260509-codex
```

Identifiers:

- Modal app: `ap-vYcwU9fpufZN5UXNBncP3L`
- W&B: `babakdam/anymal-finetune/h4oqlm9g`

Outcome:

- Stopped at synced step 16.
- Strict W&B inspector reported active `recent_grad_spikes`.
- Spike source was step 8: grad `0.024826` vs EMA `0.010866`, `2.28x`.
- No usable checkpoint.

### R1b: robust augmentation, accumulation 16

Run:

```bash
modal run modal_train.py --architecture anymal_v4 --stage finetune \
  --dataset v5_semantic_calibration_robust --max-steps 100 --batch-size 4 \
  --learning-rate 1e-5 --lora-learning-rate 1e-5 --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-stage1b248-20260509-codex
```

Identifiers:

- Modal app: `ap-TvaVjB2cLqR8upiUwcb0XC`
- W&B: `babakdam/anymal-finetune/5zj9atks`
- W&B URL: `https://wandb.ai/babakdam/anymal-finetune/runs/5zj9atks`
- Output: `/checkpoints/finetune-output/v5-stage2a-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-stage1b248-20260509-codex`

Health before stop:

- Early inspector at synced step 40:
  - Alerts: none.
  - Loss spikes: none.
  - Grad spikes: none.
  - Clip fraction: 0.
  - Accumulation: 16.
  - Latest train loss: `0.877538`.
  - Loss EMA: `0.841185`.
  - Grad norm: `0.010361`.
- Monitor recheck at synced step 50:
  - W&B state: running.
  - Alerts: none.
  - Loss spikes: none.
  - Grad spikes: none.
  - Clip fraction: 0.
  - Accumulation: 16.
  - Latest train loss: `0.636943`.
  - Loss EMA: `0.815764`.
  - Grad norm: `0.009733`.
  - Grad EMA: `0.009594`.
  - No eval loss logged yet.
  - Output dir existed but was empty; no visible `checkpoint-50`.

Outcome:

- User requested all jobs stop while step-50 eval was in progress.
- App was stopped and verified as `stopped`, 0 tasks.
- R1b should be treated as incomplete and not promotion-eligible.

## Validation performed

- `python3 -m py_compile` passed for touched Python files, including:
  - `modal_train.py`
  - `data/data_utils.py`
  - `data/instruction_dataset.py`
  - `evaluation/vqa_eval.py`
  - new scripts
- `git diff --check` passed.
- Local `pytest tests/test_evaluation.py` could not run because local Python lacks `torch`.
- Local transform import smoke also failed for the same local `torch` availability reason through `data/__init__.py`.

## Current recommendation

Do not promote V5 yet and do not launch A1 architecture yet.

Recommended next step:

1. Decide whether to resume a robust recipe test.
2. If resuming R1b, relaunch from scratch rather than relying on the interrupted step-50 state.
3. Keep accumulation 16, but consider a slightly weaker robustness recipe if R1b does not finish cleanly or if final mild-blur eval still trails V4:
   - lower blur probability or intensity,
   - apply augmentation only to a subset of semantic-calibration sources,
   - or run a paired clean/mild-blur eval at checkpoint 50 before spending to checkpoint 100.
4. Promotion gate should require:
   - W&B alerts empty at final,
   - complete checkpoint,
   - clean seed-42/43/44 does not regress vs V4,
   - mild-blur seed 42 does not trail V4,
   - strict accuracy equals clean accuracy or is within the defined tiny tolerance,
   - assistant-prefix rate remains near 0,
   - no yes/no collapse,
   - leakage audit remains clean.

## Important artifact names

Corrected clean evals:

- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed43_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed44_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed43_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed44_leftpad.json`

Robustness evals:

- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_mildblur_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_mildblur_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_v4prompt_mildblur_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_v5prompt_mildblur_leftpad.json`

Potential A1 pretrain checkpoint, not yet used for V5 architecture:

- `/checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400`
