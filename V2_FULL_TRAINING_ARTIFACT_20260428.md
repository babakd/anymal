# V2 Full Training Artifact - 2026-04-28

This file records the first meaningful V2 learned-compressor two-stage run that
completed both Stage 1 pretraining and Stage 2 fine-tuning. Future agents should
use this as the baseline V2 learned-compressor artifact unless deliberately
running a new experiment.

## Bottom Line

- Result: complete.
- Architecture: `anymal_v2`.
- Connector: `token_compressor_type=learned`.
- Stage 1 data: true LLaVA-Pretrain captions with zip-backed images.
- Stage 2 data: `balanced_mix`.
- Stage 1 token budget: 256 image placeholders.
- Stage 2 token budget: 384 image placeholders.
- Stage 1 final checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Stage 2 final checkpoint:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`

## Why This Run Was Meaningful

Earlier V2 runs were smokes/canaries or used COCO-backed fallback data for Stage
1. This run used the staged LLaVA-Pretrain zip-backed image path:

- `/checkpoints/llava_pretrain/manifest.json`
- `/checkpoints/llava_pretrain/images.zip`
- `/checkpoints/llava_data/blip_laion_cc_sbu_558k.json`

Stage 1 consumed true LLaVA-Pretrain captions/images rather than the COCO
fallback path. Stage 2 loaded the exact Stage 1 checkpoint listed above and ran
the full 3000-step balanced-mix fine-tune.

## Stage 1 Pretrain

Launch command:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --gpu-type h100 \
  --max-steps 2500 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --use-wandb \
  --run-name v2-stage1-learned-2500-20260428
```

Run metadata:

- Modal app: `ap-4Jy7OQ7ptFY77J2Zm1qg0a`
- Modal URL: `https://modal.com/apps/babakd/main/ap-4Jy7OQ7ptFY77J2Zm1qg0a`
- W&B project: `anymal-pretrain`
- W&B run: `https://wandb.ai/babakdam/anymal-pretrain/runs/x77vo36v`
- Output dir:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428`
- Final checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Runtime: 1.29 hours.
- Final train loss: `2.846356314664197`.
- Final eval loss: `2.5290`.

Stage 1 eval trend:

| Step | Eval loss |
|------|-----------|
| 250 | 3.0650 |
| 500 | 2.9030 |
| 750 | 2.8456 |
| 1000 | 2.7731 |
| 1250 | 2.6991 |
| 1500 | 2.6434 |
| 1750 | 2.6035 |
| 2000 | 2.5693 |
| 2250 | 2.5403 |
| 2500 | 2.5290 |

Verified complete checkpoints:

`250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`,
`2500`.

Expected final checkpoint artifacts were present:

- `trainer_state.pt`
- `model_meta.json`
- `token_compressor.pt`
- `projector.pt`

## Stage 2 Fine-Tune

Launch command:

```bash
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --dataset balanced_mix \
  --gpu-type a100 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-learning-rate 2e-4 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500 \
  --use-wandb \
  --run-name v2-stage2-balanced-mix-3000-20260428
```

Run metadata:

- Modal app: `ap-598nYUdjvsf1K3AS3eeA8N`
- Modal URL: `https://modal.com/apps/babakd/main/ap-598nYUdjvsf1K3AS3eeA8N`
- W&B project: `anymal-finetune`
- W&B run: `https://wandb.ai/babakdam/anymal-finetune/runs/3gyl1apj`
- W&B run name: `distinctive-oath-31`
- Output dir:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428`
- Final checkpoint:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`
- Runtime: 17.17 hours.
- Final train loss: `1.187016359488905`.
- Final eval loss: `1.1203`.
- Final VQA accuracy: `6.466666666666667%` on 500 samples.
- Final VQA avg generated tokens: `32.0`.
- Final VQA EOS rate: `0.0`.
- Final VQA time: `243.7174117565155` seconds.

Stage 2 dataset and trainable modules:

- Dataset: `balanced_mix`, `676,940` samples.
- Split: `643,093 train / 33,847 val`.
- Instruct-150K real-image filter: `157,712/157,712`, missing `0`.
- Mix-665K filtered to `338,470/665,298` cached-COCO samples.
- Stage 2 loaded Stage 1 projector and token compressor from the exact Stage 1
  final checkpoint.
- Stage 2 expanded learned compressor `pool_queries` from 256 to 384 by copying
  the pretrained rows into the Stage 2 table.
- Trainable params:
  - Vision encoder: `0 / 428,225,600`
  - Token compressor: `5,757,696 / 5,757,696`
  - Projector: `10,760,448 / 10,760,448`
  - LLM LoRA: `167,772,160 / 4,708,380,672`
  - Total: `184,290,304 / 5,153,124,416`

Stage 2 eval trend:

| Step | Eval loss |
|------|-----------|
| 300 | 1.2218 |
| 600 | 1.1935 |
| 900 | 1.1721 |
| 1200 | 1.1641 |
| 1500 | 1.1551 |
| 1800 | 1.1426 |
| 2100 | 1.1353 |
| 2400 | 1.1253 |
| 2700 | 1.1231 |
| 3000 | 1.1203 |

Verified complete checkpoints:

`50`, `100`, `150`, `200`, `250`, `300`, `350`, `400`, `450`, `500`,
`550`, `600`, `650`, `700`, `750`, `800`, `850`, `900`, `950`, `1000`,
`1050`, `1100`, `1150`, `1200`, `1250`, `1300`, `1350`, `1400`, `1450`,
`1500`, `1550`, `1600`, `1650`, `1700`, `1750`, `1800`, `1850`, `1900`,
`1950`, `2000`, `2050`, `2100`, `2150`, `2200`, `2250`, `2300`, `2350`,
`2400`, `2450`, `2500`, `2550`, `2600`, `2650`, `2700`, `2750`, `2800`,
`2850`, `2900`, `2950`, `3000`.

Expected final checkpoint artifacts were present:

- `trainer_state.pt`
- `model_meta.json`
- `llm/`
- `token_compressor.pt`
- `projector.pt`

## Interventions During The Run

The first Stage 2 attempt, Modal app `ap-fNA6pY1lnhYpk3mgwrOInR`, initialized
correctly but restarted before the first checkpoint. It was stopped manually
because no durable checkpoint existed yet.

The trainer was hardened before relaunch:

- Stage 1 distributed Modal timeout: 8 hours.
- Stage 2 Modal trainer timeout: 24 hours.
- Stage 2 checkpoint cadence: every 50 steps.
- `main()` prints the selected timeout.

The hardened Stage 2 run then completed successfully from scratch.

## Warnings Interpreted As Non-Blocking

HealthMonitor emitted occasional plateau, train/val gap, and finite gradient
spike warnings. These were not treated as failures because:

- Validation loss improved monotonically through the final eval.
- Gradient spikes settled within a few steps.
- No NaN/Inf, OOM, traceback, or persistent divergence occurred.
- Every checkpoint save and Modal volume commit succeeded after the hardened
  relaunch.

The initial Stage 2 loss warning about deviation from `ln(vocab_size)` is
expected for sparse answer-token supervision and should not by itself trigger an
intervention.

## Cost Notes

Using the checked Modal rates at run time:

- H100: about `$3.95/hr` per GPU.
- A100-80GB: about `$2.50/hr` per GPU.
- Stage 1: about `1.29h * 4 * $3.95 = $20.38`.
- Stage 2: about `17.17h * 1 * $2.50 = $42.93`.
- Successful-run compute estimate: about `$63` before overhead.

Actual billed cost should be higher because of startup/eval overhead and the
failed early Stage 2 attempt. A reasonable all-in expectation is still below the
earlier `$120-$160` planning envelope.

## Future Agent Guidance

- Do not describe V2 learned-compressor training as canary-only anymore. This
  file records a completed meaningful Stage 1 plus Stage 2 baseline.
- Do not auto-load legacy V1 checkpoints into V2.
- If comparing architecture variants, compare against the final checkpoints
  listed in this file.
- If resuming or evaluating Stage 2, use the final Stage 2 checkpoint unless the
  experiment explicitly requires an earlier checkpoint.
- If launching more long runs, use `TRAINING_RUN_BABYSITTING_PLAYBOOK.md` and
  verify checkpoint artifacts directly on the Modal volume at save boundaries.
