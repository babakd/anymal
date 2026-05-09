# AnyMAL V2 Stage 1/2 Training Runbook

Last updated: 2026-04-30

## Context

Recent agent changes made V2 the active path and added:

- SigLIP2 preprocessing at 384px.
- Strict V2 placeholder insertion.
- Stage 1 trainables: `token_compressor + projector`.
- Stage 2 trainables: `token_compressor + projector + LoRA`.
- LLaVA-Pretrain caption dataset plumbing.
- Balanced Stage 2 mixture support via `--dataset balanced_mix`.
- Optional V2 connector variants via `--token-compressor-type learned|perceiver|perceiver2`.

The Stage 1 smoke and first live canary fell back to COCO-backed instruction
captions because true LLaVA-Pretrain images were not staged yet. On
2026-04-27, the exploded-file staging attempt hit the Modal volume inode/device
limit. The current path keeps `images.zip` compressed and reads from it
directly, so meaningful Stage 1 runs should use true LLaVA-Pretrain data.

## Run Gates

1. Run local validation before launching cloud jobs.
2. Run a short Stage 1 canary that saves a checkpoint.
3. Run a short Stage 2 canary from that exact Stage 1 checkpoint.
4. Stage true LLaVA-Pretrain image storage as a zip-backed dataset, not
   exploded files on the current Modal volume.
5. Run the meaningful Stage 1 job.
6. Run Stage 2 from that exact Stage 1 checkpoint.
7. Compare against Stage 1-only, Stage 2 `instruct_150k`, and Stage 2 `balanced_mix`.

## Validation

```bash
python3 -m py_compile modal_train.py training/finetune.py
pytest tests/test_model.py::TestAnyMALv2CoreModules tests/test_training.py tests/test_evaluation.py -q
```

Local note from 2026-04-28: `python3 -m py_compile modal_train.py
data/laion_dataset.py training/finetune.py training/trainer.py` passed. Local
pytest is blocked on the workstation's `/usr/bin/python3` because `torch` is
not installed there.

## Data Staging

```bash
modal run --detach modal_train.py::stage_llava_pretrain_images
```

Expected result:

- `/checkpoints/llava_data/blip_laion_cc_sbu_558k.json`
- `/checkpoints/llava_pretrain/manifest.json`
- `/checkpoints/llava_pretrain/images.zip`
- A manifest with `storage: "zip"` and `image_zip_path`.

Current status: the staging code now keeps `images.zip` compressed on the
existing `anymal-checkpoints` volume and `LlavaPretrainCaptionDataset` reads
images directly from the zip. This avoids the exploded JPEG inode/device limit
that blocked the previous staging attempt. If `manifest.json` and `images.zip`
are missing, Stage 1 still intentionally falls back to COCO-backed captions.

Trainer readiness: full-run Modal timeouts are now raised above canary budgets:
Stage 1 distributed pretraining has an 8-hour timeout, and Stage 2 single-GPU
QLoRA has a 24-hour timeout.

Train-loop held-out benchmark eval is now off by default. For full model
selection, launch explicit external VQA checkpoint reads at the predeclared
milestones instead of watching validation metrics in the training loop.

## 2026-04-29 Corrected Full Stage 2 Runs

The corrected VQA protocol is now the only checkpoint-selection protocol:
`prompt_style=training_chat`, first-stop-token trimming, EOS including
`<|eot_id|>`, and scheduled checkpoint reads only.

The full V2 Stage 2 runs from the 256-query Stage 1 / 384-query Stage 2
expansion path were stopped after corrected VQA reads showed collapse:

- `v2-stage2-balanced-mix-light-3000-20260429`, checkpoint 300: `0.10%`.
- `v2-stage2-normalized-light-direct-object-full-3000-20260429`, checkpoint
  1000: `0.13%`.
- `v2-stage2-normalized-direct-object-full-3000-20260429`, checkpoint 1000:
  `0.37%`.

Do not select from those runs, or from the earlier
`v2-stage2-direct-object-full-3000-20260429` /
`v2-stage2-direct-natural-full-3000-20260429` runs. The earlier direct-answer
filter preserved full multi-turn source conversations, and the later corrected
branches failed the corrected VQA gate.

The active path is the true 384-query Stage 1 checkpoint:

`/checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000`

Stage 1 details:

- Modal app: `ap-wvvjKR4Du8Jq3lNTMXZpHQ`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/jmu6xbiu`
- Data: true zip-backed LLaVA-Pretrain images/captions.
- Usable samples: `555,258`; split `527,496 train / 27,762 val`.
- Token budget: `384` image tokens in Stage 1, matching Stage 2.
- Final eval loss: `2.4883`.

Active 384-query Stage 2 runs:

```bash
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --dataset balanced_mix \
  --gpu-type h100 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 5e-6 \
  --lora-learning-rate 7e-5 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000 \
  --use-wandb \
  --run-name v2-stage2-384q-balanced-mix-light-3000-20260429
```

```bash
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --dataset concat_mix_direct_object_light_trainprompt \
  --gpu-type h100 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 5e-6 \
  --lora-learning-rate 7e-5 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000 \
  --use-wandb \
  --run-name v2-stage2-384q-light-direct-object-3000-20260429
```

The VQAv2-train enhanced branch adds short-answer grounding data from train
splits only. It uses a deterministic COCO train2014 image subset and a cached
`vqa_train2014_direct_150000.json` file built from VQAv2 train2014
questions/annotations. This branch disables in-training held-out VQA eval.

```bash
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --dataset balanced_mix_vqa_direct_object_trainprompt \
  --gpu-type h100 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 5e-6 \
  --lora-learning-rate 7e-5 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000 \
  --use-wandb \
  --run-name v2-stage2-384q-vqa-direct-object-balanced-noeval-3000-20260429
```

Independent connector ablation:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v2 \
  --token-compressor-type perceiver \
  --gpu-type h100 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --pretrain-image-tokens 384 \
  --use-wandb \
  --run-name v2-stage1-perceiver-384q-3000-20260429
```

Selection protocol: fixed VQAv2 validation slice, `max_eval_samples=1000`,
`subset_seed=42`, scheduled checkpoints `300`, `1000`, `2000`, `3000`.
Run these as explicit external checkpoint reads. Canaries are diagnostic only,
and any inspected failures must not be recycled into same-run model selection.

## 2026-04-28 Full Learned-Connector Baseline

The first meaningful V2 learned-compressor Stage 1 + Stage 2 run is complete.
Use `V2_FULL_TRAINING_ARTIFACT_20260428.md` as the durable record.

Summary:

- Stage 1: true LLaVA-Pretrain zip-backed data, 2500 steps on 4x H100.
- Stage 1 Modal app: `ap-4Jy7OQ7ptFY77J2Zm1qg0a`.
- Stage 1 W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/x77vo36v`.
- Stage 1 checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`.
- Stage 1 final eval loss: `2.5290`.
- Stage 2: `balanced_mix`, 3000 steps on 1x A100.
- Stage 2 Modal app: `ap-598nYUdjvsf1K3AS3eeA8N`.
- Stage 2 W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/3gyl1apj`.
- Stage 2 checkpoint:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`.
- Stage 2 final eval loss: `1.1203`.
- Stage 2 final VQA: `6.47%` on 500 samples.

The first Stage 2 attempt restarted before its first checkpoint. The hardened
successful run saved every 50 steps, committed the Modal volume after each save,
and verified every checkpoint through `checkpoint-3000`.

2026-04-27 zip staging launch:

- Cleanup run: `ap-sHLZ3NRqxKFjE1CUPhIUVY` found no current
  `/checkpoints/llava_pretrain` directory.
- Zip staging run: `ap-AeBGzIo9qJZf2nidkichq6`.
- Result: staged `558,128` images as
  `/checkpoints/llava_pretrain/images.zip`.

## 2026-04-27 COCO-Fallback Canary Runs

Stage 1 learned-compressor canary:

- Modal app: `ap-z3aicTQumdMENCVIamGWXP`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/hyklgupy`
- Checkpoint: `/checkpoints/pretrain-output/v2-stage1-learned-canary-250-20260427-live/checkpoint-250`
- Final train loss: `1.9768349359547155`
- Step-250 eval loss: `1.6163`
- Data used: COCO-backed fallback, not true LLaVA-Pretrain.

Stage 2 learned-compressor canary:

- First attempt failed before training because the Stage 1 compressor query table
  was 256 rows and Stage 2 expects 384 rows.
- The loader now copies the 256 pretrained rows into the 384-row initialized
  Stage 2 table and leaves the additional rows initialized.
- Retry Modal app: `ap-P100EgvL0VcbAksOc0I6OS`
- W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/3d43l7fy`
- Checkpoint: `/checkpoints/finetune-output/v2-stage2-balanced-mix-canary-100-20260427-live-r2/checkpoint-100`
- Final train loss: `1.3395453007036215`
- Step-50 eval loss: `1.3110`
- Step-100 eval loss: `1.2880`
- Data used: `balanced_mix` over COCO-backed `instruct_150k` and `mix_665k`
  samples.

## 2026-04-28 Zip-Backed Canary Runs

These are the first V2 learned-compressor canaries that used true
LLaVA-Pretrain images for Stage 1.

Stage 1 learned-compressor canary:

- Modal app: `ap-qN6F3z0U7WcxeiK8Rgo6jh`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/7529i1tv`
- Checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-canary-250-20260427-zip/checkpoint-250`
- Data used: true LLaVA-Pretrain captions with zip-backed images.
- Samples: `555,258` usable samples after caption de-dupe; split
  `527,496 train / 27,762 val`.
- Token budget: Stage 1 uses 256 image placeholders.
- Final train loss: `3.6008024646259056`
- Step-50 eval loss: `3.6400`
- Step-250 eval loss: `3.0545`

Stage 2 learned-compressor canary:

- Modal app: `ap-KUlojDRbAzrhYRpASgzEYr`
- W&B final run:
  `https://wandb.ai/babakdam/anymal-finetune/runs/2wbl4ugt`
- Checkpoint:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-canary-100-20260427-zip/checkpoint-100`
- Data used: `balanced_mix`; Mix-665K filtered to `338,470/665,298`
  cached-COCO samples, with `157,712` Instruct-150K samples.
- Split: `643,093 train / 33,847 val`.
- Token budget: Stage 2 uses 384 image placeholders.
- Loaded Stage 1 projector and token compressor from the exact zip-backed
  Stage 1 checkpoint above. The Stage 2 loader expanded `pool_queries` by
  copying 256 pretrained rows into the 384-row Stage 2 table.
- Final train loss: `1.3780661925804034`
- Step-50 eval loss: `1.3188`
- Step-100 eval loss: `1.2886`
- Final VQA: `1.00%` on 500 samples; avg generated tokens `32.0`;
  EOS rate `0.0`. This is a canary sanity metric, not a quality claim.

Operational note: the first Stage 2 attempt in this Modal app was preempted at
about step 66 and restarted from scratch because the old config only saved at
step 100. The runner is now hardened so short canaries save at step 50, commit
the Modal volume on checkpoint save, and auto-resume from the latest checkpoint
in the run directory when restarted with the same run name.

## Baseline V2 Runs

Use the existing learned compressor first. This gives a clean V2 baseline for
the data/eval changes before testing the new connector.

Stage 1 canary that saves one checkpoint:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --max-steps 250 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --use-wandb \
  --run-name v2-stage1-learned-canary-250-20260427
```

Meaningful Stage 1 baseline:

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --max-steps 2500 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --use-wandb \
  --run-name v2-stage1-learned-2500-20260427
```

Only launch this after the zip staging run has written
`/checkpoints/llava_pretrain/manifest.json`; otherwise it will be a systems
check on the COCO-backed fallback path, not a real V2 Stage 1 run.

Then run Stage 2 from the Stage 1 checkpoint:

```bash
modal run modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type learned \
  --dataset balanced_mix \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-learning-rate 2e-4 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-learned-2500-20260427/checkpoint-2500 \
  --use-wandb \
  --run-name v2-stage2-balanced-mix-3000-20260427
```

## Connector Ablation

After the learned-compressor baseline, run the 2-layer Perceiver connector with
the same data and token budgets. Do not auto-load learned-compressor Stage 1
checkpoints into Perceiver Stage 2; the Modal runner now filters auto-discovery
by `token_compressor_type` and the finetune loader raises on explicit mismatch.

```bash
modal run --detach modal_train.py \
  --stage pretrain \
  --architecture anymal_v2 \
  --token-compressor-type perceiver \
  --max-steps 2500 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --use-wandb \
  --run-name v2-stage1-perceiver-2500-20260427
```

```bash
modal run modal_train.py \
  --stage finetune \
  --architecture anymal_v2 \
  --token-compressor-type perceiver \
  --dataset balanced_mix \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-learning-rate 2e-4 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v2-stage1-perceiver-2500-20260427/checkpoint-2500 \
  --use-wandb \
  --run-name v2-stage2-perceiver-balanced-mix-3000-20260427
```

## Reporting Template

For each run, record:

- Modal app run ID.
- W&B run link.
- Checkpoint path.
- Data source actually used by Stage 1: true LLaVA-Pretrain or COCO fallback.
- Stage 2 dataset: `instruct_150k`, `mix_665k`, or `balanced_mix`.
- Token budget: Stage 1 256, Stage 2 384.
- Connector: `learned`, `perceiver`, or `perceiver2`.
- Train/eval loss, generated-token rate, EOS rate.
- Final VQA metrics, with image coverage caveat.
- Qualitative predictions JSON or viewer link.
