# AnyMAL V2 Stage 1/2 Training Runbook

Last updated: 2026-04-28

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
captions because true LLaVA-Pretrain images are not staged. On 2026-04-27,
attempting to extract `images.zip` into the existing Modal volume failed with
`OSError: [Errno 28] No space left on device` after the volume hit its inode
limit. Do not judge real V2 quality from COCO-fallback Stage 1 alone.

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
