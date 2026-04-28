# AnyMAL V2 Stage 1/2 Training Runbook

Last updated: 2026-04-27

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
4. Move true LLaVA-Pretrain image storage away from exploded files on the
   current Modal volume.
5. Run the meaningful Stage 1 job.
6. Run Stage 2 from that exact Stage 1 checkpoint.
7. Compare against Stage 1-only, Stage 2 `instruct_150k`, and Stage 2 `balanced_mix`.

## Validation

```bash
python3 -m py_compile modal_train.py training/finetune.py
pytest tests/test_model.py::TestAnyMALv2CoreModules tests/test_training.py tests/test_evaluation.py -q
```

## Data Staging

```bash
modal run --detach modal_train.py::stage_llava_pretrain_images
```

Expected result:

- `/checkpoints/llava_data/blip_laion_cc_sbu_558k.json`
- `/checkpoints/llava_pretrain/manifest.json`
- A manifest `image_dir` that resolves the annotation image refs.

Current status: this failed on the existing `anymal-checkpoints` volume because
the exploded JPEG tree exhausts the volume inode/device budget. Until we switch
to tar/WebDataset shards, object storage, or a larger/different volume, Stage 1
intentionally falls back to COCO-backed captions.

## 2026-04-27 Live Canary Runs

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
