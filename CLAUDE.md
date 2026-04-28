# CLAUDE.md - AnyMAL Project Context

## Project Status (Last Updated: Feb 2026)

**Current State**: Two-stage training pipeline working end-to-end on Modal. Stage 1 pretrain completed on 4x A100-80GB (2500 steps, loss 12 -> 1.5). Stage 2 initial run (500 steps, LR=1e-5, LLaVA-Instruct-150K only) showed flat loss and no improvement over pretrain-only. Root causes identified and fixed: separate LoRA/projector LRs, cosine LR floor, multi-task dataset support (LLaVA-1.5 Mix-665K filtered to COCO), projector warmup, VQA eval image filtering, and fair inference comparison (single quantization). Ready for validation run.

| Component | Status |
|-----------|--------|
| Model architecture | Complete |
| Stage 1 pretrain (Modal, 4 GPU DDP) | Complete (2500 steps, loss 12 -> 1.5, checkpoint-2500) |
| Stage 2 finetune (Modal, 1 GPU QLoRA) | Fixed — ready for validation run with multi-task data + separate LRs |
| Inference / prediction viewer | Working (`modal_inference.py` + `prediction_viewer.html`) |
| W&B integration | Working (both stages, now logs per-group LRs) |
| LLaVA dataset loader | Working (InstructionDataset, supports instruct_150k and mix_665k) |
| Multi-task dataset (Mix-665K) | Implemented (auto-downloads, filters to cached COCO images) |
| Stage 1 caption dataset | Working (COCOCaptionDataset, 157K samples from 81K images) |
| COCO images | Cached on Modal Volume (81,479 images) |
| Health monitoring | Working (loss/grad anomaly detection) |
| Throughput tracking | Working (tokens/sec, samples/sec) |
| Train/val split | Working (deterministic 95/5 split, both stages) |
| In-training eval | Working (clipped to 200 batches, debug logging added) |
| Pretrain checkpoint auto-discovery | Working (Stage 2 auto-loads Stage 1 projector) |
| Checkpoint resume | Working (restores optimizer, scheduler, scaler, RNG states) |
| VQA evaluation | Fixed (filters to available images, graceful error handling) |
| Separate LR (LoRA vs projector) | Implemented (default: 2e-4 LoRA / 2e-5 projector) |
| Cosine LR floor | Implemented (10% of peak, prevents decay-to-zero) |
| Projector warmup | Implemented (freeze projector for first 200 steps) |
| Fair inference comparison | Fixed (single model with use_qlora=True for all checkpoints) |
| Unit tests | 101 passing, 1 skipped |

### Architecture Split (Mar 2026)

Two architectures now exist and should be treated as separate runtime paths:

- `anymal_v1`: CLIP + Perceiver Resampler. Stable default.
- `anymal_v2`: SigLIP2 + token compressor + MLP projector.

Important implications:

- preprocessing is architecture-specific (`v1` CLIP transform, `v2` SigLIP2 processor)
- projector warmup is now conceptually visual-bridge warmup
- checkpoint loading and Modal inference are architecture-aware and should not mix `v1` and `v2` checkpoints

### Two-Stage Training Pipeline

**Stage 1 -> Stage 2 flow**:
1. `modal run --detach modal_train.py --stage pretrain --max-steps 2500 --use-wandb` (4 GPUs, DDP)
2. Checkpoint saved to `/checkpoints/pretrain-output/checkpoint-N/projector.pt`
3. `modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb --dataset mix_665k` (1 GPU, QLoRA)
4. Stage 2 auto-discovers the pretrain checkpoint and loads `projector.pt`
5. `modal run modal_inference.py` to generate predictions from all checkpoints

**Note**: Use `--detach` for long runs (Stage 1 and Stage 2 production runs). Without it, Modal kills the run if the local client disconnects.

### Verified Runs

**Stage 1 full run (2500 steps, 4x A100-80GB)**:
- Loss: ~12 -> ~1.5 (plateaus around step 500, expected with frozen LLM)
- 4 GPUs via `mp.spawn` + DDP
- 157,712 caption samples from 81,479 COCO images
- Checkpoints saved every 500 steps (checkpoint-500 through checkpoint-2500)
- Resumed from checkpoint-1500 after Modal timeout (checkpoint resume working)
- Wall time: ~1.5 hours total, Cost: ~$22
- W&B: `anymal-pretrain` project

**Stage 2 initial run (500 steps, 1x A100-80GB) — BASELINE, before fixes**:
- Loss: ~1.35 -> ~1.22 (flat, bouncy between 1.0-1.7 per step)
- Used LLaVA-Instruct-150K only, LR=1e-5 for both projector and LoRA
- No clear improvement over pretrain-only; some regressions (e.g., wrong dog breed)
- Root causes: single-task data, LR too low for LoRA, no LR floor, unfair inference comparison
- All root causes now fixed (see Key Bug Fixes 12-18)

**Stage 2 inference comparison (20 examples, 9 checkpoints)**:
- Step 0 (random projector): complete garbage (repeating tokens)
- Steps 1500-2500 (pretrain only): coherent, image-grounded responses (correctly IDs objects, colors, counts)
- Steps 250-500 (finetune + LoRA): similar quality to pretrain-only, no clear improvement
- Note: comparison was confounded by different quantization (pretrain=full precision, finetune=4-bit). Now fixed.
- See `predictions_20260209_063338.json` + `prediction_viewer.html` for full comparison

### Key Bug Fixes Applied

1. **LoRA alpha**: Changed from 64 to 16 (scaling 0.25, matches paper) in `modal_train.py`, `modal_inference.py`, `configs/base.yaml`, `configs/finetune.yaml`
2. **Stage 1 real images**: Rewrote `load_llava_pretrain_dataset()` to use real COCO images instead of `torch.rand()` dummies
3. **Inference placeholder tokens**: Fixed `_run_inference()` to build `input_ids` with explicit placeholder tokens (SPLICE mode), matching training
4. **Multi-GPU pretrain**: Added `pretrain_distributed()` with `mp.spawn` + DDP for 4-GPU Stage 1
5. **Pretrain checkpoint wiring**: Added `--pretrain-checkpoint` CLI arg with auto-discovery from `/checkpoints/pretrain-output`
6. **DDP pickling**: Moved `COCOCaptionDataset` to module level (was unpicklable inside function)
7. **DDP device_map**: Set `llm_device_map=None` for distributed mode (avoids cross-GPU tensor errors)
8. **DDP dataloader workers**: Set `num_workers=0` in distributed mode (avoids nested fork issues with `mp.spawn`)
9. **Checkpoint resume**: Added full checkpoint resume support (optimizer, scheduler, scaler, RNG states, health monitor) with micro-batch fast-forward on resume
10. **Modal function timeout**: Increased `pretrain_distributed` timeout from 7200s (2 hours) to 14400s (4 hours) to avoid timeout on long Stage 1 runs
11. **Modal client disconnect**: Long-running `modal run` commands fail if the local client disconnects. Fix: use `modal run --detach` for Stage 1
12. **Separate LRs for LoRA vs projector**: Added `--lora-learning-rate` CLI param. Default: 2e-4 for LoRA, 2e-5 for projector (matches LLaVA-1.5 pattern). Infrastructure already existed in `TrainerConfig.lora_learning_rate` and `_create_optimizer()` param groups.
13. **Cosine LR floor**: Replaced `CosineAnnealingLR` (decays to 0) with `LambdaLR` cosine-with-floor. `min_lr_ratio=0.1` means LR floors at 10% of peak. Respects per-group LRs via LambdaLR multiplier.
14. **Multi-task dataset**: Added `--dataset mix_665k` option. Downloads `llava_v1_5_mix665k.json`, filters to cached COCO images (~300-350K samples), normalizes image paths to filenames. Renamed `load_llava_instruct_dataset()` to `load_finetune_dataset()`.
15. **Projector warmup**: Added `projector_warmup_steps=200` to `FinetuneConfig`. Freezes projector for first 200 steps so LoRA warms up first, then unfreezes and rebuilds optimizer/scheduler.
16. **VQA eval image filtering**: Added `filter_to_available_images=True` to `VQADataset.__init__()`. Scans image_dir for available files, filters questions to matching images. Added try/except safety net in `__getitem__()`. Updated collate to filter None.
17. **Eval loss debug logging**: Added `print_rank_0()` in `evaluate()` showing batch count and avg_loss. Warns if 0 valid batches.
18. **Fair inference comparison**: Rewrote `modal_inference.py` to use a single model with `use_qlora=True` for ALL checkpoints. For pretrain-only checkpoints, zeros LoRA B matrices (no-op adapter = equivalent to no LoRA). Eliminates quantization confound.

---

## What is This Project?

Educational replication of the **AnyMAL paper** (arXiv:2309.16058) - a multimodal LLM that understands images + text.

**Architecture**: CLIP ViT-L/14 -> Perceiver Resampler -> LLaMA-3-8B-Instruct

```
Image (224x224)
    |
    v
+-------------------------+
|  CLIP ViT-L/14          |  <- FROZEN
|  Output: [B, 257, 1024] |
+-------------------------+
    |
    v
+-------------------------+
|  Perceiver Resampler    |  <- TRAINABLE
|  Output: [B, 64, 4096]  |
+-------------------------+
    |
    v
+-------------------------+
|  LLaMA-3-8B-Instruct    |  <- FROZEN + LoRA (Stage 2 only)
+-------------------------+
    |
    v
  Generated Text
```

---

## Training on Modal (Cloud GPUs)

### Prerequisites

1. **Modal account**: https://modal.com (sign up, add payment)
2. **HuggingFace token** with LLaMA access: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. **W&B account** (optional): https://wandb.ai

### One-Time Setup

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxx
modal secret create wandb WANDB_API_KEY=wandb_xxxxxxxxxxxxx
```

### Running Training

```bash
# Stage 1: Alignment pretraining (4 GPUs, ~1.5 hours for 2500 steps)
# IMPORTANT: Use --detach for long runs to avoid Modal killing the run on client disconnect
modal run --detach modal_train.py --stage pretrain --max-steps 2500 --batch-size 2 --use-wandb

# Stage 1: Resume from checkpoint (if interrupted)
modal run --detach modal_train.py --stage pretrain --max-steps 2500 --batch-size 2 --use-wandb --resume-checkpoint /checkpoints/pretrain-output/checkpoint-1500

# Stage 2: Instruction finetuning with multi-task data (RECOMMENDED)
# Uses LLaVA-1.5 Mix-665K filtered to cached COCO images, separate LRs, projector warmup
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --learning-rate 2e-5 --lora-learning-rate 2e-4 --dataset mix_665k

# Stage 2: With original LLaVA-Instruct-150K only (for comparison)
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --dataset instruct_150k

# Stage 2 with explicit pretrain checkpoint
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --pretrain-checkpoint /checkpoints/pretrain-output/checkpoint-2500 --dataset mix_665k

# Stage 2: Resume from checkpoint (if interrupted)
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --resume-checkpoint /checkpoints/finetune-output/checkpoint-1000

# Inference: generate predictions across all checkpoints
modal run modal_inference.py --num-examples 20
```

### Modal Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--max-steps` | Training steps | 100 |
| `--batch-size` | Per-device batch size | 4 |
| `--stage` | `"finetune"` or `"pretrain"` | finetune |
| `--use-wandb` | Enable W&B logging | False |
| `--learning-rate` | Projector learning rate | 2e-5 (finetune), 2e-4 (pretrain) |
| `--lora-learning-rate` | Separate LR for LoRA params | 2e-4 (finetune only) |
| `--dataset` | `"instruct_150k"` or `"mix_665k"` | instruct_150k |
| `--pretrain-checkpoint` | Path to Stage 1 checkpoint (auto-discovered if omitted) | None |
| `--track-per-layer-grad-norms` | Log per-layer gradient norms | True |
| `--run-eval-benchmarks` | Run VQA eval during finetune | True |
| `--resume-checkpoint` | Path to checkpoint to resume training from | None |

### Cost Estimates

| Run | GPU | Steps | Time | Cost |
|-----|-----|-------|------|------|
| Pretrain full | 4x A100-80GB | 2500 | ~1.5 hours | ~$22 |
| Pretrain smoke test | 4x A100-80GB | 20 | ~2.5 min | ~$0.50 |
| Finetune short | 1x A100-80GB | 200 | ~8 min | ~$1.00 |
| Finetune validation | 1x A100-80GB | 2000 | ~2-3 hours | ~$10-15 |
| Finetune full | 1x A100-80GB | 3000 | ~3-5 hours | ~$15-25 |

### Viewing Results

- **W&B Dashboard**: https://wandb.ai (project: `anymal-pretrain` or `anymal-finetune`)
- **Modal Logs**: https://modal.com/apps (see run history)
- **Checkpoints**: Saved to Modal Volume `anymal-checkpoints`
- **Predictions**: Open `prediction_viewer.html` and load `predictions.json`

---

## Local Development

### Project Structure

```
anymal/
├── modal_train.py              # Modal cloud training (Stage 1 + Stage 2)
├── modal_inference.py          # Modal inference + prediction viewer JSON
├── prediction_viewer.html      # Browser-based prediction comparison viewer
├── MODAL_SETUP.md              # Modal setup instructions
├── configs/
│   ├── base.yaml               # Shared settings (lora_alpha=16)
│   ├── pretrain_image.yaml     # Stage 1 config
│   └── finetune.yaml           # Stage 2 config (lora_alpha=16)
├── data/
│   ├── data_utils.py           # Transforms, collators, TextProcessor
│   ├── dataset_splitter.py     # Deterministic train/val split (5% val, seed=42)
│   ├── laion_dataset.py        # LAION loader (Stage 1, local)
│   └── instruction_dataset.py  # Instruction loader (Stage 2)
├── models/
│   ├── anymal.py               # Main model class + save/load_pretrained
│   ├── encoders/
│   │   └── image_encoder.py    # CLIP ViT-L wrapper
│   ├── projectors/
│   │   ├── perceiver_resampler.py  # Cross-attention projector
│   │   └── linear_projector.py     # Simple baseline
│   └── llm/
│       └── llama_wrapper.py    # LLaMA + QLoRA
├── training/
│   ├── trainer.py              # Base trainer (DDP, AMP, warmup, clipped eval, per-group LR)
│   ├── health_monitor.py       # Loss/gradient anomaly detection
│   ├── throughput_tracker.py   # Tokens/sec, samples/sec tracking
│   ├── pretrain.py             # Stage 1 trainer
│   ├── finetune.py             # Stage 2 trainer (projector warmup, pretrain checkpoint loading)
│   └── distributed.py          # Multi-GPU utilities (DDP setup/cleanup)
├── evaluation/
│   ├── vqa_eval.py             # VQAv2 evaluation (with image filtering)
│   ├── eval_runner.py          # In-training eval wrapper (graceful errors)
│   └── captioning_eval.py      # COCO captioning
├── scripts/
│   ├── train_finetune.py       # Local Stage 2 entry point
│   ├── train_pretrain.py       # Local Stage 1 entry point
│   └── download_checkpoints.py # Download LLaMA/CLIP
└── tests/
    ├── test_model.py           # Model unit tests
    ├── test_training.py        # Training pipeline tests
    └── test_health_monitor.py  # Health monitor tests
```

### Running Tests

```bash
pytest tests/ -v                     # All tests (101 passing)
pytest tests/test_model.py -v        # Model unit tests
pytest tests/test_training.py -v     # Training pipeline tests
pytest tests/test_health_monitor.py -v  # Health monitor tests
```

### Local Training (requires GPU)

```bash
python scripts/train_finetune.py --config configs/finetune.yaml --debug
torchrun --nproc_per_node=8 scripts/train_finetune.py --config configs/finetune.yaml
```

---

## Key Implementation Details

### Training Stages

**Stage 1 (Alignment Pretraining)** — `run_pretrain()` in `modal_train.py`:
- Freeze: CLIP + LLaMA (no LoRA, `use_qlora=False`)
- Train: Perceiver Resampler only (`set_training_stage(1)`)
- Data: `COCOCaptionDataset` — real COCO images + captions from LLaVA JSON
- GPU: 4x A100-80GB via `pretrain_distributed()` + `mp.spawn` + DDP
- Saves: `projector.pt` only (no LoRA weights)
- LR: 2e-4, grad_accum: 8, effective batch: 64 (4 GPUs x 2 x 8)

**Stage 2 (Instruction Tuning)** — `run_finetune()` in `modal_train.py`:
- Freeze: CLIP + LLaMA base
- Train: Perceiver Resampler + LoRA adapters (`use_qlora=True`, `lora_alpha=16`)
- Data: `InstructionDataset` — LLaVA-Instruct-150K or Mix-665K (filtered to cached COCO images)
- GPU: 1x A100-80GB (QLoRA has DDP edge cases)
- Loads: Stage 1 `projector.pt` (auto-discovered or via `--pretrain-checkpoint`)
- LR: 2e-5 projector / 2e-4 LoRA (separate param groups), cosine with 10% floor
- Projector warmup: frozen for first 200 steps (LoRA warms up alone)
- grad_accum: 8, effective batch: 16 (1 GPU x 2 x 8)

### Separate Learning Rates (Stage 2)

The optimizer creates separate param groups for projector and LoRA with different LRs:
- **Projector**: `learning_rate` (default 2e-5) — lower to preserve Stage 1 alignment
- **LoRA adapters**: `lora_learning_rate` (default 2e-4) — higher because LoRA is low-rank + has 0.25 scaling

This matches the LLaVA-1.5 pattern (10x higher LR for LoRA vs projector). The cosine scheduler with `min_lr_ratio=0.1` respects per-group LRs via `LambdaLR` (multiplies each group's `initial_lr`).

Plumbing: CLI `--lora-learning-rate` -> `Trainer.train()` -> `run_finetune(lora_learning_rate=...)` -> `FinetuneConfig(lora_learning_rate=...)` -> `Trainer._create_optimizer()` param groups.

### Projector Warmup (Stage 2)

`FinetuneConfig.projector_warmup_steps=200` freezes the projector for the first N steps. This gives LoRA a head start before projector drifts from Stage 1 alignment. At step N, projector is unfrozen and optimizer/scheduler are rebuilt to include projector params.

### Multi-Task Dataset

`--dataset mix_665k` loads `llava_v1_5_mix665k.json` (auto-downloaded during container setup) and filters to samples whose images exist in `/checkpoints/coco_images/`. The JSON uses paths like `coco/train2017/000000123456.jpg` — the loader extracts just the filename and matches against cached files. Expected ~300-350K surviving samples (COCO + some GQA/VG overlap). The filtered data covers diverse task types: conversation, VQA, reasoning, grounding — vs single-task conversation in the 150K dataset.

### LoRA Configuration

Paper-correct values: `lora_r=64`, `lora_alpha=16` (scaling = alpha/r = 0.25).
The default in `models/anymal.py:106` is already correct. The configs and modal scripts also use 16.

### Image Token Insertion

The model uses two modes in `_splice_image_tokens()` (`models/anymal.py`):
- **SPLICE**: When `input_ids` contain placeholder tokens (`image_placeholder_token_id`), image embeddings replace them at that position. Used during Stage 2 training and inference.
- **PREPEND**: When no placeholders are found, image tokens are prepended. Used during Stage 1 (captioning has no `<image>` markers).

The inference script builds `input_ids` with explicit placeholder tokens to ensure SPLICE mode matches training.

### Pretrain Checkpoint Auto-Discovery

When `--pretrain-checkpoint` is not passed to Stage 2, `run_finetune()` scans `/checkpoints/pretrain-output/` for the highest-numbered checkpoint with a `projector.pt` file. This means `--stage pretrain` followed by `--stage finetune` "just works".

The plumbing: CLI `main()` -> `Trainer.train()` -> `run_finetune(pretrain_checkpoint=...)` -> `FinetuneConfig(pretrain_checkpoint=...)` -> `FinetuneTrainer._load_pretrain_checkpoint()`.

### Multi-GPU Distributed Pretraining

Stage 1 uses `pretrain_distributed()` which requests `gpu="A100-80GB:4"` on Modal and calls `mp.spawn(_pretrain_worker, ...)`. Each worker:
1. Sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
2. Calls `setup_distributed()` from `training/distributed.py`
3. Creates model with `llm_device_map=None` (critical — `device_map="auto"` breaks DDP)
4. Creates dataloaders with `num_workers=0` (nested multiprocessing breaks `mp.spawn`)
5. The `Trainer` base class auto-wraps with DDP when `world_size > 1`

### Inference Pipeline

`modal_inference.py` loads a **single model with `use_qlora=True`** for ALL checkpoints (pretrain and finetune). For pretrain-only checkpoints, LoRA B matrices are zeroed out (`_reset_lora_weights()`) so the adapter is a no-op — equivalent to no LoRA but with the same 4-bit quantization. This ensures fair comparison across all checkpoints.

### Memory Requirements

| Config | VRAM |
|--------|------|
| Stage 1 (no LoRA, DDP) | ~35GB per GPU |
| Stage 2 (QLoRA + gradient checkpointing) | ~45GB |
| Full precision | ~80GB+ |

### Training Health Monitoring

The trainer includes automatic health monitoring (`training/health_monitor.py`) that detects:
- **Initial loss check**: Verifies first loss is near `ln(vocab_size)` = 11.76 (20% tolerance)
- **Loss spikes**: Loss > 2x the EMA triggers an alert
- **Loss divergence**: Sustained loss increase over 50 steps
- **Loss plateau**: EMA doesn't decrease over 200 steps
- **Gradient spikes**: Grad norm > 5x running average
- **Gradient vanishing**: Grad norm < 0.01 for 10 consecutive steps

Alerts are logged to console with `[HEALTH]` prefix and to W&B as `health/alert_*` metrics.

### Checkpoint Resume

The trainer supports resuming from any checkpoint via `--resume-checkpoint`. The checkpoint contains:
- Model state (projector weights, LoRA if applicable)
- Optimizer and scheduler state
- AMP GradScaler state
- Global step, epoch number
- CUDA/CPU RNG states (for reproducibility)
- Health monitor state

On resume, the trainer fast-forwards through `global_step * gradient_accumulation_steps` micro-batches in the first epoch to reach the correct data position. This takes ~3-13 minutes depending on step count (15-55 it/s).

### Clipped Evaluation

In-training eval is clipped to `max_eval_batches=200` (both stages) to avoid blocking training. With batch_size=2, this evaluates 400 samples in ~57 seconds.

### W&B Logging

The trainer logs per-group learning rates (`train/lr_projector`, `train/lr_lora`) and component gradient norms (`train/grad_norm_projector`, `train/grad_norm_lora`). Training config (including `lora_learning_rate`, `min_lr_ratio`) is logged to W&B config at start.

---

## Known Issues & TODO

### Issues

1. **Flash Attention**: Not installed in Modal image (build takes too long). Training works with SDPA attention instead.

2. **OOM on A100-40GB**: Model + optimizer states exceed 40GB. Must use A100-80GB.

3. **Stage 2 without Stage 1**: If no pretrain checkpoint exists, Stage 2 warns and runs with random perceiver weights. The model will learn to ignore image tokens. Always run Stage 1 first.

4. **Stage 1 loss plateau**: Loss plateaus at ~1.5 after ~500 steps. This is expected — the LLM is completely frozen in Stage 1, so the perceiver quickly learns the projection and then the frozen LLM becomes the bottleneck. More steps beyond 2500 are unlikely to help.

### Paper vs Our Implementation — Data Comparison

| | AnyMAL Paper | Our Replica (with fixes) |
|--|-------------|-------------|
| **Stage 1 data** | Cleaned LAION-2B subset (billions of pairs) | 157K COCO captions from LLaVA JSON (81K images) |
| **Stage 1 steps** | 100K steps, batch 2048 | 2500 steps, batch 64 (4 GPUs x 2 x 8) |
| **Stage 2 data** | ~210K custom (60K human + 150K LLaMA-2-70B synthetic) | Mix-665K filtered to COCO (~300-350K multi-task samples) |
| **Stage 2 steps** | 3K steps, batch 128 (384K samples) | 2000 steps, batch 16 (32K samples) |
| **Stage 2 LR** | 1e-5 | 2e-5 projector / 2e-4 LoRA |
| **Base LLM** | LLaMA-2 (13B/70B) | LLaMA-3-8B-Instruct |
| **LoRA** | r=64, alpha=16 | r=64, alpha=16 |

LLaVA-1.5 LoRA reference: LR=2e-4 for LoRA / 2e-5 for projector, r=128, alpha=256, 1 epoch over 665K samples.

### TODO

- [x] Run full Stage 1 alignment (2500 steps, loss 12 -> 1.5, checkpoint-2500)
- [x] Run Stage 2 finetune with pretrained perceiver (500 steps — baseline, flat loss)
- [x] Run inference comparison across checkpoints (pretrain-only vs finetune)
- [x] Implement multi-task dataset: Mix-665K filtered to cached COCO images
- [x] Implement separate LR for LoRA vs projector (2e-4 / 2e-5)
- [x] Add cosine LR floor (10% of peak)
- [x] Add projector warmup (freeze first 200 steps)
- [x] Fix VQA eval to filter to available images only
- [x] Fix eval loss debug logging
- [x] Fix inference to use same quantization for all checkpoints
- [ ] **Run validation: 2000 steps with mix_665k + separate LRs + projector warmup** (~$10-15)
- [ ] If validation improves, run full 3000 steps
- [ ] Add flash-attn to Modal image (use pre-built wheel)
- [ ] Support multi-image input
- [ ] Add more eval benchmarks (captioning, etc.)

---

## Model Dimensions

| Component | Dimension |
|-----------|-----------|
| CLIP ViT-L output | 1024 |
| CLIP patches | 257 (256 + CLS) |
| LLaMA hidden | 4096 |
| LLaMA layers | 32 |
| LLaMA vocab | 128,256 |
| Perceiver latents | 64 |
| Perceiver layers | 6 |

---

## Credentials & Secrets

**Stored as Modal Secrets** (do not commit to code):
- `huggingface`: HF_TOKEN for LLaMA access
- `wandb`: WANDB_API_KEY for logging

**LLaMA Weights**: Cached in Modal Volume `anymal-checkpoints` at `/checkpoints/llama3-8b-instruct/`

---

## Common Commands

```bash
# Run tests
pytest tests/ -v

# Stage 1: pretrain (4 GPUs, use --detach for long runs)
modal run --detach modal_train.py --stage pretrain --max-steps 2500 --batch-size 2 --use-wandb

# Stage 1: resume from checkpoint
modal run --detach modal_train.py --stage pretrain --max-steps 2500 --batch-size 2 --use-wandb --resume-checkpoint /checkpoints/pretrain-output/checkpoint-1500

# Stage 2: finetune with multi-task data (RECOMMENDED)
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --learning-rate 2e-5 --lora-learning-rate 2e-4 --dataset mix_665k

# Stage 2: finetune with original 150K data only
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --dataset instruct_150k

# Stage 2: resume from checkpoint
modal run --detach modal_train.py --stage finetune --max-steps 2000 --batch-size 2 --use-wandb \
  --resume-checkpoint /checkpoints/finetune-output/checkpoint-1000

# Inference: generate predictions from all checkpoints
modal run modal_inference.py --num-examples 20

# Check Modal run logs
modal app logs anymal-training

# List Modal volumes
modal volume list

# Download checkpoint from Modal volume
modal volume get anymal-checkpoints /checkpoints/pretrain-output ./local_pretrain
modal volume get anymal-checkpoints /checkpoints/finetune-output ./local_finetune
```

---

## References

- **AnyMAL Paper**: [arXiv:2309.16058](https://arxiv.org/abs/2309.16058)
- **Perceiver Resampler**: [Flamingo arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- **QLoRA**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **LLaVA**: [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **LLaVA-1.5**: [CVPR 2024 — Improved Baselines with Visual Instruction Tuning](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf)
- **Modal Docs**: [modal.com/docs](https://modal.com/docs)
