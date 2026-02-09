# CLAUDE.md - AnyMAL Project Context

## Project Status (Last Updated: Feb 2026)

**Current State**: Two-stage training pipeline working end-to-end on Modal. Stage 1 pretrain completed on 4x A100-80GB (2500 steps, loss 12 -> 1.5, checkpoint-2500 saved). Stage 2 finetune verified on 1x A100-80GB (500 steps, loss 7.5 -> 1.4). Checkpoint resume and `--detach` mode working for long runs. Inference script produces predictions across checkpoints.

| Component | Status |
|-----------|--------|
| Model architecture | Complete |
| Stage 1 pretrain (Modal, 4 GPU DDP) | Complete (2500 steps, loss 12 -> 1.5, checkpoint-2500) |
| Stage 2 finetune (Modal, 1 GPU QLoRA) | Verified (500-step run, real COCO images) |
| Inference / prediction viewer | Working (`modal_inference.py` + `prediction_viewer.html`) |
| W&B integration | Working (both stages) |
| LLaVA dataset loader | Working (InstructionDataset with real images) |
| Stage 1 caption dataset | Working (COCOCaptionDataset, 157K samples from 81K images) |
| COCO images | Cached on Modal Volume (81,479 images) |
| Health monitoring | Working (loss/grad anomaly detection) |
| Throughput tracking | Working (tokens/sec, samples/sec) |
| Train/val split | Working (deterministic 95/5 split, both stages) |
| In-training eval | Working (clipped to 200 batches, both stages) |
| Pretrain checkpoint auto-discovery | Working (Stage 2 auto-loads Stage 1 projector) |
| Checkpoint resume | Working (restores optimizer, scheduler, scaler, RNG states) |
| VQA evaluation | Partial (fails on missing images, see Known Issues) |
| Unit tests | 101 passing, 1 skipped |

### Two-Stage Training Pipeline

**Stage 1 -> Stage 2 flow**:
1. `modal run --detach modal_train.py --stage pretrain --max-steps 2500 --use-wandb` (4 GPUs, DDP)
2. Checkpoint saved to `/checkpoints/pretrain-output/checkpoint-N/projector.pt`
3. `modal run modal_train.py --stage finetune --max-steps 500 --use-wandb` (1 GPU, QLoRA)
4. Stage 2 auto-discovers the pretrain checkpoint and loads `projector.pt`
5. `modal run modal_inference.py` to generate predictions from all checkpoints

**Note**: Use `--detach` for Stage 1 (long runs). Without it, Modal kills the run if the local client disconnects.

### Verified Runs

**Stage 1 full run (2500 steps, 4x A100-80GB)**:
- Loss: ~12 -> ~1.5 (plateaus around step 500, expected with frozen LLM)
- 4 GPUs via `mp.spawn` + DDP
- 157,712 caption samples from 81,479 COCO images
- Checkpoints saved every 500 steps (checkpoint-500 through checkpoint-2500)
- Resumed from checkpoint-1500 after Modal timeout (checkpoint resume working)
- Wall time: ~1.5 hours total, Cost: ~$22
- W&B: `anymal-pretrain` project

**Stage 2 verified run (500 steps, 1x A100-80GB)**:
- Loss: 7.5 -> 1.4 (converged)
- Eval loss: computed every 50 steps, ~57s each (200 batches)
- Grad norms: stable at 0.7-1.5
- Wall time: 35 min, Cost: ~$2.50

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

# Stage 2: Instruction finetuning (auto-loads Stage 1 checkpoint)
modal run modal_train.py --stage finetune --max-steps 500 --batch-size 2 --use-wandb

# Stage 2 with explicit pretrain checkpoint
modal run modal_train.py --stage finetune --max-steps 500 --pretrain-checkpoint /checkpoints/pretrain-output/checkpoint-2500

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
| `--use-dummy-data` | Ignored (real images always used, warns if passed) | False |
| `--learning-rate` | Learning rate | 1e-5 (finetune), 2e-4 (pretrain) |
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
| Finetune verified | 1x A100-80GB | 500 | ~35 min | ~$2.50 |

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
│   ├── trainer.py              # Base trainer (DDP, AMP, warmup, clipped eval)
│   ├── health_monitor.py       # Loss/gradient anomaly detection
│   ├── throughput_tracker.py   # Tokens/sec, samples/sec tracking
│   ├── pretrain.py             # Stage 1 trainer
│   ├── finetune.py             # Stage 2 trainer (loads pretrain checkpoint)
│   └── distributed.py          # Multi-GPU utilities (DDP setup/cleanup)
├── evaluation/
│   ├── vqa_eval.py             # VQAv2 evaluation
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
- Data: `InstructionDataset` — LLaVA-Instruct-150K with real COCO images
- GPU: 1x A100-80GB (QLoRA has DDP edge cases)
- Loads: Stage 1 `projector.pt` (auto-discovered or via `--pretrain-checkpoint`)
- LR: 1e-5, grad_accum: 4

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

---

## Known Issues & TODO

### Issues

1. **Flash Attention**: Not installed in Modal image (build takes too long). Training works with SDPA attention instead.

2. **OOM on A100-40GB**: Model + optimizer states exceed 40GB. Must use A100-80GB.

3. **VQA eval image coverage**: VQA evaluation samples 500 questions randomly, but these may reference COCO val2014 images not present in our cached set. Fix: filter the VQA dataset to only questions whose images exist locally.

4. **Stage 2 without Stage 1**: If no pretrain checkpoint exists, Stage 2 warns and runs with random perceiver weights. The model will learn to ignore image tokens. Always run Stage 1 first.

5. **Eval loss logging**: During Stage 1 pretrain, eval steps run (200 batches, progress bars visible) but logged eval loss shows `0.0000`. The eval loop runs but the loss value isn't captured correctly in stdout.

6. **Stage 1 loss plateau**: Loss plateaus at ~1.5 after ~500 steps. This is expected — the LLM is completely frozen in Stage 1, so the perceiver quickly learns the projection and then the frozen LLM becomes the bottleneck. More steps beyond 2500 are unlikely to help.

### TODO

- [x] Run full Stage 1 alignment (2500 steps, loss 12 -> 1.5, checkpoint-2500)
- [ ] Run Stage 2 finetune with pretrained perceiver and verify multimodal output
- [ ] Fix eval loss logging (currently shows 0.0000 in pretrain)
- [ ] Add flash-attn to Modal image (use pre-built wheel)
- [ ] Fix VQA eval to filter to available images only
- [ ] Support multi-image input
- [ ] Download COCO train images for full-scale training
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

# Stage 2: finetune (1 GPU, auto-loads pretrain checkpoint)
modal run modal_train.py --stage finetune --max-steps 500 --batch-size 2 --use-wandb

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
- **Modal Docs**: [modal.com/docs](https://modal.com/docs)
