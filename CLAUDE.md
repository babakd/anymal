# CLAUDE.md - AnyMAL Project Context

## Project Status (Last Updated: Feb 2026)

**Current State**: Training pipeline is verified end-to-end on Modal. A 500-step finetuning run completed successfully with real COCO images, health monitoring, and W&B logging.

| Component | Status |
|-----------|--------|
| Model architecture | Complete |
| Local training scripts | Complete (bug fixes applied) |
| Modal cloud training | Verified (500-step run with real COCO images) |
| W&B integration | Working |
| LLaVA dataset loader | Fixed (uses hf_hub_download + InstructionDataset) |
| COCO images | Cached on Modal Volume (81,479 val2014 images) |
| Health monitoring | Working (loss/grad anomaly detection) |
| Throughput tracking | Working (tokens/sec, samples/sec) |
| Train/val split | Working (deterministic 95/5 split) |
| In-training eval | Working (clipped to 200 batches, ~57s per eval) |
| VQA evaluation | Partial (fails on missing images, see Known Issues) |
| Unit tests | 87 passing, 1 skipped |

**Verified Training Run (500 steps)**:
- Loss: 7.5 -> 1.4 (converged)
- Eval loss: computed every 50 steps, ~57s each (200 batches)
- Grad norms: stable at 0.7-1.5
- Wall time: 35 min on A100-80GB
- Cost: ~$2.50
- W&B: all metrics logged correctly

**Recent Bug Fixes Applied**:
1. Config YAML inheritance (`scripts/train_finetune.py`)
2. Label masking off-by-one (`data/data_utils.py`)
3. Autocast PyTorch 2.0+ compatibility (`training/trainer.py`)
4. Warmup scheduler with SequentialLR (`training/trainer.py`)
5. DDP local rank fix (`training/trainer.py`)
6. Checkpoint download path (`scripts/download_checkpoints.py`)
7. Config key path backward compatibility (`scripts/train_finetune.py`)
8. Evaluation batching with proper collate (`evaluation/*.py`)
9. LLaVA dataset loader fix + Modal efficiency optimizations (`modal_train.py`)
10. Clipped eval to avoid blocking training (`training/trainer.py`, `modal_train.py`)

---

## What is This Project?

Educational replication of the **AnyMAL paper** (arXiv:2309.16058) - a multimodal LLM that understands images + text.

**Architecture**: CLIP ViT-L/14 → Perceiver Resampler → LLaMA-3-8B-Instruct

```
Image (224x224)
    │
    ▼
┌─────────────────────────┐
│  CLIP ViT-L/14          │  ← FROZEN
│  Output: [B, 257, 1024] │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Perceiver Resampler    │  ← TRAINABLE
│  Output: [B, 64, 4096]  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LLaMA-3-8B-Instruct    │  ← FROZEN + LoRA (Stage 2)
└─────────────────────────┘
    │
    ▼
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
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Add secrets
modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxx
modal secret create wandb WANDB_API_KEY=wandb_xxxxxxxxxxxxx
```

### Running Training

```bash
# Quick test with dummy data (~$0.50, 2 min)
modal run modal_train.py --use-wandb --use-dummy-data --max-steps 50 --batch-size 2

# Longer run with dummy data
modal run modal_train.py --use-wandb --use-dummy-data --max-steps 500 --batch-size 2

# Stage 1 pretraining
modal run modal_train.py --stage pretrain --use-dummy-data --max-steps 1000
```

### Modal Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--max-steps` | Training steps | 100 |
| `--batch-size` | Per-device batch size | 4 |
| `--stage` | "finetune" or "pretrain" | finetune |
| `--use-wandb` | Enable W&B logging | False |
| `--use-dummy-data` | Use synthetic data | False |
| `--learning-rate` | Learning rate | 1e-5 (finetune), 2e-4 (pretrain) |

### Cost Estimates

| Run | GPU | Steps | Time | Cost |
|-----|-----|-------|------|------|
| Quick test (dummy) | A100-80GB | 50 | ~2 min | ~$0.25 |
| Short run (dummy) | A100-80GB | 200 | ~8 min | ~$1.00 |
| Verified run (real images) | A100-80GB | 500 | ~35 min | ~$2.50 |
| Medium run | A100-80GB | 1000 | ~70 min | ~$5.00 |

*Note: Times above include model loading (~2 min) and 10 eval points (~57s each). Training throughput is ~3.5 it/s at batch_size=2.*

### Viewing Results

- **W&B Dashboard**: https://wandb.ai/your-username/anymal-finetune
- **Modal Logs**: https://modal.com/apps (see run history)
- **Checkpoints**: Saved to Modal Volume `anymal-checkpoints`

---

## Local Development

### Project Structure

```
anymal/
├── modal_train.py              # Modal cloud training script
├── MODAL_SETUP.md              # Modal setup instructions
├── checkpoints/
│   └── llama3-8b-instruct/     # LLaMA weights (15GB, cached on Modal)
├── configs/
│   ├── base.yaml               # Shared settings
│   ├── pretrain_image.yaml     # Stage 1 config
│   └── finetune.yaml           # Stage 2 config
├── data/
│   ├── data_utils.py           # Transforms, collators, TextProcessor
│   ├── dataset_splitter.py     # Deterministic train/val split (5% val, seed=42)
│   ├── laion_dataset.py        # LAION loader (Stage 1)
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
│   ├── finetune.py             # Stage 2 trainer
│   └── distributed.py          # Multi-GPU utilities
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
    └── test_health_monitor.py  # Health monitor tests
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Model tests only
pytest tests/test_model.py -v

# Health monitor tests only
pytest tests/test_health_monitor.py -v
```

### Local Training (requires GPU)

```bash
# Debug mode (50 steps)
python scripts/train_finetune.py --config configs/finetune.yaml --debug

# Multi-GPU
torchrun --nproc_per_node=8 scripts/train_finetune.py --config configs/finetune.yaml
```

---

## Key Implementation Details

### Model Initialization

```python
from models import AnyMAL

model = AnyMAL(
    llm_model_name="./checkpoints/llama3-8b-instruct",
    vision_model_name="ViT-L-14",
    vision_pretrained="openai",
    projector_type="perceiver",      # or "linear"
    num_image_tokens=64,             # Perceiver output tokens
    use_qlora=True,                  # 4-bit quantization
    lora_r=64,
    lora_alpha=16,
    use_flash_attention=False,       # Set True if flash-attn installed
    gradient_checkpointing=True,
)
```

### Training Stages

**Stage 1 (Alignment Pretraining)**:
- Freeze: CLIP + LLaMA
- Train: Perceiver Resampler only
- Data: Image-caption pairs (LAION/CC3M)
- LR: 2e-4, Steps: 100K

**Stage 2 (Instruction Tuning)**:
- Freeze: CLIP + LLaMA base
- Train: Perceiver Resampler + LoRA adapters
- Data: LLaVA-Instruct-150K
- LR: 1e-5, Steps: 3K

### Memory Requirements

| Config | VRAM |
|--------|------|
| QLoRA + gradient checkpointing | ~45GB |
| Full precision | ~80GB+ |

Use A100-80GB for training. A100-40GB will OOM.

### Training Health Monitoring

The trainer includes automatic health monitoring (`training/health_monitor.py`) that detects:

- **Initial loss check**: Verifies first loss is near `ln(vocab_size)` = 11.76 (20% tolerance)
- **Loss spikes**: Loss > 2x the EMA triggers an alert
- **Loss divergence**: Sustained loss increase over 50 steps
- **Loss plateau**: EMA doesn't decrease over 200 steps (model stopped learning)
- **Gradient spikes**: Grad norm > 5x running average
- **Gradient vanishing**: Grad norm < 0.01 for 10 consecutive steps

Alerts are logged to console with `[HEALTH]` prefix and to W&B as `health/alert_*` metrics. Alerts have a 100-step cooldown to avoid spam.

The `ThroughputTracker` (`training/throughput_tracker.py`) reports tokens/sec and samples/sec over a 50-step sliding window, logged to W&B as `throughput/*`.

### Clipped Evaluation

In-training eval is clipped to `max_eval_batches` (default: 200 in modal_train.py) to avoid blocking training. With batch_size=2, this evaluates 400 samples in ~57 seconds instead of the full val set (7,885 samples, ~18 min).

Full evaluation runs once at the end of training.

Config field in `TrainerConfig`:
```python
max_eval_batches: Optional[int] = None  # None = full eval, 200 = clipped
```

---

## Known Issues & TODO

### Issues

1. **Flash Attention**: Not installed in Modal image (build takes too long). Training works with SDPA attention instead.

2. **OOM on A100-40GB**: Model + optimizer states exceed 40GB. Must use A100-80GB.

3. **VQA eval image coverage**: VQA evaluation samples 500 questions randomly, but these may reference COCO val2014 images not present in our cached set. The EvalRunner's random subset can pick questions whose images weren't downloaded. Fix: either download all ~40K val2014 images needed by VQA, or filter the VQA dataset to only questions whose images exist locally.

4. **COCO images**: 81,479 val2014 images are cached on Modal Volume. Training uses real images. However, train images (for full-scale training) are not cached.

### TODO

- [ ] Add flash-attn to Modal image (use pre-built wheel)
- [ ] Implement inference script
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

## Git Configuration

**Author for commits**:
- Name: Your Name
- Email: your-email@example.com

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

# Train on Modal (quick test with dummy data)
modal run modal_train.py --use-wandb --use-dummy-data --max-steps 50 --batch-size 2

# Train on Modal (real COCO images, recommended)
modal run modal_train.py --use-wandb --max-steps 500 --batch-size 2

# Check Modal run logs
modal app logs anymal-training

# List Modal volumes
modal volume list

# Download checkpoint from Modal volume
modal volume get anymal-checkpoints /checkpoints/finetune-output ./local_checkpoint
```

---

## References

- **AnyMAL Paper**: [arXiv:2309.16058](https://arxiv.org/abs/2309.16058)
- **Perceiver Resampler**: [Flamingo arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- **QLoRA**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **LLaVA**: [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **Modal Docs**: [modal.com/docs](https://modal.com/docs)
