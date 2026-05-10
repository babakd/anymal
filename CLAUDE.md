# CLAUDE.md - AnyMAL Project Context

## Project Status (Last Updated: May 2026)

**Current State**: V3 is the strongest stable path. V1 is the legacy baseline, V2 is preserved for diagnosis, and V3 is the current incumbent to beat. The best corrected fast-screen result is V3 direct-calibration LoRA-only checkpoint 100: `9.10%` overall on VQAv2 val2014 `1000` samples, seed `42`, `training_chat`, versus V1 ablation-F at `7.57%`. V3 wins overall, number, other, EOS, and max-token behavior, while V1 still has stronger yes/no calibration. The next major workstream should be V4: a new architecture closer to modern VLMs, using the V1/V2/V3 lessons and predeclared architecture/recipe ablations from `docs/V4_ARCHITECTURE_PLAN.md`.

**Repo layout note (May 2026):** active design docs and runbooks live under
`docs/`; frozen historical reports and old plans live under `docs/history/`.
Eval JSON artifacts live under `results/` — root-level `vqa_eval_*.json` and
similar patterns are gitignored to prevent re-pollution.

| Component | Status |
|-----------|--------|
| Model architecture | V3 incumbent; V4 should be a new architecture label; V2 preserved for diagnosis; V1 legacy baseline |
| Stage 1 pretrain (Modal, 4 GPU DDP) | V3 uses SigLIP2 384px + 128-token 6-layer Perceiver connector; preserve V2/V3 checkpoint metadata guards |
| Stage 2 finetune (Modal, 1 GPU QLoRA) | V3 stable recipe: direct-calibration LoRA-only, connector frozen, 100 steps |
| Inference / prediction viewer | Working (`modal_inference.py` + `prediction_viewer.html`) |
| W&B integration | Working (both stages) |
| LLaVA dataset loader | Working (InstructionDataset with real images) |
| Stage 1 caption dataset | Working for V2 through zip-backed LLaVA-Pretrain images; COCO fallback remains available |
| COCO images | Cached on Modal Volume (81,479 images) |
| Health monitoring | Working (loss/grad anomaly detection) |
| Throughput tracking | Working (tokens/sec, samples/sec) |
| Train/val split | Working (deterministic 95/5 split, both stages) |
| In-training eval | Working (clipped to 200 batches, both stages) |
| Pretrain checkpoint auto-discovery | Working (Stage 2 auto-loads Stage 1 projector) |
| Checkpoint resume | Working (restores optimizer, scheduler, scaler, RNG states) |
| VQA evaluation | Corrected `training_chat` fast screen is the current continuity benchmark; report answer types, EOS, and max-token hits |
| Unit tests | 110 passing, 1 skipped |

### V4 Handoff Notes (May 2026)

Future architecture work should start from `docs/V4_ARCHITECTURE_PLAN.md`. That file
records the incumbent V3 baseline, the lessons from V1/V2/V3, the proposed V4
global/local spatial Perceiver direction, recipe ablations, promotion gates, and
babysitting rules for long Modal runs.

The short version: do not repeat the V2 mistake of adding many visual tokens
through a weak connector. V4 should keep V3's compact 128-token discipline while
adding better spatial selection: separate global/local latents, 2D position
features, optional high-resolution or tiled inputs, and strict caps on the final
visual-token budget. Start with frozen SigLIP2 and connector-only Stage 1/1B,
then freeze the connector and run short LoRA-only direct-calibration Stage 2.

The V4 target is V3 checkpoint 100, not merely V1. A candidate must preserve
generation hygiene (`EOS >= 0.98`, max-token hits `<= 0.02`) and narrow the V1
yes/no gap without giving up V3's `other` gains. Extend
`scripts/check_v3_promotion.py` or create a generic VLM promotion guard before
calling any V4 checkpoint stable.

### V3 Handoff Notes (May 2026)

V3's stable recipe lives in `configs/finetune_v3.yaml`: `anymal_v3`,
`v3_direct_calibration`, connector frozen, LoRA-only Stage 2, 100 steps,
checkpoints every 50 steps. The run that produced the stable checkpoint was
started as a 300-step candidate and deliberately stopped after checkpoint 150:
validation regressed from `0.6071` at checkpoint 100 to `0.6229`, and held-out
VQA fell from `9.10` to `8.70`. Treat checkpoint 100 as the stable fast-screen
win.

V3's key lesson is that recipe and generation discipline matter as much as the
connector. Short direct-answer supervision worked; generic long instruction data
mostly changed style. Connector+LoRA Stage 2 hurt generation hygiene. For direct
calibration, keep the connector frozen unless a specific architecture ablation
requires otherwise.

### V2 Handoff Notes (Apr 2026)

Use `--architecture anymal_v2` for current work. V2 uses SigLIP2 preprocessing at 384px through `get_vision_transform()` / `get_siglip_image_transform()`; do not feed V2 CLIP-normalized 224px images. Both local scripts and Modal now route V2 datasets through SigLIP-compatible transforms.

V2 strict image insertion requires a contiguous block of placeholder tokens whose count exactly matches the compressed image-token count. Stage 1 uses 256 image tokens; Stage 2 currently uses 384 image tokens. VQA eval now inserts the same placeholder block for V2, so eval no longer silently exercises the old V1 prepend fallback.

Stage 1 V2 trains `token_compressor + projector` with the vision tower and LLM frozen. Stage 2 V2 trains `token_compressor + projector + LoRA`. Optimizer grouping treats `token_compressor` as part of the multimodal adapter, not as `other`. Stage 2 V2 defaults `projector_warmup_steps=0`; the old optimizer/scheduler rebuild warmup path should not be reintroduced.

Pretrain checkpoint auto-discovery is architecture-aware. Legacy V1 checkpoints without `model_meta.json` are refused for V2. If no V2 checkpoint exists, Modal Stage 2 will warn and train from random V2 adapter weights rather than loading the V1 projector.

Verified Modal smoke runs:
- V2 Stage 2 finetune smoke: `v2-smoke-20260427-codex-2`, app run `ap-SLiGhwZe4E5UQlcXm4cE5i`
- V2 Stage 1 pretrain smoke: `v2-pretrain-smoke-20260427-codex`, app run `ap-WK9FApkcZviH9IgaWYfGHZ`

Recommended next real experiment: evaluate the completed learned-compressor baseline, then run the Perceiver connector ablation from the same zip-backed Stage 1 data path. Do not compare V2 Stage 2 numbers against V1 Stage 1-loaded runs.

V2 quality roadmap: see `docs/history/V2_QUALITY_PLAN.md`. In short, prioritize eval hardening, real Stage 1 pretraining data, balanced Stage 2 data, dynamic/high-resolution visual tokens, a stronger spatial connector, and hallucination/verbosity alignment before doing more LR sweeps.

V2 quality-plan batch 1 is implemented and smoke-tested. It adds V2-compatible captioning eval metrics, LLaVA-Pretrain caption dataset plumbing, balanced Stage 2 instruction mixtures, config-gated `token_compressor_type="perceiver"` / `"perceiver2"`, and Modal support for `--dataset balanced_mix`. Validation: `pytest tests -q` -> 118 passed, 1 skipped; Modal smoke runs `ap-krX0b1NJwdAD0WGO4VdJAD` (Stage 1) and `ap-jEdIuwx1tvvHe4TiSdlE1v` (Stage 2 balanced mix). Current V2 Stage 1 uses `/checkpoints/llava_pretrain/images.zip` directly; if the zip manifest is missing, it still falls back to COCO-backed captions.

V2 Stage 1/2 run prep lives in `docs/history/V2_TRAINING_RUNBOOK.md`. It includes Modal canary run IDs, learned-compressor baseline commands, Perceiver connector ablation commands, and the required reporting template. True LLaVA-Pretrain image staging now works through `/checkpoints/llava_pretrain/images.zip`, read directly from the zip to avoid the existing Modal volume inode/device limit.

The completed 2026-04-28 full baseline is recorded in
`docs/history/V2_FULL_TRAINING_ARTIFACT_20260428.md`. Do not treat V2 learned-compressor
results as canary-only anymore: Stage 1 ran 2500 true LLaVA-Pretrain steps and
Stage 2 ran 3000 `balanced_mix` steps from that exact checkpoint.

Long-run monitoring policy lives in `docs/TRAINING_RUN_BABYSITTING_PLAYBOOK.md`.
Future agents should use it when supervising Modal training: prioritize correct
run identity, checkpoint recoverability, validation trend, and persistent
gradient behavior over noisy single-batch train loss.

### Two-Stage Training Pipeline

**Stage 1 -> Stage 2 flow**:
1. `modal run --detach modal_train.py --stage pretrain --max-steps 2500 --use-wandb` (4 GPUs, DDP)
2. Checkpoint saved to `/checkpoints/pretrain-output/checkpoint-N/projector.pt`
3. `modal run modal_train.py --stage finetune --max-steps 500 --use-wandb` (1 GPU, QLoRA)
4. Stage 2 auto-discovers the pretrain checkpoint and loads `projector.pt`
5. `modal run modal_inference.py` to generate predictions from all checkpoints

**Note**: Use `--detach` for Stage 1 (long runs). Without it, Modal kills the run if the local client disconnects.

### Verified Runs

**V2 learned Stage 1 full run (2500 steps, 4x H100, 2026-04-28)**:
- True LLaVA-Pretrain zip-backed data.
- Final checkpoint: `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Final eval loss: `2.5290`
- W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/x77vo36v`
- Durable record: `docs/history/V2_FULL_TRAINING_ARTIFACT_20260428.md`

**V2 learned Stage 2 full run (3000 steps, 1x A100, 2026-04-28)**:
- Loaded the exact Stage 1 checkpoint above; data was `balanced_mix`.
- Final checkpoint: `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`
- Final eval loss: `1.1203`
- Final VQA: `6.47%` on 500 samples
- W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/3gyl1apj`
- Durable record: `docs/history/V2_FULL_TRAINING_ARTIFACT_20260428.md`

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
├── modal_train.py              # Modal cloud training (Stage 1 + Stage 2, all V1-V4)
├── modal_inference.py          # Modal inference + prediction viewer JSON
├── modal_viewer.py             # FastAPI ASGI viewer for prediction JSONs
├── model_metadata.py           # Architecture-aware checkpoint metadata helpers
├── arch_sxs_inference.py       # V1/V2/V3/V4 side-by-side inference
├── compare_inference.py        # Two-checkpoint compare driver
├── three_way_inference.py      # Three-model compare driver
├── v1_v2_compare_inference.py  # V1/V2 side-by-side (used as helper by other eval scripts)
├── v2_compare_inference.py     # V2 Stage 1/Stage 2 compare driver
├── v2_quality_diagnostics.py   # V2 audit/probe diagnostics
├── vqa_checkpoint_eval.py      # Modal VQAv2 fast-screen evaluator (continuity benchmark)
├── pope_checkpoint_eval.py     # POPE hallucination eval
├── gqa_checkpoint_eval.py      # GQA eval
├── analyze_v2_compare.py / analyze_v2_probe.py  # Local analyzers for V2 JSONs
├── prediction_viewer.html      # Browser-based prediction comparison viewer
├── docs/                       # Active design docs and runbooks
│   ├── MODAL_SETUP.md
│   ├── TRAINING_RUN_BABYSITTING_PLAYBOOK.md
│   ├── V4_ARCHITECTURE_PLAN.md
│   ├── V5_RESEARCH_PLAN_20260509.md
│   ├── V6_CAUSAL_FALSIFICATION_PLAN.md
│   ├── V6_PLANNER_NEXT_STEPS.md
│   └── history/                # Frozen plans/reports kept for the record
│       ├── V2_DEBUG_REPORT_20260429.md
│       ├── V2_FULL_TRAINING_ARTIFACT_20260428.md
│       ├── V2_QUALITY_PLAN.md
│       ├── V2_TRAINING_RUNBOOK.md
│       ├── V3_ARCHITECTURE_PLAN.md
│       ├── V4_RESEARCH_RECIPE_20260508.md
│       ├── V4_V5_ABLATION_AGENT_BRIEF.md
│       ├── codex.md
│       └── v5_progress.md
├── results/                    # Tracked eval JSON artifacts (vqa_eval_*.json etc.)
├── configs/                    # finetune_v{1..5}*.yaml, pretrain_v{2..4}*.yaml, etc.
├── data/                       # Loaders, splitter, collators, dataset transforms
├── models/                     # anymal_v{1..4}.py, encoders/, projectors/, llm/
├── training/                   # trainer.py, pretrain.py, finetune.py, health_monitor.py, distributed.py
├── evaluation/                 # vqa_eval.py, captioning_eval.py, eval_runner.py
├── scripts/                    # Local entry points, promotion checkers, downloads, w&b inspector
├── notebooks/                  # 01_understanding_architecture.ipynb
└── tests/                      # test_model.py, test_training.py, test_evaluation.py, test_health_monitor.py
```

`results/` is gitignored at the root for `vqa_eval_*.json`, `vqa_sample_*.json`,
`vqa_checkpoint_eval_*.json`, `v1_v2_compare_*.json`, `v2_compare_*.json`,
`v2_probe_*.json`, `v2_quality_*.json` so new runs that drop JSONs at the repo
root are silently ignored — write eval outputs to `results/` instead.

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

V1-V3 work is complete and recorded in `EXPERIMENTS.md`. The active priorities
are V4 (and any V5/V6 follow-ups). New work should be tracked in
`docs/V4_ARCHITECTURE_PLAN.md` and `docs/V6_PLANNER_NEXT_STEPS.md` rather than
in this list.

Carryover items still relevant across versions:
- [ ] Fix eval loss logging in Stage 1 pretrain (still shows `0.0000` in stdout)
- [ ] Add flash-attn to the Modal image (pre-built wheel)
- [ ] Filter VQA eval to questions whose COCO images are present locally
- [ ] Support multi-image input
- [ ] Download COCO train images for full-scale training
- [ ] Split `modal_train.py` (currently 4,300+ lines) into per-stage modules

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
