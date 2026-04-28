# LoRA Configuration Sweep Results

## Sweep Config
- Steps per experiment: 200
- Batch size: 2
- Gradient accumulation: 8
- Effective batch size: 16
- GPU: A100-80GB
- Total experiments: 15 unique runs (+ 3 reused)
- Estimated cost: ~$15-18

## Stage A: Target Module Selection (r=64, alpha=16)
| ID | target_modules | eval_loss | train_loss | status |
|----|---------------|-----------|------------|--------|
| A1 | q,v (2 modules) | 1.3149 | 1.3425 | done |
| A2 | q,k,v,o (4 attn) | 1.3030 | 1.3324 | done |
| **A3** | **all 7 (default)** | **1.2855** | 1.3187 | **winner** |
| A4 | q,v,gate,up (4 mixed) | 1.2885 | 1.3218 | done |

**Finding**: More modules = better. All 7 wins, but diminishing returns after 4 modules.

## Stage B: Rank Selection (all 7 modules, alpha/r=0.25)
| ID | lora_r | lora_alpha | eval_loss | train_loss | status |
|----|--------|------------|-----------|------------|--------|
| B1 | 4 | 1 | 1.3111 | 1.3430 | done |
| B2 | 8 | 2 | 1.3045 | 1.3367 | done |
| B3 | 16 | 4 | 1.2974 | 1.3303 | done |
| B4 | 32 | 8 | 1.2913 | 1.3242 | done |
| **B5** | **64** | **16** | **1.2855** | 1.3187 | **winner (=A3)** |

**Finding**: Monotonic improvement with rank. r=64 wins. Diminishing returns above r=32.

## Stage C: Alpha Scaling (r=64, all 7 modules)
| ID | lora_alpha | alpha/r ratio | eval_loss | train_loss | status |
|----|------------|---------------|-----------|------------|--------|
| C1 | 16 | 0.25 (paper default) | 1.2855 | 1.3187 | reused from A3 |
| C2 | 32 | 0.5 | 1.2788 | 1.3134 | done |
| C3 | 64 | 1.0 | 1.2720 | 1.3092 | done |
| **C4** | **128** | **2.0** | **1.2705** | 1.3108 | **winner** |

**Finding**: The paper's alpha=16 (ratio 0.25) is suboptimal. Ratio 2.0 beats it by 1.2%. This was the biggest single improvement in the sweep.

## Stage D: Learning Rate (r=64, alpha=128, all 7 modules)
| ID | lr (projector) | lora_lr | eval_loss | train_loss | status |
|----|---------------|---------|-----------|------------|--------|
| D1 | 5e-6 | 1e-4 | 1.2750 | 1.3087 | done |
| D2 | 1e-5 | 2e-4 | 1.2705 | 1.3108 | reused from C4 |
| **D3** | **2e-5** | **2e-4** | **1.2692** | 1.3107 | **winner** |
| D4 | 1e-5 | 5e-4 | 1.2944 | 1.3684 | done |
| D5 | 2e-5 | 5e-4 | 1.2941 | 1.3672 | done |

**Finding**: lora_lr=2e-4 is the sweet spot. 5e-4 overshoots with alpha=128. Doubling projector LR to 2e-5 helps slightly.

---

## Overall Winner

**Config**: `lora_r=64, lora_alpha=128, target_modules=all 7, lr=2e-5, lora_lr=2e-4`

| Metric | Paper Default | Optimized | Improvement |
|--------|--------------|-----------|-------------|
| eval_loss | 1.2855 | **1.2692** | **-1.27%** |
| train_loss | 1.3187 | 1.3107 | -0.61% |
| lora_alpha | 16 | **128** | 8x higher |
| lr (projector) | 1e-5 | **2e-5** | 2x higher |

### Key Takeaways

1. **Alpha scaling was the biggest lever** — changing alpha from 16 to 128 (ratio 0.25 → 2.0) gave ~1.2% eval loss improvement. The paper's default was too conservative.

2. **Target modules and rank matched the defaults** — all 7 modules and r=64 were already the right choices. No savings to be found by reducing these.

3. **LoRA LR is sensitive with high alpha** — 5e-4 LoRA LR works fine with alpha=16 but overshoots with alpha=128 (since effective LoRA update = alpha/r * gradient, higher alpha amplifies the step size).

4. **Projector LR can be higher** — 2e-5 is slightly better than 1e-5, suggesting the projector can absorb faster updates.

### Recommended `modal_train.py` command
```bash
modal run modal_train.py --stage finetune \
  --max-steps 500 \
  --batch-size 2 \
  --lora-r 64 \
  --lora-alpha 128 \
  --learning-rate 2e-5 \
  --lora-learning-rate 2e-4 \
  --use-wandb
```
