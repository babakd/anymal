# V10 Qwen3 Ceiling Results

Date: 2026-05-13

## Decision

No V10 Qwen3 ceiling candidate is promoted.

The campaign found several ways to recover the V9/Batch-A plateau, but no
candidate cleared the minimum viable GQA floor while preserving the corrupted
image controls and POPE requirements. The best GQA movement came from C1
learned 2D patch features, but that branch damaged blank/shuffled/wrong-image
controls and was not promotable.

## Baseline

Current Qwen/V9 candidate:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Important confirmed metrics:

```text
Clean VQA n1000:        66.167
Blank n1000:            39.400
Shuffled n1000:         37.367
Wrong-image n1000:      37.967
POPE n1000:             79.100
GQA trusted n1000:      43.100
```

Minimum viable floor for this campaign required GQA trusted n1000 >= 43.8 and
better than the V3 robust same-slice GQA, with controls preserved.

## Branch Summary

| Branch | Best or screened result | Decision |
| --- | --- | --- |
| Batch A: scale/anchor/interpolation search | Best materialized candidate at `/checkpoints/pretrain-output/v10-qwen3-stage1b2400-scale105-batcha-best/checkpoint-2400-scale105`, GQA 43.4 | Not enough GQA |
| B1: GQA + anti-shuffle connector continuation | checkpoint-100 GQA 43.0 | Stop |
| B2: contrastive image-dependence loss | checkpoint-50 GQA 39.6 | Stop |
| B3: trainable global/per-token scale | Best GQA 43.1 | Stop |
| C1: learned 2D patch table | checkpoint-150 scale 1.125 GQA 44.5 | Not promotable: blank 40.0, shuffled 37.8, wrong 38.5, perturb mean about 65.6 |
| C2: coordinate MLP | checkpoint GQA sequence 40.3, 42.8, 41.1, 41.7 | Stop |
| D1: query-conditioned scalar scale | checkpoint-25 GQA 43.1, checkpoint-100 GQA 42.7 | Stop |
| D2: query-conditioned per-token scale | checkpoint-25 GQA 42.8, checkpoint-100 GQA 43.2 | Stop |
| D3: neutral query-conditioned patch selector | checkpoint-25 GQA 43.2, checkpoint-50 GQA 43.1, checkpoint-100 GQA 43.0 | Stop |
| E1: gated visual cross-attention adapters | checkpoint-25 GQA 43.4, checkpoint-50 GQA 43.4 | Stop |

## E1 Notes

E1 was implemented as a frozen-Qwen, frozen-connector visual cross-attention
adapter path with selected decoder layers `12,18,24,30`, adapter bottleneck
512, and zero gate initialization.

Smoke fix:

```text
/checkpoints/pretrain-output/v10-qwen3-e1-vxattn-gated-smoke-step1b/checkpoint-1
```

The first smoke caught a gradient-checkpointing/autograd issue when all layer
inputs were frozen. After forcing `inputs_embeds` to require grad for this
frozen-connector adapter mode, the one-step smoke completed and saved.

Main E1 run:

```text
/checkpoints/pretrain-output/v10-qwen3-e1-vxattn-gated-cont100-lr1e4-save25/checkpoint-25
/checkpoints/pretrain-output/v10-qwen3-e1-vxattn-gated-cont100-lr1e4-save25/checkpoint-50
https://wandb.ai/babakdam/anymal-pretrain/runs/7v0sdm8e
```

The run was intentionally stopped at step 50 after the early screen failed.
W&B reported `train/grad_norm: 0.0` at step 50, while both checkpoints produced
the same GQA score and answer histogram:

```text
GQA trusted n1000:       43.4
Strict accuracy:         43.4
EOS rate:                1.0
Max-token-hit rate:      0.0
Assistant-prefix rate:   0.0
Top answers:             no=184, yes=161, man=35, right=27, left=25
```

This suggests the zero-gated adapter stayed effectively at the Batch-A plateau
under this short continuation, so no full control bundle was run.

## Verification

Local checks:

```bash
python3 -m py_compile models/anymal_v3.py scripts/modal/train.py modal_train.py models/visual_cross_attention.py training/pretrain.py training/trainer.py
python3 -m compileall -q models training evaluation data scripts
python3 scripts/repo_health_check.py
```

Modal artifacts were written under:

```text
/checkpoints/v10_qwen_ceiling/
```

## Conclusion

The active Qwen3 AnyMAL ceiling remains the V9/Batch-A family. Qwen3 remains
strong on clean VQA and POPE, but this campaign did not find a connector or
adapter repair that improves GQA trusted n1000 enough without damaging the
safety/control metrics. The most informative near-miss was C1: spatial patch
features can move GQA, but the movement currently comes with unacceptable
control regressions.
