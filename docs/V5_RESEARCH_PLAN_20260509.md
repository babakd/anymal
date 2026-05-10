# AnyMAL V5 Research Plan - 2026-05-09

## Objective

V5 should beat the promoted V4 bundle in a way that survives stricter evidence:
raw generated answers should be clean, the architecture gain should be isolated
from the recipe gain, and the result should hold beyond one VQAv2 seed. The
goal is not to hide V4's weaknesses behind a stronger post-processor. It is to
turn the lessons from V4 into a cleaner architecture and training recipe.

## What V4 Actually Proved

V4 proved that the AnyMAL stack can be pushed far above V3 on the corrected
VQAv2 fast screen when a spatial connector checkpoint is paired with semantic
calibration data. The strongest reported checkpoint was:

- `anymal_v4`
- SigLIP2-So400m at 384px
- 64 global + 64 local visual tokens
- Stage1B248 legacy-width spatial connector checkpoint
- frozen connector, LoRA-only Stage 2
- `v4_semantic_calibration`, batch size 4, `loss_scale=0.03`
- cleaned VQAv2 seed-42 score: `52.40` overall, `76.09` yes/no, EOS `1.0`

The critique changes how that result should be interpreted:

- The original raw-output evidence was contaminated by VQA eval right-padding:
  batched decoder-only generation was starting shorter prompts from padded
  positions, which produced apparent `assistant\n\n<answer>` prefixes at roughly
  the expected `(batch_size - 1) / batch_size` rate.
- V5 evaluation therefore left-pads decoder prompts. Under the corrected
  seed-42 eval, the V4 semantic-calibration checkpoint scores `51.27`
  strict/clean with `0.0` assistant-prefix rate.
- The promoted checkpoint uses the older direct 4096-wide connector, not the
  desired lean bottleneck connector.
- The global/local split, 2D positions, token budget, and V4 recipe transfer to
  V3/V1 were not causally isolated.
- Two VQAv2 seeds are encouraging, but still too narrow for a new default.

## V5 Position

V5 is an evidence-first modernization:

1. Keep the compact visual-token contract. Do not repeat V2's raw token-growth
   failure.
2. Prefer the lean bottleneck spatial connector as the V5 architecture target:
   64 global + 64 local tokens, connector width 1024, 3 layers, 8 heads, 2D
   position features, output gate `0.0001`.
3. Treat role-clean generation as a first-class training and promotion target.
   A model that needs evaluator stripping for most answers is not promotable.
4. Keep semantic calibration, but make it answer-only at the prompt contract:
   no role labels, no explanations, no duplicated assistant headers.
5. Run the minimum causal grid needed to distinguish architecture gains from
   recipe gains before claiming V5 is an architecture win.

## Promotion Gates

V5 must clear all V4/V3 gates plus stricter V5-only gates:

| Gate | Requirement |
|---|---:|
| Clean VQAv2 seed-42 accuracy | `>=` promoted V4 cleaned score once V4 is evaluated with the corrected left-padded raw-clean path |
| Strict/raw VQAv2 seed-42 accuracy | within `1.0` point of cleaned accuracy |
| Assistant role prefix rate | `<= 0.01` overall and `<= 0.02` in every answer-type bucket |
| EOS rate | `>= 0.98` |
| Max-token-hit rate | `<= 0.02` |
| Yes/no on non-yes/no questions | `<= 0.05` |
| Confirmation | at least two additional VQAv2 seeds or one larger locked slice |
| Robustness | must not lose to promoted V4 on locked resize, blur, crop, or translation perturbations |
| Leakage | VQAv2 train/calibration image IDs disjoint from evaluated val2014 image IDs |

The new eval diagnostics are `strict_accuracy`, `assistant_role_prefix_rate`,
per-answer-type prefix rates, top answer histograms, and predicted answer-kind
rates by answer type.

## Ablation Sequence

### Tier 0: No-GPU Sanity

Run before training:

- Analyze V4 prediction dumps for role-prefix rate and marginal answer
  distribution.
- Add strict/raw scoring to VQA eval so prefix stripping cannot silently produce
  the headline metric.
- Fix decoder-only VQA eval to left-pad prompts; right-padded batched evals are
  invalid for raw-generation hygiene decisions.
- Add a leakage audit for Stage 2 VQAv2 sources versus every locked eval slice.
- Re-run the promoted V4 checkpoint with full prediction dumps under the stricter
  diagnostics.

### Tier 1: Cheap V5 Recipe Isolation

Run short frozen-connector LoRA-only Stage 2 branches:

| ID | Base checkpoint | Dataset | Purpose |
|---|---|---|---|
| V5-R0 | Stage1B248 legacy V4 | `v5_semantic_calibration` | Does a stricter answer-only prompt remove role-prefix emission without losing V4's cleaned score? |
| V5-R1 | Stage1B248 legacy V4 | `v5_semantic_calibration_robust` | Does light VQA-safe image augmentation recover the blur robustness loss without sacrificing raw hygiene? |
| V5-A1 | A1 bottleneck Stage1B400 | `v5_semantic_calibration` | Does the lean modern connector inherit the semantic-calibration gain? |
| V5-C1 | V3 Stage 1B | `v5_semantic_calibration` | Does the recipe alone explain the gain? |

Selection rule: do not compare cleaned accuracy alone. Use strict accuracy and
role-prefix rate as co-primary metrics.

Stop/go rule: V5-R0 can justify continuing the role-clean recipe, but it cannot
by itself justify V5-A1. Before spending architecture-transfer compute, R0 must
clear the raw/strict gate, W&B must finish green, and seed/robustness
confirmation must show that the observed gain is not just evaluator cleanup or
seed-42 noise. If a cheap robustness check flips the V5 margin, run a
recipe-level fix before launching an architecture-transfer branch.

Current R0 evidence after the evaluator fix:

- Corrected seed-42/43/44 strict mean: V5-R0 `51.81`, V4 `51.56`, mean margin
  `+0.26`; all runs have `0.0` assistant-prefix rate.
- Seed-42 `mild_blur`: V5-R0 `51.47`, V4 `51.60`, so the margin flips by
  `-0.13`.
- Prompt isolation on `mild_blur` points to the checkpoint/training mix, not the
  inference prompt: V5-R0 with the V4 prompt scores `51.30`; V4 with the V5
  prompt remains `51.60`.
- Decision: do not launch V5-A1 yet. Run V5-R1 as a light visual-robustness
  recipe canary from the same Stage1B248 base.
- V5-R1 at accumulation `8` was stopped before step 50 because the strict W&B
  inspector still had an active `recent_grad_spikes` alert at synced step `16`
  (`2.28x` spike source at step `8`). No checkpoint from that run is usable.
  R1b increases finetune accumulation to `16` to reduce augmented-step variance.

### Tier 2: Architecture Attribution

Only after V5-A1 is mechanically healthy:

| ID | Change | Question |
|---|---|---|
| V5-A2 | no 2D position features | Are the spatial positions doing work? |
| V5-A3 | 48 global / 48 local tokens | Is 128 tokens necessary? |
| V5-A4 | 64 global / 128 local tokens | Does more local capacity help without hygiene loss? |
| V5-A5 | prompt-conditioned patch selector before Perceiver | Can learned token routing improve robustness without prefix growth? |

### Tier 3: Robustness And Second Benchmark

Run the chosen V5 checkpoint and the promoted V4 checkpoint on:

- VQAv2 perturbations: `resize_up`, `mild_blur`, `center_crop_90`,
  `translate_5pct`
- POPE or an equivalent hallucination/negative yes-no benchmark
- A public multimodal benchmark if cached or cheap to stage

## First V5 Experiment

Launch `V5-R0` first because it tests the critique's highest-leverage issue:
the V4 score depends heavily on role-prefix stripping. The run is cheap, uses
the already-proven Stage1B248 base, and answers whether the semantic-calibration
recipe can be made clean at the source.

If R0 fails raw/strict cleanliness or loses the V4 semantic-calibration score,
stop and diagnose the prompt/source contract rather than spending on
architecture. If R0 passes, run causal and robustness checks before `V5-A1`:
the next spend should establish whether the role-clean recipe itself transfers
and whether the V4 baseline reproduces under the stricter diagnostics. Only
then launch `V5-A1` on the A1 bottleneck Stage1B400 checkpoint as the first
architecture-modernization candidate.

## Run Shapes

V5-R0:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-bs4-lossscale003-from-stage1b248-20260509-codex
```

V5-R1:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-stage1b248-20260509-codex
```

V5-A1:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.01 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-bs4-lossscale001-from-a1stage1b400-20260509-codex
```

Evaluate with full prediction samples so the V5-specific metrics are present in
the artifact:

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/<run-name>/checkpoint-100 \
  --candidate-label <label> \
  --candidate-architecture anymal_v4 \
  --no-include-baselines \
  --max-samples 1000 \
  --seed 42 \
  --prompt-style training_chat \
  --system-prompt 'Answer with only the final answer. Do not include role labels, explanations, or the word assistant. End after the answer.' \
  --prediction-samples 1000 \
  --output <artifact>.json
```
