# V11 Qwen3 Ceiling Hillclimb Plan

Date: 2026-05-13

Audience: an execution agent with access to the AnyMAL working directory, Modal training/eval tooling, prior V8/V9/V10 artifacts, and GPU budget. This file is self-contained enough to start work without reading the prior chat, but the agent should inspect the previous ledgers and artifacts when available.

## Bottom line

The goal is no longer merely to pass the old LLaMA/V3 incumbent gates. The goal is to find and climb the most promising path for a **Qwen3-8B AnyMAL model that clearly surpasses the LLaMA-based V3 robust baseline**, especially on compositional visual grounding.

Do **not** stop exploration just because a checkpoint clears the old viability floor. The previous V9 candidate already mostly did that. V11 is a ceiling search.

The current situation is:

```text
V9 Qwen scale1.05 is strong on clean VQA, POPE, perturbations, and basic controls.
V9/Qwen still underperforms or only matches the LLaMA/V3 family on the trusted GQA n1000 slice.
V10 broad search did not promote a better candidate.
The only V10 branch that moved GQA materially was learned 2D patch features, C1, but it damaged blank/shuffled/wrong-image controls.
E1 gated cross-attention was not a conclusive negative because the adapter appeared not to train meaningfully.
```

The main V11 hypothesis:

> Qwen has enough language capacity and direct-answer ability. The ceiling is blocked by how visual evidence is injected and constrained. The highest-value hillclimb is to preserve the V9 scale-calibrated Qwen alignment while adding stronger spatial/compositional visual evidence and better evidence-dependence objectives.

## Current baselines to track

### LLaMA/V3 robust baseline

Use this as the historical baseline to beat, not as a stopping point.

```text
Architecture: anymal_v3
Decoder: LLaMA-3-8B-Instruct
Connector: V3 Perceiver, 128 image tokens
Known aligned V3 robust metrics from prior runs:
clean VQA: around 62.967 on current-cache n1000 seed42
POPE: around 77.100
GQA trusted n1000: around 43.700 on the same slice used in the V10 confirmation
```

### V9 Qwen scale1.05 baseline

This is the current Qwen baseline and the checkpoint to compare against.

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Known confirmation metrics:

```text
Clean VQA n1000:        66.167
Blank n1000:            39.400
Shuffled n1000:         37.367
Wrong-image n1000:      37.967
POPE n1000:             79.100
GQA canonical n500:     44.000
GQA trusted n1000:      43.100
EOS/max-token/prefix:   clean
```

This checkpoint is an inference/eval checkpoint, not an optimizer-resume checkpoint. It contains `model_meta.json` and `projector.pt` with `connector_output_scale: 1.05`. For resume training, use the underlying Stage1B2350 training checkpoint and reproduce or materialize the scale behavior explicitly.

Underlying Stage1B2350 anchor:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

### V10 best clue: C1 learned 2D patch features

C1 was not promotable, but it is the most informative branch because it moved GQA.

```text
C1 learned 2D patch table
Best reported point: checkpoint-150 with scale around 1.125
GQA trusted n1000: 44.5
Failure: blank about 40.0, shuffled about 37.8, wrong about 38.5, perturb mean about 65.6
```

Interpretation: spatial features can move GQA, but the current implementation/training lets control leakage rise. V11 should focus on recovering the C1 GQA gain while preserving V9 control behavior.

## What counts as success in V11

This is an exploration/hillclimb campaign, not a narrow pass/fail promotion run. Keep a running frontier of candidates.

### Primary ceiling objective

Maximize:

```text
GQA trusted n1000 and, if available, larger trusted GQA slices
```

while tracking:

```text
clean VQA
blank/shuffled/wrong-image controls
POPE
perturbations
strict/raw hygiene
answer distributions
```

The desired direction is:

```text
GQA trusted n1000: clearly above V9 43.1 and V3 robust 43.7
Target exploratory milestone: >= 44.5
Stretch target: >= 45.0
Ambitious target: >= 46.0 while keeping VQA/POPE strong
```

Do not stop the campaign just because a candidate reaches 43.8 or 44.0. If a branch is still improving and failures are understandable or repairable, keep climbing.

### Exploration tolerance

During exploration, temporary regressions are allowed. A branch may be worth continuing if it shows any of the following:

```text
GQA improves by >= 0.5 over V9 on trusted n1000
GQA improves on a specific taxonomy bucket that was known weak
controls worsen but by a small, monotonic, or explainable amount
clean VQA and POPE stay strong while GQA rises
attention/gradient diagnostics show a previously untested mechanism is actually learning
```

Stop or deprioritize a branch if:

```text
GQA does not move after multiple real training checkpoints
control metrics collapse with no compensating GQA signal
clean VQA drops hard and does not recover
POPE collapses
generation hygiene regresses
W&B health is bad or the claimed trainable path has zero/near-zero gradients
```

Promotion can be stricter later. V11’s first job is to identify the most promising hill.

## Required evaluation ladder

Use staged evaluation to avoid wasting time but do not overfit to tiny screens.

### Cheap screen

For early checkpoints and grids:

```text
GQA trusted n1000
shuffled n1000
blank n1000
clean n1000
wrong-image n1000 if the first four look promising
POPE n1000 for top candidates
```

### Candidate screen

For any candidate that looks meaningfully better than V9 on GQA:

```text
clean VQA n3000 seed42
blank n3000 seed42
shuffled n3000 seed42
wrong-image n3000 seed42
POPE n1000
GQA trusted n1000
mild blur n1000
center crop 90 n1000
translate 5 pct n1000
strict/raw diagnostics
answer histograms and answer-kind rates
```

### Confirmation screen

For a serious ceiling candidate:

```text
clean VQA n3000 seeds 42/43/44
GQA trusted n1000 repeat if possible, or larger GQA slice if available
full leakage audit
materialized checkpoint test with no eval-time override
pairwise analysis versus V9 and V3 robust
```

## Before new training: run GQA taxonomy

Do this first unless an urgent pre-existing run is already underway.

Compare:

```text
V9 scale1.05
V10 C1 best 2D branch
V3 robust
Batch-A best if distinct from V9
```

Produce pairwise files:

```text
V9 correct / C1 wrong
C1 correct / V9 wrong
V3 robust correct / V9 wrong
V9 correct / V3 robust wrong
C1 correct / V3 robust wrong
V3 robust correct / C1 wrong
```

Break GQA errors into available categories, for example:

```text
spatial relation
left/right
object identity
attribute
color
counting
comparison
logical/compositional
yes/no object presence
```

If the dataset metadata does not expose perfect categories, use question text heuristics and report uncertainty. The purpose is not publication-grade taxonomy; it is to decide what hill to climb.

Decision use:

```text
If C1 mainly improves spatial/relation questions, prioritize constrained 2D spatial features.
If C1 mainly improves attributes or object identity, prioritize patch routing, resolution, or multi-level features.
If C1 mostly shifts yes/no priors, prioritize contrastive/control objectives.
If V9 is weak on counting, prioritize count/object data and larger token budget.
```

## Priority 1: C1 spatial-feature salvage

This is the highest-priority path because C1 is the only branch that clearly moved GQA.

### 1A. No-training C1/V9 interpolation and scale search

Goal: recover some C1 GQA gain without inheriting its control leakage.

Inputs:

```text
V9 scale1.05 projector/checkpoint
C1 best projector/checkpoint from V10 learned 2D patch table branch
```

Create interpolated connectors:

```text
projector_interp = (1 - alpha) * projector_V9 + alpha * projector_C1
alpha values: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75
```

For each interpolation, sweep connector output scale:

```text
1.00, 1.025, 1.05, 1.075, 1.10, 1.125
```

Evaluate first:

```text
GQA trusted n1000
shuffled n1000
blank n1000
clean n1000
```

Promote top grid points to candidate screen.

Why this matters: it is cheap and directly tests whether the GQA gain is recoverable as a small movement in connector space rather than a full retrain.

### 1B. 2D branch strength sweep

If C1 architecture has a separable 2D branch or can be modified to expose one, introduce a `spatial_branch_scale` or `spatial_gate`.

Sweep:

```text
base_connector_output_scale: 1.00, 1.025, 1.05, 1.075, 1.10, 1.125
spatial_branch_scale:       0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00
```

Evaluate as in 1A. Look for an intermediate point where:

```text
GQA rises above V9
blank/shuffled/wrong controls are not materially worse than V9
POPE and clean VQA remain strong
```

Do not require perfect control caps during this sweep. The goal is to map the Pareto frontier.

### 1C. Constrained spatial-branch training

If 1A/1B shows a promising region, train a constrained C1 variant.

Start from the V9/Stage1B2350 connector. Add zero-initialized learned 2D patch features.

Phase 1:

```text
freeze original V3 connector path
train only:
  2D patch table
  spatial branch gate/scale
  optional final tiny scale parameters
Qwen frozen
vision frozen
```

Objective:

```text
GQA compositional CE
original v3_grounding replay CE
anti-shuffle / wrong-image contrastive answer-dependence loss
POPE-style object presence/absence replay
optional KL to V9 outputs on replay examples
```

Suggested mixture for first run:

```text
60% original v3_grounding replay
20% GQA compositional direct-answer
10% anti-shuffle / wrong-image contrastive pairs
10% POPE-style object presence/absence
```

Train short first:

```text
checkpoints at 25, 50, 100, 150, 200
```

Evaluate every checkpoint on GQA trusted n1000 and controls. Continue only while GQA is moving or controls are repairable.

Phase 2, only if Phase 1 works partially:

```text
unfreeze a small subset of connector projection parameters
very low LR
keep spatial gate regularized
continue 100-300 steps
```

### 1D. C1 with anti-control regularization

If C1 improves GQA but controls rise, add an explicit control loss:

```text
logp(answer | correct image, question)
  > logp(answer | shuffled/wrong image, question) + margin
```

Start conservative:

```text
lambda = 0.05, 0.10, 0.20
margin = 0.25, 0.50
```

Avoid universal `cannot determine` labels as the primary mechanism. Prefer contrastive suppression of the correct answer under invalid images.

## Priority 2: repair and retest E1 gated visual cross-attention

V10 E1 should not be considered a real negative because W&B reported zero gradient norm and checkpoint-25/50 had identical GQA/answer histograms.

### 2A. Gradient proof

Before a real E1 run, run a one-batch diagnostic that verifies:

```text
cross-attention adapter gradients are nonzero
visual gate gradients are nonzero
parameter deltas after one optimizer step are nonzero
logits change on a fixed batch after one optimizer step
generated answers or logprobs change on a tiny fixed probe
```

Log per-parameter-group norms:

```text
cross_attn.q_proj
cross_attn.k_proj
cross_attn.v_proj
cross_attn.o_proj
adapter bottleneck
visual gate
any layernorm or scale params
```

If any trainable group has zero gradients unexpectedly, fix before continuing.

### 2B. Nonzero-gate E1 retry

Zero gate may be too conservative. Try:

```text
gate init: 1e-4, 1e-3, 1e-2
layers: 12, 18, 24, 30 initially
adapter bottleneck: 512 initially; 256 if unstable
```

Training setup:

```text
Qwen base frozen
V9 connector frozen at first
train cross-attention adapters and gates
use GQA + replay + contrastive objective, not old semantic Stage2
```

Evaluate checkpoints:

```text
25, 50, 100, 200, 300
```

Continue only if GQA moves or the diagnostic shows real learning with a plausible path. If GQA remains at 43.1-43.4 with no answer/logit movement, stop.

### 2C. E1 with connector lightly unfrozen

If adapter-only learns but plateaus, try:

```text
train cross-attention adapters
unfreeze final connector projection or a small visual scale/gate
low LR for connector params
```

Rationale: repeated visual access may need the connector to expose visual memory in a form adapters can use.

## Priority 3: larger visual token budget

This remains untested for the Qwen ceiling. Old V2 showed that more weak tokens did not help, but the Qwen/V9 connector is much better calibrated. GQA may need more visual memory.

### 3A. 192-token V3 Perceiver

Keep everything else stable:

```text
Qwen3-8B
SigLIP2 So400m 384
V3-style Perceiver
connector output scale calibrated
same Qwen RMS/gate lessons
```

Try:

```text
image tokens: 192
Perceiver layers/heads unchanged if feasible
Stage1A/Stage1B not necessarily full first; use staged screens
```

Run a shortened but meaningful alignment first, then extend if promising:

```text
Stage1A: enough to reach healthy RMS and nontrivial fixed probes
Stage1B: evaluate checkpoints densely around the point where clean/GQA emerge
```

Do not promote from short runs, but use them to decide whether 192 tokens has a GQA signal.

### 3B. 256-token V3 Perceiver

Run only if 192 shows signal or if GPUs allow parallel broad exploration.

Risk: more tokens may increase answer-prior leakage or compute without improving GQA. Track controls closely.

### 3C. Token budget plus 2D features

If 192 tokens improves GQA or C1 salvage improves GQA, combine them:

```text
192 tokens + learned 2D patch table + scale/gate regularization
```

This is more confounded, so run after individual signals exist.

## Priority 4: Qwen visual-grounding calibration stage

Do not use old LLaMA-style robust semantic Stage2 as-is. Qwen does not need ordinary answer calibration. It needs visual-dependence calibration.

### 4A. Trainable components

Prefer one of:

```text
spatial branch only
cross-attention adapters only
small visual-intake LoRA with KL
connector final projection/scale only
```

Avoid full decoder LoRA unless a clear diagnostic shows it is needed.

### 4B. Loss

Use:

```text
CE on clean visual QA examples
contrastive answer-dependence loss for wrong/shuffled images
KL to V9 outputs on replay examples
optional POPE binary CE
```

Suggested objective:

```text
loss = CE_clean
     + lambda_contrastive * contrastive_margin_loss
     + lambda_kl * KL(candidate || V9) on replay
```

Start:

```text
lambda_contrastive = 0.05 or 0.10
margin = 0.25 or 0.50
lambda_kl = 0.05
```

Increase only if controls stay too high.

### 4C. Data mixture

Use targeted data, not broad generic VQA oversampling:

```text
50-60% original v3_grounding replay
20-25% GQA compositional direct-answer
10-15% wrong/shuffled contrastive pairs
5-10% POPE-style object presence/absence
optional 5% hard counting/spatial relation prompts if available
```

Report exact source files and image-ID leakage audit for any new source.

## Priority 5: higher-risk visual improvements

Run these after C1 salvage, E1 repair, and token-budget experiments unless the GQA taxonomy strongly points to them.

### 5A. Higher image resolution

Try 448 or 512 if the SigLIP path and preprocessing support it.

First keep final visual token count fixed:

```text
resolution up, final tokens unchanged
```

Then combine with 192 tokens if resolution alone helps.

Why: GQA may need smaller objects, attributes, and spatial relations.

### 5B. Multi-level SigLIP features

Old DeepStack-style features failed in V4/LLaMA, but Qwen ceiling is a different regime. Try only if taxonomy suggests attribute/local-detail failures.

Keep the final token contract compact:

```text
multi-level input features -> 128 or 192 final tokens
```

Do not simply dump more tokens into Qwen.

### 5C. Vision-side adapters

Only after connector/cross-attention/token-budget paths are exhausted.

Conservative version:

```text
last-block SigLIP LoRA or tiny vision adapter
very low LR
GQA + controls objective
frozen Qwen base
```

This is high-risk and should not be the first V11 run.

## Things to deprioritize

Do not spend much time on:

```text
generic longer Stage1A caption alignment
generic longer Stage1B with the old objective
ordinary long robust semantic Stage2
full decoder LoRA calibration
failed additive latent-shift question conditioning
coordinate MLP 2D features in the same form as V10 C2
standalone query-conditioned scalar/per-token scale if unchanged from V10
post-processing or answer cleanup tricks
```

These have either already failed or target the wrong bottleneck.

## Suggested execution order

### Day 0 / setup

1. Load and verify V9 baseline and C1 best artifacts.
2. Reproduce GQA trusted n1000 for V9 and C1 if needed.
3. Run GQA taxonomy and pairwise analysis.
4. Confirm all eval scripts can handle materialized scale and any C1 metadata.

### Batch 1: C1 no-training salvage

1. V9-C1 projector interpolation grid.
2. C1 spatial branch scale grid if branch is separable.
3. Base connector output scale sweep for top interpolation/branch-scale points.
4. Promote top 3 to candidate screen.

Expected value: high. Cheap and directly targets the only proven GQA-moving branch.

### Batch 2: constrained C1 training

1. Freeze V9 path, train only 2D branch/gate.
2. Use GQA + replay + contrastive objective.
3. Evaluate checkpoints at 25/50/100/150/200.
4. If promising, lightly unfreeze final connector projection and continue.

Expected value: high. This is the main hill.

### Batch 3: E1 repair and real retest

1. Run gradient-proof diagnostic.
2. Try nonzero gate inits.
3. Run adapter-only E1 continuation with GQA/control objective.
4. If real learning appears, continue while promising.

Expected value: medium-high but only after training path is proven.

### Batch 4: token-budget expansion

1. 192-token Qwen V3 Perceiver branch.
2. Dense checkpoint screening around first strong GQA emergence.
3. Combine with scale calibration.
4. Try 256 only if 192 has signal.

Expected value: medium-high, more expensive.

### Batch 5: higher-risk visual paths

1. Higher resolution.
2. Multi-level SigLIP inputs.
3. Vision-side adapter.

Expected value: potentially high, but more confounded and should follow the cheaper paths.

## Reporting requirements

Every run should update a V11 ledger with:

```text
run name
checkpoint path
base checkpoint
trainable parameters
frozen parameters
connector metadata
scale/gate metadata
objective and data mixture
W&B run id
health status
GQA trusted n1000
clean/blank/shuffled/wrong screens
POPE if run
perturbations if run
answer histogram
reason for continuing or stopping
```

For any candidate that appears to beat V9 meaningfully, include:

```text
materialized checkpoint path
no-override smoke eval
full candidate screen
leakage audit
pairwise analysis against V9 and V3 robust
```

## Final note

The important strategic change is this:

```text
V9 was about making Qwen viable.
V11 is about finding the Qwen ceiling.
```

The agent should therefore not stop at the first checkpoint that crosses an old floor. Treat old gates as diagnostics, not as the objective. The current best evidence says the most promising hill is:

```text
V9 scale-calibrated Qwen connector
+ controlled spatial/2D visual evidence
+ GQA/compositional objective
+ anti-shuffle/wrong-image contrastive constraints
```

Climb that hill first, then repair cross-attention and explore token budget if the C1 hill plateaus.
