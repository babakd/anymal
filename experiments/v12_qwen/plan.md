# V12 Qwen3 Aggressive Ceiling Search Plan

Date: 2026-05-13

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior experiment artifacts, and prior V8/V9/V10/V11 notes.

## One-line objective

This is **not** a conservative promotion run. This is an aggressive Qwen3 ceiling search.

The goal is to discover how far Qwen/Qwen3-8B AnyMAL can be pushed above the LLaMA/V3 robust baseline, especially on compositional visual reasoning. Do not stop simply because one candidate is promotable. Run the high-ceiling branches, learn which hill is real, then recover controls for the best hill.

---

# Current state

## Baselines

### LLaMA/V3 robust reference

```text
/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
```

Important same-slice reference:

```text
GQA trusted n1000: 43.7
```

This is the old LLaMA/V3 baseline to beat clearly.

### V9 Qwen scale-calibrated checkpoint

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Important metrics:

```text
Clean VQA n1000:   66.167
Blank n1000:       39.400
Shuffled n1000:    37.367
Wrong n1000:       37.967
POPE n1000:        79.100
GQA trusted n1000: 43.100
```

This is the strong Qwen baseline. It is good on clean VQA, POPE, and controls, but weak on GQA relative to the LLaMA/V3 same-slice reference.

### V11 Qwen frontier

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Important metrics:

```text
GQA trusted n1000:     44.900
Clean VQA n3000 mean:  65.922
Blank n3000 seed42:    39.078
Shuffled n3000 seed42: 36.767
Wrong n3000 seed42:    37.178
POPE n1000:            80.100
```

This is the active Qwen frontier.

Key interpretation from V11:

```text
C1-style training moved common projector weights into a better GQA region.
The explicit learned patch-position table was not the winning component.
The best state disabled patch-position contribution:
  patch_position_feature_scale = 0.0
and used:
  connector_output_scale = 1.125
```

The V11 frontier is a strong local result, but V11 stopped after local C1 salvage. This V12 plan must push beyond that.

---

# Exploration philosophy

## Separate exploration from promotion

Do not apply promotion gates during early exploration.

During exploration, a branch is allowed to temporarily worsen:

```text
blank-image control
shuffled-image control
wrong-image control
POPE
clean VQA
```

if it teaches us how to move GQA, spatial reasoning, left/right reasoning, compositional grounding, or visual dependence.

The question during exploration is:

```text
Can this mechanism move the ceiling?
```

not:

```text
Is this checkpoint immediately promotable?
```

Promotion gates apply only to final candidates.

## Do not stop after one local candidate

V11 found a good candidate and stopped. V12 must not repeat that.

Before declaring the campaign complete, the agent must run at least these high-ceiling mechanisms unless there is a hard technical blocker:

```text
1. Repaired gated visual cross-attention, with verified gradients.
2. Larger image-token budget, at least 192 tokens and preferably 256.
3. Controlled continuation from the V11 frontier with new objectives and trainable subsets.
4. At least one higher-risk visual-capacity branch:
   - higher resolution,
   - multi-level SigLIP features,
   - or vision-side adapter.
```

If any branch shows a clear GQA or taxonomy movement, hillclimb that branch for multiple iterations before stopping.

---

# Primary metrics

## Exploration metrics

Primary:

```text
GQA trusted n1000
GQA pairwise deltas versus V11, V9, and V3
GQA taxonomy buckets:
  spatial
  left/right
  object identity
  attribute
  counting
  comparison
  yes/no presence
  other
```

Secondary:

```text
Clean VQA n1000 and n3000
Blank image
Shuffled image
Wrong-image same-answer-type
POPE
Perturbation mean
EOS
Max-token-hit
Assistant-prefix
Strict-clean gap
```

## Exploration milestones

These are not hard stop gates. They are milestones.

```text
Milestone A: GQA trusted n1000 >= 45.5
Milestone B: GQA trusted n1000 >= 46.5
Milestone C: GQA trusted n1000 >= 48.0
```

If a branch reaches any milestone, run a recovery pass for controls rather than rejecting it.

## Promotion criteria

Only final materialized candidates need to pass promotion-style checks.

A final candidate should be evaluated with no eval-time overrides and should include:

```text
GQA trusted n1000, and larger GQA if available
Clean VQA n3000 seeds 42/43/44
Blank n3000 seed42
Shuffled n3000 seed42
Wrong-image n3000 seed42
POPE n1000
Mild blur n1000
Center crop 90 n1000
Translate 5 pct n1000
Leakage audit
Full metadata
Strict/raw diagnostics
Answer histograms
Pairwise GQA analysis
```

For final promotion, preserve or recover the V9/V11 control behavior. But do not use these final gates to stop exploratory runs early.

---

# Hard stop conditions

Stop a branch only for hard technical or scientific failure:

```text
NaNs or infs
persistent W&B active alerts
placeholder contract failure
EOS < 0.95 in exploration screens
max-token-hit > 0.05 in exploration screens
assistant-prefix > 0.02
zero gradients where training should occur
no parameter deltas after optimizer step
no logit change after optimizer step for adapter branches
three consecutive checkpoints with no movement on GQA or taxonomy and no useful control insight
```

For branches with control regression but GQA improvement, do not stop immediately. Run recovery:

```text
scale sweep
interpolation back toward V11 frontier
KL regularization
control-aware contrastive continuation
branch gate attenuation
```

---

# Required setup work

## 0A. Rehydrate the frontier leaderboard

Create or update a V12 leaderboard table with at least:

```text
V3 robust
V9 Qwen scale1.05
V10 Batch-A best
V10 C1 best
V11 frontier
```

Include:

```text
checkpoint path
architecture metadata
connector_output_scale
patch_position_feature_scale
image-token count
GQA n1000
clean VQA
blank
shuffled
wrong
POPE
perturb mean
generation hygiene
leakage status
```

## 0B. Pairwise GQA taxonomy refresh

Run pairwise/taxonomy analysis for:

```text
V11 frontier vs V9
V11 frontier vs V3
V11 frontier vs V10 C1
V11 frontier vs any new high-GQA branch
```

Use existing script:

```text
scripts/analyze_gqa_pairwise.py
```

If metadata is missing, use heuristic question-text labels as in V11, but clearly label them as heuristic.

---

# Track A: Repaired gated visual cross-attention

V10 E1 was not a real negative. It reported:

```text
train/grad_norm: 0.0
same GQA at checkpoint 25 and 50
same answer histogram
```

So V12 must first fix and prove the adapter training path.

## A0. Gradient-proof diagnostic

Before a real run, perform a one-batch proof from the V11 frontier.

Base:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Adapter:

```text
frozen Qwen base
frozen or lightly trainable connector
train gated visual cross-attention adapters
selected decoder layers initially: 12, 18, 24, 30
adapter bottleneck: 512
```

Run diagnostics:

```text
1. forward pass on fixed batch
2. backward pass
3. verify nonzero gradients on cross-attention parameters
4. verify nonzero gradient on gate parameter
5. optimizer step
6. verify nonzero parameter deltas
7. verify logits change on fixed samples
8. verify generated outputs can change on a small fixed probe
```

Do not launch a full E1 run until this passes.

## A1. Nonzero-gate visual cross-attention grid

Zero gate may have prevented useful learning. Run nonzero gates:

```text
gate_init = 1e-4
gate_init = 1e-3
gate_init = 1e-2
```

Try at least two layer sets:

```text
mid_upper: 12,18,24,30
upper_dense: 18,22,26,30,34
```

Try at least two bottleneck sizes if the first grid is stable:

```text
512
1024
```

Trainable parameters:

```text
cross-attention adapters
gate parameters
optionally adapter layer norms
```

Keep Qwen base frozen.

Initial training objective:

```text
GQA compositional direct-answer CE
VQA replay CE
POPE presence/absence replay
anti-shuffle / wrong-image contrastive loss
KL to V11 frontier outputs on VQA replay
```

Run schedule:

```text
checkpoints: 10, 25, 50, 100, 200, 400
screen GQA at every checkpoint
screen controls at promising checkpoints
```

Continue a branch if:

```text
GQA improves by >= 0.3 over V11 frontier
or taxonomy shows improvement on left/right/spatial/object identity
or controls improve while GQA stays near V11
```

Do not stop only because controls regress if GQA moves upward. First attempt control recovery.

## A2. If cross-attention works, hillclimb it

If any cross-attention candidate reaches:

```text
GQA >= 45.5
```

then run:

```text
scale sweep around connector_output_scale
adapter gate sweep
KL weight sweep
contrastive weight sweep
longer continuation to 800 or 1200 steps if GQA is still trending
```

Materialize any promising checkpoint and run the full promotion bundle only after the hillclimb.

---

# Track B: Larger image-token budget

The old V2 lesson was that more tokens through a weak connector did not help. That does not settle the Qwen3 setting. Qwen is now scale-calibrated, GQA is the bottleneck, and V11 suggests connector geometry matters.

## B1. 192-token V3/Qwen Perceiver branch

Use the V11 frontier as the best available initialization.

Target:

```text
image_tokens = 192
same SigLIP2 384 vision tower
same Qwen3 decoder
V3-style Perceiver topology
connector_output_scale initialized near 1.125
patch_position_feature_scale = 0.0 unless explicitly testing spatial branch
```

Initialization strategy:

```text
copy common projector / Perceiver weights from V11 frontier
copy first 128 latent queries from V11
initialize extra 64 latent queries by:
  small-noise copies of existing latents
  or interpolation/mean of existing latents
record the method in metadata
```

Training:

```text
connector-only first
Qwen frozen
vision frozen
```

Objective:

```text
GQA compositional CE
original v3_grounding replay
VQA direct-answer replay
anti-shuffle / wrong-image contrastive loss
POPE replay
optional KL to V11 frontier outputs on VQA replay
```

Schedule:

```text
short canary: 10 steps
if healthy: checkpoints 25, 50, 100, 200, 400, 800, 1500
```

Screen:

```text
GQA n1000 first
then clean/blank/shuffled/wrong n1000 for any checkpoint with GQA movement
then full controls for top candidates
```

Do not kill the branch for early control regressions if GQA rises meaningfully.

## B2. 256-token branch

Run if B1 is stable or if compute allows parallelism.

Same setup as B1, but:

```text
image_tokens = 256
extra 128 latent queries initialized from V11 latent distribution
```

Use a lower LR if early probes show shortcut pressure.

## B3. Token-budget plus scale sweep

For promising 192/256 checkpoints, sweep:

```text
connector_output_scale:
  1.00
  1.05
  1.10
  1.125
  1.15
  1.175
```

Do this before rejecting the branch. V9 and V11 both showed scale can determine viability.

---

# Track C: Controlled continuation from the V11 frontier

V11 tried one all-projector 10-step continuation and it regressed GQA. That was too blunt. V12 should test narrower trainable subsets and better objectives.

Base:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

## C1. Trainable subset grid

Run short continuations with these trainable subsets:

```text
1. latent queries only
2. output projection only
3. last 2 Perceiver layers only
4. cross-attention blocks inside the Perceiver only
5. scale/gate parameters only
6. last 2 Perceiver layers + output projection
7. full connector, but only after the narrower subsets are screened
```

Each subset should run:

```text
steps: 10, 25, 50, 100, 200
LRs: 1e-5 and 5e-6
loss scale: 0.3 unless diagnostics suggest otherwise
```

Objective:

```text
GQA CE
VQA replay CE
anti-shuffle/wrong-image contrastive loss
KL to V11 outputs on VQA replay
```

Continue a subset if it produces:

```text
GQA >= 45.0
or a useful taxonomy-specific gain
or strong control improvement without GQA loss
```

## C2. Contrastive image-dependence objective

For GQA and VQA examples, build negatives:

```text
correct image + question + answer
shuffled image + same question + same answer
wrong same-answer-type image + same question + same answer
blank image + same question + same answer
```

Loss:

```text
CE(correct image)
+ lambda * max(0, margin - logp(answer | correct image) + logp(answer | negative image))
+ KL_to_frontier_on_replay
```

Start with:

```text
lambda = 0.05, 0.10, 0.20
margin = 0.25, 0.5
KL weight = 0.05, 0.10
```

The goal is to improve visual dependence without forcing "cannot determine" answer labels.

## C3. Control recovery protocol for high-GQA checkpoints

If a checkpoint reaches high GQA but fails controls:

```text
1. connector scale sweep
2. interpolate toward V11 frontier:
   alpha = 0.05, 0.10, 0.20, 0.30
3. run short control-aware continuation:
   anti-shuffle contrastive
   VQA replay
   KL to high-GQA checkpoint and/or V11 frontier
4. attenuate any new branch gate
```

Do not discard high-GQA checkpoints before attempting recovery.

---

# Track D: Spatial/compositional connector expansion

C1 proved that a C1-style training path can move common projector weights into a better GQA region. V11 showed that the explicit learned 2D table was not the final component. But spatial/compositional evidence is still the best clue.

## D1. New constrained spatial branch

Start from V11 frontier.

Add a new spatial branch, but constrain it:

```text
original V11 connector path frozen for phase 1
new spatial branch trainable
spatial branch gate initialized very small
patch_position_feature_scale initially 0.0
```

Spatial branch options:

```text
learned 2D table
low-rank 2D table
separable row/column embeddings
small coordinate MLP only if learned table variants fail
```

Train only:

```text
spatial branch
spatial branch gate
optionally final scale
```

Objective:

```text
GQA CE
anti-shuffle/wrong contrastive
VQA replay
POPE replay
KL to V11 frontier
```

Run:

```text
steps: 25, 50, 100, 200, 400
```

If GQA moves, unfreeze last Perceiver layer only and continue.

## D2. Spatial branch gate sweep

For any spatial branch checkpoint, sweep:

```text
spatial branch gate / scale:
  0.00
  0.05
  0.10
  0.20
  0.35
  0.50
  0.75
  1.00
```

and connector output scale:

```text
1.05
1.10
1.125
1.15
1.175
```

Keep this separate from training. Materialize any strong candidate.

## D3. Spatial-taxonomy objective

If taxonomy shows left/right and spatial remain weaknesses, build a focused training mixture:

```text
40% GQA spatial / left-right / relation examples
30% VQA and v3_grounding replay
15% anti-shuffle and wrong-image contrastive examples
10% POPE replay
5% attribute/count replay
```

Do not oversample yes/no presence unless the taxonomy proves it is the target. C1 improved yes/no presence but controls can regress through yes/no-prior movement.

---

# Track E: Higher visual capacity

Run this after the first cross-attention and token-budget canaries, or in parallel if resources allow.

## E1. Higher resolution with fixed final token count

Try:

```text
SigLIP2 input resolution:
  384 baseline
  448
  512
```

Keep final image-token count fixed at first:

```text
128 or 192, depending on B1 results
```

Do not change decoder or Qwen prompt contract except placeholder count if token budget changes.

Goal:

```text
improve GQA spatial/object detail without increasing answer-prior leakage
```

Screen:

```text
GQA n1000
clean n1000
blank/shuffled/wrong n1000
POPE
```

## E2. Multi-level SigLIP features

DeepStack-lite failed in a previous LLaMA/V4 regime, but Qwen3 ceiling is a different regime.

Try a controlled version:

```text
use SigLIP hidden levels [-3, -2, -1]
fuse into the same final token budget
do not dump extra raw tokens into Qwen
start with 128 or 192 final image tokens
use Qwen scale calibration
```

Screen strictly for GQA and controls. Do not promote if it only improves clean VQA.

## E3. Vision-side adapter

This is higher risk but fair game for ceiling search.

Try conservative vision adaptation:

```text
SigLIP last-block LoRA or small adapter
very low LR
connector trainable or frozen depending on stability
Qwen frozen
```

Objective:

```text
GQA CE
VQA replay
POPE replay
anti-shuffle contrastive
```

Do not full-finetune the entire vision tower in the first pass. Start with last-block adapter only.

---

# Track F: Qwen visual-grounding calibration stage

Do not run old LLaMA-style semantic Stage2. It repeatedly hurt or failed the relevant geometry.

Instead, if decoder-side adaptation is needed, use a Qwen-specific calibration stage:

```text
small trainable component:
  visual cross-attention adapters
  or visual-intake attention LoRA
  or selected q/v LoRA layers

not full MLP LoRA
not broad semantic calibration
```

Loss:

```text
CE on GQA and VQA clean
contrastive image-dependence loss
KL to V11 frontier outputs on replay
POPE replay
```

Candidate LoRA variants:

```text
rank 4 q_proj/v_proj only
rank 8 q_proj/v_proj only
selected layers only:
  mid-upper layers
  upper layers
no MLP LoRA
```

A decoder-side branch must pass the gradient/delta/logit-change proof before real training.

---

# Evaluation protocol

## Cheap screen

For every checkpoint:

```text
GQA trusted n1000
EOS
max-token-hit
assistant-prefix
top answer histogram
```

If GQA improves or taxonomy changes, run:

```text
clean VQA n1000
blank n1000
shuffled n1000
wrong n1000
POPE n1000
```

## Intermediate screen

For top branch candidates:

```text
GQA trusted n1000
clean n3000 seed42
blank n3000 seed42
shuffled n3000 seed42
wrong n3000 seed42
POPE n1000
mild blur n1000
center crop 90 n1000
translate 5 pct n1000
pairwise taxonomy
```

## Final screen

For final candidates:

```text
GQA trusted n1000 and larger GQA if available
clean n3000 seeds 42/43/44
blank n3000 seed42
shuffled n3000 seed42
wrong n3000 seed42
POPE n1000
perturbation suite
leakage audit
strict/raw diagnostics
metadata verification
materialized no-override checkpoint confirmation
```

---

# Required reporting

The final report must not simply say "candidate promoted" or "no candidate promoted."

It must include:

```text
1. Full leaderboard of all branches.
2. Best GQA regardless of controls.
3. Best control-safe candidate.
4. Best POPE candidate.
5. Best clean VQA candidate.
6. Pairwise taxonomy for top candidates.
7. Which mechanisms moved GQA and which did not.
8. Which mechanisms caused control leakage.
9. Whether control leakage was recoverable.
10. Recommended hill to climb next.
```

This campaign is allowed to end with:

```text
high-GQA but control-leaky branch
```

as long as it provides a clear next recovery path. Do not discard such a branch as "failed" without analysis.

---

# Minimum work before stopping

Do not stop the campaign until all of the following are true:

```text
1. Repaired cross-attention adapter received a real gradient-verified test.
2. 192-token branch received a real training and GQA/control screen.
3. 256-token branch was either tested or explicitly blocked by implementation constraints.
4. At least three trainable subsets from Track C were tested.
5. At least one constrained spatial branch from Track D was tested.
6. At least one high-visual-capacity branch from Track E was tested, unless blocked.
7. Any branch with GQA >= 45.5 received at least one control-recovery attempt.
8. Any branch with GQA >= 46.5 received a deeper hillclimb unless technically unhealthy.
```

This is the key difference from V11: do not stop after finding a local promotable frontier.

---

# Suggested execution order

## Phase 1: immediate high-ceiling tests

```text
A0 gradient-proof E1 cross-attention
A1 nonzero-gate cross-attention grid
B1 192-token branch canary and continuation
C1 trainable-subset continuation from V11 frontier
```

Run these in parallel if resources allow.

## Phase 2: expand the first promising hill

If cross-attention moves GQA:

```text
hillclimb cross-attention gates, layers, KL, contrastive weights, and training length
```

If 192 tokens moves GQA:

```text
try 256 tokens
scale sweep
longer continuation
combine with spatial branch
```

If C1 subset continuation moves GQA:

```text
run deeper continuation
scale/interpolation recovery
try adjacent trainable subsets
```

## Phase 3: higher visual capacity

```text
higher resolution
multi-level SigLIP
vision last-block adapter
```

Only after at least one of these is run should the agent call the V12 ceiling search complete.

---

# Current best starting checkpoint

Use this as the default base unless a branch requires otherwise:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Metadata expectations:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
connector: V3-style Perceiver
image tokens: 128
connector_output_scale: 1.125
patch_position_feature_scale: 0.0
Qwen base frozen unless branch explicitly trains small adapters
vision tower frozen unless Track E3
```

---

# Final instruction

The agent should be aggressive.

A safe local frontier at GQA 44.9 is not enough. The point of this run is to find whether Qwen3 can reach a much higher compositional-grounding ceiling.

Temporary control regressions are acceptable during exploration. What matters is identifying mechanisms that move the ceiling, then recovering controls for the best mechanisms.
