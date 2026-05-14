# V14 Focused Qwen3 Substrate Attempt Plan

Date: 2026-05-14

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior checkpoints, and V8-V13 experiment artifacts.

## Purpose

V14 is not a broad sweep. It is a focused attempt to answer one question:

> Can a new visual substrate learn to imitate the V11 Qwen frontier, then exceed it on compositional visual reasoning?

The main failure of V13 was not lack of ambition. It was too much breadth. Several tracks were counted as “completed-negative” from old or shallow artifacts, but the central substrate question was not deeply answered.

V14 fixes that by requiring one primary branch to be trained deeply enough to prove or disprove the key hypothesis.

---

# Current baseline

## V11 Qwen frontier

Use this as teacher and baseline:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Representative metrics from prior runs:

```text
GQA search n1000:       44.9
GQA confirm n3000:      42.6
Clean VQA n3000 mean:   65.922
Blank n3000 seed42:     39.078
Shuffled n3000 seed42:  36.767
Wrong n3000 seed42:     37.178
POPE n1000:             80.100
```

## Key lessons so far

1. V11 is a real local basin. Many V12/V13 local edits returned to or fell below it.
2. V11 was not reproduced by the no-branch C1 diagnostic. The training-time branch mattered, even though the final patch-position contribution was disabled.
3. Ordinary Qwen Stage2 LoRA is not the answer. It repeatedly failed to improve GQA or visual dependence.
4. Existing AnyRes/spatial-grid artifacts do not count as a full V13/V14 substrate test unless they were trained with V11 teacher KL and imitation metrics.
5. The next serious attempt must first prove V11 imitation before trying to beat V11.

---

# Core rule

Do not satisfy this assignment by touching many branches lightly.

V14 should produce a deep result for:

```text
one primary substrate
```

and, only if time/resources allow or the primary hits a hard blocker:

```text
one backup substrate
```

The agent should not run five branches shallowly.

---

# Recommended primary substrate

Default primary:

```text
spatial-preserving compressed grid
```

Reason:

```text
It breaks the 128-token learned-Perceiver bottleneck.
It preserves spatial layout better than the V3 Perceiver.
It is less implementation-heavy than full AnyRes.
It is the cleanest next substrate to test deeply.
```

The agent may choose a different primary only after writing a short decision memo explaining why.

Allowed primary choices:

```text
A. spatial-preserving compressed grid
B. AnyRes / MLP pass-through
C. hybrid V11 + spatial tail
```

Default choice should be A.

Backup should usually be B.

---

# What counts as a valid test

A substrate is not tested unless all of the following are true:

```text
1. It is launched fresh for V14, or the agent proves that an existing run exactly matches V14 requirements.
2. V11 full answer-token KL is active during the imitation phase.
3. Teacher-imitation metrics are logged.
4. Visual-token RMS/scale diagnostics are logged.
5. The branch receives a real imitation phase before GQA hillclimbing.
6. The branch receives a compositional phase after imitation is healthy.
7. It is evaluated on GQA search and confirm, not only n128/n200.
8. The report includes whether the branch imitated V11 before trying to surpass it.
```

Old artifacts can be used for context, debugging, or initialization. They cannot by themselves close the primary branch unless they meet the above criteria.

---

# Phase 0: decision memo and setup

Before training, produce a short memo in the run log:

```text
primary substrate selected:
backup substrate selected:
why this substrate is likely to escape V11:
what architecture changes relative to V11:
expected token count:
placeholder contract changes needed:
teacher KL implementation status:
data sources used:
leakage risks:
minimum training exposure planned:
```

Then complete setup:

```text
1. Reconfirm V11 teacher checkpoint loads and evaluates.
2. Build or identify GQA search and confirm slices.
3. Confirm bootstrap/Wilson confidence intervals are available.
4. Prepare V11 teacher-logit or teacher-probability cache.
5. Confirm leakage audit sources for GQA train/eval image IDs and question IDs.
6. Run a small Qwen/V11 replay batch through teacher KL code and verify finite KL.
```

---

# Teacher loss

Use one teacher loss:

```text
full answer-token KL to V11
```

Do not use visual-token MSE as the primary teacher loss because the new substrate may have different token topology or token count.

Do not use gold-answer-only imitation because it is too weak.

## Teacher KL definition

For replay examples:

```text
image, question, answer
```

Compute V11 distribution over supervised answer-token positions.

Student loss:

```text
KL(student answer-token distribution, V11 teacher answer-token distribution)
```

Prompt and image positions remain masked.

If full-vocab KL is too expensive, use:

```text
top-k teacher distribution plus remainder bucket
```

but keep the loss distributional.

---

# Primary substrate: spatial-preserving compressed grid

## Architecture goal

Replace the learned 128-latent Perceiver bottleneck with a spatial grid representation.

Input:

```text
SigLIP2 384 patch grid
```

Output:

```text
spatially ordered visual tokens projected into Qwen hidden space
```

Start with:

```text
256 visual tokens
```

Optional second stage:

```text
512 visual tokens
```

Only move to 512 if 256 either imitates V11 or shows a clear partial signal.

## Architecture sketch

```text
image
-> SigLIP2 patch grid
-> spatial pooling or strided grid selection
-> 2-layer MLP projector to Qwen hidden size
-> RMS normalization / scale calibration
-> contiguous visual placeholder splice into Qwen
```

Required metadata:

```text
architecture name
llm_backbone = Qwen/Qwen3-8B
vision image size
source patch grid shape
output grid shape
image_tokens
pooling mode
position encoding type
connector_output_scale
visual RMS statistics
teacher checkpoint
teacher KL weight
```

## Placeholder contract

The old system often assumes 128 image placeholders. Generalize it for this branch.

Audit/update as needed:

```text
models/anymal_v3.py
evaluation/vqa_eval.py
evaluation/checkpoint_eval/gqa_checkpoint_eval.py
evaluation/checkpoint_eval/vqa_checkpoint_eval.py
evaluation/checkpoint_eval/pope_checkpoint_eval.py
data collators
chat template utilities
checkpoint metadata guards
scripts that assume image_tokens == 128
```

Stage 0 contract tests must pass:

```text
placeholder count equals image_tokens
placeholders are contiguous where required
inputs_embeds path works
left-padded generation works
batched generation works
strict/raw diagnostics work
checkpoint metadata records image token count
```

---

# Training phases for the primary substrate

## Phase 1: imitation only

Goal:

```text
make the new substrate reproduce V11 behavior
```

Trainable:

```text
new visual substrate / projector
scale/gate/norm parameters
```

Frozen:

```text
Qwen base
SigLIP vision tower
```

Loss:

```text
V11 full answer-token KL
clean CE on replay examples
```

Data:

```text
VQA replay
GQA replay
POPE replay
existing v3_grounding replay
```

Do not add heavy new compositional pressure until imitation is healthy.

### Imitation metrics

Log at every checkpoint:

```text
teacher KL
clean CE
student-V11 exact-answer agreement on replay
student-V11 top-answer overlap
student-V11 GQA pairwise deltas on a small replay slice
VQAv2 clean n200 or n1000
GQA search subset
visual-token RMS
connector scale/gate values
average generated tokens
EOS / max-token-hit / prefix
```

A branch is not ready for GQA improvement training until it shows meaningful V11 imitation.

Indicative imitation targets:

```text
GQA search within ~1.5 points of V11
VQA clean not collapsed
generation hygiene clean
student-V11 answer agreement increasing over checkpoints
teacher KL decreasing
```

Do not overinterpret exact thresholds. The trend matters.

## Phase 2: compositional grounding

Only after imitation is healthy.

Add data:

```text
GQA relation/spatial/left-right examples
Visual Genome relation/attribute if available
RefCOCO-style grounding if available
POPE replay
VQA replay
hard negatives
```

Loss:

```text
clean CE
full answer-token KL to V11 on replay
contrastive image-dependence loss
```

Contrastive examples:

```text
correct image + question + answer
negative image + same question + same answer
```

Negatives:

```text
blank
shuffled
wrong same-answer-type
same-object different-relation
left/right swapped
attribute swapped
```

Start conservative:

```text
lambda = 0.05 or 0.10
margin = 0.25 or 0.50
```

## Phase 3: scale and control recovery

For promising checkpoints, sweep:

```text
connector_output_scale
visual branch gate
RMS normalization mode
```

If GQA improves but controls fail, try recovery before rejecting:

```text
increase replay KL
add VQA/POPE replay
add anti-shuffle contrastive
attenuate branch gate
interpolate toward the best imitation checkpoint
```

---

# Minimum training exposure

A primary substrate may not be declared negative from a short smoke.

Minimum before declaring completed-negative:

```text
1. Stage 0 contract tests pass.
2. Imitation phase runs long enough to show trend, not just one checkpoint.
3. At least three imitation checkpoints are evaluated.
4. If imitation improves but has not reached V11, continue or adjust LR/scale once.
5. Compositional phase runs only after imitation is healthy.
6. GQA search and confirm are both run for the best imitation and best compositional checkpoints.
```

If the branch never improves teacher imitation despite finite gradients and sane RMS, then it is a real negative.

---

# Backup substrate: AnyRes / MLP pass-through

Only run after the primary is blocked or after the primary reaches a clear plateau.

## Goal

Break the 128-token Perceiver substrate more radically by preserving more image structure.

## Token budget

Start small:

```text
global 384 image -> 128 visual tokens
2 local crops at 384 -> 64 visual tokens each
total = 256 visual tokens
```

If small version imitates V11:

```text
global -> 256 tokens
4 local crops -> 128 tokens each
total = 768 visual tokens
```

Do not jump directly to thousands of visual tokens.

## Pooling

Use:

```text
fixed spatial grid pooling
or strided patch selection
```

Record:

```text
crop count
crop order
crop coordinates
tokens per crop
grid shape per crop
position embeddings
connector scale
RMS stats
```

## Training

Same three phases:

```text
imitation
compositional grounding
control recovery
```

Do not declare AnyRes dead from existing partial artifacts unless they meet V14 validity requirements.

---

# Optional diagnostic: C1 auxiliary branch mechanism

The no-branch C1 result showed that the branch mattered during training.

If time allows, run a focused auxiliary-branch diagnostic:

```text
start from the original C1 anchor
add an auxiliary spatial branch during training
train common projector plus auxiliary branch
disable auxiliary branch at evaluation
compare common projector movement and GQA
```

Questions:

```text
Can a training-only auxiliary branch reproduce or exceed V11?
Does the auxiliary branch act as an optimization scaffold?
Can a different auxiliary branch produce a better common-projector basin?
```

This is secondary to the primary substrate attempt.

---

# Evaluation protocol

## Cheap screens

During training:

```text
GQA search subset
VQA clean/control small slice
student-V11 agreement
teacher KL
generation hygiene
```

## Serious screens

For best imitation and best compositional checkpoints:

```text
GQA search n1000
GQA confirm n3000
VQA clean n1000 or n3000
blank n1000 or n3000
shuffled n1000 or n3000
wrong n1000 or n3000
POPE n1000
pairwise taxonomy vs V11
```

## Final screens

Only for candidates that show real signal:

```text
GQA final/largest available
VQA clean n3000 seeds 42/43/44
blank n3000
shuffled n3000
wrong n3000
POPE adversarial and popular if available
mild blur
center crop 90
translate 5 pct
ChartQA or TextVQA if available
MMStar if available
leakage audit
bootstrap confidence intervals
full prediction samples
```

---

# Expanded eval and data, but keep focus

Do not let expanded benchmark work consume the whole run.

Add at least one expanded benchmark:

Preferred order:

```text
TextVQA
ChartQA
MMStar
MM-Vet if scoring is already reliable
```

ChartQA n200 showed the current system is weak, but a new substrate may improve fine-grained visual/text structure. Use expanded benchmarks to detect capability movement, not as the only selection metric.

---

# What not to do

Do not:

```text
run five branches shallowly
count pre-existing partial artifacts as full V14 evidence
run old semantic Stage2 LoRA as the main path
judge a new substrate from n128/n200 alone
declare failure before checking teacher imitation
chase one GQA n1000 bump without confirm split
train broad decoder LoRA before the new substrate can imitate V11
ignore RMS/scale diagnostics
```

---

# Decision logic

## If primary substrate cannot imitate V11

First try:

```text
scale/RMS adjustment
lower LR
higher teacher KL
simpler pooling
smaller token count
better replay mix
```

If it still cannot imitate V11 after one serious adjustment, document it as an interface-imitation failure.

## If primary substrate imitates V11 but cannot exceed it

Then change:

```text
compositional data
contrastive objective
relation/left-right data
branch scale/gate
```

Do not immediately switch architecture.

## If primary substrate beats V11 but controls fail

Run recovery:

```text
teacher KL increase
VQA/POPE replay increase
control contrastive data
scale/gate attenuation
checkpoint interpolation
```

## If primary substrate beats V11 and controls recover

Promote to full final evaluation.

## If primary fails and backup is available

Run backup under the same imitation-first discipline.

---

# Final report requirements

The final report must answer these questions explicitly:

```text
1. Which primary substrate was selected and why?
2. Did it pass contract tests?
3. Did teacher KL decrease?
4. Did student-V11 agreement improve?
5. Did it imitate V11 before GQA training?
6. What happened after compositional training?
7. Did controls fail, and if so, was recovery attempted?
8. Did the branch beat V11 on search and confirm GQA?
9. What did pairwise taxonomy show?
10. Was the backup substrate run? If not, why not?
11. What is the next most rational hill?
```

Do not simply label branches “negative” without explaining whether the failure was:

```text
implementation
imitation
compositional improvement
control recovery
evaluation variance
```

---

# Success definition

A successful V14 does not necessarily need a promoted model.

A successful V14 must produce one of:

```text
1. A new substrate that imitates V11 and beats it on confirm GQA.
2. A new substrate that imitates V11 but does not beat it, proving the bottleneck is not just the visual interface.
3. A clear demonstration that a specific substrate cannot imitate V11 despite proper teacher KL and real exposure.
4. A high-GQA but control-leaky branch with a documented recovery path.
```

The key is depth and clarity, not breadth.
