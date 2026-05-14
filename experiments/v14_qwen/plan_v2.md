# V14 Focused Qwen3 Substrate Attempt Plan, V2

Date: 2026-05-14
Purpose: focused post-V13 plan for escaping the V11 Qwen basin without repeating shallow broad sweeps.

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior checkpoints, and V8-V13 experiment artifacts.

---

# One-line objective

V14 is a focused substrate attempt, not a broad sweep.

The core question is:

> Can a new visual substrate first imitate the V11 Qwen frontier, then exceed it on compositional/fine-grained multimodal reasoning?

The agent should go deep on one primary substrate. Do not satisfy this assignment by sampling many branches lightly.

---

# Current baseline

## V11 Qwen frontier

Use this as teacher and baseline:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Representative prior metrics:

```text
GQA search n1000:       44.9
GQA confirm n3000:      42.6
Clean VQA n3000 mean:   65.922
Blank n3000 seed42:     39.078
Shuffled n3000 seed42:  36.767
Wrong n3000 seed42:     37.178
POPE n1000:             80.100
ChartQA val n200:       6.0
```

## Key lessons from V8-V13

1. Qwen3 can be aligned, but it is highly sensitive to visual-token scale and connector geometry.
2. V11 is a narrow, strong local basin for the current frozen SigLIP2 + V3 128-token Perceiver + frozen Qwen setup.
3. No-branch C1 did not reproduce V11. The training-time auxiliary branch mattered even though its final patch-position output was disabled.
4. Ordinary Qwen Stage2 LoRA is not the answer. It repeatedly failed to improve GQA or visual dependence.
5. V12/V13 local edits around V11 mostly returned to the same basin or regressed.
6. A new substrate must first prove it can imitate V11 before being asked to improve beyond it.
7. GQA n1000 bumps are noisy. Search/confirm/final splits and paired statistics are required.

---

# Core rule

Do not run five shallow branches.

A valid V14 result must deeply test:

```text
one primary substrate
```

and optionally:

```text
one backup substrate
```

The primary substrate must be launched under V14 or proven to exactly match V14 requirements. Pre-existing partial artifacts do not count as a valid test unless they satisfy all validity requirements below.

---

# Validity requirements for a substrate test

A substrate is not considered tested unless all of the following are true:

```text
1. It was launched fresh for V14, or an existing run is proven to exactly match V14 requirements.
2. V11 full answer-token KL is active during the imitation phase.
3. Cached V11 teacher distributions are used or a documented equivalent is used.
4. Teacher-imitation metrics are logged.
5. Visual-token RMS/scale/gate diagnostics are logged.
6. The branch receives the minimum imitation exposure before being judged.
7. Phase 2 compositional training begins only after imitation health is achieved.
8. GQA search and confirm are both run for the best imitation and compositional checkpoints.
9. The final report states whether failure was due to implementation, imitation, improvement, control recovery, or evaluation variance.
```

---

# Recommended primary substrate

Default primary:

```text
spatial-preserving compressed grid
```

Reason:

```text
It breaks the 128-token learned-Perceiver bottleneck.
It preserves spatial layout better than V3 Perceiver latents.
It is less implementation-heavy than full AnyRes.
It is the cleanest next substrate to test deeply.
```

Allowed primary choices:

```text
A. spatial-preserving compressed grid
B. AnyRes / MLP pass-through
C. hybrid V11 + spatial tail
```

Default is A. If the agent chooses B or C, write a short decision memo before launch explaining why.

Backup should usually be:

```text
AnyRes / MLP pass-through
```

---

# Phase 0: mandatory setup and diagnostics

Before any main substrate training, complete these.

## 0A. Decision memo

Write a short memo in the run log:

```text
primary substrate selected:
backup substrate selected:
why this substrate is likely to escape V11:
what specifically was wrong or incomplete about prior related attempts:
what changes in V14:
architecture changes relative to V11:
expected token count:
placeholder contract changes needed:
teacher KL cache status:
data sources used:
leakage risks:
minimum imitation exposure:
minimum compositional exposure:
```

The memo must explicitly answer:

```text
How is this different from V12/V13 spatial-grid, AnyRes, token-budget, or tail attempts?
```

## 0B. Eval split setup

Build or identify:

```text
GQA search split
GQA confirm split
GQA final / largest reliable split available
```

Minimum target:

```text
search: n1000
confirm: n3000 if available
final: largest reliable trusted slice available
```

Add paired bootstrap or equivalent paired significance testing for candidate-vs-V11 on identical examples.

## 0C. Reconfirm V11

Rerun V11 on:

```text
GQA search
GQA confirm
ChartQA or TextVQA chosen expanded metric
```

Record:

```text
accuracy
correct/total
yes/no and other breakdown when available
bootstrap/Wilson CIs
prediction samples
answer histograms
```

## 0D. Cached V11 teacher distributions

Use cached V11 teacher top-k + remainder answer-token distributions.

Online V11 teacher forward is allowed only for debug or tiny proof runs.

Cache fields:

```text
sample_id
image ref
question
answer
answer token positions
teacher top-k token ids per supervised position
teacher top-k probabilities or logits
teacher remainder probability mass
teacher greedy answer
teacher raw answer
teacher checkpoint metadata
prompt template metadata
```

Use this cache for replay examples in imitation, compositional training, and recovery.

## 0E. Mandatory C1 auxiliary-branch diagnostic

The no-branch C1 result showed that the branch mattered. V14 must study this mechanism, not leave it optional.

Run or prepare a diagnostic that tests:

```text
auxiliary branch present during training
common projector trained with the auxiliary branch
auxiliary branch disabled at evaluation
```

Compare against:

```text
historical V11
no-branch C1 diagnostic
```

Minimum question:

```text
Can a training-only auxiliary branch move common projector weights toward or beyond the V11 basin?
```

This can run in parallel with primary substrate setup, but it should not be skipped.

## 0F. Leakage setup

For any GQA train data, audit:

```text
GQA eval image IDs
GQA eval question IDs
raw refs
split labels
numeric filename collisions
```

Prior audits had numeric false positives. Use split-aware inspection and document any collision.

---

# Teacher loss

Use one teacher loss as the main imitation loss:

```text
full answer-token KL to V11
```

Do not use visual-token MSE as the primary teacher loss. New substrates may have different token topology or token count.

Do not use gold-answer-only imitation. It is too weak.

## KL definition

For replay examples:

```text
image, question, answer
```

Use V11 teacher distribution over supervised answer-token positions.

Student loss:

```text
KL(student answer-token distribution, V11 teacher answer-token distribution)
```

Prompt and image positions are masked.

If full-vocab KL is too expensive:

```text
use top-k teacher distribution plus remainder bucket
```

but keep distributional imitation.

---

# Primary substrate: spatial-preserving compressed grid

## Architecture goal

Replace the learned 128-latent V3 Perceiver bottleneck with a spatial grid representation.

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

Optional second scale:

```text
512 visual tokens
```

Move to 512 only if 256 shows either V11 imitation or a strong partial capability signal.

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

Generalize the old 128-placeholder assumptions.

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
checkpoint metadata records image-token count
generation hygiene is sane on a tiny probe
```

---

# Minimum evidence exposure

These are evidence floors, not budget limits.

A new substrate cannot be declared completed-negative before these are satisfied unless it has a hard technical failure.

## Imitation phase minimum

```text
minimum 1500 optimizer steps
at least 3 evaluated imitation checkpoints
```

Hard technical failures that can stop early:

```text
NaNs or infs
persistent W&B active alerts
placeholder contract failure
generation broken beyond recovery
EOS < 0.90 on tiny probes after debugging
no gradients where gradients should exist
teacher KL not finite
visual-token RMS pathologically unstable despite scale/RMS fixes
```

## Compositional phase minimum

Only after imitation health is reached:

```text
minimum 1500 optimizer steps
at least 3 evaluated compositional checkpoints
```

## Recovery phase minimum

For a high-GQA or high-expanded-metric candidate with control failures:

```text
minimum 500-1000 optimizer steps or equivalent recovery attempts
```

Recovery attempts may include:

```text
scale/gate sweep
increase teacher KL
increase VQA/POPE replay
add control contrastive examples
interpolation toward best imitation checkpoint
branch-gate attenuation
```

---

# Phase 1: imitation-only training

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
full answer-token KL to V11
clean CE on replay examples
```

Data:

```text
VQA replay
GQA replay
POPE replay
existing v3_grounding replay
```

Do not add heavy new compositional or contrastive pressure during Phase 1.

## Imitation metrics

Log at every checkpoint:

```text
teacher KL
clean CE
student-V11 exact-answer agreement on replay
student-V11 top-answer overlap
student-V11 GQA pairwise deltas on a small replay slice
VQAv2 clean n200 or n1000
GQA search subset
ChartQA/TextVQA if available
visual-token RMS
connector scale/gate values
average generated tokens
EOS / max-token-hit / prefix
```

## Hard rule for entering Phase 2

Phase 2 may begin only after:

```text
1. teacher KL declines across 3 consecutive evaluated checkpoints,
2. student-V11 exact-answer agreement improves across those checkpoints,
3. GQA search is within 1.0 point of V11 search,
4. VQA clean is not collapsed,
5. generation hygiene is clean.
```

If this never happens after the minimum imitation exposure and one serious adjustment, document it as an imitation failure.

A serious adjustment can be:

```text
scale/RMS fix
lower LR
higher KL weight
simpler pooling
smaller token count
better replay mix
```

---

# Phase 1.5: KL vs contrastive diagnostic fork

Before a long Phase 2, run a short diagnostic fork from the best imitation checkpoint.

## Fork A: KL-only continuation

```text
full answer-token KL
clean CE
no contrastive loss
```

## Fork B: KL + GQA CE

```text
full answer-token KL
clean CE
GQA CE
no contrastive loss
```

## Fork C: KL + GQA CE + contrastive

```text
full answer-token KL
clean CE
GQA CE
contrastive image-dependence loss
```

Compare:

```text
teacher KL
student-V11 agreement
GQA search
blank/shuffled/wrong controls
POPE
ChartQA/TextVQA if available
```

Purpose:

```text
Determine whether contrastive negatives improve visual dependence or pull the model out of the V11 basin.
```

Do not proceed to long compositional training until this fork identifies the safest direction.

---

# Phase 2: compositional grounding

Only after imitation health and Phase 1.5.

Data:

```text
GQA relation/spatial/left-right examples
Visual Genome relation/attribute if available
RefCOCO-style grounding if available
POPE replay
VQA replay
hard negatives
ChartQA/TextVQA if capability-expansion branch is active
```

Loss depends on Phase 1.5 result.

Default if Fork C helps:

```text
clean CE
full answer-token KL to V11 on replay
contrastive image-dependence loss
```

Default if Fork C hurts:

```text
clean CE
full answer-token KL to V11
GQA CE
limited or no contrastive loss
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

---

# Phase 3: control recovery

For checkpoints with GQA or expanded-metric gains but control regressions, run recovery before rejecting.

Recovery tools:

```text
increase teacher KL
increase VQA/POPE replay
add anti-shuffle contrastive examples
sweep connector_output_scale
sweep branch gate
attenuate new visual branch
interpolate toward best imitation checkpoint
```

Do not immediately kill high-GQA candidates for control failures.

---

# Statistical win definition

Use paired comparisons against V11 on the same examples.

## Exploration improvement

A candidate deserves follow-up if:

```text
GQA search delta >= +0.5 over V11
or ChartQA/TextVQA improves materially
or pairwise taxonomy shows meaningful left/right/spatial gain
```

## Provisional GQA win

```text
GQA confirm delta >= +1.0 over V11 confirm
```

## Strong GQA win

```text
GQA confirm delta >= +1.5 over V11 confirm
and paired bootstrap supports a positive candidate-vs-V11 delta
```

## Promotion-level win

```text
largest/final GQA supports the gain
controls recover
POPE retained or improved
generation hygiene clean
leakage audit passes
```

Do not claim a win from one n1000 slice.

---

# ChartQA and TextVQA as first-class secondary metrics

ChartQA/TextVQA are not optional decoration. They measure whether a new substrate preserves fine-grained visual/text information that the V11 128-token interface currently lacks.

Current reference:

```text
V11 ChartQA val n200: about 6%
```

A substrate that improves ChartQA/TextVQA substantially may be worth continuing even if GQA initially ties V11.

Required:

```text
Run ChartQA or TextVQA on V11 and on the primary substrate.
Use a larger slice than n200 if available.
Record exact-match accuracy, answer histograms, EOS, avg tokens.
```

Interpretation:

```text
GQA improvement = compositional reasoning claim.
ChartQA/TextVQA improvement = fine-grained visual/text capability claim.
```

Do not conflate the two, but treat both as important.

---

# Backup substrate: AnyRes / MLP pass-through

Only run after the primary is blocked or reaches a clear plateau, unless resources allow parallel work.

## Goal

Break the 128-token Perceiver substrate more radically by preserving more image structure.

## Token budget

Start small:

```text
global 384 image -> 128 visual tokens
2 local crops at 384 -> 64 visual tokens each
total = 256 visual tokens
```

If small AnyRes imitates V11:

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

Same phases:

```text
Phase 1 imitation
Phase 1.5 KL-vs-contrastive fork
Phase 2 compositional
Phase 3 recovery
```

Do not declare AnyRes dead from existing partial artifacts unless they satisfy V14 validity requirements.

---

# Decoder adaptation fork if imitation succeeds but GQA ties

This is a required fork.

If the primary substrate successfully imitates V11 but ties V11 on GQA confirm, do not immediately propose another substrate.

Instead run Qwen visual-intake adaptation on that substrate.

## Visual-intake LoRA

Start with:

```text
q_proj/v_proj only
rank 4 or 8
selected mid/upper layers
no MLP LoRA
```

Loss:

```text
full answer-token KL to substrate teacher or V11
GQA CE
VQA replay CE
POPE replay CE
optional contrastive loss if Phase 1.5 showed it helps
ChartQA/TextVQA data if capability branch is active
```

If q/v LoRA moves the needle, then try:

```text
q/k/v/o LoRA
rank 8 or 16
selected layers first
```

Do not run broad decoder LoRA before the substrate can imitate V11.

---

# Vision adaptation fork if decoder adaptation ties

If the new substrate imitates V11 and decoder visual-intake adaptation ties, then try vision adaptation on that substrate.

Start with:

```text
SigLIP last 4 blocks adapter or LoRA
low LR
new substrate trainable or lightly trainable
Qwen frozen
full answer-token KL
GQA/compositional data
ChartQA/TextVQA if relevant
VQA/POPE replay
```

Do not use a 25-step n200 result as the final judgment for vision adaptation.

---

# What not to do

Do not:

```text
run five branches shallowly
count pre-existing partial artifacts as V14 evidence
skip teacher KL cache
start Phase 2 before imitation health
judge a new substrate from n128/n200 alone
claim a GQA win without confirm and paired comparison
run old semantic Stage2 LoRA as the main path
train broad decoder LoRA before substrate imitation
ignore ChartQA/TextVQA capability movement
ignore RMS/scale diagnostics
```

---

# Final report requirements

The final report must answer:

```text
1. Which primary substrate was selected and why?
2. What was specifically different from prior failed attempts?
3. Did Stage 0 contract tests pass?
4. Did the V11 teacher cache work?
5. Did teacher KL decrease?
6. Did student-V11 agreement improve?
7. Did the substrate imitate V11 before Phase 2?
8. What did the Phase 1.5 KL-vs-contrastive fork show?
9. What happened during compositional training?
10. Did controls fail, and was recovery attempted?
11. Did the candidate beat V11 on GQA search and confirm?
12. What did paired bootstrap / paired deltas show?
13. What happened on ChartQA/TextVQA?
14. Was the backup substrate run? If not, why not?
15. If imitation succeeded but GQA tied, was decoder adaptation attempted?
16. What is the next most rational hill?
```

For any negative result, classify failure type:

```text
implementation failure
imitation failure
compositional-improvement failure
control-recovery failure
evaluation-variance failure
capability-mismatch failure
```

---

# Success definition

A successful V14 does not require a promoted checkpoint.

It must produce one of:

```text
1. A new substrate that imitates V11 and beats it on confirm GQA.
2. A new substrate that imitates V11 and improves ChartQA/TextVQA substantially.
3. A new substrate that imitates V11 but cannot beat it, localizing the bottleneck beyond connector geometry.
4. A clear demonstration that the chosen substrate cannot imitate V11 despite proper teacher KL, real exposure, and one serious adjustment.
5. A high-GQA or high-capability but control-leaky branch with documented recovery attempts.
```

Depth and diagnosis matter more than branch count.
