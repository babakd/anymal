# V15 Qwen3 Vision-Adaptation and Final Connector-Isolation Plan

Date: 2026-05-14

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior checkpoints, and V8-V14 experiment artifacts.

---

# One-line objective

V15 is the transition away from connector-only hillclimbing.

The goal is to run the last decisive connector-isolation checks, then move to **vision-side adaptation with connector co-adaptation and frozen Qwen3**, using ChartQA/TextVQA and compositional grounding as first-class capability metrics.

This is not another broad connector search.

---

# Current state

## Active Qwen frontier

Use this as the main baseline and retention teacher:

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
ChartQA val n200:       about 6%
```

## Key lessons from V8-V14

1. Qwen3 alignment became viable only after visual-token scale/RMS was handled.
2. V11 is a strong, narrow, path-dependent connector basin.
3. No-branch C1 did not reproduce V11, so the training-time auxiliary branch mattered even though its final output was disabled.
4. Ordinary Qwen Stage2 LoRA repeatedly failed to improve visual dependence.
5. Local connector edits and substrate probes around V11 mostly returned to V11 or regressed.
6. Pure 256-token spatial-grid replacement failed to imitate V11 under cached answer-token KL.
7. Current ChartQA performance is extremely poor, so the next campaign should measure new visual capability, not only GQA.
8. Vision adaptation has only received shallow probes. It is now the most plausible underexplored lever.

---

# High-level strategy

V15 has three parts:

```text
Part A: close the last connector-isolation questions.
Part B: run capability-led vision adaptation with Qwen frozen.
Part C: only if Part B shows signal, optionally add small Qwen visual-intake adaptation.
```

The default scientific claim to test:

```text
The current bottleneck is not the Qwen decoder and not another local connector tweak.
The likely next lever is improving the visual features feeding the already-calibrated V11 connector.
```

---

# Non-negotiable operating rules

## 1. Do not restart broad connector search

The connector has one final isolating ablation left:

```text
128-token spatial-grid connector
patch_position_feature_scale = 0.0
cached V11 KL
same V14-style imitation protocol
```

After that, do not spend the campaign on more connector variants unless this ablation unexpectedly imitates V11.

## 2. Qwen stays frozen in the main vision-adaptation campaign

Do not unfreeze Qwen first.

Decoder adaptation is allowed only after vision adaptation shows signal and must be small/visual-intake focused.

## 3. KL is retention-only

Do not distill V11 onto tasks where V11 is bad.

Use:

```text
VQA/GQA/POPE/COCO replay:
  CE + V11 answer-token KL

ChartQA/TextVQA/OCR/Doc/AI2D capability data:
  CE only, no V11 KL, or negligible formatting-only KL if explicitly justified

Counterfactual/hard negatives:
  contrastive image-dependence loss
```

If V11 gets about 6% on ChartQA, applying strong KL on ChartQA would teach the student to remain bad at ChartQA.

## 4. Attribute gains with controls

Every serious vision-adaptation result must be compared to:

```text
connector-only same data/objective
vision-adapter-only, connector frozen
joint vision + connector co-adaptation
high-resolution frozen-SigLIP + connector pilot
```

Without these, the agent cannot tell whether a gain came from data, vision adaptation, connector drift, or resolution.

## 5. Do not judge from n200 only

n200 is a smoke screen. Serious claims require search/confirm or larger slices.

---

# Phase 0: setup, final connector checks, and capability harness

## 0A. Reconfirm baselines

Re-evaluate V11 on:

```text
GQA search n1000
GQA confirm n3000
VQAv2 clean/control if needed
POPE adversarial
ChartQA
TextVQA if harness exists or can be added quickly
```

Record:

```text
accuracy
correct / total
answer-type breakdown when available
generation hygiene
prediction samples
bootstrap or paired confidence intervals where possible
```

## 0B. Final connector-isolation ablation

Run exactly one final connector-isolation experiment before moving on.

### Experiment

```text
spatial-grid connector
128 image tokens
patch_position_feature_scale = 0.0
connector_output_scale = 1.125
cached V11 answer-token KL
Qwen frozen
SigLIP frozen
V14-style imitation protocol
```

### Why

V14’s negative confounded:

```text
Perceiver -> spatial grid
128 tokens -> 256 tokens
patch_position_feature_scale 0.0 -> coord-MLP scale 1.0
```

This ablation asks:

```text
Can spatial-grid architecture itself imitate V11 if token count and position-feature contribution are controlled?
```

### Required logging

```text
teacher KL
student-V11 exact-answer agreement
GQA search
GQA confirm if promising
VQA clean
ChartQA
visual-token RMS
connector scale
generation hygiene
```

### Decision

If it imitates V11:

```text
V14 failure was likely token-count / position-feature / implementation interaction.
Continue this branch only if it shows a real path to capability gain.
```

If it fails:

```text
Close pure connector-replacement as the main hill and proceed to vision adaptation.
```

Do not launch many additional connector variants.

## 0C. C1 auxiliary-branch mechanism diagnostic

The no-branch C1 result is too important to leave unresolved.

### Goal

Test whether an auxiliary branch can act as a training-time optimization scaffold.

### Experiment

Start from a V9/V11-compatible anchor.

During training:

```text
original V3/V11 Perceiver path active
auxiliary spatial branch active
common projector trainable
auxiliary branch trainable
Qwen frozen
SigLIP frozen
```

At eval, compare:

```text
auxiliary branch on
auxiliary branch attenuated
auxiliary branch disabled
common-projector-only materialization
```

### Question

```text
Can the auxiliary branch move the common projector toward or beyond the V11 basin even when the auxiliary output is disabled at inference?
```

This is secondary to the vision campaign, but it is the last important unanswered connector-mechanism question.

## 0D. Capability eval harness

Make ChartQA first-class.

Also add TextVQA if feasible.

Minimum:

```text
ChartQA val larger than n200 if available
TextVQA validation if harness can be added quickly
GQA search/confirm
POPE adversarial and popular if available
VQAv2 clean/control
```

For ChartQA/TextVQA, record:

```text
exact match
relaxed numeric match if available
answer histograms
average generated tokens
EOS / max-token-hit / prefix
```

## 0E. Teacher replay cache

Use or build a larger V11 replay cache than the small V14 imitation cache.

Recommended sources:

```text
VQA direct-answer replay
GQA replay
POPE replay
COCO object/color/count replay
```

Target: tens of thousands of balanced examples if practical.

Cache:

```text
sample_id
source
image ref
question
answer
answer token positions
teacher top-k + remainder distribution
teacher raw answer
teacher checkpoint metadata
prompt metadata
```

---

# Phase 1: high-resolution frozen-SigLIP pilot

This tests whether pixel density and training at higher resolution can move capability before unfreezing vision.

## Experiment

Base:

```text
V11 connector/Qwen interface
Qwen frozen
SigLIP frozen
connector trainable or lightly trainable
```

Resolution:

```text
448 or 512 first
672 only if memory/harness is stable
```

Data:

```text
ChartQA
TextVQA/OCR if available
VQA/GQA/POPE replay
```

Loss:

```text
capability data: CE only
retention replay: CE + V11 KL
counterfactuals: contrastive if included
```

## Purpose

Separate:

```text
pixel-density / resolution bottleneck
```

from:

```text
vision-feature semantic bottleneck
```

If higher-resolution frozen-SigLIP improves ChartQA/TextVQA materially, the next campaign may focus on high-resolution interface/data rather than unfreezing SigLIP.

## Evaluation

Primary:

```text
ChartQA
TextVQA if available
```

Retention:

```text
GQA confirm
VQAv2 clean/control
POPE
```

---

# Phase 2: vision adaptation with connector co-adaptation

This is the main V15 campaign.

## Base

Start from:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Keep:

```text
Qwen/Qwen3-8B frozen
V11/V3 Perceiver visual interface
connector_output_scale = 1.125
```

Do not start from the failed pure spatial-grid substrate.

## Trainable components

Run a controlled set of branches.

### Branch A: connector-only control

Purpose:

```text
Tests whether data/objective alone moves the model.
```

Trainable:

```text
connector only
```

Frozen:

```text
SigLIP
Qwen
```

Use the same data/objective as the joint vision branch.

### Branch B: vision-adapter-only diagnostic

Purpose:

```text
Tests whether vision adaptation can move the model without connector drift.
```

Trainable:

```text
SigLIP adapter/LoRA only
```

Frozen:

```text
connector
Qwen
```

### Branch C: joint vision + connector co-adaptation

Purpose:

```text
Real candidate.
```

Trainable:

```text
SigLIP late-block adapters/LoRA
connector with constrained LR / constrained subset
```

Frozen:

```text
Qwen
```

### Branch D: high-resolution frozen-SigLIP pilot

May be completed in Phase 1. Include in comparison table.

---

# Vision adaptation scope

## First run: last 2 SigLIP blocks

Train:

```text
last 2 SigLIP blocks via LoRA/adapters
```

Suggested targets:

```text
attention projections
MLP projections only if attention-only is stable or underpowered
layer norms if needed
```

## Second run: last 4 blocks

Only if last-2 is stable or shows signal.

## Last 6 blocks

Only after last-4 shows signal.

Do not full-unfreeze SigLIP in the first V15 pass.

---

# Connector co-adaptation and drift control

The connector should be allowed to adapt in the joint branch, but the V11 basin must be protected.

## Trainable connector options

Start with one of:

```text
output projection + norm
last Perceiver block + output projection + norm
full connector at very low LR
```

Do not blindly train the full connector at old connector-training LR.

## LR guidance

Use conservative vision LR:

```text
SigLIP adapter LR: 1e-6 to 3e-6
```

Connector LR depends on trainable scope:

```text
full connector: 5e-7 to 2e-6
last block / output / norm only: 2e-6 to 5e-6
```

If the agent has strong reason to invert this ratio, document why before launch.

## Drift metrics

Log on a fixed image/question set:

```text
connector output RMS
connector output cosine similarity to V11
connector output MSE to V11
answer-token KL to V11 on replay
student-V11 exact-answer agreement
GQA/VQA/POPE retention
```

Interpretation:

```text
If vision adaptation improves capability but connector drift is extreme, try stronger KL or lower connector LR.
If connector-only matches joint gains, the gain is from data/objective, not vision adaptation.
If vision-only moves capability but joint does not, connector co-adaptation may be too aggressive.
```

---

# V15 data mix

Do not use naive concatenation. Large OCR-VQA-like sources can swamp ChartQA and retention replay.

Start with a balanced target mixture.

## Suggested first mix

```text
30% V11 retention replay:
  VQA, GQA, POPE, COCO object/color/count

25% ChartQA

20% TextVQA / OCR-VQA / DocVQA if available

15% GQA relation/spatial/left-right

10% counterfactual / hard negatives:
  wrong-image same-answer-type
  shuffled
  blank
  same-object different-relation
  left/right swapped where possible
```

If TextVQA/OCR/DocVQA are unavailable, reallocate:

```text
+10% ChartQA
+10% GQA relation/spatial/left-right
```

## Loss by data family

### Retention replay

```text
CE + V11 KL
```

KL weight starting point:

```text
1.0
```

Sweep if needed:

```text
0.5
2.0
```

### Capability data: ChartQA/TextVQA/OCR/DocVQA/AI2D

```text
CE only
```

No V11 KL by default.

Rationale:

```text
V11 is bad at chart/text capability, so distilling V11 onto these samples suppresses learning.
```

### Counterfactual negatives

Use contrastive image-dependence loss.

Starting values:

```text
lambda = 0.05 or 0.10
margin = 0.25 or 0.50
```

---

# Minimum exposure

Do not judge vision adaptation from a 25-step n200 probe.

## Smoke

```text
1-10 steps
```

Purpose:

```text
gradient proof
checkpoint save/load
vision adapter save/load
generation hygiene
no NaNs
```

## Main run

Minimum:

```text
3000 optimizer steps
```

Save/evaluate:

```text
250
500
1000
1500
2000
3000
```

If a branch is trending up on ChartQA/TextVQA or GQA, extend.

## Early stop only for hard failure

Hard failures:

```text
NaNs / infs
persistent W&B active alerts
broken generation
EOS collapse
vision gradients zero when trainable
teacher KL non-finite
severe VQA/POPE collapse that does not recover after one adjustment
```

---

# Evaluation protocol

## During training

At checkpoints:

```text
ChartQA n200 or larger
TextVQA if available
GQA search subset
VQA clean/control small slice
POPE small/standard
student-V11 retention agreement
connector drift metrics
generation hygiene
```

## Serious evaluation for promising checkpoints

Run:

```text
ChartQA larger validation slice
TextVQA validation if available
GQA search n1000
GQA confirm n3000
VQAv2 clean n3000 seed42
blank n3000
shuffled n3000
wrong n3000
POPE adversarial n1000
POPE popular if available
mild blur / crop / translate if relevant
```

## Final candidate eval

For any candidate showing real capability movement:

```text
ChartQA full available validation
TextVQA full available validation
GQA final / largest trusted slice
VQAv2 clean seeds 42/43/44
full corrupted-image controls
POPE
leakage audit
prediction samples
pairwise taxonomy
bootstrap / paired confidence intervals
```

---

# Success criteria

## Exploration success

Any of these is meaningful:

```text
ChartQA improves materially over V11
TextVQA improves materially over V11
GQA confirm improves over V11
left/right/spatial taxonomy improves
capability improves while GQA/VQA/POPE retention remains acceptable
```

## Strong capability success

Example target:

```text
ChartQA: 6% -> 20%+
```

or a comparable TextVQA jump, with VQA/POPE/GQA not catastrophically degraded.

## GQA success

```text
GQA confirm >= V11 confirm + 1.0
```

Strong:

```text
GQA confirm >= V11 confirm + 1.5
paired bootstrap supports positive candidate-vs-V11 delta
```

## Retention success

```text
VQAv2 clean roughly retained
blank/shuffled/wrong controls not badly degraded or recoverable
POPE retained or improved
generation hygiene clean
```

---

# If results are mixed

## Capability improves but GQA ties

Continue capability branch.

This is still valuable. ChartQA/TextVQA gains would mean the visual side learned something V11 cannot do.

## Capability improves but controls degrade

Run recovery:

```text
increase retention replay
increase V11 KL on replay only
lower vision LR
lower connector LR
freeze more connector
scale/gate sweep
contrastive hard negatives
```

## Connector-only matches joint vision branch

Conclusion:

```text
gain is likely data/objective, not vision adaptation
```

Then either:

```text
continue with cheaper connector/data recipe
or change vision-adaptation scope
```

## Vision-only improves but joint fails

Conclusion:

```text
connector co-adaptation may be disturbing V11
```

Try:

```text
connector frozen
or connector output/norm only
or lower connector LR
```

## Joint works and controls recover

Promote to final eval and consider adding small Qwen visual-intake LoRA.

---

# Optional Phase 3: small Qwen visual-intake adaptation

Only after vision adaptation shows signal.

Start with:

```text
q_proj/v_proj only
rank 4 or 8
selected mid/upper layers
no MLP LoRA
```

Loss:

```text
retention replay: CE + V11 or candidate-teacher KL
capability data: CE only
counterfactuals: contrastive
```

Do not run broad Qwen SFT unless small visual-intake adaptation shows a real hill.

---

# What not to do

Do not:

```text
run old semantic Stage2 LoRA
unfreeze Qwen before vision adaptation shows signal
run many connector variants
train full SigLIP from the start
apply V11 KL to ChartQA/TextVQA/OCR capability data
judge from n200 only
ignore connector drift
ignore connector-only same-data control
treat ChartQA as a decorative metric
```

---

# Required final report

The final report must include:

```text
1. Final connector-isolation ablation result.
2. C1 auxiliary-branch diagnostic status/result.
3. V11 baseline reconfirmation.
4. Data mix and exact loss assignment by data family.
5. High-resolution frozen-SigLIP pilot result.
6. Connector-only same-data control result.
7. Vision-only diagnostic result.
8. Joint vision + connector result.
9. Vision adapter scope and LR details.
10. Connector trainable scope and LR details.
11. Connector drift metrics.
12. ChartQA/TextVQA results.
13. GQA search/confirm results.
14. VQAv2 clean/control and POPE retention.
15. Leakage audit.
16. Clear attribution:
    data/objective vs vision adaptation vs connector co-adaptation.
17. Recommendation:
    continue vision adaptation, switch to data-only, add Qwen visual-intake LoRA, or stop.
```

Do not summarize a capability-gaining but control-leaky model as simply failed. Explain the gain, the damage, and the recovery attempts.

---

# Recommended execution order

```text
1. Run final 128-token no-position spatial-grid connector ablation.
2. Run or schedule C1 auxiliary-branch scaffold diagnostic.
3. Build/verify ChartQA and TextVQA eval harnesses.
4. Run high-resolution frozen-SigLIP pilot.
5. Launch connector-only same-data control.
6. Launch vision-only diagnostic.
7. Launch joint last-2 SigLIP adapter + connector co-adaptation.
8. If signal, launch last-4 SigLIP adapter + connector co-adaptation.
9. If signal persists, run final eval and consider small Qwen visual-intake LoRA.
```

This is the first campaign where the leading metric is a capability the current stack cannot do well. Treat ChartQA/TextVQA movement as real progress, not as a side note.
