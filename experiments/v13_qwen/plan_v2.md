# V13 Qwen3 Substrate-Break Plan, Revised

Date: 2026-05-14
Revision: incorporates feedback after the first V13 plan review.

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, checkpoints, previous experiment files, and V8/V9/V10/V11/V12 artifacts.

---

# One-line objective

V13 is a **substrate-break campaign** for Qwen/Qwen3-8B AnyMAL.

The goal is to escape the current V11 basin by replacing or augmenting the 128-token V3 Perceiver visual interface with a higher-capacity, spatially faithful visual substrate, while using V11 as a teacher so new interfaces first learn to preserve the behavior that already works.

This is not a 25-step smoke-test campaign. New substrates require real alignment exposure before being judged.

---

# Current state

## LLaMA/V3 robust reference

```text
/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
```

Key same-slice reference:

```text
GQA trusted n1000: 43.7
```

## V9 Qwen scale-calibrated reference

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Representative metrics:

```text
Clean VQA n1000:   66.167
Blank n1000:       39.400
Shuffled n1000:    37.367
Wrong n1000:       37.967
POPE n1000:        79.100
GQA trusted n1000: 43.100
```

## V11 Qwen frontier

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

Representative metrics:

```text
GQA trusted n1000:     44.900
Clean VQA n3000 mean:  65.922
Blank n3000 seed42:    39.078
Shuffled n3000 seed42: 36.767
Wrong n3000 seed42:    37.178
POPE n1000:            80.100
```

Key interpretation:

```text
V11 is the best known Qwen connector basin.
The useful state came from C1-trained common projector weights.
The learned 2D patch-position contribution was disabled at inference:
  patch_position_feature_scale = 0.0
The connector scale was:
  connector_output_scale = 1.125
```

## V12 conclusion

V12 tried local and medium-size moves around V11:

```text
visual cross-attention
larger token-budget continuations
projector subset continuations
spatial residual branches
query-conditioned selectors/scales
decoder-side LoRA
higher resolution
multi-level / DeepStack bridges
vision-side SigLIP adapters
projector soups / interpolation
focused GQA data
contrastive / anti-shuffle objectives
```

No branch robustly beat V11 on matched GQA checks. Some n1000 bumps collapsed on n3000. Many branches regressed to the 42-43 GQA region.

Interpretation:

```text
The current frozen SigLIP2 384 + V3 128-token Perceiver + frozen Qwen substrate is likely exhausted.
V13 must stop polishing that corner and train new visual interfaces with V11 teacher preservation.
```

---

# Major changes in this revision

This revised plan makes five important changes.

```text
1. The no-branch C1 diagnostic is Phase 0, not a later side experiment.
2. The first main substrate branch is the spatial-preserving compressed grid, not V11 + extra tail tokens.
3. AnyRes / MLP pass-through is fully specified around token counts and placeholder plumbing.
4. The teacher loss is fixed: use full answer-token KL from V11, not visual-token MSE or gold-answer-only imitation.
5. The plan includes explicit exposure budgets, triage rules, a plateau contingency, and a smaller expanded eval suite.
```

---

# V13 operating principles

## 1. V11 is the teacher

Every new visual substrate should first prove that it can imitate the V11 behavior.

The key training pattern is:

```text
Stage A: imitate V11 on replay.
Stage B: add compositional and counterfactual grounding pressure.
Stage C: recover controls if needed.
Stage D: optionally adapt Qwen/vision only after the new visual interface is stable.
```

## 2. Exploration and promotion are separate

During exploration, a branch is allowed to temporarily regress on:

```text
blank-image control
shuffled-image control
wrong-image control
clean VQA
POPE
```

if it shows a meaningful capability gain such as:

```text
GQA improvement
left/right improvement
spatial/relation improvement
broad benchmark improvement
text/chart/OCR improvement
```

Do not kill a high-signal branch just because it is not immediately promotable. First attempt control recovery.

## 3. Do not kill a new substrate from a 25-step probe

A 10-25 step run is a smoke test. It proves plumbing, generation, gradients, and checkpointing. It is not enough to declare a new substrate scientifically dead unless there is a hard failure.

A substrate branch can be called completed-negative only after:

```text
smoke
teacher-imitation phase
compositional phase
scale/gate sweep
GQA search and confirm screen
basic control screen
```

## 4. Retire old Qwen Stage2 semantic LoRA

Do not run old LLaMA-style semantic calibration as a main path.

If a V13 candidate gets decoder adaptation, it should be a Qwen-specific visual-grounding SFT stage with:

```text
GQA/compositional CE
VQA replay
POPE replay
contrastive image-dependence loss
full answer-token KL to teacher
small visual-intake LoRA or selected adapters
```

not broad answer-prior rewriting.

---

# Evaluation plan

## Phase 0 eval setup

Build or emulate:

```text
GQA search split
GQA confirm split
GQA final / largest trusted split available
```

Minimum target:

```text
search: n1000
confirm: n3000 if available
final: largest reliable trusted slice available
```

Do not use the same small slice for both model selection and final claims.

## Core eval suite

For serious candidates:

```text
GQA trusted search / confirm / final
VQAv2 clean n3000 seeds 42/43/44
VQAv2 blank n3000 seed42
VQAv2 shuffled n3000 seed42
VQAv2 wrong-image same-answer-type n3000 seed42
POPE adversarial n1000
POPE popular n1000 if available
mild blur n1000
center crop 90 n1000
translate 5 pct n1000
generation hygiene:
  EOS
  max-token-hit
  assistant-prefix
  strict-clean gap
leakage audit
answer histograms
GQA pairwise/taxonomy
```

## Expanded eval suite, narrowed

Prioritize four additions rather than a broad unfocused list.

Run these if the harness/data are available:

```text
1. MMStar
2. MM-Vet or MM-Vet v2 if scoring is reliable
3. TextVQA
4. ChartQA
```

Also keep:

```text
POPE adversarial + POPE popular
```

Skip MMMU initially unless the judge/scoring path is already reliable and cheap enough not to distract from the substrate question. Skip MMBench unless an existing local harness is already trusted.

## Bootstrap confidence intervals

Add bootstrap confidence intervals for:

```text
GQA
VQAv2 clean
blank/shuffled/wrong controls
POPE
expanded benchmarks where possible
```

V11/V12 showed that n1000 GQA movement can be misleading.

## Thinking-mode diagnostic

Add a cheap diagnostic for Qwen3 thinking mode.

Run on a GQA subset:

```text
V11 non-thinking baseline
V11 thinking-enabled prompt
best V13 substrate non-thinking
best V13 substrate thinking-enabled
```

Requirements:

```text
strict/raw hygiene must be measured
no hidden post-processing promotion
record average generated tokens
record latency if practical
```

This is diagnostic only unless answer hygiene stays clean.

---

# Teacher loss: required choice

Use **full answer-token KL** from V11 as the teacher loss.

Do not use visual-token MSE as the main teacher loss because new substrates may have different token counts and different visual-token topology.

Do not use only gold-answer logprob imitation because it is too weak and does not transfer V11’s wrong-answer suppression behavior.

## Teacher KL definition

For replay examples:

```text
image, question, answer
```

Run the V11 teacher and cache or compute teacher distributions over supervised answer-token positions.

Loss:

```text
KL(student_distribution || teacher_distribution)
```

or equivalent teacher-to-student distillation over answer-token positions.

Prompt and image positions remain masked.

If full-vocab KL is too expensive, use:

```text
top-k teacher distribution + remainder bucket
```

but keep it distributional.

## Where to use teacher KL

Use teacher KL during:

```text
new substrate teacher-imitation phase
compositional training replay batches
control recovery
decoder visual-grounding SFT
vision adaptation
```

---

# Data plan

## Required data inventory

Create a data inventory table before training.

For each source, record:

```text
source path
split
image-id format
question-id format if applicable
task type
whether it may overlap VQAv2 val2014
whether it may overlap POPE
whether it may overlap GQA eval
whether it may overlap expanded evals
```

## Leakage checks

When adding GQA train data, audit:

```text
GQA eval image IDs
GQA eval question IDs
raw refs
split labels
numeric filename collisions
```

Prior audits saw numeric false positives. Handle these with split-aware inspection, but document them.

## Data families to use

Use locally available/cached sources first.

### Replay / teacher preservation

```text
V11 replay examples
VQA direct-answer replay
POPE replay
COCO object/color/count replay
existing v3_grounding replay
```

### General visual alignment

```text
caption/dense-caption data
LLaVA-style visual instruction mix
Mix-665K-style direct-answer/instruction subset
ShareGPT4V-style detailed captions if available
```

### Compositional visual reasoning

```text
GQA train
GQA spatial/relation/left-right splits
Visual Genome relation/attribute data if available
same-image different-question pairs
same-question different-image pairs
```

### Grounding / region

```text
RefCOCO / RefCOCO+ / RefCOCOg if available
Visual Genome object/attribute/relation annotations
COCO object/color/count prompts
```

### Text/chart/document

```text
TextVQA train
ChartQA train
OCR-VQA
AI2D
DocVQA if practical
```

### Counterfactual / hard negatives

Build or reuse:

```text
blank-image negatives
shuffled-image negatives
wrong-image same-answer-type negatives
same-object different-relation negatives
left/right swapped pairs
attribute-swapped pairs
object presence/absence negatives
```

---

# Training objectives

## Clean CE

```text
CE(answer | correct image, question)
```

Use on normal VQA, GQA, POPE, caption/direct-answer, text/chart/OCR, and other clean examples.

## Full answer-token KL to V11

```text
KL(student || V11 teacher)
```

Use on replay and control-preservation batches.

## Contrastive image-dependence loss

For a correct and negative pair:

```text
correct image + question + answer
negative image + same question + same answer
```

Loss:

```text
CE(correct)
+ lambda * max(0, margin - logp(answer | correct image) + logp(answer | negative image))
+ KL_to_teacher_on_replay
```

Start:

```text
lambda = 0.05, 0.10, 0.20
margin = 0.25, 0.50
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

Do not globally force “cannot determine” unless explicitly testing a diagnostic dataset.

---

# Phase 0: required diagnostics before main substrate training

## 0A. No-branch C1 diagnostic

This must run before the main substrate branches unless technically blocked.

Question:

```text
Was V11 caused by C1 data/objective alone,
or did the added 2D branch change gradient flow even though its output was later disabled?
```

Experiment:

```text
start from the same Qwen anchor used for the original C1 path
use the same C1 objective/data and schedule
do not add a learned 2D patch-position table
do not add coord MLP
do not add patch-position branch
materialize/evaluate with connector_output_scale = 1.125
```

Required eval:

```text
GQA search and confirm
VQA clean/control
POPE
GQA pairwise/taxonomy
```

Interpretation:

```text
If no-branch C1 reproduces V11:
  V11 was primarily data/objective/path, not spatial branch.
  Future substrate training should prioritize teacher imitation + objective design.

If no-branch C1 does not reproduce V11:
  the branch altered optimization dynamics even though its output was disabled.
  Future substrate training should consider auxiliary training paths or branch-gated mechanisms.
```

## 0B. Teacher-logit cache

Prepare a cache for V11 teacher distributions on replay data.

Minimum sources:

```text
VQA replay
GQA replay
POPE replay
counterfactual-control replay if available
```

Cache:

```text
input metadata
answer token positions
teacher top-k logits/probs or full logits if practical
teacher answer
teacher raw output
```

## 0C. Eval split and CI setup

Create:

```text
GQA search / confirm / final slices
bootstrap CI script or wrapper
baseline reruns for V3, V9, V11
```

---

# Main substrate Track 1: spatial-preserving compressed grid

This is the first main substrate branch.

Reason:

```text
It is a clean break from learned Perceiver compression.
It preserves spatial layout.
It is more operationally controlled than full AnyRes.
It has lower plumbing risk than full pass-through.
```

## 1A. Architecture

Replace the 128 learned Perceiver latent bottleneck with a spatial grid path.

Input:

```text
SigLIP2 patch grid at 384
```

Process:

```text
spatial grid pooling or strided pooling
preserve spatial order
project pooled grid tokens to Qwen hidden size with a 2-layer MLP
RMS-normalize visual tokens
apply learned visual scale/gate
include 2D/grid position embeddings
```

Token counts:

```text
256 spatial tokens
512 spatial tokens
```

Start with 256.

Metadata must include:

```text
image_tokens
grid_height
grid_width
pooling mode
position embedding type
connector_output_scale
visual RMS stats
```

## 1B. Training schedule

Do not kill this from a 25-step smoke.

Minimum:

```text
smoke: 1-10 steps
teacher imitation: at least 300-500 optimizer steps
compositional grounding: at least 500-1000 optimizer steps if imitation is healthy
scale/gate sweep
GQA search and confirm screens
```

If GQA is trending up or taxonomy improves, extend.

## 1C. Loss

Phase 1 teacher imitation:

```text
CE on replay
full answer-token KL to V11
```

Phase 2 compositional grounding:

```text
GQA CE
VQA replay CE
POPE replay CE
contrastive image-dependence loss
KL to V11 on replay
```

## 1D. Screens

At checkpoint ladder:

```text
50
100
250
500
1000
1500
```

Screen:

```text
GQA search
GQA taxonomy
VQA clean/control n1000
POPE
```

For top candidates, run confirm.

---

# Main substrate Track 2: hybrid V11 + spatial tail

This is second priority. It preserves the V11 pathway while adding extra visual evidence.

## 2A. Architecture

Visual tokens:

```text
128 V11 Perceiver tokens
+ spatial tail tokens
```

Tail sizes:

```text
128 tail tokens
256 tail tokens
512 tail tokens
```

Total visual tokens:

```text
256
384
640
```

Tail source:

```text
MLP-projected SigLIP patch/grid tokens
or spatially pooled grid tokens
```

Keep separate scales:

```text
v11_token_scale = 1.125
tail_token_scale
tail_gate
```

Initialize:

```text
tail_gate = 0.0, 1e-4, or 1e-3
tail RMS matched to Qwen token RMS
```

## 2B. How this differs from failed V12 token-budget runs

V12 token-budget branches appended or changed tokens without a proper teacher-imitation phase.

This track must first prove:

```text
with tail enabled, the model can imitate V11
```

before asking it to improve GQA.

## 2C. Training

Phase 1:

```text
freeze V11 path
train only tail projection/gate/norm
CE + full answer-token KL to V11
```

Phase 2:

```text
train tail on GQA/compositional + contrastive negatives + replay KL
```

Phase 3 if promising:

```text
unfreeze a small fusion layer or last V11 Perceiver block
```

## 2D. Required exposure

Minimum before completed-negative:

```text
teacher imitation: 300-500 steps
compositional: 500-1000 steps
scale/gate sweep
confirm-slice eval
```

---

# Main substrate Track 3: MLP pass-through / AnyRes

This is the radical branch. It intentionally drops the 128-token Perceiver interface.

The plan must specify token counts and plumbing before training.

## 3A. Visual-token targets

Do not start with thousands of visual tokens. Use staged token budgets.

### Small AnyRes

```text
global image tokens: 128
local crops: 2
tokens per crop: 64
total visual tokens: 256
```

### Medium AnyRes

```text
global image tokens: 256
local crops: 4
tokens per crop: 128
total visual tokens: 768
```

### Large AnyRes, only after small/medium are healthy

```text
global image tokens: 256
local crops: 4
tokens per crop: 192 or 256
total visual tokens: 1024-1280
```

If memory or sequence length is problematic, use:

```text
spatial pooling
patch stride
crop-level pooling
```

but preserve spatial order.

## 3B. Pooling strategy

Use one of:

```text
fixed spatial grid pooling
strided patch selection
adaptive pooling to target grid size
```

Record:

```text
input patch count
output tokens per crop
grid shape per crop
crop order
crop-position embeddings
2D position embeddings
```

## 3C. Placeholder and model plumbing

The codebase historically enforces 128 image placeholders. Generalize this.

Required files to audit/update:

```text
models/anymal_v3.py
evaluation/vqa_eval.py
vqa_checkpoint_eval.py
pope_checkpoint_eval.py
gqa_checkpoint_eval.py
data collators
chat template utilities
checkpoint metadata guards
scripts that assume 128 image tokens
```

Stage 0 contract tests:

```text
tokenizer accepts correct placeholder count
prompt contains exactly image_tokens placeholders
placeholders are contiguous where required
inputs_embeds path works
left-padded generation works
batched generation works
strict/raw diagnostics work
checkpoint metadata records image_tokens and crop settings
```

## 3D. Architecture

Visual path:

```text
SigLIP global image
+ local crops/tiles
-> patch/grid features
-> 2-layer MLP projector to Qwen hidden size
-> RMS normalization
-> learned visual scale/gate
-> Qwen prompt splice
```

No Perceiver compression in the first version.

## 3E. Training

This branch requires real exposure.

Minimum:

```text
smoke and contract tests
teacher imitation phase: 500+ steps
interface alignment: comparable to successful Qwen Stage1 exposure if stable
compositional grounding phase
scale/gate sweep
confirm eval
```

Do not call Track 3 dead after one short weak GQA screen.

---

# Main substrate Track 4: repaired visual cross-attention with full training

V12 fixed the gradient path but only tested local/short versions. This track retries cross-attention in a teacher-imitation-first regime.

## 4A. Architecture

Use:

```text
V11 visual tokens as visual memory
optional spatial tokens from Track 1 or 2 as additional memory
gated visual cross-attention in selected Qwen layers
```

Layer sets:

```text
upper: 18,22,26,30,34
mid-upper: 12,18,24,30
```

Gate initialization:

```text
1e-4
1e-3
```

Avoid exactly zero unless gradient proof is repeated and passes.

## 4B. Required proof

Before real training:

```text
nonzero gradients
nonzero parameter deltas
logits change after one optimizer step
generated outputs can change on a tiny probe
```

## 4C. Training

Phase 1:

```text
teacher imitation to V11
```

Phase 2:

```text
GQA/compositional + contrastive negatives + replay KL
```

Trainable:

```text
cross-attention adapters
gates
possibly visual-memory projection
```

Do not start with broad decoder LoRA.

---

# Decoder adaptation after substrate training

Do not start here.

Only run after at least one new visual substrate can match or beat V11.

## 5A. Qwen visual-intake LoRA

Train:

```text
q_proj/v_proj only
rank 4 or 8
selected mid/upper layers
no MLP LoRA
```

Loss:

```text
GQA CE
VQA replay CE
POPE replay CE
contrastive image-dependence
full answer-token KL to teacher
```

## 5B. Broader LoRA

Only if 5A shows a hill.

Try:

```text
q/k/v/o LoRA
rank 8 or 16
selected layers first
```

## 5C. Full decoder SFT

Only if the new substrate plus smaller LoRA shows clear improvement without losing visual dependence.

---

# Vision adaptation after substrate training

Only after the new visual interface is stable.

## 6A. SigLIP last-block adapter

Start with:

```text
last 4 blocks adapter or LoRA
low LR
new visual interface trainable
Qwen frozen
```

## 6B. Vision + decoder co-adaptation

Only if 6A and decoder LoRA show signal.

Use:

```text
full answer-token KL
VQA/POPE replay
contrastive negatives
GQA/compositional data
```

---

# Phase 2.5 decision point: if all substrates plateau

After Tracks 1, 2, and 3 receive real exposure:

## If all three land at V11 ± 0.5 GQA on confirm/final

Stop proposing more connector-only substrate variants.

Interpretation:

```text
The bottleneck is likely frozen Qwen, frozen SigLIP, data/objective, or the Qwen visual reasoning interface more generally, not just connector geometry.
```

Next step:

```text
run decoder adaptation and/or vision adaptation on the best substrate
```

## If one substrate improves broad benchmarks but not GQA

Keep it as a capability substrate and test decoder/vision adaptation.

## If one substrate improves GQA but fails controls

Run control recovery before rejection:

```text
scale/gate sweep
contrastive controls
VQA/POPE replay
KL to V11
interpolation toward teacher
```

---

# Execution order

## Phase 0

```text
1. Build/reconfirm GQA search/confirm/final splits.
2. Add bootstrap CIs.
3. Run no-branch C1 diagnostic.
4. Prepare V11 teacher-logit cache.
5. Inventory data and leakage risks.
6. Run Qwen thinking-mode diagnostic on GQA subset.
```

## Phase 1

```text
Track 1: spatial-preserving compressed grid
```

This is the first full substrate attempt.

## Phase 2

```text
Track 2: hybrid V11 + spatial tail
```

Run in parallel with Track 1 only if resources allow.

## Phase 3

```text
Track 3: MLP pass-through / AnyRes
```

Begin plumbing early, but do not start full training until placeholder/token-count contract tests pass.

## Phase 4

```text
Track 4: repaired visual cross-attention full-training retry
```

Run after gradient proof and teacher cache are ready.

## Phase 5

```text
Qwen visual-grounding SFT / decoder adaptation on best substrate
```

## Phase 6

```text
vision adaptation on best substrate
```

---

# Minimum work before stopping V13

Do not stop the campaign until:

```text
1. No-branch C1 diagnostic is completed or explicitly blocked.
2. Spatial-preserving compressed grid gets real teacher-imitation and compositional training.
3. Hybrid V11 + spatial tail gets real teacher-imitation and compositional training, unless Track 1 clearly dominates and resources are constrained.
4. AnyRes / MLP pass-through passes contract tests and receives real alignment training.
5. GQA search/confirm/final or equivalent split evaluation is available.
6. Bootstrap CIs are available for main comparisons.
7. At least one expanded benchmark beyond VQAv2/GQA/POPE is run.
8. GQA train-data leakage is audited by image ID and question ID.
9. If all substrate tracks plateau, decoder/vision adaptation is attempted on the best substrate.
```

---

# Success definitions

## Exploration success

A branch is useful if it shows:

```text
GQA confirm improvement >= +1.0 over V11
or meaningful left/right/spatial taxonomy gain
or broad benchmark improvement
or text/chart/OCR improvement
or successful imitation of V11 with a higher-capacity visual interface
```

Immediate promotion is not required.

## Promotion success

A final V13 candidate should ideally show:

```text
GQA final >= V11 + 1.5
or broad benchmark win with no GQA loss
clean VQA roughly retained
VQA corrupted controls recovered
POPE retained or improved
generation hygiene clean
leakage audit pass
confidence intervals that support the claimed gain
```

Do not promote based on one n1000 slice.

---

# Required final report

The final report must include:

```text
1. Baseline table for V3, V9, V11.
2. Evaluation split definitions and bootstrap CIs.
3. Data inventory and leakage notes.
4. No-branch C1 diagnostic result.
5. Spatial-preserving compressed-grid results.
6. Hybrid V11 + spatial-tail results.
7. AnyRes / MLP pass-through results.
8. Cross-attention full-training result.
9. Decoder/vision adaptation results if run.
10. Best candidate by GQA.
11. Best candidate by broad benchmark.
12. Best candidate by controls.
13. Best overall candidate.
14. Pairwise taxonomy for top models.
15. Explicit recommendation for the next hill.
```

Do not summarize a high-GQA but control-leaky branch as simply “failed.” Explain what it improved and what control recovery was attempted.

---

# Final instruction

V13 should leave the V11 basin.

Do not spend the campaign tuning scalars around the V11 checkpoint. Run the no-branch C1 diagnostic, build a real evaluation split with confidence intervals, train a spatial-preserving compressed grid, train a specified AnyRes/pass-through path, and use V11 as a teacher throughout.

The key test is not whether a 25-step probe beats V11. The key test is whether a new visual substrate can first imitate V11 and then exceed it after real compositional training.
