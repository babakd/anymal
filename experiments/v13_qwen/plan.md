# V13 Qwen3 Substrate-Break Plan

Date: 2026-05-14

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, checkpoints, prior experiment files, and prior V8/V9/V10/V11/V12 results.

## One-line objective

V13 is not another local V11 hillclimb. V13 is a **substrate-break campaign** for Qwen/Qwen3-8B AnyMAL.

The goal is to leave the current V3 128-token Perceiver basin and determine whether Qwen3 can reach a materially higher visual-reasoning ceiling with:

1. a stronger visual interface,
2. a broader and more compositional training mixture,
3. a more reliable evaluation harness,
4. staged decoder and vision adaptation only after the new visual interface is stable.

The central thesis:

```text
V11 is a strong local frontier for the current frozen SigLIP2 384 + 128-token V3 Perceiver + frozen Qwen interface.
V12 strongly suggests that local edits around that interface are exhausted.
V13 should treat V11 as a teacher/baseline, not as the architecture to keep polishing.
```

---

# Current state and why V13 exists

## LLaMA/V3 robust reference

```text
/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
```

Important same-slice reference:

```text
GQA trusted n1000: 43.7
```

This is the old LLaMA/V3 reference. V13 should beat it clearly, but the real goal is to move well beyond it.

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

This showed Qwen could become a viable-ish core decoder once the visual-token scale was calibrated.

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

Important V11 finding:

```text
The best state came from C1-trained common projector weights.
The explicit learned 2D patch-position table was disabled:
  patch_position_feature_scale = 0.0
The connector scale was:
  connector_output_scale = 1.125
```

## V12 conclusion

V12 aggressively tried local and medium-size moves:

```text
visual cross-attention
larger token budgets by continuation
controlled projector continuations
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

No branch robustly beat V11 on matched GQA checks. Some n1000 bumps collapsed on n3000. Many training branches regressed to the 42-43 GQA region.

Interpretation:

```text
The current 128-token V3 Perceiver substrate is likely exhausted.
V11 should be treated as a narrow, useful connector basin.
The next ceiling attempt must break the substrate instead of polishing V11.
```

---

# V13 operating principles

## 1. Evaluation upgrades run in parallel, not as a blocker

Do not spend weeks building every benchmark before launching architecture runs. But do build a better scorecard immediately and use it to avoid chasing GQA n1000 noise.

## 2. New visual interfaces require real training exposure

For a genuinely new interface, a 10-step or 25-step continuation is a smoke test only. Do not declare a new substrate dead from one short probe unless it has a hard technical failure such as NaNs, broken generation, no gradients, or impossible memory.

New architecture branches should get:

```text
smoke
short alignment
full alignment
scale sweep
GQA/control screen
then optional decoder adaptation
```

## 3. V11 is the teacher

When building a new visual interface, first teach it to imitate V11 behavior on replay data. Then ask it to surpass V11 on compositional and counterfactual data.

This is the key change from V12:

```text
Stage A: preserve V11 basin.
Stage B: add new visual capacity.
Stage C: add compositional grounding pressure.
Stage D: recover controls.
```

## 4. Exploration and promotion are separate

During exploration, branches may temporarily fail controls if they move a meaningful capability.

Do not discard a branch with high GQA, high OCR, high chart performance, or strong spatial gains merely because blank/shuffle controls regress. First try recovery:

```text
scale sweep
branch-gate attenuation
KL to V11
VQA/POPE replay
contrastive control training
interpolation toward teacher
```

Promotion gates apply only to final candidates.

## 5. The old Stage2 recipe is retired for Qwen ceiling

Do not run plain LLaMA-style semantic Stage2 LoRA as a main branch. For Qwen, ordinary Stage2 repeatedly hurt or failed to repair visual dependence.

Decoder adaptation is allowed, but only as a Qwen-specific visual-grounding stage after the new visual interface is stable.

---

# V13 evaluation plan

## Core eval suite

Every serious candidate should eventually be evaluated on:

```text
GQA trusted n3000 or larger available trusted slice
GQA trusted n1000 for fast comparison
VQAv2 clean n3000 seeds 42/43/44
VQAv2 blank n3000 seed42
VQAv2 shuffled n3000 seed42
VQAv2 wrong-image same-answer-type n3000 seed42
POPE adversarial n1000
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

## Expanded eval suite

Add these if the harness/data is available in the working environment.

Priority order:

```text
1. Fuller GQA / GQA split-confirmation
2. MMBench or comparable broad multimodal benchmark
3. MMStar or comparable multi-skill VLM benchmark
4. TextVQA
5. ChartQA
6. AI2D
7. OCRBench or DocVQA
8. RealWorldQA
9. MMMU only if judge/scoring path is reliable and cost is acceptable
```

Do not let missing expanded benchmarks block the first architecture runs. But do not claim a final V13 default without at least one broad non-VQA benchmark and at least one GQA-confirmation slice.

## GQA split protocol

Create or emulate three GQA slices:

```text
gqa_search: fast iteration slice, e.g. n1000
gqa_confirm: separate n1000 or n3000 slice
gqa_final: largest available trusted slice
```

The same slice should not be used for all model selection and final confirmation.

## Bootstrap confidence intervals

Add bootstrap confidence intervals for:

```text
GQA
VQAv2 clean
blank/shuffled/wrong controls
POPE
```

This is required because V11/V12 showed n1000 movement can be slice noise.

## Reporting requirement for eval

The final V13 report must include:

```text
best candidate by GQA
best candidate by controls
best candidate by POPE
best candidate by broad benchmark
best candidate by text/chart/OCR benchmark if available
best overall candidate
confidence intervals for main comparisons
pairwise taxonomy against V11 and V3
```

---

# V13 data expansion plan

The current Stage1B-style mixture is too narrow for a ceiling run. Expand data in targeted layers.

## Data families to add or verify

Use what is available locally or cached. If a source is not available, record it as missing and move to the next.

### Visual instruction / dense caption / general VLM data

Examples of desired data families:

```text
LLaVA-1.5-style instruction mixture
Mix-665K-style direct-answer and instruction subset
ShareGPT4V-style detailed captions
high-quality dense captions
synthetic short-answer visual instruction data
```

Purpose:

```text
teach the new interface broad visual-to-Qwen compatibility
```

### Compositional and relation data

```text
GQA train
GQA balanced-style relation/spatial splits
Visual Genome relation/attribute questions if available
same-image different-question pairs
same-question different-image pairs
left/right and relation-heavy examples
```

Purpose:

```text
target the GQA weakness directly
```

### Grounding / region data

```text
RefCOCO / RefCOCO+ / RefCOCOg if available
Visual Genome object/attribute/relationship annotations
COCO object/color/count prompts
region caption or referring-expression tasks
```

Purpose:

```text
improve object identity, attribute binding, relation grounding
```

### Text/chart/document data

```text
TextVQA train
OCR-VQA
ChartQA train
AI2D train
DocVQA if practical
```

Purpose:

```text
evaluate and improve whether new visual interface carries fine-grained spatial/text information
```

### Counterfactual and hard negative data

Build or reuse:

```text
shuffled-image negatives
wrong-image same-answer-type negatives
same-object different-relation negatives
attribute-swapped negatives
left/right swapped questions or images where possible
object presence/absence negatives
blank-image reliability examples
```

Purpose:

```text
force visual dependence rather than language-prior answering
```

## Leakage policy

For every source:

```text
record source path
record image-id format
record split
record whether it overlaps with VQAv2 val2014, POPE, GQA eval, and any expanded evals
```

Fail closed on exact image-id overlap unless the evaluation intentionally uses train-like data for diagnostics only.

Numeric filename collisions should be inspected manually and documented, as prior audits showed some false positive collisions.

---

# V13 training objectives

## Teacher imitation loss

Use V11 as teacher on replay data.

For a sample:

```text
image, question, answer
```

Record or compute V11 output distribution / answer logprob.

Loss options:

```text
KL(new_model || V11) on answer-token distribution
or
MSE / cosine distillation on visual token embeddings if compatible
or
answer-level logprob imitation
```

Purpose:

```text
new interfaces must first preserve the known-good Qwen visual-token behavior
```

## Clean CE loss

Standard answer cross-entropy:

```text
CE(answer | correct image, question)
```

Use on VQA, GQA, POPE, caption/direct-answer and other clean examples.

## Contrastive image-dependence loss

For clean and negative pairs:

```text
correct image + question + answer
negative image + same question + same answer
```

Use:

```text
CE(correct)
+ lambda * max(0, margin - logp(answer | correct image) + logp(answer | negative image))
```

Negatives:

```text
blank
shuffled
wrong same-answer-type
same-object different-relation
left/right swapped where possible
attribute swapped
```

Start conservative:

```text
lambda = 0.05, 0.10, 0.20
margin = 0.25, 0.50
```

## Control recovery loss

For branches with high GQA but high controls:

```text
VQA replay CE
POPE replay CE
contrastive negatives
KL to V11 on ordinary replay
```

Do not force "cannot determine" globally. That can harm POPE and clean VQA.

## Stage naming

Avoid calling everything Stage2. Use these terms:

```text
Interface alignment
Teacher imitation
Compositional grounding
Control recovery
Decoder visual adaptation
Vision adaptation
```

---

# Architecture Track A: Hybrid V11 + spatial tail

This is the preferred first substrate-break branch because it preserves the known-good V11 pathway and adds new visual evidence.

## A1. Architecture

Base:

```text
V11 frontier
```

Visual tokens:

```text
128 V11 Perceiver tokens
+ N spatial tail tokens
```

Test N:

```text
128
256
512
```

Tail options:

```text
MLP-projected SigLIP patch tokens
spatially pooled grid tokens
strided patch tokens
top-k or fixed-grid selected patch tokens
```

Initial total visual tokens:

```text
256 tokens = 128 V11 + 128 tail
384 tokens = 128 V11 + 256 tail
640 tokens = 128 V11 + 512 tail
```

Keep two separate scale/gate systems:

```text
v11_token_scale
tail_token_scale
tail_gate
```

Initialization:

```text
v11_token_scale = 1.125
tail_gate = 0.0 or very small, e.g. 1e-4 / 1e-3
tail tokens RMS-matched to Qwen token RMS
```

## A2. Training phases

### Phase A: teacher imitation

Train only:

```text
tail projection
tail gate
optional tail normalization
```

Freeze:

```text
V11 Perceiver path
Qwen base
SigLIP
```

Objective:

```text
KL to V11 on VQA/GQA/POPE replay
CE on clean replay
```

Goal:

```text
the hybrid model should behave exactly like V11 when tail_gate is small
```

### Phase B: compositional tail training

Unfreeze only the tail path and perhaps final visual-token normalization.

Data:

```text
GQA compositional
Visual Genome/referring/relation if available
VQA replay
POPE replay
counterfactual negatives
```

Objective:

```text
CE + contrastive image-dependence + KL to V11
```

### Phase C: controlled unfreezing

If GQA moves, unfreeze:

```text
last Perceiver layer
or V11 output projection
or small fusion layer between V11 tokens and tail tokens
```

Do not unfreeze the whole connector first.

## A3. Screens

Do not reject after only 10 or 25 steps unless technically broken.

Minimum schedule:

```text
smoke: 1-10 steps
short: 50, 100, 200
full: 500, 1000, 1500 or matched exposure if trend is positive
```

Screen:

```text
GQA search
VQAv2 clean/control
POPE
pairwise taxonomy
```

## A4. Success patterns

Promising:

```text
GQA improves even if controls regress temporarily
left/right or spatial taxonomy improves
tail gate learns nonzero useful value
scale sweep recovers controls
```

Bad:

```text
tail gate stays zero
logits do not change
generation breaks
GQA stays below V11 after full teacher-imitation and compositional training
```

---

# Architecture Track B: MLP pass-through + AnyRes tiling

This is the radical branch. It intentionally drops the 128-token Perceiver compression.

## B1. Architecture

Visual features:

```text
global 384 image
+ local 384 crops/tiles
```

Start with:

```text
global + 2 local crops
```

Then scale to:

```text
global + 4 local crops
```

Projector:

```text
2-layer MLP from SigLIP hidden size to Qwen hidden size
RMS normalization
learned visual scale/gate
optional 2D/crop-position embeddings
```

Visual tokens:

```text
all or pooled SigLIP patch tokens
no Perceiver compression in first version
```

If sequence length is too large, use spatial pooling:

```text
grid pooling per crop
or patch stride/downsample
```

But keep spatial order.

## B2. Training

This branch requires full training. A 25-step run is not meaningful.

Training stages:

```text
1. Smoke and scale/RMS validation.
2. Interface alignment on caption/dense-caption/general visual instruction.
3. V11 teacher imitation on replay.
4. Compositional grounding with GQA/relation/counterfactual data.
5. Optional decoder visual adaptation.
```

Recommended minimum exposure:

```text
At least comparable to the successful Qwen full-length Stage1A/Stage1B attempt.
Do not stop early because first short GQA screens are weak.
Use checkpoint ladder and trend analysis.
```

## B3. Why this branch exists

The likely bottleneck is that the Perceiver compresses away the spatial and object detail needed for GQA. Pass-through/tiled tokens preserve more structure.

This branch should be judged on:

```text
GQA
left/right taxonomy
object identity
attribute binding
TextVQA/ChartQA/OCR if available
```

not only VQAv2 clean.

---

# Architecture Track C: spatial-preserving compressed grid

This is a middle ground between V11 and full pass-through.

## C1. Architecture

Pipeline:

```text
SigLIP patch grid
-> spatial grid pooling / strided pooling
-> 256 or 512 spatial tokens
-> MLP projector to Qwen hidden size
-> optional learned 2D/crop embeddings
```

No learned Perceiver latents at first.

Variants:

```text
256 spatial tokens
512 spatial tokens
global + local pooled grids
```

## C2. Training

Same as Track B but cheaper.

This branch is useful if full pass-through is too expensive or unstable.

## C3. Success criterion

Look for improvements in:

```text
GQA spatial/left-right/object taxonomy
TextVQA/OCR/ChartQA if available
VQA corrupted-image controls after recovery
```

---

# Architecture Track D: no-branch C1 diagnostic

This is a cheap diagnostic recommended before or alongside the larger branches.

## D1. Question

V11 came from a C1 branch where the learned 2D patch-position contribution was later disabled. Was the gain caused by:

```text
C1 objective/data alone
```

or by:

```text
the presence of the 2D branch changing gradient flow and optimizer trajectory
```

## D2. Experiment

Start from the same Qwen anchor used for the original C1 run.

Train the same C1 objective/data and schedule, but:

```text
do not add learned 2D patch-position branch
do not add patch-position table
do not add coord MLP
```

Then materialize/evaluate with:

```text
connector_output_scale = 1.125
```

Use the same GQA trusted slices and VQA/POPE controls.

## D3. Interpretation

If it reproduces V11:

```text
C1 was mainly data/objective/path, not spatial architecture.
```

If it does not:

```text
the branch affected gradient flow even though its output was later disabled.
This matters for designing new substrate training.
```

This is not the main ceiling branch, but it clarifies what V11 really is.

---

# Architecture Track E: repaired visual cross-attention, full training version

V12 fixed the gradient path but only tested local/short versions. V13 can retry cross-attention only if it is embedded in a teacher-imitation and full-exposure regime.

## E1. Architecture

Use:

```text
V11 128 visual tokens as visual memory
optional spatial tail from Track A as additional memory
gated cross-attention in selected Qwen layers
```

Layer sets:

```text
upper: 18,22,26,30,34
mid-upper: 12,18,24,30
```

Gate:

```text
nonzero init: 1e-4 or 1e-3
not exactly zero unless gradient proof confirms learning
```

## E2. Training

Phase 1:

```text
teacher imitation to V11
```

Phase 2:

```text
GQA/relation/counterfactual training
```

Phase 3:

```text
control recovery
```

Trainable:

```text
cross-attention adapters
gates
possibly small visual memory projection
```

Do not train broad Qwen LoRA at first.

## E3. When to stop

Stop only if:

```text
gradient proof fails
adapter deltas are zero
logits do not change
or after full teacher-imitation + compositional training it still cannot match V11
```

Do not stop based only on a 25-step GQA screen.

---

# Decoder adaptation track

Only start this after one visual-interface branch can at least match V11.

## F1. Small visual-intake LoRA

Train:

```text
Qwen q_proj/v_proj only
rank 4 or 8
selected mid/upper layers
no MLP LoRA
```

Objective:

```text
CE on GQA/VQA
contrastive image dependence
KL to V11 or to the new visual-interface teacher
POPE replay
```

## F2. Broader LoRA

Only if F1 shows a hill.

Try:

```text
q/k/v/o LoRA
rank 8 or 16
selected layers first
```

## F3. Full decoder SFT

Only if the new interface and LoRA stages show evidence that Qwen can improve without losing visual dependence.

Do not start with full decoder finetuning.

---

# Vision adaptation track

Only after the new visual interface is stable.

## G1. SigLIP last-block adapter

Try:

```text
last 4 blocks adapter or LoRA
very low LR
new visual interface trainable
Qwen frozen
```

## G2. Vision + decoder co-adaptation

Only after G1 and decoder LoRA show signal.

Use strong KL/replay controls to avoid losing V11 behavior.

---

# V13 execution order

## Phase 0: eval and data setup

1. Build GQA search/confirm/final split or equivalent.
2. Add bootstrap confidence intervals.
3. Add or verify expanded evals.
4. Inventory expanded data sources and leakage risks.
5. Prepare V11 teacher outputs/logprobs for replay data.

## Phase 1: cheap diagnostics

1. No-branch C1 diagnostic.
2. Reconfirm V11 on new GQA confirm/final split.
3. Build initial leaderboard with CIs.

## Phase 2: substrate branches

Run these in parallel if possible:

```text
Track A: Hybrid V11 + spatial tail
Track B: MLP pass-through + AnyRes tiling
Track C: spatial-preserving compressed grid
```

Each must get more than a smoke test.

Minimum per branch:

```text
smoke
teacher-imitation phase
compositional phase
scale/gate sweep
GQA search and confirm screens
```

## Phase 3: adapt Qwen to the best new interface

For the best branch only:

```text
small visual-intake LoRA
then broader LoRA if signal exists
```

## Phase 4: vision adaptation

For the best branch only:

```text
SigLIP last-block adapters or LoRA
```

## Phase 5: final comparison

Evaluate the best V13 candidates against:

```text
V3 robust
V9 Qwen
V11 Qwen
best high-GQA but control-leaky branch
```

Use the full final screen and report CIs.

---

# Minimum work before stopping V13

Do not stop V13 until at least:

```text
1. No-branch C1 diagnostic is run or explicitly blocked.
2. Hybrid V11 + spatial tail receives real teacher-imitation and compositional training.
3. MLP pass-through / AnyRes receives real alignment training, not just a 25-step smoke.
4. Spatial-preserving compressed grid receives a real alignment run.
5. Expanded GQA evaluation with confidence intervals is available.
6. At least one expanded benchmark beyond VQAv2/GQA/POPE is run.
7. Any branch that beats V11 by >= 1.0 GQA point on search gets a confirm split before being rejected or promoted.
8. Any branch that beats V11 by >= 2.0 GQA points receives a control-recovery attempt, even if controls initially regress.
```

This is the main difference from V12.

---

# Success definitions

## Exploration success

A branch is useful if it shows any of:

```text
GQA confirm improvement >= +1.0 over V11
large left/right or spatial taxonomy improvement
broad benchmark improvement
text/chart/OCR improvement from a richer visual interface
successful imitation of V11 with a new higher-capacity interface
```

It does not need to be immediately promotable.

## Promotion success

A final V13 candidate should ideally show:

```text
GQA final >= V11 + 1.5
or broad benchmark win plus no GQA loss
Clean VQA roughly retained
VQA corrupted controls recovered
POPE retained or improved
generation hygiene clean
leakage audit pass
confidence intervals not overlapping trivially with V11 on the main claimed metric
```

Do not promote based on one n1000 slice.

---

# Expected failure modes

## New interface cannot imitate V11

Then the architecture or initialization is wrong. Fix imitation before pursuing GQA gains.

## New interface improves GQA but breaks controls

Do not reject immediately. Try:

```text
scale/gate sweep
KL to V11
control contrastive loss
interpolation toward teacher
lower branch gate
```

## New interface improves text/OCR/chart but not GQA

Keep as a separate capability branch. It may still be valuable.

## GQA gains appear only on search slice

Treat as noise until confirm/final slice agrees.

## Decoder adaptation improves clean VQA but hurts GQA/controls

Reject old-style decoder adaptation. Try smaller/selected visual-intake adapters only.

---

# Required final report structure

The final report must include:

```text
1. Baseline table with V3, V9, V11.
2. Evaluation harness changes and confidence intervals.
3. Data inventory and leakage notes.
4. Results for no-branch C1 diagnostic.
5. Results for Hybrid V11 + spatial tail.
6. Results for MLP pass-through / AnyRes.
7. Results for spatial-preserving compressed grid.
8. Any decoder/vision adaptation results.
9. Best candidate by GQA.
10. Best candidate by broad benchmark.
11. Best candidate by controls.
12. Best overall candidate.
13. Pairwise taxonomy for top models.
14. Clear recommendation for next hill.
```

Do not summarize a high-GQA but control-leaky branch as simply "failed." Explain what it improved and how control recovery was attempted.

---

# Final instruction

V13 should step out of the V11 basin.

Do not spend the campaign tuning another scalar around the V11 checkpoint. Use V11 as a teacher, build a higher-capacity visual interface, train it long enough to be meaningful, and evaluate it with enough breadth to distinguish real progress from GQA slice noise.
