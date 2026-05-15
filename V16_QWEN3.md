# V16 Qwen3 Capability-Led Data Recipe Plan

Date: 2026-05-14

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior checkpoints, and V8-V15 experiment artifacts.

---

# One-line objective

V16 is the first campaign to optimize the V15 Pareto trade rather than search for another architecture.

The goal is to produce a candidate that **clearly beats V11 on a Pareto-front comparison**: ChartQA and TextVQA materially improved, GQA at least retained, VQA/POPE retained. This is achieved by refining the V15 balanced data/objective recipe under stop-point and retention sweeps, with explicit connector-drift instrumentation, a fair vision-only retry, and finally executing the C1 auxiliary-branch diagnostic that has been deferred for four campaigns.

This is not another connector-architecture campaign. The connector axis is closed.

---

# Current state

## Active Qwen frontier and retention teacher

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

V11 representative metrics:

```text
GQA search n1000:       44.9
GQA confirm n3000:      42.6
Clean VQA n3000 mean:   65.922
Blank n3000 seed42:     39.078
Shuffled n3000 seed42:  36.767
Wrong n3000 seed42:     37.178
POPE adversarial n1000: 80.100
ChartQA val n200:       6.0
ChartQA val n1000:      7.8
TextVQA val n1000:      exact 28.5 / soft 26.47
```

## V15 candidate to confirm

The V15 balanced connector-only run produced the first non-V11-basin movement in the campaign sequence. The most interesting checkpoint is not the final one.

```text
/checkpoints/pretrain-output/v15-qwen3-balanced-v11-cachekl-lr2e6-3000/checkpoint-500
```

V15 step ladder, connector-only balanced, n=200:

```text
step 250:  GQA 43.5, ChartQA 6.0, TextVQA exact 33.0
step 500:  GQA 44.0, ChartQA 8.0, TextVQA exact 36.0
step 1000: GQA 44.0, ChartQA 6.0, TextVQA exact 35.5
step 1500: GQA 44.0, ChartQA 7.0, TextVQA exact 37.0
step 2000: GQA 42.5, ChartQA 7.0, TextVQA exact 38.0
step 3000: GQA 42.5, ChartQA 7.0, TextVQA exact 38.0
```

Step 500 is a Pareto-frontier candidate: GQA at V11 parity, ChartQA +2, TextVQA +3 vs V11 at n=200. V15 did not confirm step 500 at larger slices. V16 must.

## Key lessons V8-V15

1. The connector geometry axis is now closed. Pure architecture replacement has failed across V12/V13/V14/V15.
2. The decoder is not the bottleneck. Every Qwen LoRA variant has regressed.
3. V11 is a real, narrow basin. Multiple unrelated mechanisms tie it on n=3000.
4. The actual missing lever was **training data on capability tasks**. V15's connector-only run gained ChartQA/TextVQA by simply being exposed to that data.
5. Vision-only adaptation preserves GQA better than connector adaptation but does not gain capability on its own under the V15 recipe and exposure.
6. Joint vision+connector and high-res frozen-SigLIP underperformed connector-only at matched exposure.
7. KL retention-only is the correct configuration: V11 KL on retention data, CE only on capability data.
8. n=200 movement does not always survive n=1000 or n=3000. Stop-point selection must use larger slices.

---

# Non-negotiable operating rules

## 1. The connector architecture is fixed for V16

Use the V11 V3 Perceiver, 128 tokens, `connector_output_scale=1.125`, `patch_position_feature_scale=0.0`. Do not propose another connector replacement, spatial branch, AnyRes variant, or token-budget change. The V15 final spatial-grid ablation closed that axis.

## 2. Qwen3 stays frozen

Do not unfreeze the Qwen base. A small visual-intake LoRA is allowed only in Phase 4, only after Phase 1-3 plateau, and only at q/v rank ≤ 8.

## 3. KL is retention-only

Apply V11 cached answer-token KL on retention rows (VQA, GQA replay, POPE, COCO). Capability rows (ChartQA, TextVQA, OCR-VQA, DocVQA, AI2D) use CE only. Counterfactual rows use the contrastive image-dependence loss.

## 4. Connector drift must be instrumented

Every training run logs at every saved checkpoint:

```text
connector output RMS on fixed probe set
connector output cosine similarity to V11 on fixed probe set
connector output MSE to V11 on fixed probe set
answer-token KL to V11 on retention replay
student-V11 exact-answer agreement on retention replay
```

Fixed probe set: 64 examples drawn deterministically from VQAv2 val, seed 42. Same set across all runs.

Without these metrics, a checkpoint cannot be promoted.

## 5. Leakage audits are mandatory before any capability claim

Every new capability source must be audited against the corresponding eval slice.

Required audits:

```text
ChartQA train images vs ChartQA val images
ChartQA train question_ids vs ChartQA val question_ids
TextVQA train images vs TextVQA val images
TextVQA train question_ids vs TextVQA val question_ids
GQA train_balanced images vs GQA testdev_balanced images
GQA train_balanced question_ids vs GQA testdev_balanced question_ids
OCR-VQA / DocVQA / AI2D similarly when added
```

Fail closed on exact overlap.

## 6. Do not promote from n=200

Promotion candidates must be confirmed at:

```text
ChartQA val: full available
TextVQA val: full available or at least n=2000
GQA: search n=1000 and confirm n=3000
VQAv2: clean n=3000 seed 42; blank/shuffled/wrong n=3000 seed 42
POPE: adversarial n=1000, popular n=1000
```

With paired bootstrap CIs on candidate-vs-V11 deltas where slices are matched.

---

# Phase 0: confirmation, instrumentation, and the C1 diagnostic

This phase is the floor. Nothing in Phase 1-4 launches until Phase 0 is complete.

## 0A. Confirm V15 checkpoint-500 at the larger slices

Run the full promotion eval on V15 balanced checkpoint-500.

```text
checkpoint: /checkpoints/pretrain-output/v15-qwen3-balanced-v11-cachekl-lr2e6-3000/checkpoint-500
```

Required evals:

```text
GQA search n1000
GQA confirm n3000
VQAv2 clean n3000 seed 42
VQAv2 blank n3000 seed 42
VQAv2 shuffled n3000 seed 42
VQAv2 wrong-image n3000 seed 42
POPE adversarial n1000
POPE popular n1000
ChartQA val full
TextVQA val full or n2000
```

For each eval, report:

```text
accuracy
correct / total
answer-type breakdown
generation hygiene (EOS, max-hit, prefix, thinking)
paired bootstrap CI vs V11 where applicable
prediction samples
```

Decision logic:

- If checkpoint-500 confirms ChartQA/TextVQA gains and matches V11 GQA n3000 within paired-bootstrap CI overlap, V15 checkpoint-500 is the V16 starting frontier. Carry it forward.
- If checkpoint-500 underperforms at n=3000, fall back to V11 as the frontier and document the n=200-to-n=3000 collapse.

## 0B. Retroactive V15 leakage audit

Run leakage audits against the V15 capability training sources used for the connector-only balanced run:

```text
ChartQA train file used by v15_chartqa_train
TextVQA train file used by v15_textvqa_train
GQA spatial/relation train used by v15 mixture
```

Required audit outputs:

```text
exact image-id overlap
exact question-id overlap
numeric-stem collisions
raw ref overlap
audit verdict per eval slice
```

If any audit fails, the V15 ChartQA/TextVQA gains are not creditable and must be reproduced after the audit is fixed.

## 0C. Build the fixed connector-drift probe set

Create and persist a deterministic fixed probe set used by every V16 run for drift metrics.

```text
artifact: /checkpoints/v16_qwen/drift_probe_set_v1.json
size: 64
source: VQAv2 val seed 42 deterministic offset 0
contents: image refs, questions, V11 connector output tensors per example
```

The V11 connector outputs must be captured once, with the V11 checkpoint, and stored. Subsequent drift metrics compare against this cached tensor set.

## 0D. C1 auxiliary-branch diagnostic (mandatory)

This experiment has been deferred in V12, V13, V14, and V15. V16 cannot proceed without running it.

### Hypothesis

V11 came from a C1 training path with a learned 2D patch-position branch. The branch's output was disabled at inference (`patch_position_feature_scale=0.0`), but no-branch C1 did not reproduce V11. The branch's *training-time presence* may have acted as an optimization scaffold.

### Experiment

```text
base: V9 anchor at scale 1.05
  /checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105

training:
  add learned 2D patch-position branch
  patch_position_feature_scale = 1.0 during training
  common projector trainable
  branch trainable
  Qwen frozen
  SigLIP frozen
  dataset: v10_qwen_gqa_antishuffle_stage1b (C1 historical match)
  steps: 200 (match historical C1)
  lr: 5e-5

evaluation matrix:
  branch on, scale 1.0
  branch off, scale 0.0 (V11 historical setting)
  branch attenuated, scale 0.5
```

### Decision

For each evaluation mode, run GQA search n1000.

```text
If "branch off, scale 0.0" reaches V11 GQA: V11 path is reproducible.
  This is a positive scaffolding result.
If only "branch on" matches V11: the branch contributes at inference, not just training.
If neither matches V11: V11 may be a frozen lottery and not directly reproducible.
```

In all cases, document the result. This is the last unanswered question about V11's actual mechanism.

---

# Phase 1: refined balanced recipe with stop-point and retention sweep

The V15 finding is that a single recipe at 3000 steps overshot the Pareto-frontier. V16 sweeps the stop point and retention strength.

## 1A. Stop-point sweep

Use the V15 balanced recipe as the base:

```text
base checkpoint: V11 frontier
trainable: full V3 connector
frozen: Qwen, SigLIP
data mix: V15 balanced (30/25/20/15/10)
lr: 2e-6
KL: cached V11 retention-only, weight 1.0
counterfactual: contrastive lambda=0.05, margin=0.5
```

Save checkpoints at: `100, 250, 400, 500, 600, 800, 1000, 1500, 2000, 3000`.

Eval every saved checkpoint at n=200:

```text
GQA n200
ChartQA val n200
TextVQA val n200
VQAv2 clean n200
connector-drift metrics on the 0C probe set
```

For the top-3 checkpoints by Pareto position (GQA-vs-ChartQA-vs-TextVQA), run the full Phase 0A confirmation suite.

## 1B. Retention-weight sweep

If the stop-point sweep produces a candidate with capability gain but GQA regression > 1.0 at n=3000, run a retention-strength sweep:

```text
KL weight: {1.0, 2.0, 3.0, 5.0}
retention mix proportion: {30%, 40%, 50%}
```

Each sweep cell: 1000 steps, eval at 500 and 1000. Promote the best to 3000 steps and Phase 0A confirmation.

Hard rule: do not sweep retention to the point where capability gains disappear. If a cell pushes GQA up but flattens ChartQA/TextVQA back to V11, it is not a valid V16 candidate.

## 1C. Counterfactual ablation

The V15 main run used CE on counterfactual rows, not the explicit contrastive loss. Run one branch with the contrastive loss properly activated:

```text
contrastive lambda: 0.10
margin: 0.5
on shuffled/wrong-image/blank rows
```

At 1000 steps, eval GQA + ChartQA + TextVQA + VQA controls + POPE. If it materially improves the Pareto position, promote to 3000 + Phase 0A confirmation.

---

# Phase 2: fair vision-only retry

The V15 vision-only branch was stopped at step 500 despite the plan calling for 3000-step minimums. It uniquely preserved GQA. It deserves a fair run.

## Phase 2 hard exposure rule

Hard exposure floor for Phase 2A: minimum 3000 optimizer steps with checkpoints at 250, 500, 1000, 1500, 2000, 3000.

Phase 2A may stop early only on hard technical failure:

```text
NaNs or infs
broken generation
zero gradients where gradients should exist
persistent W&B active alerts after debugging
teacher KL non-finite
placeholder contract failure
```

Phase 2A may NOT stop early because:

```text
Phase 1 looked better at matched step counts
n=200 probes plateaued
the trajectory was not monotonic before step 1500
the mix selection from Phase 1 changed
compute pressure from other branches
the agent suspects vision adaptation will not work
```

If Phase 2A appears flat at step 500 or step 1000, document the trajectory and keep training to 3000. Vision-side adaptation may need longer exposure than connector adaptation to surface a signal; that is itself a finding worth recording.

Phase 2A is in the must-run subset. If compute is constrained, drop other branches before truncating 2A.

## 2A. Vision-only last-2 at full exposure

```text
base: V11 frontier
trainable:
  SigLIP last-2 blocks: layers 25 and 26
    self_attn, layer_norm1, layer_norm2
frozen: connector, Qwen
data mix: V15 balanced or the best Phase 1 mix
KL: retention-only, weight 1.0
lr: 1e-6 vision
steps: 3000
save: 250, 500, 1000, 1500, 2000, 3000
```

## 2B. Vision-only last-4

Gate (all evidence must come from 2A at step 3000, not earlier):

```text
ChartQA val n=1000 >= V11 + 1.0
OR
TextVQA val n=2000 exact >= V11 + 1.5
```

The trigger metric must be at n=1000 or larger. Do not gate this branch on n=200 probes.

If the gate is met:

```text
trainable: SigLIP layers 23, 24, 25, 26
  same components as 2A
lr: 1e-6
steps: 3000
save: 250, 500, 1000, 1500, 2000, 3000
```

Same hard-exposure rule as Phase 2A: 3000 steps minimum, hard-technical-failure stops only.

## 2C. Vision + connector joint, properly exposed

Gate (both conditions required, both from n>=1000 evidence):

```text
Phase 1 has produced a Pareto-promotable candidate confirmed at n=3000.
AND
Phase 2A at step 3000 has shown capability movement at n>=1000:
  ChartQA val n=1000 >= V11 + 1.0
  OR
  TextVQA val n=2000 exact >= V11 + 1.5
```

Do not run 2C on n=200 evidence. Do not run 2C if Phase 1 is the only signal source; without 2A signal, the joint is just connector-only with extra trainable parameters.

If both gates are met, run the joint:

```text
trainable: SigLIP last-2 + connector
  SigLIP lr: 1e-6
  connector lr: 2e-6 (matches Phase 1 winner)
steps: 3000
```

This is the only re-test of the joint setup. Do not run more joint variants unless this one improves over both Phase 1 and Phase 2A.

---

# Phase 3: capability-data scaling

Phase 1 is the main hill. Phase 3 explores whether **more capability data** changes the Pareto position.

## 3A. Inventory available data

Audit the working directory and Modal volume for:

```text
ChartQA: train, augmented
TextVQA: train
OCR-VQA: train
DocVQA: train
AI2D: train
ScienceQA: train (image-bearing subset)
Visual Genome: relations, attributes, region descriptions
RefCOCO/RefCOCO+/RefCOCOg
```

Record:

```text
available size per source
license / pretraining-compatibility notes
expected leakage risk against eval slices
```

## 3B. Scaled capability mix

Build a scaled mixture relative to V15's 20k ChartQA + 20k TextVQA:

```text
suggested target sizes:
  ChartQA train:  50k
  TextVQA train:  50k
  OCR-VQA:        50k (if available)
  DocVQA:         20k (if available)
  AI2D:           10k (if available)

retention replay:
  scale proportionally to keep 30-40% retention share
```

Run the best Phase 1 recipe on the scaled mixture at the best Phase 1 stop point.

## 3C. Decision

```text
If scaled mix improves ChartQA and TextVQA materially without GQA loss:
  scaling is the lever; this is the V16 promotion candidate.

If scaled mix saturates:
  the recipe is data-bounded only at the V15 scale; larger data won't add.

If scaled mix breaks retention:
  retention replay needs to scale faster than capability data.
```

---

# Phase 4: optional small Qwen visual-intake LoRA

Only run if Phase 1-3 have plateaued and a clear winning recipe exists.

```text
base: best V16 Pareto checkpoint
trainable: q_proj and v_proj LoRA
  rank: 4 or 8
  alpha: 16
  dropout: 0.05
  target layers: mid-upper (18, 22, 26, 30)
no MLP LoRA
no broad q/k/v/o LoRA initially
steps: 500 minimum, with checkpoints at 100, 250, 500
lr: 5e-6
lora_lr: 5e-6
loss_scale: 0.03
```

Hard exposure floor: ≥500 steps with ≥3 checkpoints. Do not declare LoRA tied from a 50-step probe.

Decision:

```text
If Phase 4 moves the Pareto position positively, promote.
If Phase 4 ties or regresses, the campaign promotes the best Phase 1-3 candidate without LoRA.
```

---

# Evaluation protocol

## Cheap screens during training

At every saved checkpoint:

```text
GQA n200
ChartQA val n200
TextVQA val n200
VQAv2 clean n200
connector-drift metrics on 0C probe set
generation hygiene
```

## Intermediate promotion screen

For any checkpoint that on n=200 shows:

```text
GQA within 1.0 of V11
AND ChartQA >= V11 + 1.5
AND TextVQA exact >= V11 + 2.0
```

run:

```text
GQA search n1000
ChartQA val full
TextVQA val n2000 or full
VQAv2 clean n1000
POPE adversarial n1000
```

## Final promotion screen

For the V16 candidate:

```text
GQA search n1000
GQA confirm n3000
VQAv2 clean n3000 seed 42
VQAv2 blank n3000 seed 42
VQAv2 shuffled n3000 seed 42
VQAv2 wrong-image n3000 seed 42
POPE adversarial n1000
POPE popular n1000
ChartQA val full
TextVQA val full
paired bootstrap CIs vs V11 on every matched slice
leakage audit on every training source
pairwise taxonomy vs V11 for GQA and ChartQA
prediction samples for every artifact
```

---

# Success criteria

## Pareto promotion candidate

V16 promotes a candidate only if:

```text
GQA confirm n3000 >= V11 - 0.5  (paired bootstrap CI overlap permitted)
ChartQA val full >= V11 + 1.5   (paired bootstrap CI excludes zero)
TextVQA val full exact >= V11 + 2.0
VQAv2 clean n3000 >= V11 - 1.5
VQAv2 corrupted controls within V11 + 0.7 absolute
POPE adversarial n1000 >= V11 - 0.5
generation hygiene clean
leakage audits pass
```

## Strong Pareto win

```text
GQA confirm n3000 >= V11 - 0.0
ChartQA val full >= V11 + 3.0
TextVQA val full exact >= V11 + 4.0
all other retention metrics retained
```

## Pure GQA win

Possible but not the target. If a V16 candidate beats V11 on GQA confirm n3000 by ≥+1.0 with paired bootstrap support, document it but do not let it crowd out Pareto evaluation.

## Localizing negative result

If no candidate meets the Pareto criteria, the report must still explain:

```text
Did capability metrics move and GQA fall too far?
Did vision-only fail to gain capability under proper exposure?
Did scaling data saturate?
Was retention KL the binding constraint?
```

Localizing the bottleneck is itself a successful outcome.

---

# What not to do

```text
Do not propose new connector architectures.
Do not propose 192/256/512 token budgets.
Do not run AnyRes variants.
Do not unfreeze Qwen except as Phase 4 with the specified scope.
Do not unfreeze SigLIP beyond last-4 blocks.
Do not apply V11 KL to capability data.
Do not promote a candidate from n=200 alone.
Do not promote a candidate without leakage audit.
Do not skip the C1 auxiliary-branch diagnostic again.
Do not stop a planned 3000-step branch at step 500 because of n=200 noise.
Do not skip connector-drift instrumentation.
Do not chase a single GQA n1000 bump without n=3000 confirmation.
```

---

# Recommended execution order

```text
1. Phase 0A: confirm V15 checkpoint-500 at full slices.
2. Phase 0B: retroactive V15 leakage audit.
3. Phase 0C: build the connector-drift probe set.
4. Phase 0D: run the C1 auxiliary-branch diagnostic.
5. Phase 1A: stop-point sweep.
6. Phase 1B: retention-weight sweep (only if 1A candidate fails GQA retention).
7. Phase 1C: counterfactual ablation.
8. Phase 2A: vision-only last-2 at 3000 steps.
9. Phase 2B: vision-only last-4 only if 2A shows movement.
10. Phase 2C: vision + connector joint at 3000 steps only if both halves show signal.
11. Phase 3A: data inventory.
12. Phase 3B: scaled capability mix on best Phase 1 recipe.
13. Phase 4: optional Qwen q/v LoRA only if Phase 1-3 plateau with a clear winning recipe.
14. Final promotion eval and report.
```

If compute is constrained, the must-run subset is:

```text
0A, 0B, 0C, 0D, 1A, 2A.
```

The rest are conditional on signal from the must-run subset.

---

# Final report requirements

The report must answer every item explicitly. Do not collapse multiple negatives into "branch failed."

```text
1. Phase 0A: did V15 checkpoint-500 confirm at n=3000 and full capability slices?
2. Phase 0B: did the V15 leakage audit pass for ChartQA and TextVQA?
3. Phase 0C: was the drift probe set built and used by every run?
4. Phase 0D: what did the C1 auxiliary-branch diagnostic show across the three eval modes?
5. Phase 1A: what was the stop-point Pareto curve?
6. Phase 1B: did retention-weight sweeps recover GQA at the candidate stop point?
7. Phase 1C: did the contrastive counterfactual loss change the Pareto position?
8. Phase 2A: did vision-only last-2 at 3000 steps move capability with GQA preserved?
9. Phase 2B/2C: did wider or joint vision adaptation help?
10. Phase 3A: what data was actually available?
11. Phase 3B: did scaled capability data move the Pareto position?
12. Phase 4: did q/v LoRA add value on the best non-LoRA candidate?
13. Connector drift summary across runs.
14. Leakage audit results per training source.
15. Pairwise taxonomy for GQA and ChartQA on the promotion candidate.
16. Final V16 candidate metadata and full promotion eval table.
17. Paired bootstrap CIs vs V11 on every matched slice.
18. Failure classification per branch:
    - implementation
    - imitation
    - capability-improvement
    - control-recovery
    - evaluation-variance
    - capability-mismatch
    - exposure-incomplete
19. Recommendation for V17:
    continue Pareto refinement, switch to data scaling, add Qwen LoRA, or stop.
```

The promotion candidate, if it exists, must include:

```text
materialized checkpoint path
full model_meta.json
prediction samples for every reported eval
paired bootstrap CIs
leakage audit artifacts
connector drift summary
training script and exact command preserved
W&B run IDs preserved
```

---

# One-sentence mandate

V16 is the first campaign that already knows what the lever is — capability data + connector adaptation with retention-only KL — and the job is to refine that lever along the Pareto frontier with proper confirmation, fair exposure on the underexplored branches, the C1 diagnostic finally executed, and explicit drift and leakage discipline.
