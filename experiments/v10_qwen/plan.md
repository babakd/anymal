# V10 Qwen3 Ceiling Experiment Plan

Date: 2026-05-12
Audience: execution agent with access to the AnyMAL working directory, prior V8/V9 artifacts, Modal, W&B, and checkpoints.

## Executive Summary

The project is no longer asking whether Qwen/Qwen3-8B can merely replace the LLaMA-3-8B AnyMAL V3 decoder. V9 nearly solved that. The new goal is:

> Find the Qwen3 ceiling: push the Qwen3 AnyMAL V3 path clearly above the best LLaMA/V3 incumbent on clean VQA, perturbations, POPE, and especially GQA/compositional grounding, while preserving the corrupted-image controls.

The current best Qwen candidate is:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

This is a materialized eval/inference checkpoint with:

```text
model_meta.json
projector.pt
connector_output_scale: 1.05
```

It is not an optimizer-resume checkpoint.

The candidate passed the original materialized-checkpoint confirmation bundle except for a larger GQA concern:

```text
Clean VQA n1000:        66.167  pass
Blank n1000:            39.400  pass
Shuffled n1000:         37.367  pass, exactly on old cap
Wrong-image n1000:      37.967  pass
Wrong-image n3000:      36.756  pass
POPE n1000:             79.100  pass
GQA canonical n500:     44.000  pass
GQA trusted n1000:      43.100  concern
V3 robust GQA n1000:    43.700  same slice
```

The old minimum GQA gate was 43.800, but both Qwen and the V3 robust incumbent were below that on the newer trusted GQA n1000 slice. Since V3 still beat Qwen on that slice, promotion is held.

## Core Interpretation

V9 showed that Qwen3 is strong when its visual-token scale is calibrated:

```text
Stage1B2350 + connector_output_scale 1.05
```

The old LLaMA-style Stage2 calibration was not the winning path. Stage2 variants repeatedly failed to improve the active gates and often damaged GQA or corrupted-image controls.

Current best scientific read:

1. Qwen3 does not need ordinary robust semantic Stage2 to be useful.
2. The best Qwen state is near the Stage1B connector trajectory, before Stage2 LoRA.
3. Connector output scale is a first-class alignment knob for Qwen.
4. The new bottleneck is GQA/compositional grounding and robust visual dependence, not answer formatting.
5. The next useful work is connector calibration, checkpoint/scale search, GQA-aware connector repair, and minimal spatial/query-aware connector changes.

Do not spend this round on another plain Stage2 robustcal run. Do not change the decoder. Do not revive the failed additive latent-shift question-conditioning path.

## Current Baselines And Anchors

### LLaMA/V3 robust incumbent

Use this as the old baseline to beat:

```text
/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
```

Previously aligned V3 robust metrics:

```text
Clean n1000:             62.967
Blank n1000:             39.733
Shuffled n1000:          37.367
Wrong-image n1000:       38.900
Perturb mean:            60.189
POPE:                    77.100
GQA n500/canonical:      43.800 or 43.800 gate context
GQA trusted n1000:       43.700
EOS/max/prefix:          clean
```

### B1 no-architecture grounding control

Useful reference, not the target:

```text
Clean:                   63.267
Blank:                   39.433
Shuffled:                36.633
Wrong:                   37.833
Perturb mean:            63.556
POPE:                    75.000
GQA:                     44.600
```

B1 improved some grounding controls but regressed POPE, so it is not the current target.

### Qwen V9 materialized candidate

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Materialized checkpoint. Use for inference/eval only.

Important confirmed metrics:

```text
Clean n1000:             66.167
Blank n1000:             39.400
Shuffled n1000:          37.367
Wrong n1000:             37.967
Wrong n3000:             36.756
POPE n1000:              79.100
GQA canonical n500:      44.000
GQA trusted n1000:       43.100
```

### Qwen Stage1B2350 training anchor

Use this for training continuations and checkpoint/scale experiments:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Original unscaled metrics from V9:

```text
Clean:                   66.233
Blank:                   39.333
Shuffled:                37.567
Wrong:                   38.167
Mild blur:               67.667
Center crop 90:          66.067
Translate 5 pct:         66.733
Perturb mean:            66.822
POPE:                    79.200
GQA n500:                44.200
```

### Nearby Qwen Stage1B anchors

Try to use any available checkpoints in this family:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2100
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2150
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2250
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2300
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2400
```

Some may not exist on the Modal volume. Verify before launch. In V9, checkpoint-2100 was not available on the Modal volume during one POPE fill-in attempt.

## New Success Targets

Since the goal is now to exceed the LLaMA/V3 incumbent clearly, use stronger targets than the original replacement gate.

### Primary ceiling targets

```text
Clean VQA n3000 seed42:       >= 66.0, ideally >= 67.0
Clean VQA seeds 42/43/44:     stable, no seed collapse
Blank n3000:                  <= 38.8
Shuffled n3000:               <= 36.9, ideally <= 36.5
Wrong-image n3000:            <= 37.5
Perturb mean n1000:           >= 66.5
POPE n1000:                   >= 79.0
GQA trusted n1000:            >= 44.5, ideally >= 45.0
EOS:                          >= 0.98
Max-token-hit:                <= 0.02
Assistant-prefix:             <= 0.01
Strict-clean gap:             <= 1.0, ideally 0.0
Leakage audit:                pass
```

### Minimal “still viable” floor

If a candidate improves GQA but does not hit the full ceiling target, it must at least satisfy:

```text
Clean n3000:                  >= 66.0
Blank n3000:                  <= 39.4
Shuffled n3000:               <= 37.0
Wrong n3000:                  <= 38.0
POPE n1000:                   >= 79.0 or no worse than V9 scale1.05 by >0.3
GQA trusted n1000:            > V3 robust same-slice GQA and >= 43.8
```

Do not promote a candidate that only improves GQA by damaging corrupted-image controls or POPE.

## Evaluation Protocol

Every candidate that passes a cheap screen must be evaluated with:

```text
VQA clean seed42 n1000, then n3000 if promising
VQA blank seed42 n1000, then n3000 if promising
VQA shuffled seed42 n1000, then n3000 if promising
VQA wrong-image same-answer-type seed42 n1000, then n3000 if promising
VQA mild blur n1000
VQA center crop 90 n1000
VQA translate 5 pct n1000
POPE n1000
GQA trusted n1000
strict/raw diagnostics
EOS/max-token/prefix metrics
top answer histograms
answer-kind rates
leakage audit
```

GQA n500 is now a smoke metric only. Use **GQA trusted n1000** for selection and promotion.

Do not tune only on the same GQA n1000 forever. If feasible, split GQA into:

```text
GQA-dev-search:    500 examples
GQA-dev-confirm:  500 examples
GQA-final:        separate 1000 examples if available
```

At minimum, record whether a GQA result came from search or confirmation.

## Experiment Batch A: No-Training Pareto Search

Run this first. V9 showed that connector scale alone was decisive, so explore this cheap space before additional training.

### A1. Fine-grained scale sweep on Stage1B2350

Use the original Stage1B2350 anchor, not the materialized scale checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Evaluate scale overrides:

```text
1.000
1.015
1.025
1.035
1.045
1.050
1.060
1.075
1.090
1.100
```

Minimum screen per scale:

```text
GQA trusted n1000
Shuffled n1000
Blank n1000
Clean n1000
```

For any scale that looks promising, run:

```text
Clean n3000
Blank n3000
Shuffled n3000
Wrong n3000
POPE n1000
Perturb suite n1000
```

Rationale: scale 1.05 fixed the old gates but fell on GQA n1000. There may be a narrow Pareto point around 1.015-1.10 that improves GQA while preserving controls.

### A2. Anchor x scale grid

Evaluate nearby Stage1B anchors with scale overrides.

Candidate anchors:

```text
2100
2150
2250
2300
2350
2400, if available
```

Scale values:

```text
1.000
1.025
1.050
1.075
1.100
```

Minimum screen:

```text
GQA trusted n1000
Shuffled n1000
Clean n1000
Blank n1000
```

Then full bundle for the top 3.

Rationale: checkpoint choice and scale likely interact. V9 found 2100 and 2350 both near-miss. A different anchor with the right scale may clear GQA and controls without new training.

### A3. Connector weight interpolation

Interpolate projector weights between nearby Stage1B checkpoints.

Candidate pairs:

```text
2100 -> 2350
2000 -> 2350
2150 -> 2350
2300 -> 2350
2350 -> 2500
```

Interpolation coefficients:

```text
alpha = 0.25, 0.50, 0.75
```

Definition:

```text
projector_interp = (1 - alpha) * projector_A + alpha * projector_B
```

Do not interpolate Qwen weights. Qwen is frozen. Only interpolate connector/projector weights.

For each interpolated connector, test scale:

```text
1.000
1.025
1.050
1.075
```

Rationale: the Stage1B trajectory showed tradeoffs among clean score, shuffled control, blank control, and GQA. Weight interpolation may find a smoother point than any saved checkpoint.

### A4. Materialize top candidates

If a candidate only works through an eval-time scale override, materialize it into checkpoint metadata and smoke-test it without override, as V9 did.

Use or extend:

```text
scripts/materialize_v9_scale_checkpoint.py
```

Output path convention:

```text
/checkpoints/pretrain-output/v10-qwen3-<anchor-or-interp>-scale<scale>-candidate/checkpoint-<id>-scale<scale>
```

Run at least:

```text
shuffled n100 smoke
clean n100 smoke
metadata inspection
```

before full confirmation.

## Experiment Batch B: Tiny Connector Repair

Only run Batch B after Batch A establishes the best available anchor/scale candidate.

Use the best Batch A candidate as the base. Prefer training from a real optimizer-resume checkpoint if continuing training. If the best candidate is materialized/inference-only, map it back to the corresponding training checkpoint and set the same scale behavior in metadata/config.

### B1. GQA + anti-shuffle connector-only continuation

Train:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
vision: frozen SigLIP2 So400m 384
Qwen base: frozen
trainable: connector only
image tokens: 128
connector output scale: initialize from best Batch A scale
steps: 50, 100, 150, 200
```

Mixture:

```text
70% original v3_grounding replay
15% GQA compositional direct-answer
10% shuffled/wrong-image contrastive negatives
5% POPE-style object presence/absence
```

Evaluate every checkpoint:

```text
GQA trusted n1000 first
Shuffled n1000 second
Clean n1000 third
Blank n1000 fourth
```

Run full bundle only if the candidate beats the Batch A base on GQA without damaging shuffled/blank.

Rationale: the old GQA-heavy continuation was too blunt. This mixture preserves most original grounding while nudging GQA and anti-shuffle behavior.

### B2. GQA contrastive image-dependence loss

For GQA/compositional examples, create negatives:

```text
correct image + question + answer
wrong image + same question + same answer
shuffled image + same question + same answer
```

Add margin objective:

```text
logp(answer | correct image, question)
>
logp(answer | negative image, question) + margin
```

Suggested starting parameters:

```text
lambda = 0.1
margin = 0.5
steps = 50, 100, 150, 200
```

Use this as connector-only Stage1B continuation, not ordinary Stage2 LoRA.

Rationale: the goal is to strengthen visual dependence without teaching a global "cannot determine" answer and without decoder answer-prior rewrites.

### B3. Low-risk scale-only training

If per-scale behavior appears close but unstable, train only a very small set of connector scale parameters:

```text
global visual scale
or
128 per-image-token scales
```

Freeze all other connector weights. Initialize to the best Batch A scale.

Train for:

```text
50-200 steps
```

on GQA + anti-shuffle mixture.

Rationale: if scale is the key Qwen lever, learn it minimally rather than retraining the full connector.

## Experiment Batch C: Minimal Spatial Connector Upgrades

Run Batch C if Batch A/B cannot push GQA trusted n1000 to at least 44.5 while preserving controls.

### C1. V3 Perceiver + zero-initialized 2D patch features

Keep V3 topology. Add 2D patch information before the Perceiver:

```text
SigLIP patch features
+ learned 2D patch position embedding
-> V3 Perceiver Resampler
-> 128 image tokens
-> Qwen
```

Constraints:

```text
No V4 global/local split.
No new token count.
No decoder change.
No ordinary Stage2.
2D path initialized to zero or near-zero so the model starts equivalent to V9.
Connector output scale initialized from best Batch A candidate.
```

Train:

```text
connector only
50, 100, 200, 400 steps
GQA + original grounding replay + anti-shuffle negatives
```

Primary metrics:

```text
GQA trusted n1000
Shuffled n3000
Clean n3000
Blank n3000
POPE
```

Rationale: the remaining bottleneck is compositional/spatial grounding. This is the cleanest real connector change.

### C2. Coordinate MLP position features

Alternative to learned 2D embedding table:

```text
input: x, y, x^2, y^2, x*y
small MLP -> position feature
add to SigLIP patch feature
```

Run this only if C1 is partially promising or unstable.

Rationale: coordinate MLP may generalize better under crops/perturbations than a learned table tied to one grid.

### C3. Low-probability spatial corruption training

Add gentle patch-level augmentation during connector continuation:

```text
small local patch dropout
patch masking
low-probability local patch shuffle within regions
small random crop
```

Do not use destructive global shuffle during clean training.

Rationale: the model should learn coherent spatial evidence rather than bag-of-objects priors.

Stop if clean or GQA degrades in the first 100-200 steps.

## Experiment Batch D: Query-Dependent Visual Routing

Run after or alongside Batch C if compute is available.

### D1. Query-conditioned scalar visual scale

Architecture:

```text
question embedding -> one bounded scalar visual scale
```

Initialize to the best fixed scale, e.g.:

```text
scale = 1.05
```

Bound the scale range:

```text
0.95 <= scale <= 1.15
```

Rationale: different question types may need different visual-token strength. This is far safer than the failed additive latent-shift architecture.

### D2. Query-conditioned per-token scale

Architecture:

```text
question embedding -> 128 bounded scale factors
visual_token_i *= scale_i
```

Initialize all scale factors to the best fixed scale.

Rationale: different Perceiver latents may specialize. Question-conditioned per-token scale may improve GQA while preserving VQA/POPE.

### D3. Neutral query-conditioned patch selector

Do not reuse the failed additive latent-shift A1. Use patch routing:

```text
question embedding + SigLIP patch features
-> soft residual patch weights
-> weighted patch features
-> V3 Perceiver
-> 128 image tokens
```

Initialize to neutral:

```text
weighted_patch = patch * (1 + 0)
```

No hard top-k initially.

Train with:

```text
GQA + original grounding replay + anti-shuffle objective
```

Rationale: GQA often needs specific objects, attributes, and relations. Query-conditioned patch routing may help the connector focus on relevant evidence.

## Experiment Batch E: Repeated Visual Access In Qwen

Only run this if B/C/D fail to push GQA above the ceiling target.

### E1. Gated visual cross-attention adapters

Add small gated cross-attention modules in selected Qwen layers:

```text
visual memory = 128 connector tokens
Qwen hidden states cross-attend to visual memory
gate initialized to 0
```

Candidate layers:

```text
12, 18, 24, 30
```

Training:

```text
Qwen base frozen
connector frozen or lightly trainable
cross-attn adapters trainable
possibly attention-only low-rank LoRA
```

Rationale: prefix-only image insertion may be enough for VQAv2/POPE but insufficient for compositional GQA. Cross-attention lets the decoder revisit visual memory during answer generation.

This is higher risk and higher implementation cost. Do not start here.

## Analysis Work Required

### GQA error taxonomy

For every serious candidate and for the current V9 scale1.05 checkpoint, produce pairwise GQA comparisons:

```text
Qwen correct, V3 wrong
V3 correct, Qwen wrong
scale1.00 correct, scale1.05 wrong
scale1.05 correct, scale1.00 wrong
candidate correct, scale1.05 wrong
scale1.05 correct, candidate wrong
```

Categorize errors if metadata permits:

```text
relation
spatial
attribute
counting
object identity
color
comparison
logical/compositional
```

If GQA failure is relation/spatial-heavy, prioritize Batch C. If attribute/object-heavy, prioritize Batch D. If counting-heavy, adjust number/count data.

### Answer distribution checks

For each top candidate:

```text
top cleaned answers
top raw answers
answer-kind rates by question type
yes/no on non-yes/no
number distribution
blank/shuffled/wrong answer histograms
```

Candidates that improve GQA by collapsing answer distribution are invalid.

### Leakage audits

Run leakage audits for every final candidate and every new dataset component used in training. Treat exact/raw overlap as blocker. Numeric-only collisions require inspection and documentation.

## Stop Rules

Stop a candidate branch if any of the following happens:

```text
GQA trusted n1000 falls below V9 scale1.05 by >0.5
Shuffled n1000 rises above 38.0
Blank n1000 rises above 40.0
Clean n1000 falls below 65.0
POPE falls below 78.0
EOS < 0.98
Max-token-hit > 0.02
Assistant-prefix > 0.01
W&B active alerts persist after a short watch window
Checkpoint metadata does not match run label
Connector scale not recorded in metadata
```

For cheap screens, evaluate the active failure metric first. For example, if a candidate is primarily trying to fix GQA, run GQA first, then shuffled/blank. If it fails the first active gate badly, stop before spending on full VQA/POPE.

## What Not To Do

Do not:

```text
Run another ordinary robust semantic Stage2 and expect it to solve GQA.
Use full MLP LoRA Stage2 as the first repair lever.
Train generic Stage1A longer.
Blindly train Stage1B longer without a new objective.
Change the decoder away from Qwen3-8B.
Relax the corrupted-image gates to declare victory.
Use the failed additive latent-shift question-conditioning method.
Claim 2D/spatial features failed unless Batch C is actually run.
Claim query patch routing failed unless Batch D3 is actually run.
Tune only on GQA n1000 without a held-out confirmation split.
Promote an eval-time scale override without materializing it into metadata.
```

## Recommended Execution Order

### First pass: highest expected value

1. **A1 fine-grained scale sweep on Stage1B2350.**
2. **A2 anchor x scale grid.**
3. **A3 connector interpolation + scale on top pairs.**
4. Full bundle for the top 3 no-training candidates.
5. Materialize any scale/interpolation winner into checkpoint metadata.

Rationale: scale already solved the original V9 gate. This is cheap and likely to find the next Pareto point.

### Second pass: if no-training search does not clear GQA

6. **B1 GQA + anti-shuffle connector-only continuation** from the best Batch A candidate.
7. **B2 GQA contrastive image-dependence loss** if B1 does not move GQA enough.
8. **B3 scale-only training** if scale remains the obvious lever.

Rationale: the best model is pre-Stage2, so repair the connector/objective rather than doing decoder LoRA.

### Third pass: real connector architecture

9. **C1 zero-initialized 2D patch features.**
10. **D1 query-conditioned scalar scale.**
11. **D2 query-conditioned per-token scale.**
12. **D3 neutral query-conditioned patch selector.**

Rationale: if GQA remains the bottleneck, add spatial and query-aware visual evidence pathways while keeping the successful V3/Qwen interface.

### Last resort

13. **E1 gated visual cross-attention adapters.**

Rationale: highest ceiling, but highest implementation and attribution risk.

## Promotion Decision

A V10 Qwen ceiling candidate should be promoted only if it satisfies:

```text
Clean n3000 seed42 >= 66.0 and preferably >= 67.0
Clean seeds 42/43/44 stable
Blank n3000 <= 38.8
Shuffled n3000 <= 36.9
Wrong n3000 <= 37.5
Perturb mean >= 66.5
POPE >= 79.0
GQA trusted n1000 >= 44.5, preferably >= 45.0
Generation hygiene clean
Leakage audit pass
Checkpoint materialized with all scale/connector metadata
```

A candidate that only passes the old replacement gate but does not improve GQA trusted n1000 should be recorded as viable but not as the Qwen ceiling result.

## Final Note

The most important lesson from V9 is that Qwen3 performance is governed by the multimodal interface, not by generic answer calibration. The strongest path is to stay near the Stage1B connector trajectory and improve:

```text
visual-token scale calibration
GQA/compositional connector alignment
spatial information
question-relevant visual routing
```

Do not let this regress into a broad LoRA/prompt-tuning sweep. The goal is a higher-ceiling Qwen visual connector.
