# V9 / Qwen Connection V2 Experiment Plan

**Date:** 2026-05-12  
**Audience:** execution agent with access to the AnyMAL working directory, checkpoints, Modal, W&B, and the V8 Qwen3 results summarized in the repository/artifact logs.

## Executive summary

The Qwen/Qwen3-8B decoder swap is no longer an integration problem. The model is mechanically integrated, trainable, and capable of outperforming the LLaMA-based V3 robust incumbent on clean VQAv2, perturbation robustness, and POPE. It is **not yet a completely viable replacement** because the final extended Qwen checkpoint failed two gates:

```text
shuffled-image absolute control
GQA compositional/generalization gate
```

The next phase should therefore be **Qwen Connection V2**, not another generic longer Qwen run. The goal is to keep Qwen3-8B and fix the connection between SigLIP2/V3 visual tokens and Qwen’s visual-evidence use.

The highest-value hypothesis is:

> Qwen3 learned the direct-answer VQA interface and object-level evidence, but the current V3 connector plus LLaMA-derived robust semantic-calibration recipe lets Qwen lean too much on answer priors under corrupted-image controls and does not preserve enough GQA-style compositional grounding.

Start with **recipe/objective repair from the best Qwen Stage1B near-miss checkpoint**, then move to connector V2 only if the cheaper recipe/objective fixes fail.

## Strict scope

Use **only**:

```text
Qwen/Qwen3-8B
anymal_v3-compatible 128-image-token interface unless an experiment explicitly modifies the connector
SigLIP2-So400m at 384px
frozen vision tower
Qwen base frozen unless an experiment explicitly trains small LoRA visual-intake adapters
```

Do **not** evaluate other decoders in this phase. Do **not** switch to a native VLM. Do **not** relax the incumbent gates. Do **not** re-run the exact same Qwen V8 recipe for more generic steps unless a new objective, connector, or adapter hypothesis is being tested.

## Current state to internalize

### LLaMA-based incumbent gates

Use the aligned current-cache V3 robust model as the primary incumbent.

```text
Label: V3 robust incumbent
Checkpoint: /checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
Architecture: anymal_v3
Connector: perceiver_resampler
Image tokens: 128
```

Primary metrics:

| Model | Clean | Blank | Shuffle | Wrong | Blank gap | Shuffle gap | Wrong gap | Mild blur | Center crop 90 | Translate 5 pct | Perturb mean | POPE | GQA | EOS | Max hit | Prefix | Strict-clean gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 robust incumbent | 62.967 | 39.733 | 37.367 | 38.900 | 23.233 | 25.600 | 24.067 | 59.900 | 60.200 | 60.467 | 60.189 | 77.100 | 43.800 | 1.000 | 0.000 | 0.000 | 0.000 |

These are the **minimum viable replacement gates** for Qwen V2:

```text
clean >= 62.967
blank <= 39.733
shuffled <= 37.367
wrong-image same-answer-type <= 38.900
perturb mean >= 60.189
POPE >= 77.100
GQA >= 43.800
EOS >= 0.98
max-token-hit <= 0.02
assistant-prefix <= 0.01 overall
strict-clean gap <= 1.0, ideally 0.0
leakage audit passes
```

A true Qwen upgrade should ideally clear a higher bar:

```text
clean >= 65.900
perturb mean >= 64.000
POPE >= 78.000 or at least >= 77.100
GQA >= 44.000
blank/shuffled/wrong all at or below incumbent caps
hygiene perfect or near-perfect
```

### Useful LLaMA-side control

B1 is not the target decoder path, but it is useful context for grounding/data effects.

```text
B1 no-architecture counterfactual control
Checkpoint: /checkpoints/finetune-output/v7-b1-v3-counterfactual-grounding-robustcal-acc16-bs4-lossscale003/checkpoint-100
Clean: 63.267
Blank: 39.433
Shuffled: 36.633
Wrong: 37.833
POPE: 75.000
GQA: 44.600
```

B1 showed that counterfactual grounding data can improve VQAv2 control gaps, but it regressed POPE. Do not blindly copy B1’s data policy into Qwen without tracking POPE.

### Qwen V8 final result

```text
Final Qwen V8 checkpoint:
/checkpoints/finetune-output/v8-qwen3-8b-v3-stage2-lora-from-stage1b1500-robustcal-800-lr1e5-loss003-20260510/checkpoint-800
```

| Model | Clean | Blank | Shuffle | Wrong | Blank gap | Shuffle gap | Wrong gap | Perturb mean | POPE | GQA | EOS | Max hit | Prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 V8 final ckpt800 | 65.900 | 38.467 | 39.167 | 38.100 | 27.433 | 26.733 | 27.800 | 64.844 | 78.000 | 42.600 | 1.000 | 0.000 | 0.000 |

Interpretation:

```text
Worked:
  clean VQA improved over incumbent
  perturbation mean improved
  POPE improved
  blank and wrong-image controls were acceptable
  generation hygiene was perfect

Failed:
  shuffled-image absolute control was too high
  GQA was too low
```

### Best Qwen Stage1B near-miss

This is the most important anchor for V9.

```text
Best Stage1B near-miss:
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350

Observed metrics:
clean:    66.233
blank:    39.333
shuffled: 37.567
wrong:    38.167
GQA500:   44.200
EOS:      1.000
max-hit:  0.000
prefix:   0.000
```

This checkpoint is not promotable because shuffled misses the incumbent cap by about 0.20. But it is the best available Qwen anchor because it already passes clean, blank, wrong-image, and GQA.

Also keep checkpoint 2100 available as a secondary anchor:

```text
Secondary Stage1B near-miss:
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2100

Observed metrics:
clean:    66.000
blank:    39.670
shuffled: 37.730
wrong:    38.200
GQA500:   44.000
```

### Critical Qwen scale lesson

Early Qwen Stage1A failed because visual connector outputs were far too large relative to Qwen token embeddings:

```text
bad initial connector/Qwen token RMS ratio: ~39x
healthy Stage1A ratio after fix: ~1.0
healthy Stage1B ratio: ~1.18-1.23
```

Keep the Qwen-specific scaling setup unless an experiment explicitly tests scale:

```text
connector output gate init: 0.0256
Stage 1 backward loss scale: 0.3
monitor connector/Qwen RMS ratio continuously
```

Do not run Qwen connector experiments without RMS diagnostics.

## Overall experiment strategy

Run the work in three batches.

```text
Batch 0: evaluate and analyze the best pre-Stage2 Qwen anchors.
Batch 1: fix Stage2 recipe/objective from checkpoint 2350.
Batch 2: repair Stage1B connector alignment from checkpoint 2350.
Batch 3: only if needed, implement Qwen connector V2 architecture.
```

If bandwidth is limited, focus first on **Batch 0 and Batch 1**. They are the highest expected-value path because Qwen is already a near-miss before Stage2, and the old Stage2 appears to damage shuffled control and GQA.

## Non-negotiable evaluation protocol

Every branch that is considered for promotion must produce:

```text
VQAv2 current-cache seed42 n=1000:
  clean
  blank
  shuffled
  wrong-image same-answer-type

Perturbations:
  mild blur
  center crop 90
  translate 5 pct

Second benchmarks:
  POPE
  GQA

Diagnostics:
  EOS
  max-token-hit
  assistant-prefix
  strict-clean gap
  top raw answers
  top cleaned answers
  answer-kind rates by answer type
  prediction samples with image IDs
  leakage audit
  checkpoint model_meta and connector_meta
```

Use n=100 probes only as cheap screens. A branch is not promotable from n=100.

For expensive branches, the recommended eval order is:

```text
1. blank n=1000 if blank was the active failure mode
2. shuffled n=1000 if shuffled was the active failure mode
3. clean n=1000
4. wrong-image n=1000
5. GQA
6. POPE
7. perturbation suite
```

If a branch misses an active control gate by more than 0.5 points on n=1000, stop that branch. If it misses by less than or equal to 0.3 points, it is a near-miss and should receive the rest of the core metrics before being rejected. Final promotion still requires the exact gates.

Use the existing launcher shape when applicable:

```bash
python3 scripts/launch_v8_eval_bundle.py \
  --checkpoint <CHECKPOINT> \
  --label "<LABEL>" \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --artifact-prefix <ARTIFACT_PREFIX> \
  --remote-dir /checkpoints/v9_qwen_v2_remote \
  --parallelism 5
```

Adapt the script name or arguments if the repository has renamed V8 utilities to V9 utilities. Preserve the same metrics and artifact schema.

## Batch 0: close the pre-Stage2 evidence

### Experiment 0A: full eval bundle for Stage1B checkpoint 2350, no Stage2

**Purpose:** determine whether the best Qwen state is before LoRA Stage2.

Checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Run full bundle:

```text
clean
blank
shuffled
wrong-image same-answer-type
mild blur
center crop 90
translate 5 pct
POPE
GQA
answer diagnostics
leakage audit
```

Decision:

```text
If checkpoint 2350 passes perturbation and POPE and only misses shuffled narrowly:
  use it as the primary Qwen V2 anchor.

If checkpoint 2350 already fails POPE or perturbation badly:
  the old Stage2 was providing important repairs, and Qwen V2 must preserve those repairs while fixing controls.
```

### Experiment 0B: full or partial eval bundle for Stage1B checkpoint 2100, no Stage2

Checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2100
```

Purpose: confirm whether the 2100 to 2350 region is generally GQA-good and control-near-miss, or whether 2350 is an isolated point.

Minimum eval:

```text
clean
blank
shuffled
wrong-image
GQA
POPE if the four VQA controls look competitive
```

### Experiment 0C: Stage1B2350 vs Stage2800 comparison analysis

Compare:

```text
Stage1B2350 no Stage2
Stage2 ckpt800 final
V3 robust incumbent
B1 control, for data-policy context
```

Produce pairwise files:

```text
2350_correct_stage2_wrong
stage2_correct_2350_wrong
2350_correct_v3_wrong
v3_correct_2350_wrong
```

Inspect:

```text
top answers on clean and shuffled images
answer-kind rates by ground-truth answer type
yes/no rate on non-yes/no questions
number-answer priors
GQA misses by answer category if available
examples where Stage1B2350 is correct and Stage2800 is wrong
examples where Stage2800 is correct and Stage1B2350 is wrong
```

Expected hypothesis:

```text
Stage2 buys direct-answer calibration and POPE/perturbation behavior, but damages some visual/compositional dependence.
```

## Batch 1: Qwen-specific Stage2 recipe/objective fixes

These are the first real repair experiments. Start from checkpoint 2350 unless Batch 0 strongly favors checkpoint 2100.

### Shared Stage2 settings

Base checkpoint by default:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Keep:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
vision tower frozen
connector frozen, unless experiment explicitly says otherwise
Qwen base frozen
left-padded generation
Qwen non-thinking chat template
128 contiguous <|image|> placeholders
```

Do **not** repeat the old full LoRA Stage2 recipe as the main experiment. Full-rank/full-target LoRA already failed to repair the gates.

### Experiment 1A: Stage2-lite attention-only LoRA

**Question:** Did full Stage2 LoRA damage shuffled control and GQA by giving Qwen too much capacity to rewrite answer priors?

Implement or expose flags for:

```text
LoRA target modules: q_proj,k_proj,v_proj,o_proj only
No MLP LoRA: no gate_proj, no up_proj, no down_proj
LoRA ranks: 16 first, then 8 if needed
Learning rates: 5e-6 and/or 1e-5
Stage2 loss scale: 0.03
Checkpoints: 50,100,200,400
```

Suggested first run:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 400 \
  --batch-size 4 \
  --learning-rate 5e-6 \
  --lora-learning-rate 5e-6 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350 \
  --freeze-connector \
  --lora-rank 16 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --preserve-checkpoints 50,100,200,400 \
  --use-wandb \
  --run-name v9-qwen3-stage2-lite-attn-r16-from-stage1b2350-lr5e6-loss003
```

If the flags do not exist, implement them. Do not silently fall back to the old full-target LoRA.

Success:

```text
clean >= 65.0
blank <= 39.733
shuffled <= 37.367
wrong <= 38.900
GQA >= 43.800
POPE >= 77.100
hygiene passes
```

Interpretation:

```text
If attention-only low-rank LoRA passes:
  Qwen V2 is mostly a Stage2 capacity/targeting fix.

If it preserves GQA but misses shuffled narrowly:
  combine with control-aware Stage2 or contrastive objective.

If it underfits clean badly:
  try rank 32 attention-only or 1e-5 LR before rejecting.
```

### Experiment 1B: control-aware Stage2 mixture

**Question:** Can the failing gates be fixed by adding explicit control examples to Stage2 without changing the connector?

Implement a dataset, suggested name:

```text
v9_qwen_controlaware_stage2
```

Initial mixture:

```text
70% existing v5_semantic_calibration_robust
10% shuffled-image counterfactual VQA
10% wrong-image same-answer-type counterfactual VQA
5% blank/corrupted-image examples
5% GQA-style compositional direct-answer examples
```

Use the same image corruptions or mismatches as the evaluation controls. Do not invent a mismatched corruption that does not match eval.

Prompt/label policy:

Clean VQA examples:

```text
Answer with only the final answer.
```

Corrupted-control examples:

```text
The image may be blank, corrupted, or mismatched. If the image does not contain enough reliable visual evidence, answer: cannot determine.
```

Target for corrupted examples:

```text
cannot determine
```

POPE-style object presence/absence examples, if included:

```text
Answer yes or no.
```

Do not let “cannot determine” become a frequent answer on normal VQAv2 clean eval. Track top answers carefully.

Training should use Stage2-lite attention-only LoRA first:

```text
rank 16
q/k/v/o only
lr 5e-6 or 1e-5
checkpoints 50/100/200/400
```

Success:

```text
shuffled drops below cap
GQA does not regress
POPE does not regress below 77.1
clean remains >= 65 if possible, definitely >= 62.967
```

Failure modes:

```text
clean drops hard -> counterfactual ratio too high or label policy too strong
POPE drops -> cannot-determine or absence policy is interfering with object decisions
blank passes but shuffled fails -> wrong/corrupted-image sampling is too weak
GQA drops -> too little compositional supervision or LoRA still over-calibrates priors
```

### Experiment 1C: contrastive answer-suppression Stage2

**Question:** Can we reduce confidence in the correct answer under corrupted/wrong images without teaching a universal “cannot determine” response?

This is probably the most principled fix for the exact failure. Implement if feasible.

For a clean VQA example:

```text
positive: (correct_image, question, correct_answer)
negative 1: (shuffled_image, question, correct_answer)
negative 2: (wrong_same_answer_type_image, question, correct_answer)
negative 3: (blank_image, question, correct_answer)
```

Objective:

```text
CE positive direct-answer loss
+ lambda * max(0, margin - logp_pos + logp_neg)
```

Start conservative:

```text
lambda = 0.1
margin = 0.5
```

Variants:

```text
lambda 0.1, margin 0.5
lambda 0.2, margin 0.5
lambda 0.1, margin 1.0
```

Use Stage2-lite attention-only LoRA from checkpoint 2350.

Success:

```text
shuffled decreases without clean collapse
GQA holds or improves
POPE holds
normal VQAv2 top answers do not show cannot-determine pollution
```

### Experiment 1D: GQA-preserving Stage2 mixture

**Question:** Is the GQA failure caused by VQA-only semantic calibration washing out compositional grounding?

Implement a Stage2 dataset, suggested name:

```text
v9_qwen_gqa_preserving_stage2
```

Initial mixture:

```text
60% existing v5_semantic_calibration_robust
20% GQA train or GQA-style direct-answer examples
10% VQA counterfactual controls
10% POPE-style object absence/presence
```

Use Stage2-lite attention-only LoRA first.

Success:

```text
GQA >= 44.0
shuffled <= 37.367
clean >= 65.0 preferred, >= 62.967 required
POPE >= 77.1
```

## Batch 2: Stage1B connector/objective repair

Run Batch 2 if Batch 1 does not produce a viable Qwen V2 checkpoint or if Batch 0 shows that the best Qwen state is pre-Stage2.

The purpose is to repair the failure at the connector-learning stage rather than relying on LoRA to fix it later.

### Experiment 2A: control-aware Stage1B continuation from checkpoint 2350

**Question:** Can a short connector-only continuation repair shuffled-image control using the right objective?

Base:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Dataset, suggested name:

```text
v9_qwen_controlaware_stage1b
```

Initial mixture:

```text
70% existing v3_grounding clean
10% shuffled-image counterfactual
10% wrong-image same-answer-type counterfactual
5% GQA-style compositional direct-answer
5% POPE-style object absence/presence
```

Train:

```text
connector only
Qwen base frozen
vision frozen
Stage 1 loss scale 0.3
preserve Qwen RMS diagnostics
steps 250 to 500
save every 50 or 100 steps
```

Evaluate:

```text
n=100 cheap probes every save
promote promising checkpoints to n=1000 blank/shuffled/clean/wrong
GQA for checkpoints that pass or near-pass controls
POPE for finalists
```

Success:

```text
shuffled <= 37.367
blank <= 39.733
wrong <= 38.900
clean >= 65.0 preferred, >= 62.967 required
GQA >= 43.8
```

### Experiment 2B: GQA-heavy Stage1B continuation

**Question:** Does compositional supervision need to be in Stage1B rather than late Stage2?

Dataset:

```text
v9_qwen_gqa_stage1b
```

Initial mixture options:

```text
80% v3_grounding
20% GQA train / GQA-style direct-answer
```

or, if controls are still too weak:

```text
70% v3_grounding
15% GQA
15% counterfactual controls
```

Train connector-only from checkpoint 2350 for 250 to 500 steps.

Success:

```text
GQA >= 44.0
shuffled does not rise
clean holds near 66
```

### Experiment 2C: Stage1B connector plus visual-intake LoRA

**Question:** Does Qwen need a small decoder-side visual-intake adapter to use visual prefix tokens properly?

This is more invasive than connector-only continuation, but still much smaller than changing the full decoder.

Train during Stage1B:

```text
connector trainable
Qwen attention-only LoRA trainable
Qwen base frozen
no Qwen MLP LoRA
LoRA rank 8 or 16
LoRA targets: q_proj,k_proj,v_proj,o_proj
```

Use the control-aware Stage1B mixture from Experiment 2A.

Compare directly against Experiment 2A:

```text
connector-only control-aware Stage1B
vs
connector + visual-intake LoRA control-aware Stage1B
```

Success:

```text
improves GQA and shuffled control without increasing blank/wrong
```

If this works and connector-only does not, the correct Qwen V2 recipe likely includes decoder-side visual-intake adaptation.

## Batch 3: Qwen connector V2 architecture

Do not start Batch 3 until Batch 1 and Batch 2 have enough evidence, unless spare GPUs are available. The current Qwen V3 connector is already close; architecture changes should be targeted and minimal.

### Experiment 3A: inference-time visual-token scale sweep

**Question:** Is the residual shuffled/blank failure sensitive to connector output scale?

Before changing architecture, evaluate checkpoint 2350 and Stage2800 with visual-token output scaling at inference:

```text
scale = 0.85
scale = 0.90
scale = 0.95
scale = 1.00
scale = 1.05
scale = 1.10
```

Metrics:

```text
clean
blank
shuffled
wrong
GQA
POPE if promising
```

Interpretation:

```text
If a lower scale drops shuffled below cap without hurting clean/GQA much:
  implement RMS-locked connector or trainable scale regularization.

If scale changes mainly reduce clean or do not affect shuffled:
  scale is not the main remaining lever.
```

### Experiment 3B: RMS-locked V3 connector

**Question:** Should Qwen visual-token scale be structurally controlled rather than only initialized correctly?

Connector V2 modification:

```text
SigLIP features
-> V3 Perceiver
-> projection to Qwen hidden size 4096
-> RMSNorm or per-token RMS normalization
-> learned scalar gate initialized near Qwen token RMS
-> optional RMS regularizer
```

Suggested regularizer:

```text
rms_loss = alpha * (log(connector_rms / qwen_token_rms))^2
alpha = 0.01 initially
```

Keep all other V3/Qwen settings unchanged.

Train either:

```text
zero-disruption continuation from checkpoint 2350, if implementation permits
```

or:

```text
fresh Stage1A/Stage1B with the known Qwen scale/gate setup
```

Start with continuation if possible.

### Experiment 3C: V3 plus explicit 2D patch-position features

**Question:** Do shuffled-image and GQA failures indicate missing spatial/compositional evidence?

Do **not** revive full V4 global/local architecture. Keep the V3 interface simple:

```text
SigLIP patch features + learned or sinusoidal 2D patch position features
-> same V3 Perceiver
-> 128 image tokens
-> Qwen hidden size
```

Keep:

```text
128 image tokens
6 Perceiver layers
16 heads
FF mult 4
direct projection to Qwen hidden size
Qwen RMS/gate fix
```

Preferred first version:

```text
zero-init 2D position embeddings
load existing checkpoint 2350 connector weights
continue Stage1B with control-aware objective
```

If continuation is not technically clean, run a fresh Stage1A/Stage1B only after the zero-init continuation route is ruled out.

Success:

```text
shuffled-image control improves
GQA improves or holds
clean does not drop below incumbent
```

### Experiment 3D: query-conditioned soft patch selector

**Question:** Does Qwen need question-relevant visual patch routing rather than static patch compression?

Do not repeat the failed V7 additive latent-shift design. That design was:

```text
pooled prompt embedding -> additive latent shift
```

It failed badly and is not the same as patch routing.

Use a neutral-initialized soft selector:

```text
question embedding q
patch features p_i
score_i = small MLP([p_i, q, p_i * q_projected])
weight_i = neutral residual weight near 1.0
weighted_patch_i = p_i * weight_i
weighted patches -> V3 Perceiver -> 128 tokens
```

Initialization requirement:

```text
selector gate initialized to 0 or weights initialized near 1.0
initial behavior should match existing V3 connector closely
```

Avoid hard top-k at first. Hard selection is more brittle and may destabilize Stage1A.

Train:

```text
continue from checkpoint 2350 if possible
otherwise fresh Stage1A/Stage1B with Qwen scale diagnostics
use control-aware Stage1B objective
```

Success:

```text
GQA improves
shuffled control improves
clean remains competitive
```

### Experiment 3E: gated visual cross-attention adapter, later only

This is the most invasive architecture lever. Run it only after the smaller connector/objective repairs fail.

Idea:

```text
visual memory = 128 V3 connector tokens
add gated cross-attention adapters to selected Qwen layers, e.g. layers 16, 24, 32
gates initialized to 0
train connector + small cross-attention adapters
Qwen base frozen
```

Why this may help:

```text
prefix-spliced visual tokens may be diluted through the decoder stack
cross-attention gives Qwen repeated access to visual evidence during answer generation
```

Why not first:

```text
larger code change
more confounds
harder checkpoint comparability
risk of overfitting or generation instability
```

## Prompt and placement diagnostics

These are cheap and can run in parallel with Batch 0/1.

Using checkpoint 2350 and Stage2800, test:

```text
A. current image block placement
B. image block after question but before answer
C. image block with explicit visual delimiters
D. shorter system prompt
E. grounding-biased system prompt
```

Example grounding-biased diagnostic prompt:

```text
Answer using the image. If the image does not provide enough reliable evidence, do not guess. Give only the final answer.
```

This is diagnostic only. Do not promote a checkpoint solely from a prompt trick unless it preserves clean VQA, GQA, POPE, and normal answer distributions.

Success signal:

```text
shuffled drops by ~1 point or more without clean/GQA collapse
```

If prompt placement substantially changes corrupted-image control, reflect that in the Qwen V2 prompt contract before retraining.

## Checkpoint-selection rules

Do not select checkpoints by internal loss alone. V8 showed that lower Stage1A/Stage1B/Stage2 loss often did not correlate with downstream control gates.

Use this order:

```text
1. Checkpoint completeness and metadata correctness
2. W&B health: no active alerts, no NaNs, no loss/grad spikes, clip fraction acceptable
3. Qwen RMS diagnostics for Stage1 branches
4. n=100 cheap VQA control probe, if used
5. n=1000 control gates
6. GQA and POPE
7. perturbation suite
8. leakage audit
```

A checkpoint can be diagnostic if it misses gates. A checkpoint is promotable only if it clears gates exactly.

## Metadata requirements

Every run must write or preserve:

```text
architecture
llm_backbone
llm_model_type
chat_template_family
image_placeholder_token and token ID
image_placeholder_count
connector type
connector layers/heads/ff_mult
image token count
question conditioning setting
2D position setting, if any
RMS/gate settings
Stage1 loss scale
Stage2 loss scale
LoRA rank
LoRA alpha
LoRA dropout
LoRA target modules
trainable parameter groups
base checkpoint path
training dataset mixture
corruption/control data policy
```

For Stage1 Qwen runs, log:

```text
connector_output_rms
qwen_token_embedding_rms
connector/qwen RMS ratio
connector output gate value
placeholder contract valid rate
grad norm
grad clip fraction
loss/grad spike flags
```

For Stage2 Qwen runs, log trainable sanity:

```text
adapter params: expected 0 unless explicitly enabled
LoRA params: expected value for rank/targets
connector trainable: false unless experiment explicitly trains it
vision trainable: false
Qwen base trainable: false
```

## Leakage policy

Run leakage audits for every finalist and for any new data mixture involving counterfactual controls, GQA, or POPE-style examples.

Minimum audit:

```text
VQAv2 eval image IDs vs all Stage1 and Stage2 training sources
POPE eval image IDs vs training sources
GQA eval image IDs vs training sources
exact string overlap
numeric image ID overlap
```

If any overlap is unexplained, the checkpoint is not promotable.

## What not to do

Do not:

```text
change decoder away from Qwen/Qwen3-8B
relax the shuffled or GQA gates
run the old full LoRA Stage2 recipe again and expect a different result
train longer on the same objective without a new hypothesis
select checkpoints by lowest loss
reuse the failed additive latent-shift question-conditioning design
start with a large cross-attention stack before cheaper levers
let cannot-determine become a normal clean-VQA answer
skip POPE after adding counterfactual/negative examples
skip GQA after any Stage2 calibration change
skip RMS diagnostics in Stage1
```

## Recommended execution order

### First focus: Batch 0 + Batch 1

This is the most important subset. It is cheaper and most directly addresses the observed failure.

1. Full eval Stage1B2350 no Stage2.
2. Eval Stage1B2100 enough to confirm anchor behavior.
3. Pairwise Stage1B2350 vs Stage2800 vs V3 robust.
4. Stage2-lite attention-only LoRA from checkpoint 2350.
5. Control-aware Stage2 from checkpoint 2350.
6. Contrastive answer-suppression Stage2 from checkpoint 2350 if feasible.
7. GQA-preserving Stage2 if GQA remains below gate.

### Second focus: Batch 2

Run if Batch 1 does not produce a viable checkpoint.

8. Control-aware Stage1B connector-only continuation from checkpoint 2350.
9. GQA-heavy Stage1B continuation from checkpoint 2350.
10. Connector + visual-intake attention-only LoRA Stage1B.

### Third focus: Batch 3

Run only after Batch 1/2 results clarify whether the problem is recipe/objective or connector structure.

11. Inference-time visual-token scale sweep.
12. RMS-locked V3 connector.
13. V3 + 2D patch-position features.
14. Query-conditioned soft patch selector.
15. Gated cross-attention adapter only if the smaller changes fail.

## Decision table

| Result pattern | Interpretation | Next action |
|---|---|---|
| Stage1B2350 full bundle passes everything except shuffled by <= 0.3 | Pre-Stage2 Qwen is the right anchor | Focus on Stage2-lite or contrastive repair only |
| Stage2-lite passes gates | Full LoRA over-calibrated Qwen | Promote Stage2-lite recipe and confirm larger slice |
| Control-aware Stage2 passes but POPE drops | Counterfactual policy too blunt | Reduce corrupted ratio or separate POPE/object-presence labels |
| Contrastive Stage2 passes | Best recipe fix | Promote contrastive objective as Qwen V2 recipe |
| Stage2 variants all fail but Stage1B2350 remains near-pass | Repair belongs in Stage1B connector objective | Run Batch 2 |
| Control-aware Stage1B continuation passes | Qwen V2 is a data/objective fix | Use connector-only continuation plus minimal Stage2-lite |
| Connector + visual-intake LoRA passes and connector-only does not | Qwen needs decoder-side visual intake adaptation | Make visual-intake LoRA part of Qwen V2 |
| 2D position features improve shuffled/GQA | Missing spatial evidence was key | Promote V3+2D Qwen connector V2 |
| Query-conditioned patch selector improves GQA/control | Patch routing was key | Promote neutral-initialized selector connector |
| All smaller levers fail | Prefix-splice interface may be limiting | Consider gated cross-attention adapter |

## Final promotion requirements

A Qwen Connection V2 checkpoint can be called a viable core decoder replacement only if it clears all minimum gates:

```text
clean >= 62.967
blank <= 39.733
shuffled <= 37.367
wrong-image same-answer-type <= 38.900
perturb mean >= 60.189
POPE >= 77.100
GQA >= 43.800
EOS >= 0.98
max-token-hit <= 0.02
assistant-prefix <= 0.01
strict-clean gap <= 1.0
leakage audit passes
metadata complete
W&B health clean
```

A Qwen V2 checkpoint can be called an actual upgrade only if it additionally shows most of:

```text
clean >= 65.900
perturb mean >= 64.000
POPE >= 78.000 or materially above incumbent
GQA >= 44.000
all corrupted-image controls at or below incumbent caps
strong pairwise wins over V3 robust without answer-prior collapse
```

After a checkpoint clears the n=1000 gates, run:

```text
VQAv2 clean n=3000 confirmation
VQAv2 corrupted controls n=3000 if the n=1000 control margins are <= 0.5 points
all clean seeds 42/43/44 if promoting as new default
```

Do not call Qwen V2 complete until the exact gates clear. Near-misses are useful diagnostics, not replacements.
