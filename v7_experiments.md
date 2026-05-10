# V7 Architecture Grounding Experiment Plan

Date: 2026-05-10

Audience: an execution agent with access to the working directory, Modal, W&B, checkpoints, GPUs, and the current AnyMAL codebase.

This is a standalone plan. It assumes no access to prior planning context beyond this document.

---

## Bottom line

The next phase should start real architecture-grounding experiments now.

Do not wait for every promotion-grade benchmark sweep before changing architecture. The prior V6 campaign already falsified the simple claim that the legacy V4 spatial connector uniquely caused the low-50s VQAv2 result. The strongest current model is V3 with the robust semantic-calibration recipe, and the right next question is:

> Can architecture changes improve visual grounding beyond V3 robust, rather than merely improving answer calibration or VQAv2 priors?

The key rule for this phase:

> Every architecture experiment must be evaluated against image-use controls and a no-architecture grounding baseline.

Architecture exploration can begin immediately. Promotion still requires larger benchmark confirmation, leakage audits, robustness checks, and independent benchmarks.

---

## Current experimental state

The completed V6 campaign compared legacy V4, V3, V1, and lean V4 under the modern robust semantic-calibration recipe.

| Candidate | Clean mean | Perturb mean | POPE | GQA | Interpretation |
|---|---:|---:|---:|---:|
| V4 incumbent | 52.067 | 52.083 | 69.5 | 35.0 | Historical incumbent |
| V6-R1b legacy V4 robust recipe | 52.489 | 52.083 | 69.9 | 37.2 | Recipe win on legacy V4 |
| V6-C1b V3 robust recipe | 61.078 | 60.108 | 77.1 | 43.8 | Current best controlled model |
| V6-C2b V1 robust recipe | 52.956 | 51.933 | 69.2 | 36.8 | Clean-band diagnostic, perturb miss |
| V6-A1 lean V4 robust transfer | 37.944 | 38.350 | 49.4 | 29.6 | Failed transfer |

V3 robust also passed the previous image-use gates:

```text
clean seed42:                 59.667
blank image:                  37.400
shuffled image:               33.600
wrong-image same-answer-type: 36.800
```

Interpretation:

1. The prior V4-family architecture story is falsified.
2. V3 was under-tested under the modern recipe.
3. Recipe and evaluation hygiene matter a lot.
4. Lean V4 does not inherit the recipe gain.
5. V3 robust is now the diagnostic platform.
6. The main remaining weakness is not clean VQAv2 accuracy. It is grounding margin.

---

## New objective

The goal is not simply to beat V3 robust on clean VQAv2.

The goal is to find whether an architecture can improve visual grounding relative to V3 robust.

Use these primary quantities:

```text
grounding gap blank   = clean accuracy - blank-image accuracy
grounding gap shuffle = clean accuracy - shuffled-image accuracy
grounding gap wrong   = clean accuracy - wrong-image same-answer-type accuracy
```

A good architecture should do at least one of the following:

1. Raise clean accuracy while keeping blank, shuffled, and wrong-image accuracy low.
2. Hold clean accuracy roughly constant while lowering blank, shuffled, and wrong-image accuracy.
3. Improve POPE and GQA while preserving VQAv2 hygiene.
4. Improve grounding gaps without answer-kind collapse.

A bad architecture may appear to improve clean VQAv2 while also improving blank-image or shuffled-image accuracy. That is not grounding. That is likely answer prior exploitation or calibration.

---

## What is required before architecture exploration

Only the following are required before or alongside architecture experiments:

1. Freeze V3 robust as the current diagnostic incumbent.
2. Lock the image-use evals for every future candidate:
   - clean image,
   - blank image,
   - shuffled image,
   - wrong-image same-answer-type,
   - POPE,
   - GQA diagnostic.
3. Run one larger VQAv2 confirmation slice for V3 robust, preferably in parallel with first architecture smoke runs.

Do not wait for all of the following before starting architecture exploration:

- full MME or MMBench,
- full VQAv2 val,
- exhaustive V3 versus V4 error analysis,
- multiple fresh Stage 1 seeds,
- V1 strict-splice cleanup,
- full Stage1B checkpoint-selection study.

Those are promotion requirements or analysis workstreams, not blockers for architecture experimentation.

---

## What is required before promotion

Promotion is stricter than exploration.

No new architecture should be called the new default unless it has:

1. Larger VQAv2 confirmation, not only a 1000-sample slice.
2. Clean seeds, at least 42, 43, and 44.
3. Perturbation suite:
   - resize up,
   - mild blur,
   - center crop 90 percent,
   - translate 5 percent.
4. Image-use controls:
   - blank image,
   - shuffled image,
   - wrong-image same-answer-type.
5. POPE or equivalent object hallucination benchmark.
6. GQA or equivalent compositional grounding benchmark.
7. Leakage audit for every eval source and training source.
8. Strict-clean parity.
9. EOS and max-token hygiene.
10. Answer-kind distribution checks.
11. At least one no-architecture baseline showing the same gain is not caused only by data or objective changes.

---

## Core evaluation protocol

Every candidate in this phase must produce a comparable eval bundle.

### Required VQAv2 artifacts

For first-pass architecture experiments, use the 1000-sample locked slice first:

```text
seed42 clean
seed42 blank image
seed42 shuffled image
seed42 wrong-image same-answer-type
seed42 mild blur
seed42 center crop 90 percent
seed42 translate 5 percent
```

For candidates that pass first-pass gates, add:

```text
seed43 clean
seed44 clean
larger VQAv2 clean slice
larger VQAv2 blank/shuffled/wrong-image slice if affordable
```

Each artifact must include full prediction samples.

Required fields:

```text
clean accuracy
strict accuracy
strict-clean gap
number accuracy
other accuracy
yes/no accuracy
EOS rate
max-token-hit rate
assistant-role prefix rate
top raw answers
top cleaned answers
predicted answer-kind by ground-truth answer type
image IDs
raw predictions
cleaned predictions
candidate checkpoint
candidate architecture
connector metadata
training dataset metadata
image transform metadata
```

### Required non-VQAv2 artifacts

First pass:

```text
POPE 1000-sample or current locked subset
GQA diagnostic slice
```

Promotion pass:

```text
expanded POPE
larger GQA balanced diagnostic slice
optional MME or MMBench only if local harness is trusted
```

---

## Primary incumbent

Use V6-C1b V3 robust as the baseline for all comparisons.

Expected metadata:

```text
architecture: anymal_v3
vision encoder: SigLIP2-So400m at 384px
connector: fixed-size Perceiver Resampler
image tokens: 128
connector projection: direct to LLaMA hidden size
Stage 2 recipe: robust semantic calibration
connector frozen during Stage 2
LLM base frozen with LoRA or QLoRA adapters
prompt style: training_chat
left-padded generation
```

The exact checkpoint path should be recovered from the completed V6 campaign artifacts. Do not infer the checkpoint from a run label alone. Verify `model_meta.json`, startup logs, W&B config, and checkpoint metadata.

---

## Experiment tracks

Run two tracks in parallel:

1. Architecture-grounding experiments.
2. No-architecture grounding baselines.

The no-architecture baseline is essential. Without it, improvements from architecture may actually be caused by counterfactual data, objective changes, or calibration.

---

# Track A: Architecture-grounding experiments

## A1: Prompt-conditioned V3 Perceiver

### Priority

Highest.

### Causal question

Can question-conditioned visual resampling improve grounding over fixed V3 Perceiver latents while preserving the 128-token image contract?

### Motivation

V3 robust is strong, but its visual resampler uses fixed learned latent queries. It may compress the image without knowing what the question asks. A prompt-conditioned resampler should route capacity toward question-relevant visual evidence.

Current V3 pattern:

```text
image patches -> fixed learned latent queries -> 128 image tokens -> LLaMA
```

A1 pattern:

```text
question tokens + image patches -> question-conditioned latent queries -> 128 image tokens -> LLaMA
```

### Design constraints

Keep these fixed:

```text
vision encoder: SigLIP2-So400m at 384px
image tokens: 128
LLM: LLaMA-3-8B-Instruct base
Stage 2 recipe: same robust semantic calibration as V3 robust
prompt/eval: left-padded training_chat
connector frozen during Stage 2 unless explicitly testing a Stage 2 connector-update variant
```

Only change the connector conditioning mechanism.

### Implementation sketch

Add a new connector class or mode, for example:

```text
question_conditioned_perceiver_resampler
```

Inputs:

```text
vision patch features: [B, N, Dv]
question token embeddings or pooled question embedding: [B, T, Dl] or [B, Dl]
```

Possible implementation options, ranked:

1. Use pooled question embedding to generate additive shifts to learned Perceiver latents.
2. Use question token cross-attention to condition latent queries before image cross-attention.
3. Use FiLM-style scale and shift on latent queries from question embedding.

Preferred first implementation:

```text
base_latents: learned [K, H]
question_summary: pooled final hidden representation or embedding average [B, H]
conditioned_latents = base_latents + MLP(question_summary).view(B, K, H)
conditioned_latents attend to image patches through the normal Perceiver stack
```

This is simple and should preserve most V3 behavior.

### Training plan

Stage 1 is required if the connector changes.

Use the V3 Stage 1 recipe as the starting point, not the V4 lean recipe.

Recommended fixed schedule:

```text
Stage1A: connector-only, fixed step count matching V3 baseline if known
Stage1B: connector-only grounding continuation, fixed checkpoint step
Stage2A: robust semantic calibration, connector frozen, LoRA-only
```

If time is constrained, run a shorter mechanical smoke first:

```text
Stage1A canary: 20 to 50 steps
Stage1A smoke: 200 to 300 steps
Stage1B smoke: 100 to 200 steps
Stage2A: 50 and 100 step checkpoints
```

Do not compare the smoke to final V3 robust as a promotion candidate. Use it only to detect whether the architecture is viable.

### Evaluation

First-pass required:

```text
VQAv2 seed42 clean
VQAv2 seed42 blank image
VQAv2 seed42 shuffled image
VQAv2 seed42 wrong-image same-answer-type
VQAv2 seed42 mild blur
POPE locked subset
GQA diagnostic slice
```

### Success condition

A1 is a strong architecture signal if:

```text
clean >= V3 robust - 0.5
blank gap > V3 robust blank gap by at least 2.0
shuffle gap > V3 robust shuffle gap by at least 2.0
wrong-image gap > V3 robust wrong-image gap by at least 2.0
POPE >= V3 robust
GQA >= V3 robust
strict-clean gap <= 1.0
EOS >= 0.98
max-token-hit <= 0.02
```

A1 is also valuable if clean accuracy is flat but grounding gaps and POPE/GQA improve.

### Failure modes

Mark A1 as failed or diagnostic-only if:

```text
clean drops by more than 3 points
blank/shuffle/wrong-image accuracy rises with clean accuracy
POPE drops by more than 2 points
GQA drops by more than 2 points
EOS or max-token hygiene fails
connector metadata does not match the planned architecture
```

---

## A2: Query-conditioned patch selector before V3 Perceiver

### Priority

Second. Run after A1 is mechanically healthy, or in parallel if compute is abundant.

### Causal question

Can routing question-relevant image patches before compression improve grounding more than conditioning the Perceiver latents alone?

### Motivation

The model may need to select relevant visual evidence before compressing hundreds of SigLIP2 patch features into 128 tokens. V2 showed that simply pushing more weak visual tokens into the LLM is not enough. A2 tests targeted routing instead of token-count growth.

A2 pattern:

```text
question embedding -> scores image patch tokens -> soft or hard selected patch features -> V3 Perceiver -> 128 image tokens -> LLaMA
```

### Design constraints

Keep final output contract fixed:

```text
128 image tokens
same placeholder count
same LLaMA hidden size
same Stage 2 robust recipe
same eval harness
```

### Implementation options, ranked

1. Soft patch gating.
2. Differentiable top-k or sparse attention.
3. Hard top-k selector.

Preferred first implementation:

```text
question_summary = pooled question representation
patch_scores = dot_or_mlp(question_summary, patch_features)
patch_weights = sigmoid or softmax scores
weighted_patch_features = patch_features * patch_weights
V3 Perceiver attends over weighted_patch_features
```

Avoid hard top-k in the first implementation unless the soft selector is clearly ineffective.

### Training plan

Same as A1:

```text
Stage1A connector-only
Stage1B connector-only grounding continuation
Stage2A robust semantic calibration with connector frozen
```

### Evaluation

Same as A1.

### Success condition

A2 is preferred over A1 if it improves at least two of:

```text
POPE
GQA
wrong-image grounding gap
shuffled-image grounding gap
other-type VQAv2 accuracy
number-type VQAv2 accuracy
```

while preserving clean VQAv2 within 1 point of V3 robust.

---

## A3: Dual-bank V3, global plus query-conditioned local

### Priority

Third. Run only after A1 or A2 shows a useful signal.

### Causal question

Was the global/local idea useful, but V4's static spatial implementation wrong?

### Motivation

V4 used static global/local spatial tokens and failed to establish a causal architecture win. A3 tests a cleaner version of the same intuition inside the V3 family:

```text
32 global fixed latents
96 question-conditioned local latents
total = 128 image tokens
```

This preserves the compact 128-token contract and V3-like simplicity while giving the model both scene context and question-targeted local evidence.

### Design constraints

```text
final image tokens: 128
split: 32 global fixed + 96 local question-conditioned
vision encoder: SigLIP2-So400m 384px
Stage 2: robust semantic calibration
connector frozen during Stage 2
```

### Critical control

Compare against A1:

```text
A1: all 128 latents question-conditioned
A3: 32 global fixed + 96 local question-conditioned
```

If A3 does not beat A1 on grounding or robustness, do not keep the split.

### Success condition

A3 is useful if it beats A1 on at least one of:

```text
GQA
POPE
wrong-image grounding gap
other-type VQAv2 accuracy
number-type VQAv2 accuracy
```

with no clean VQAv2 or hygiene regression.

---

## A4: Gated visual cross-attention adapters in LLaMA

### Priority

Fourth, unless compute is abundant.

### Causal question

Is input-splicing image tokens into the prompt limiting visual use because visual evidence gets diluted through the decoder stack?

### Motivation

All current strong variants rely on image embeddings inserted into the input sequence. A more invasive architecture can give upper LLaMA layers repeated access to visual memory through gated cross-attention.

A4 pattern:

```text
V3 visual memory: 128 image tokens
frozen LLaMA base
LoRA as before
plus gated visual cross-attention adapters at selected upper layers
```

Suggested adapter layers:

```text
layers 16, 20, 24, 28
```

Alternative smaller smoke:

```text
layers 20, 28 only
```

### Design constraints

```text
base LLaMA remains frozen
vision encoder remains frozen
V3 visual memory starts as the visual source
adapter gates initialized near zero
Stage 2 robust semantic calibration
strict generation hygiene checks
```

### Why this is risky

A4 is more invasive than A1 to A3. It can destabilize generation, overfit answer priors, or make comparisons harder. Run A4 after A1/A2 unless there is enough compute to parallelize.

### Success condition

A4 is promising if:

```text
clean >= V3 robust
blank/shuffle/wrong-image gaps improve
POPE improves
GQA improves
strict-clean gap <= 1.0
EOS and max-token gates pass
```

A4 is not useful if it only improves clean VQAv2 while increasing blank or shuffled image accuracy.

---

# Track B: No-architecture grounding baselines

## B1: V3 robust plus counterfactual grounding data

### Priority

Must run in parallel with A1.

### Causal question

Can the grounding gap be improved by data and objective changes alone, without changing architecture?

### Motivation

If counterfactual training improves grounding as much as A1 or A2, the solution is likely data/objective rather than architecture. If A1 or A2 beats B1 under the same evals, that is a real architecture signal.

### Base architecture

Use unchanged V3 robust architecture.

```text
architecture: anymal_v3
vision encoder: SigLIP2-So400m 384px
connector: fixed V3 Perceiver Resampler
image tokens: 128
```

### Data additions

Add a counterfactual grounding mixture to the robust semantic-calibration recipe.

Recommended data types:

1. Correct image, normal answer.
2. Wrong image, same question, answer should change or become not answerable depending on task design.
3. Blank image, selected questions only, target should be not answerable.
4. Hard negative same-answer-type image.
5. POPE-style object absence questions.
6. Answer-type-balanced counterfactual examples.

### Label policy

Do not teach the model to say `unknown` for normal VQAv2 examples. Normal VQAv2 still expects direct answers.

Use explicit counterfactual labels only for counterfactual subsets.

Recommended labels:

```text
not enough information
cannot determine
no
```

For POPE-style object absence examples, direct `no` is appropriate when the object is absent.

For blank or wrong-image examples, use `not enough information` or `cannot determine` only when the task is explicitly framed as a grounding probe.

### Dataset mixture starting point

Start conservative:

```text
normal robust semantic calibration: 0.80
counterfactual grounding examples: 0.15
POPE-style object absence probes: 0.05
```

If this destabilizes normal VQAv2, reduce counterfactual examples:

```text
normal robust semantic calibration: 0.90
counterfactual grounding examples: 0.07
POPE-style object absence probes: 0.03
```

### Training

Use the same Stage 2 setup as V3 robust:

```text
connector frozen
LoRA-only Stage 2
same learning rate as robust recipe unless instability appears
same accumulation as robust recipe
max steps 100 for first pass
```

### Evaluation

Same as A1.

### Success condition

B1 is successful if:

```text
clean remains within 1 point of V3 robust
blank/shuffle/wrong-image accuracy decreases
POPE improves
GQA does not regress
strict-clean gap remains <= 1.0
EOS and max-token gates pass
```

### Interpretation

| Result | Interpretation |
|---|---|
| B1 improves grounding as much as A1/A2 | Data/objective likely explain the gain |
| A1/A2 beat B1 on grounding and POPE/GQA | Architecture signal |
| B1 improves blank/shuffle but hurts clean | Counterfactual mixture too strong or label policy misaligned |
| B1 improves POPE but not VQAv2 grounding controls | Useful hallucination calibration, not enough for general grounding |

---

## B2: V3 robust plus image-drop consistency regularization

### Priority

Optional after B1.

### Causal question

Can training explicitly separate answerable visual examples from unanswerable or corrupted image contexts?

### Idea

During Stage 2, include paired samples:

```text
correct image + question -> normal answer
blank or wrong image + same question -> not enough information / cannot determine
```

Add consistency or contrastive loss if supported. If not supported, implement as supervised paired examples first.

### Warning

This can hurt normal VQAv2 if overused because VQAv2 expects an answer even when dataset priors are strong. Keep the mixture small and evaluate clean accuracy frequently.

---

# Minimal next execution order

Run these in order, with A1 and B1 allowed to run in parallel.

## Step 1: Confirm and freeze V3 robust baseline

Tasks:

1. Locate the exact V6-C1b V3 robust checkpoint.
2. Verify metadata.
3. Rerun or fetch the existing eval bundle:
   - clean,
   - blank,
   - shuffled,
   - wrong-image same-answer-type,
   - perturbations,
   - POPE,
   - GQA.
4. Run one larger VQAv2 confirmation slice.

Do not block A1/B1 on the larger slice if the baseline metadata is already verified.

## Step 2: Launch B1 no-architecture counterfactual grounding baseline

This is the attribution control for architecture.

## Step 3: Launch A1 prompt-conditioned V3 Perceiver

This is the highest-priority architecture experiment.

## Step 4: Evaluate A1 and B1 with identical image-use controls

Compare:

```text
V3 robust
B1 counterfactual V3
A1 prompt-conditioned V3
```

Use the decision table below.

| Outcome | Next action |
|---|---|
| A1 improves grounding and B1 does not | Continue architecture path, run A2 |
| B1 improves grounding as much as A1 | Focus on data/objective, postpone more architecture |
| A1 improves clean but blank/shuffle also rise | Reject as grounding win |
| A1 hurts clean but improves POPE/GQA | Keep diagnostic, tune Stage 1 or combine with B1 |
| A1 fails mechanically | Fix implementation once, then try A2 if failure is specific to latent conditioning |
| A1 and B1 both fail | Revisit Stage 1, data labels, image-use evals, or cross-attention A4 |

## Step 5: Launch A2 if A1 is promising or inconclusive

Prioritize A2 if the failure analysis suggests patch routing is the missing piece.

## Step 6: Launch A3 or A4 only after A1/A2/B1 results

A3 if question-conditioned resampling works and you want to test a cleaner global/local split.

A4 if input-splice appears to be the bottleneck or if all resampler-only changes fail while V3 robust remains strong.

---

# Suggested command templates

The exact flags may need adjustment based on the current codebase. Verify available CLI arguments before launch.

## V3 robust baseline eval template

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint <V3_ROBUST_CHECKPOINT> \
  --candidate-label v3_robust_baseline \
  --candidate-architecture anymal_v3 \
  --no-include-baselines \
  --max-samples 1000 \
  --seed 42 \
  --prompt-style training_chat \
  --system-prompt 'Answer with only the final answer. Do not include role labels, explanations, or the word assistant. End after the answer.' \
  --prediction-samples 1000 \
  --output outputs/v7/vqa_eval_v3_robust_seed42_clean_leftpad.json
```

Run the same template for blank, shuffled, wrong-image, and perturbation modes using the existing V6 eval-bundle script if available:

```bash
python3 scripts/launch_v6_eval_bundle.py \
  --checkpoint <V3_ROBUST_CHECKPOINT> \
  --label v3_robust_baseline \
  --architecture anymal_v3 \
  --output-dir outputs/v7/v3_robust_baseline
```

If the script name or flags changed, inspect `scripts/launch_v6_eval_bundle.py` and use the same artifact schema as the completed V6 campaign.

## B1 Stage 2 training template

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage finetune \
  --dataset v7_semantic_calibration_counterfactual \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint <V3_CONNECTOR_ONLY_STAGE1_OR_STAGE1B_CHECKPOINT> \
  --freeze-connector \
  --use-wandb \
  --run-name v7-b1-v3-counterfactual-grounding-robustcal-acc16-bs4-lossscale003
```

Important: B1 should use the same connector-only V3 base that produced V3 robust. Do not use a checkpoint that already contains the previous robust Stage 2 LoRA unless the intent is continued finetuning and the run is labeled as such.

## A1 Stage 1 template

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage pretrain \
  --dataset v4_grounding \
  --v3-connector-type question_conditioned_perceiver_resampler \
  --pretrain-image-tokens 128 \
  --max-steps <FIXED_STAGE1A_STEPS> \
  --batch-size <MATCH_V3_BASELINE> \
  --learning-rate <MATCH_V3_BASELINE> \
  --use-wandb \
  --run-name v7-a1-stage1a-question-conditioned-v3-perceiver-128tok
```

## A1 Stage 2 template

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint <A1_CONNECTOR_ONLY_STAGE1B_CHECKPOINT> \
  --freeze-connector \
  --use-wandb \
  --run-name v7-a1-question-conditioned-v3-perceiver-robustcal-acc16-bs4-lossscale003
```

If loss or generation hygiene is unstable, run one stabilization variant:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 5e-6 \
  --lora-learning-rate 5e-6 \
  --finetune-loss-scale 0.01 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint <A1_CONNECTOR_ONLY_STAGE1B_CHECKPOINT> \
  --freeze-connector \
  --use-wandb \
  --run-name v7-a1b-question-conditioned-v3-perceiver-robustcal-lr5e6-lossscale001-acc16
```

---

# Early-stop and failure policy

## Training health

Stop or mark diagnostic-only if:

```text
W&B active alerts are nonempty after the warmup window
recent_loss_spikes remain active
recent_grad_spikes remain active
gradient clipping is nonzero and rising
checkpoint artifact is missing
validation loss is missing at scheduled checkpoint
model metadata is inconsistent with run label
```

## Generation hygiene

Stop or reject if:

```text
strict-clean gap > 1.0
assistant-role prefix rate > 0.01 overall
EOS < 0.98
max-token-hit > 0.02
predicted yes/no on non-yes/no questions > 0.05
```

## Grounding failure

Reject as a grounding win if:

```text
clean improves but blank-image accuracy also improves materially
clean improves but shuffled-image accuracy also improves materially
clean improves but wrong-image accuracy also improves materially
POPE regresses while VQAv2 improves
GQA regresses while VQAv2 improves
```

Use grounding gaps, not clean VQAv2 alone.

---

# Analysis required after first results

For each candidate, summarize:

```text
clean accuracy
blank-image accuracy
shuffled-image accuracy
wrong-image accuracy
blank grounding gap
shuffle grounding gap
wrong-image grounding gap
POPE
GQA
number accuracy
other accuracy
yes/no accuracy
strict-clean gap
EOS
max-token-hit
top 20 answers
top raw answers
predicted answer-kind by ground-truth answer type
```

Also produce paired comparison files for:

```text
A1 correct, V3 robust wrong
V3 robust correct, A1 wrong
B1 correct, V3 robust wrong
V3 robust correct, B1 wrong
A1 correct, B1 wrong
B1 correct, A1 wrong
```

The key qualitative questions:

1. Are improvements concentrated in visually grounded object, attribute, count, or spatial questions?
2. Are blank/shuffle failures mostly common priors such as `yes`, `no`, `1`, `2`, colors, or common objects?
3. Does the candidate reduce wrong-image confidence?
4. Does the candidate improve `other` and `number`, or only yes/no?
5. Does POPE improve for object absence as well as object presence?
6. Does GQA improve in relation, attribute, and counting categories?

---

# Do not do next

Do not run the lean V4 no-2D, compact96, or local192 grid. Lean V4 failed the prerequisite transfer.

Do not revive DeepStack-lite as the default path.

Do not introduce vision-encoder finetuning yet.

Do not spend many runs tuning tiny robust-augmentation variants before testing architecture-grounding and counterfactual baselines.

Do not call a clean VQAv2 bump an architecture win without blank, shuffled, wrong-image, POPE, and GQA evidence.

Do not compare a V1 prepend-mode interface to V3 or V4 splice-mode interfaces as a clean architecture ablation.

---

# Promotion rubric for this phase

## Diagnostic architecture signal

A candidate earns this if:

```text
it beats V3 robust on at least two grounding gaps
it ties or beats V3 robust on POPE or GQA
it keeps clean VQAv2 within 1 point of V3 robust
it passes generation hygiene
it beats or clearly differs from B1 no-architecture grounding baseline
```

## Strong architecture signal

A candidate earns this if:

```text
clean VQAv2 >= V3 robust
all three grounding gaps improve by at least 2 points
POPE improves
GQA improves
perturbation mean ties or improves
strict-clean gap <= 1.0
leakage audit passes
B1 does not explain the same gain
```

## Promotion candidate

A candidate earns this only after:

```text
larger VQAv2 slice confirms the gain
clean seeds 42/43/44 pass
the perturbation suite passes
expanded POPE passes
larger GQA passes
leakage audit passes
image-use controls pass
W&B history is green
checkpoint metadata is verified
```

---

# One-sentence mandate

Start architecture work now, but judge it on grounding gap, POPE, GQA, and image-use controls against V3 robust and a no-architecture counterfactual baseline, not on clean VQAv2 alone.
