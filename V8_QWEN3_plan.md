# V8 Qwen3-8B Compute-Matched and Extended Core Decoder Replacement Plan

**Date:** 2026-05-10  
**Audience:** a new execution agent with access to the AnyMAL working directory, Modal/GPU training, W&B, checkpoint volumes, current evaluation scripts, and the existing V7/V8 artifacts.  
**Purpose:** restart the Qwen3-8B decoder-replacement work from the prior integration checkpoint, but remove the shortcut that made the last result non-decisive. The final question is not whether Qwen3 can be wired into AnyMAL. That already passed. The final question is:

> If the LLaMA-family decoder is replaced with **Qwen/Qwen3-8B**, while keeping the successful V3 visual architecture and robust semantic recipe, and while giving Qwen3 at least the same training exposure as the best LLaMA-based model and then longer fine-tuning, can Qwen3 become a completely viable or improved core decoder replacement?

This file is self-contained. Do not assume access to the prior chat. Do not ask the previous V8 agent to continue this work. Start from the repository and artifacts described here.

---

## 0. Non-negotiable directive

Use only:

```text
Qwen/Qwen3-8B
```

Do not test another LLM. Do not run a LLaMA fallback. Do not run a native VLM baseline. This is a Qwen3-8B replacement campaign only.

The previous Qwen3 work was an **integration pass**, not a fair viability attempt. The previous training schedule was:

```text
Stage 1A: 300 optimizer steps
Stage 1B: 400 optimizer steps
Stage 2: 100 optimizer steps
```

That is not enough to conclude anything about Qwen3 as a replacement. The new agent must do the following:

1. **Use the prior Qwen3 implementation work.** Do not re-implement the integration unless something is broken.
2. **Run a short diagnostic/hyperparameter phase only to choose stable Qwen3-specific training hyperparameters.** These diagnostic runs are not viability candidates.
3. **After the best hyperparameters are chosen, run a final locked training line.** This line must train Qwen3 for at least the same Stage 1A, Stage 1B, and Stage 2 step/sample exposure as the best LLaMA-based model.
4. **For fine-tuning, go beyond the LLaMA checkpoint length.** If the best LLaMA model used Stage 2 checkpoint `N`, Qwen3 must evaluate checkpoint `N` and then continue to later checkpoints, at minimum `2N` and `4N`, and `8N` if still improving or still below viability.
5. **For pretraining, do not stop at a smoke budget.** Train at least as long as the LLaMA-based pretraining stages. If Qwen3 is still not aligned at the matched exposure but Stage 1 probes are improving, continue pretraining beyond the matched schedule until it is better on the preregistered Stage 1 probes or until two consecutive extended checkpoints stop improving.
6. **Final gates are exactly the gates set by the best LLaMA-based V3 robust model.** No relaxed Qwen3-specific standard.

The purpose is to measure a fair decoder swap with comparable or greater compute, not to save money.

---

## 1. Current incumbent to beat

The current best controlled AnyMAL model is the V3 robust LLaMA-based model.

```text
Label: v7_v3_robust_currentcache
Checkpoint: /checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
Architecture: anymal_v3
Vision tower: SigLIP2-So400m at 384px
Connector: perceiver_resampler
Image tokens: 128
Connector layers: 6
Connector heads: 16
Connector FF mult: 4
Question conditioning: none
Decoder: incumbent LLaMA-family 8B decoder
Stage 2 recipe: robust role-clean semantic calibration
```

Current-cache aligned 1000-sample VQAv2 seed-42 bundle:

| Metric | Incumbent value |
|---|---:|
| clean | 62.967 |
| blank image | 39.733 |
| shuffled image | 37.367 |
| wrong image, same answer-type | 38.900 |
| clean minus blank gap | 23.233 |
| clean minus shuffled gap | 25.600 |
| clean minus wrong-image gap | 24.067 |
| POPE | 77.100 |
| GQA | 43.800 |
| EOS | 1.000 |
| max-token-hit | 0.000 |
| assistant-prefix | 0.000 |

Larger confirmation:

| Eval | Incumbent value |
|---|---:|
| VQAv2 seed42 n=3000 clean | 63.556 |

Historical V6 perturbation reference for the same V3 robust family:

| Metric | Incumbent value |
|---|---:|
| perturbation mean | 60.108 |

If current-cache aligned perturbation artifacts for this exact incumbent are present in the working directory, use those exact current-cache values instead of the historical `60.108` number. Do not mix cache regimes without labeling them.

---

## 2. Context from V7 and the prior V8 shortcut

### 2.1 V7 context

B1 is useful as a no-architecture grounding control, but it is not the target of this plan.

```text
Checkpoint: /checkpoints/finetune-output/v7-b1-v3-counterfactual-grounding-robustcal-acc16-bs4-lossscale003/checkpoint-100
Architecture: anymal_v3
Connector: same V3 perceiver_resampler
```

B1 slightly improved VQAv2 grounding gaps but regressed POPE. Do **not** mix B1 counterfactual data into the main Qwen3 decoder-swap run. The point here is to change the decoder while preserving the successful V3 recipe as closely as possible.

The A1 question-conditioned connector failed badly:

```text
Connector: question_conditioned_perceiver_resampler
Conditioning: pooled_prompt_embedding_additive_latent_shift
Result: clean, grounding gaps, POPE, and GQA regressed heavily.
```

Do **not** use A1 here. The Qwen3 campaign must stay on the plain V3 `perceiver_resampler` connector.

### 2.2 What the prior V8 agent accomplished

The previous agent integrated Qwen/Qwen3-8B and proved the plumbing works.

Implemented and validated:

```text
--llm-backbone Qwen/Qwen3-8B support
Qwen3 cache path: /checkpoints/qwen3-8b
chat_template_family=qwen3_non_thinking
single-token <|image|> placeholder for Qwen3
strict 128-placeholder prompt contract
left-padded generation
Qwen stop-token handling for EOS and <|im_end|>
inputs_embeds generation path
LoRA target validation
VQA / POPE / GQA evaluation support for Qwen checkpoints
```

Stage 0 smoke passed:

```text
Model: Qwen/Qwen3-8B
model_type: qwen3
hidden_size: 4096
layers: 36
attention heads: 32
KV heads: 8
image placeholder token: <|image|>
image placeholder id: 151669
128 placeholders contiguous
input_ids vs inputs_embeds last-logit parity: max abs diff 0.0
generation path: HF .generate(inputs_embeds=...)
LoRA targets found: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Prior short-run checkpoints:

```text
Stage 1A short run:
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-fixed300/checkpoint-300

Stage 1B short run:
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300/checkpoint-400

Stage 2 short run:
/checkpoints/finetune-output/v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
```

Prior short-run result:

| Model | Clean | Blank | Shuffle | Wrong | Blank gap | Shuffle gap | Wrong gap | POPE | GQA | EOS | Max hit | Prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent V3 robust | 62.967 | 39.733 | 37.367 | 38.900 | 23.233 | 25.600 | 24.067 | 77.100 | 43.800 | 1.000 | 0.000 | 0.000 |
| Qwen3 short-run ckpt100 | 55.800 | 38.933 | 39.333 | 39.633 | 16.867 | 16.467 | 16.167 | 74.500 | 35.200 | 1.000 | 0.000 | 0.000 |

Interpretation:

```text
Integration: pass.
Generation hygiene: pass.
Decoder-swap viability: fail.
Fair Qwen3 replacement attempt: not yet run.
```

The failure pattern was weak visual-to-decoder alignment, not answer-format collapse. Therefore the next campaign must focus on full connector alignment and full LoRA calibration, not on prompt cleanup.

---

## 3. Hard constraints for the final replacement line

### 3.1 Decoder

Use exactly:

```text
Qwen/Qwen3-8B
```

Expected Qwen3 metadata:

```text
llm_model_type: qwen3
llm_hidden_size: 4096
chat_template_family: qwen3_non_thinking
image_placeholder_token: <|image|>
image_placeholder_count: 128
padding_side_for_generation: left
```

### 3.2 Visual architecture

Keep the main line fixed:

```text
Architecture: anymal_v3
Vision tower: SigLIP2-So400m at 384px
Vision tower training: frozen
Connector: perceiver_resampler
Image tokens: 128
Connector layers: 6
Connector heads: 16
Connector FF mult: 4
Projection: direct to decoder hidden size 4096
Question conditioning: none
Placeholder contract: exactly 128 contiguous <|image|> placeholders
```

Do not change the vision tower, token count, connector family, or question conditioning in the final comparable line.

### 3.3 Training ownership

| Stage | Trainable modules | Frozen modules |
|---|---|---|
| Stage 1A | connector only | SigLIP2, Qwen3 base |
| Stage 1B | connector only | SigLIP2, Qwen3 base |
| Stage 2 | Qwen3 LoRA adapters only | SigLIP2, connector, Qwen3 base |

Do not finetune the full Qwen3 base. Do not unfreeze the vision encoder. Do not train the connector during final Stage 2. A diagnostic run may violate this only if it is clearly labeled diagnostic and is not used as the final comparable decoder-swap result.

### 3.4 Recipe identity

The main result must answer:

```text
What happens if we keep the successful V3 recipe and swap only the core decoder to Qwen3, after fixing integration and scale issues and giving Qwen3 comparable or greater training exposure?
```

Therefore, do not mix in:

```text
B1 counterfactual grounding data
A1 question-conditioned connector
V4 spatial connector
DeepStack-lite
vision encoder finetuning
native multimodal Qwen path
```

---

## 4. Required first deliverable: incumbent schedule manifest

Before launching the final Qwen3 training line, reconstruct the exact training exposure that produced the incumbent V3 robust LLaMA-based model.

Create:

```text
outputs/v8_qwen3_full/schedule_manifest.json
```

It must include the incumbent and the planned Qwen3 exposure:

```json
{
  "incumbent_checkpoint": "/checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100",
  "incumbent_architecture": "anymal_v3",
  "incumbent_stage1a": {
    "run_name": "...",
    "checkpoint": "...",
    "dataset": "...",
    "optimizer_steps": 0,
    "effective_batch_size": 0,
    "num_gpus": 0,
    "gradient_accumulation": 0,
    "approx_training_examples_seen": 0,
    "learning_rate": "...",
    "warmup_steps": 0,
    "save_steps": "...",
    "eval_steps": "..."
  },
  "incumbent_stage1b": {
    "run_name": "...",
    "checkpoint": "...",
    "dataset": "...",
    "optimizer_steps": 0,
    "effective_batch_size": 0,
    "approx_training_examples_seen": 0,
    "learning_rate": "...",
    "warmup_steps": 0,
    "save_steps": "...",
    "eval_steps": "..."
  },
  "incumbent_stage2": {
    "run_name": "...",
    "checkpoint": "...",
    "dataset": "...",
    "optimizer_steps": 0,
    "effective_batch_size": 0,
    "approx_training_examples_seen": 0,
    "learning_rate": "...",
    "lora_learning_rate": "...",
    "loss_scale": "...",
    "warmup_steps": 0,
    "save_steps": "...",
    "eval_steps": "..."
  },
  "qwen3_hyperparameter_diagnostic_phase": {
    "allowed": true,
    "viability_candidate": false
  },
  "planned_qwen3_final_stage1a": {
    "optimizer_steps": 0,
    "approx_training_examples_seen": 0,
    "matched_or_exceeds_incumbent": true,
    "extended_checkpoints_planned": true
  },
  "planned_qwen3_final_stage1b": {
    "optimizer_steps": 0,
    "approx_training_examples_seen": 0,
    "matched_or_exceeds_incumbent": true,
    "extended_checkpoints_planned": true
  },
  "planned_qwen3_final_stage2": {
    "matched_step": 0,
    "extended_steps": [],
    "matched_or_exceeds_incumbent": true,
    "goes_beyond_incumbent": true
  }
}
```

How to reconstruct the incumbent exposure:

1. Inspect the incumbent checkpoint metadata and parent references.
2. Inspect W&B run records if run IDs are available.
3. Inspect `EXPERIMENTS.md`, `V6_CAUSAL_FALSIFICATION_PLAN.md`, prior run scripts, generated artifact metadata, and checkpoint `model_meta.json`, `training_args.json`, or equivalent files.
4. Search the checkpoint volume and working tree:

```bash
find /checkpoints -iname '*v6-c1b-v3*' -o -iname '*v3*robust*' -o -iname '*roleclean*semanticcal*'
grep -R "v6-c1b-v3-roleclean-semanticcal-robust" -n . || true
grep -R "v3.*semanticcal.*robust" -n . || true
```

5. If exact metadata is incomplete, use the strongest available evidence and record uncertainty explicitly. In ambiguous cases, the Qwen3 schedule must err upward, not downward.

Hard rule:

> A final Qwen3 checkpoint is invalid if its Stage 1A, Stage 1B, or Stage 2 sample exposure is lower than the incumbent exposure recorded in the manifest.

---

## 5. Two-phase execution structure

The campaign has two distinct phases.

### 5.1 Phase A: Qwen3 hyperparameter diagnostics

Purpose: choose stable Qwen3-specific hyperparameters and fix obvious alignment issues before the full final run.

Allowed diagnostic changes:

```text
Stage 1 learning rate
Stage 1 warmup
Stage 1 loss scale, if used
Stage 2 LoRA learning rate
Stage 2 loss scale
connector output RMS normalization or scalar output scale, if diagnosed by norm mismatch
small LayerNorm on connector outputs, if diagnosed by norm mismatch
checkpoint/eval interval
```

Not allowed in diagnostics if the result will later be compared as the main decoder swap:

```text
B1 counterfactual data
A1 question-conditioned connector
new visual token count
new vision tower
native Qwen multimodal path
full Qwen base finetuning
```

Diagnostic runs can be short. They may use budgets like 100-500 Stage 1 steps or 50-100 Stage 2 steps. But diagnostic runs must be labeled as diagnostics and may not be promoted.

Select hyperparameters by:

```text
training stability
validation loss trend
connector-output-vs-Qwen-embedding norm match
fixed tiny VQA probe behavior
blank/shuffle probe behavior
absence of prompt/role leakage
no active W&B alerts
```

Do not choose hyperparameters only by peeking at final VQAv2/POPE/GQA test bundles.

### 5.2 Phase B: final locked compute-matched and extended Qwen3 run

After hyperparameters are chosen, start the final comparable Qwen3 run. Prefer fresh connector initialization for the final line. Continuing from a diagnostic checkpoint is only allowed if the report proves that total exposure is tracked from step 0 and that no invalid checkpoint selection happened.

The final line must produce:

```text
Stage 1A matched checkpoint: >= incumbent Stage 1A steps/examples
Stage 1A extended checkpoint(s): beyond matched exposure if probes are still improving or if Qwen3 is not better on preregistered Stage 1 probes
Stage 1B matched checkpoint: >= incumbent Stage 1B steps/examples
Stage 1B extended checkpoint(s): beyond matched exposure if probes are still improving or if Qwen3 is not better on preregistered Stage 1 probes
Stage 2 matched checkpoint: same step/sample exposure as the best LLaMA Stage 2 checkpoint
Stage 2 extended checkpoints: at minimum 2x and 4x the best LLaMA Stage 2 step; 8x if still improving or still below viability
```

This means there are two different conclusions to report:

```text
compute-matched result: Qwen3 at the same recipe exposure as LLaMA
extended-ceiling result: Qwen3 after longer fine-tuning and, if needed, longer pretraining
```

Both matter. The compute-matched result answers fairness. The extended result answers how much better Qwen3 can get after integration and alignment issues are fixed.

---

## 6. Qwen3-specific alignment diagnostics

The previous failure looked like visual-to-decoder alignment weakness. Therefore all final Stage 1 runs must log these diagnostics at every eval interval:

```text
Qwen token embedding RMS
Qwen prompt embedding RMS
connector output RMS before final projection
connector output RMS after final projection
connector output mean/std per token
ratio: connector_output_RMS / Qwen_token_embedding_RMS
attention mask validity for 128 placeholders
supervised token rate
validation loss
fixed tiny VQA probe decoded answers
fixed tiny VQA probe answer-kind rates
fixed tiny blank/shuffle probe, at least 100 examples if cheap
```

A connector-scale fix is allowed only if diagnostics clearly show scale mismatch. If used, it must be included in the final locked hyperparameter manifest and trained for the full matched/extended schedule.

Allowed connector-scale fixes:

```text
output RMS normalization before final connector projection
learned scalar gate initialized to match Qwen token embedding RMS
connector output scale calibrated from a frozen batch before training
small LayerNorm on connector outputs before insertion
```

Do not use these to change the architecture question. They are permitted only to make the Qwen3 hidden-space alignment comparable to the LLaMA hidden-space alignment.

---

## 7. Stage 1A final training: full caption/alignment pretraining

Purpose: learn a stable connector from SigLIP2 visual features into Qwen3 hidden space.

Required properties:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
stage: pretrain
dataset: same Stage 1A alignment dataset family as incumbent V3, unless the manifest proves otherwise
image tokens: 128
trainables: connector only
Qwen3 base: frozen
vision tower: frozen
connector init: fresh random, unless continuation is explicitly justified
```

Step/sample budget:

```text
matched_stage1a_steps >= incumbent_stage1a.optimizer_steps
matched_stage1a_examples_seen >= incumbent_stage1a.approx_training_examples_seen
```

If the incumbent Stage 1A schedule cannot be reconstructed, use a conservative fallback:

```text
minimum matched Stage 1A: 3000 optimizer steps
save/evaluate: 500, 1000, 1500, 2000, 2500, 3000
```

Extended Stage 1A rule:

```text
If the matched Stage 1A checkpoint is still worse than the incumbent-derived Stage 1 probe or the Qwen3 probe metrics are still improving, continue Stage 1A beyond the matched checkpoint.
Suggested extensions: 4500 and 6000 steps.
If the incumbent exposure is greater than 3000, scale extensions proportionally: 1.5x and 2x the incumbent Stage 1A steps.
Stop Stage 1A extension only after either Qwen3 is better on the preregistered Stage 1 probes or two consecutive extended checkpoints show no improvement.
```

Command template:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset <INCUMBENT_STAGE1A_DATASET> \
  --pretrain-image-tokens 128 \
  --max-steps <MATCHED_OR_EXTENDED_STAGE1A_STEPS> \
  --batch-size 4 \
  --pretrain-save-steps <SAVE_INTERVAL_WITH_MILESTONES> \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1a-final-<steps>-<hp_tag>
```

Health gates:

```text
placeholder count exactly 128
no metadata mismatch
no NaNs
no persistent clipping or grad spikes
no active W&B alerts at the checkpoint used for Stage 1B
validation loss improving or plateaued, not exploding
connector norm diagnostics recorded
```

---

## 8. Stage 1B final training: full grounding pretraining

Purpose: teach the Qwen3-aligned connector direct visual grounding before Stage 2 LoRA calibration.

Required properties:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
stage: pretrain
dataset: same Stage 1B grounding dataset family as incumbent V3 / robust path
pretrain checkpoint: selected full Stage 1A matched or extended checkpoint
trainables: connector only
Qwen3 base: frozen
vision tower: frozen
```

Step/sample budget:

```text
matched_stage1b_steps >= incumbent_stage1b.optimizer_steps
matched_stage1b_examples_seen >= incumbent_stage1b.approx_training_examples_seen
```

If the incumbent Stage 1B schedule cannot be reconstructed, use a conservative fallback:

```text
minimum matched Stage 1B: 3000 optimizer steps
save/evaluate: 500, 1000, 1500, 2000, 2500, 3000
```

Extended Stage 1B rule:

```text
If the matched Stage 1B checkpoint is still worse than the incumbent-derived Stage 1B probe or the Qwen3 image-use probe metrics are still improving, continue Stage 1B beyond the matched checkpoint.
Suggested extensions: 4500 and 6000 steps.
If the incumbent exposure is greater than 3000, scale extensions proportionally: 1.5x and 2x the incumbent Stage 1B steps.
Stop Stage 1B extension only after either Qwen3 is better on the preregistered Stage 1B probes or two consecutive extended checkpoints show no improvement.
```

Command template:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset <INCUMBENT_STAGE1B_DATASET> \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-final-<stage1a_steps>-<hp_tag>/checkpoint-<stage1a_step> \
  --pretrain-image-tokens 128 \
  --max-steps <MATCHED_OR_EXTENDED_STAGE1B_STEPS> \
  --batch-size 4 \
  --pretrain-save-steps <SAVE_INTERVAL_WITH_MILESTONES> \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1b-final-<steps>-from-stage1a-<stage1a_step>-<hp_tag>
```

Do not select the Stage 1B checkpoint by hindsight from final VQA alone. Use the preregistered matched/extended milestones and health/probe criteria.

---

## 9. Stage 2 final training: matched and longer robust semantic calibration

Purpose: adapt the Qwen3 decoder behavior with LoRA while keeping the Qwen3-aligned visual connector frozen.

Required properties:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
stage: finetune
dataset: v5_semantic_calibration_robust, unless the incumbent manifest proves a different final robust recipe
pretrain checkpoint: full Qwen3 Stage 1B matched or extended checkpoint
trainables: Qwen3 decoder LoRA only
connector: frozen
vision tower: frozen
base Qwen3: frozen
chat_template_family: qwen3_non_thinking
answer labels: short role-clean direct answers only
```

Matched Stage 2 rule:

```text
Let N = optimizer step of the best LLaMA-based Stage 2 checkpoint.
For the current incumbent, this appears to be checkpoint-100, but the manifest must verify it.
The Qwen3 final Stage 2 run must evaluate checkpoint N.
```

Longer-than-LLaMA Stage 2 rule:

```text
After checkpoint N, continue and evaluate at minimum:
2N
4N

If Qwen3 has not passed viability by 4N and metrics are improving or stable, continue to:
8N

If Qwen3 passes viability at N, still continue to 2N and 4N to measure the Qwen3 ceiling.
If Qwen3 passes at 2N, still evaluate 4N.
If Qwen3 passes at 4N and 8N was not already triggered, 8N is optional unless the 4N checkpoint is still improving materially.
```

If `N=100`, the required Stage 2 milestones are:

```text
100, 200, 400
```

and the conditional extension milestone is:

```text
800
```

Command template:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps <STAGE2_FINAL_MAX_STEPS_4N_OR_8N> \
  --batch-size 4 \
  --learning-rate <LOCKED_STAGE2_LR_FROM_DIAGNOSTICS> \
  --lora-learning-rate <LOCKED_STAGE2_LORA_LR_FROM_DIAGNOSTICS> \
  --finetune-loss-scale <LOCKED_STAGE2_LOSS_SCALE_FROM_DIAGNOSTICS> \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-final-<stage1b_steps>-from-stage1a-<stage1a_step>-<hp_tag>/checkpoint-<stage1b_step> \
  --freeze-connector \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-finalstage-roleclean-semanticcal-robust-<hp_tag>
```

Stage 2 health gates:

```text
no NaNs
no active W&B alerts
no prompt placeholder failures
no generation max-token collapse
EOS >= 0.98 on smoke evals
max-token-hit <= 0.02 on smoke evals
assistant-prefix <= 0.01 on smoke evals
eval loss not exploding
LoRA gradient norms nonzero enough to update, unless diagnostics show stable known low-gradient regime
```

If gradients are effectively zero throughout Stage 2 and eval loss is flat, run exactly one Qwen3-specific Stage 2 LR variant after the full baseline:

```text
lora_learning_rate: 2e-5
learning_rate: 2e-5
same loss scale unless diagnostics demand otherwise
same full milestone schedule N, 2N, 4N, and conditional 8N
```

Do not call the LR variant final unless it also passes every final gate.

---

## 10. Evaluation protocol

### 10.1 Required eval bundle for every Stage 2 milestone

For Stage 2 milestones `N`, `2N`, `4N`, and `8N` if run:

```text
VQAv2 current-cache seed42 clean, n=1000
VQAv2 current-cache seed42 blank, n=1000
VQAv2 current-cache seed42 shuffled, n=1000
VQAv2 current-cache seed42 wrong-image same-answer-type, n=1000
VQAv2 current-cache seed42 mild blur, n=1000
VQAv2 current-cache seed42 center crop 90, n=1000
VQAv2 current-cache seed42 translate 5 percent, n=1000
POPE current-cache
GQA current-cache
full prediction samples for VQAv2 artifacts
strict/raw diagnostics
answer-kind rates
answer histograms
leakage audits
```

For the first checkpoint that passes all 1000-sample gates, run larger confirmation:

```text
VQAv2 seed42 n=3000 clean
VQAv2 seed42 n=3000 blank
VQAv2 seed42 n=3000 shuffled
VQAv2 seed42 n=3000 wrong-image same-answer-type
POPE expanded if available
GQA expanded if available
```

If a later longer checkpoint beats the first passing checkpoint on the 1000-sample gates, rerun larger confirmation for the later checkpoint too.

### 10.2 Evaluation command template

Use the existing Qwen-enabled eval bundle launcher or equivalent foreground calls.

```bash
python3 scripts/launch_v8_eval_bundle.py \
  --checkpoint /checkpoints/finetune-output/<qwen3-final-run>/checkpoint-<step> \
  --label "Qwen3 V8 final checkpoint <step>" \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --artifact-prefix v8_qwen3_v3_final_ckpt<step> \
  --remote-dir /checkpoints/v8_qwen3_final_remote \
  --parallelism 5
```

If background Modal launching is unreliable, use foreground calls with local parallelism and fetch every artifact explicitly. Do not skip evals because the launcher is inconvenient.

### 10.3 Leakage audit

Run leakage audits for:

```text
VQAv2 bundle
POPE artifact
GQA artifact
```

Training sources must include at minimum:

```text
Stage 1A alignment source files
Stage 1B grounding source files
Stage 2 robust semantic calibration source files
any generated or cached files used by Qwen3 full runs
```

A checkpoint is invalid for promotion if any exact or numeric image-ID overlap appears and cannot be explained as a legitimate split-safe overlap.

---

## 11. Final gates

A Qwen3 checkpoint is a completely viable core decoder replacement only if it satisfies every gate below.

### 11.1 Exposure gates

```text
Stage 1A exposure >= incumbent Stage 1A exposure
Stage 1B exposure >= incumbent Stage 1B exposure
Stage 2 exposure >= incumbent Stage 2 exposure
Stage 2 includes longer-than-LLaMA milestone evaluation, at least 2N and 4N, unless training health fails before those checkpoints
Qwen3 connector initialized fresh or continuation exposure is fully tracked from step 0
connector trained only against Qwen3 hidden space
Stage 2 connector frozen
```

If this gate fails, the checkpoint cannot be called viable. It can only be called diagnostic or integration-only.

### 11.2 Clean VQAv2 gates

Current-cache 1000-sample seed42:

```text
clean >= 62.967
```

Larger confirmation after passing 1000-sample gates:

```text
VQAv2 seed42 n=3000 clean >= 63.556
```

### 11.3 Grounding gates

Current-cache 1000-sample seed42:

```text
blank image accuracy <= 39.733
shuffled image accuracy <= 37.367
wrong-image same-answer-type accuracy <= 38.900
clean minus blank gap >= 23.233
clean minus shuffled gap >= 25.600
clean minus wrong-image gap >= 24.067
```

A checkpoint may not buy clean VQA by increasing blank/shuffle/wrong-image accuracy. A clean score above the incumbent with worse grounding gaps is not viable.

### 11.4 Robustness gates

```text
perturbation mean >= incumbent perturbation mean
```

Use current-cache incumbent perturbation mean if available. Otherwise use:

```text
60.108
```

No individual perturbation may collapse relative to clean by more than the incumbent's corresponding drop, if incumbent current-cache perturbation artifacts exist.

### 11.5 POPE and GQA gates

```text
POPE >= 77.100
GQA >= 43.800
```

These gates are required. A decoder replacement that matches clean VQAv2 but falls materially on POPE or GQA is not viable.

### 11.6 Generation hygiene gates

```text
strict-clean gap <= 1.0 point, ideally 0.0
EOS >= 0.98
max-token-hit <= 0.02
assistant-prefix rate <= 0.01 overall
assistant-prefix rate <= 0.02 in every answer-type bucket
predicted yes/no rate on non-yes/no questions <= 0.05
full prediction samples present
```

### 11.7 Metadata gates

Every final artifact must include:

```text
architecture=anymal_v3
llm_backbone=Qwen/Qwen3-8B
llm_model_type=qwen3
chat_template_family=qwen3_non_thinking
connector_type=perceiver_resampler
question_conditioning=null
image_tokens=128
image_placeholder_token=<|image|>
image_placeholder_count=128
padding_side_for_generation=left
generation_path recorded
LoRA target modules recorded
trainable parameter groups recorded
hyperparameter phase tag recorded
final locked hyperparameters recorded
matched-vs-extended checkpoint status recorded
```

---

## 12. Checkpoint selection and decision labels

### 12.1 Checkpoint selection

Let `N` be the best LLaMA Stage 2 checkpoint step.

1. Evaluate Qwen3 at `N`, `2N`, and `4N`.
2. Evaluate `8N` if Qwen3 is still improving or still below viability at `4N`.
3. The first checkpoint that passes all 1000-sample gates becomes the first larger-confirmation candidate.
4. If a later checkpoint also passes and improves at least two of clean, grounding-gap mean, POPE, GQA, or perturbation mean without worsening any required metric, larger-confirm the later checkpoint too.
5. If no checkpoint passes by `8N`, declare the full Qwen3 replacement not viable under this fixed V3/Qwen recipe unless there is a specific new hypothesis that justifies a new campaign.

Do not select a checkpoint with active W&B alerts, missing checkpoint artifacts, missing eval artifacts, or failed leakage audit.

### 12.2 Decision labels

Use exactly one of these labels in the final report:

```text
integration pass only
hyperparameter diagnostic only
compute-matched trained but not viable
extended trained but not viable
compute-matched viable core replacement
extended viable core replacement
viable and improved core replacement
```

Definitions:

```text
integration pass only:
  smoke/training plumbing works, but exposure gates or final eval gates fail.

hyperparameter diagnostic only:
  short runs helped choose training settings, but no final matched/extended run was completed.

compute-matched trained but not viable:
  Qwen3 reached the LLaMA-matched exposure, but one or more final gates failed.

extended trained but not viable:
  Qwen3 reached matched exposure and longer checkpoints, but one or more final gates still failed.

compute-matched viable core replacement:
  Qwen3 passes all gates at the matched LLaMA recipe exposure.

extended viable core replacement:
  Qwen3 fails or underperforms at matched exposure but passes all gates after longer pretraining and/or longer fine-tuning.

viable and improved core replacement:
  Qwen3 passes all gates and is strictly better than the incumbent on at least two of clean, grounding-gap mean, POPE, GQA, or perturbation mean without regressing any required metric.
```

---

## 13. Required final report format

The final handoff must include these sections.

### 13.1 Schedule manifest summary

| Stage | Incumbent steps | Incumbent examples seen | Qwen3 matched steps | Qwen3 matched examples | Qwen3 extended steps | Exposure gate |
|---|---:|---:|---:|---:|---:|---|
| Stage 1A |  |  |  |  |  | pass/fail |
| Stage 1B |  |  |  |  |  | pass/fail |
| Stage 2 |  |  |  |  |  | pass/fail |

### 13.2 Hyperparameter diagnostics

Report:

```text
which diagnostic runs were launched
what each tested
why the final hyperparameters were selected
connector norm ratios before and after any fix
Stage 1 probe results
Stage 2 smoke results
which diagnostics were discarded and why
```

### 13.3 Training health

Report:

```text
run names
Modal app IDs
W&B run IDs
checkpoint paths
active W&B alerts
gradient clipping rate
loss spikes / grad spikes
validation losses by eval interval
connector output norm diagnostics
Qwen embedding norm diagnostics
supervised-token rates
trainable parameter groups
```

### 13.4 Evaluation summary

Include a table comparing:

```text
incumbent V3 robust
B1 control, for context only
previous Qwen3 short-run ckpt100, for diagnostic context only
new Qwen3 matched checkpoint
new Qwen3 extended checkpoints
```

Required columns:

```text
clean
blank
shuffled
wrong image same-answer-type
blank gap
shuffle gap
wrong gap
mild blur
center crop 90
translate 5 percent
perturbation mean
POPE
GQA
EOS
max-token-hit
assistant-prefix
strict-clean gap
```

### 13.5 Decision

Use one of the decision labels in Section 12.2 and justify it against every gate.

---

## 14. Failure interpretation policy

Do not call Qwen3 nonviable until the matched and required extended exposure gates have been satisfied.

If Qwen3 fails after full exposure, diagnose honestly. Likely failure modes:

```text
connector output norm mismatch with Qwen token embeddings
Stage 1 caption alignment insufficient for Qwen's embedding manifold
Stage 1B grounding loss improves but image-use probes do not
Stage 2 LoRA gradients too small under loss_scale=0.03
Qwen direct-answer calibration overfits answer type but not visual evidence
Qwen3 needs different connector scaling despite same hidden size
Qwen3 needs longer Stage 1 but not longer Stage 2, or vice versa
```

But the previous 300/400/100 run does not satisfy the standard and must not be used as the final conclusion.

---

## 15. What not to do

Do not:

```text
run another LLM backbone
use the previous Qwen3 checkpoint-100 as the final answer
call 300/400/100 full training
skip the hyperparameter diagnostic phase if Qwen norm/scale issues are unresolved
use a diagnostic checkpoint as the final comparable run
mix in B1 counterfactual data for the main decoder-swap line
switch to the failed A1 question-conditioned connector
change the vision tower
change the image token count
unfreeze the full Qwen3 base
judge viability on clean VQAv2 only
skip POPE or GQA
skip blank/shuffle/wrong-image controls
skip leakage audits
skip larger n=3000 confirmation after a 1000-sample pass
promote any checkpoint with missing metadata or active W&B alerts
stop at the LLaMA Stage 2 step without running longer checkpoints
```

---

## 16. Minimum acceptable next action

The next agent's first visible deliverable must be:

```text
outputs/v8_qwen3_full/schedule_manifest.json
```

The next visible training deliverable must be either:

```text
one or more clearly labeled Qwen3 hyperparameter diagnostic runs
```

or, if the best hyperparameters are already obvious from existing diagnostics:

```text
a final locked Qwen3 Stage 1A run whose step/sample budget is at least the incumbent LLaMA Stage 1A exposure
```

The campaign is successful only if Qwen3 becomes a fully viable core decoder replacement under the same incumbent gates, with both compute-matched and longer-than-LLaMA checkpoints reported.
