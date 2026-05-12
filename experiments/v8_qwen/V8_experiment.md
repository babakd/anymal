# V8 Core LLM Swap Experiment Plan

**Date:** 2026-05-10  
**Audience:** execution agent with access to the AnyMAL working directory, Modal/GPU training, W&B, and evaluation artifacts.  
**Purpose:** run a controlled core-decoder swap from the current LLaMA-3-8B-Instruct AnyMAL stack to a newer open-weight LLM while preserving the cost profile, the V3 visual interface, and the V7 grounding-evaluation discipline.

This file is self-contained. Do not assume access to the conversation that produced it.

**Operator update:** the Llama 3.1 migration/canary route is intentionally removed from the option space for this execution. Keep current LLaMA-3-8B compatibility only as incumbent/default compatibility, not as an additional candidate branch.

---

## 0. One-line directive

Start a **core LLM swap** from the current V3 robust AnyMAL platform. Default target is **Qwen/Qwen3-8B**. Keep the V3 visual connector contract fixed, retrain the connector from scratch into the new decoder hidden space, rerun the robust semantic Stage 2 recipe, and judge success by clean VQA, grounding gaps, POPE, and GQA. Do not change the vision encoder, image token count, architecture family, counterfactual data policy, or native multimodal stack in the main decoder-swap line.

---

## 1. Current state to internalize

### 1.1 Current incumbent

The current strongest controlled model is V3 robust from the V6 causal-falsification campaign.

```text
Label: v7_v3_robust_currentcache
Checkpoint: /checkpoints/finetune-output/v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003/checkpoint-100
Architecture: anymal_v3
Vision tower: SigLIP2-So400m at 384px
Connector: perceiver_resampler
Image tokens: 128
Core decoder: LLaMA-3-8B-Instruct family
Stage 2 recipe: robust role-clean semantic calibration
Metadata: verified from checkpoint/model artifacts
```

Current-cache aligned 1000-sample VQAv2 seed-42 evaluation:

| Metric | Value |
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

Larger clean confirmation:

| Eval | Value |
|---|---:|
| VQAv2 seed42 n=3000 clean | 63.556 |

Interpretation: V3 robust is the actual incumbent. Treat old V4/V5 checkpoints as historical controls, not as the main baseline.

### 1.2 Current no-architecture grounding control

B1 is the active no-architecture counterfactual grounding control. It should be used to interpret decoder-swap grounding behavior, but it is not a promotion target because POPE regressed.

```text
Checkpoint: /checkpoints/finetune-output/v7-b1-v3-counterfactual-grounding-robustcal-acc16-bs4-lossscale003/checkpoint-100
Architecture: anymal_v3
```

| Metric | B1 value | Delta vs aligned V3 |
|---|---:|---:|
| clean | 63.267 | +0.300 |
| blank image | 39.433 | -0.300 |
| shuffled image | 36.633 | -0.734 |
| wrong image, same answer-type | 37.833 | -1.067 |
| clean minus blank gap | 23.833 | +0.600 |
| clean minus shuffled gap | 26.633 | +1.033 |
| clean minus wrong-image gap | 25.433 | +1.366 |
| POPE | 75.000 | -2.100 |
| GQA | 44.600 | +0.800 |
| EOS | 1.000 | +0.000 |
| max-token-hit | 0.000 | +0.000 |
| assistant-prefix | 0.000 | +0.000 |

Perturbations for B1 checkpoint-100:

| Perturbation | B1 value |
|---|---:|
| mild blur | 63.267 |
| center crop 90 | 63.333 |
| translate 5 percent | 64.067 |

Interpretation: B1 is useful as a grounding-data control. It improved VQAv2 image-use gaps slightly but regressed POPE. Do not mix B1 data into the first core LLM swap, or you will confound decoder effects with data/objective effects.

### 1.3 Failed V7 architecture smoke

A1 prompt-conditioned V3 Perceiver failed and should not be continued as the basis for this LLM-swap phase.

```text
Stage 1 checkpoint used for Stage 2:
/checkpoints/pretrain-output/v7-a1-stage1a-qcond-v3-perceiver-128tok-smoke300-from-canary20/checkpoint-225

Stage 2 checkpoint evaluated:
/checkpoints/finetune-output/v7-a1-qcond-v3-perceiver-robustcal-from-ckpt225-acc16-bs4-lossscale003/checkpoint-50

Connector: question_conditioned_perceiver_resampler
Conditioning: pooled_prompt_embedding_additive_latent_shift
Image tokens: 128
Perceiver layers: 6
Heads: 16
FF mult: 4
```

| Metric | A1 value | Delta vs aligned V3 |
|---|---:|---:|
| clean | 50.633 | -12.334 |
| blank image | 42.400 | +2.667 |
| shuffled image | 40.633 | +3.266 |
| wrong image, same answer-type | 39.567 | +0.667 |
| clean minus blank gap | 8.233 | -15.000 |
| clean minus shuffled gap | 10.000 | -15.600 |
| clean minus wrong-image gap | 11.067 | -13.000 |
| POPE | 57.200 | -19.900 |
| GQA | 35.800 | -8.000 |
| EOS | 1.000 | +0.000 |
| max-token-hit | 0.000 | +0.000 |
| assistant-prefix | 0.000 | +0.000 |

Interpretation: A1 was mechanically healthy enough to diagnose, but it is an architecture failure. Do not combine the core LLM swap with additive latent question conditioning.

---

## 2. V8 question

The V7/V6 evidence says the old “legacy V4 spatial connector caused the gain” story is false. V3 with the robust semantic recipe is now the real platform. The next causal question is:

> Does replacing the core frozen LLM with a newer, stronger, still affordable open-weight decoder improve AnyMAL’s visual grounding and answer behavior when the visual interface and recipe are held constant?

This is a decoder-swap experiment, not a new multimodal architecture experiment.

The main comparison is:

```text
V3 robust + LLaMA-3-8B-Instruct decoder
vs
V3 robust-equivalent pipeline + new decoder
```

Keep fixed in the main line:

```text
Vision tower: frozen SigLIP2-So400m at 384px
Connector family: V3 perceiver_resampler
Image token count: 128
Image placeholder contract: strict contiguous block, exactly 128 placeholders
Training shape: connector-only Stage 1, then LoRA-only Stage 2
Stage 2 data: robust role-clean semantic calibration
Evaluation: current-cache V7/V8 bundle with clean, blank, shuffled, wrong-image, POPE, GQA, hygiene, prediction samples, leakage audits
```

Change only:

```text
Core LLM backbone and corresponding tokenizer/chat-template path
```

---

## 3. LLM selection policy

### 3.1 Hard constraints

A candidate LLM must satisfy these requirements before a full run:

1. **Cost envelope**
   - Target: approximately the current 8B LLaMA cost profile.
   - Acceptable: dense or effectively dense model up to about 10B parameters.
   - Avoid: 20B+ total-parameter models, large MoE models, or native VLM stacks as the main core-swap line.

2. **Open-weight availability**
   - Weights must be available locally or on Hugging Face under a license the project can use.
   - Prefer Apache 2.0.
   - Llama Community License is acceptable only for the incumbent/current compatibility path if project policy permits it.

3. **Decoder API compatibility**
   - Must support causal LM operation through `transformers` or a clean local model wrapper.
   - Must support, or be made to support, `inputs_embeds` for the first generation step.
   - Must expose token embeddings, hidden size, attention mask handling, pad/eos/bos IDs, and generation stopping.

4. **Tokenizer and chat-template controllability**
   - The agent must be able to create a strict direct-answer prompt with exactly one contiguous image-placeholder block.
   - The agent must be able to disable thinking/reasoning verbosity if the model supports a thinking mode.
   - The model must not require native image tokens or image processors for the main AnyMAL core-swap line.

5. **LoRA/QLoRA targetability**
   - The project must be able to attach LoRA adapters to the decoder attention and MLP projection modules.
   - Module names must be verified from the loaded model, not assumed from LLaMA.

6. **No old connector reuse**
   - Existing V3 connector weights are trained into the LLaMA hidden space. They are not portable to Qwen, Mistral, Gemma, or a different LLaMA generation without retraining.
   - Even if hidden size remains 4096, retrain the connector from scratch for the new decoder.

### 3.2 Ranking

Use this ranking unless the working directory reveals a blocking dependency issue.

#### Rank 1: Qwen/Qwen3-8B

Default target.

Why:

- It is a recent open-weight 8B-class dense causal LM.
- Hugging Face config reports `hidden_size=4096`, `num_hidden_layers=36`, `num_attention_heads=32`, `num_key_value_heads=8`, and `model_type=qwen3`.
- It uses Apache 2.0 licensing.
- It has modern instruction-following, multilingual, and reasoning capabilities.
- The 4096 hidden size keeps the connector output width equal to the current LLaMA-3-8B hidden width, reducing shape churn. It does not eliminate the need to retrain the connector.
- Qwen3 supports thinking and non-thinking modes. For VQA, use non-thinking/direct-answer behavior.

Default model ID:

```text
Qwen/Qwen3-8B
```

Do not use `Qwen/Qwen3-8B-Base` as the default unless you deliberately want a base-model decoder experiment. The current AnyMAL pipeline relies on short Stage 2 instruction calibration and direct-answer behavior, so the chat/instruct Qwen3-8B path is the first target.

#### Rank 2: mistralai/Ministral-3-8B-Instruct-2512-BF16

Second serious alternative, but only after Qwen.

Why:

- Recent 8B/9B-class open model.
- Apache 2.0.
- Model card describes an 8.4B language model plus 0.4B vision encoder, with 256K context and strong small-model benchmark claims.
- BF16 and FP8 formats exist.

Caution:

- It has native vision capabilities. If you use those, it is no longer a clean AnyMAL core-decoder swap.
- Use only the text decoder path if it is cleanly accessible. Otherwise classify it as an external multimodal baseline, not the main decoder-swap experiment.
- Verify hidden size, model wrapper, tokenizer, and LoRA target modules before full training.

#### Rank 3: Gemma 4 E4B or Gemma 3 4B

Small/diagnostic only.

Why:

- Recent Google open-weight multimodal family.
- Gemma 4 documentation describes E2B, E4B, 26B A4B, and 31B sizes, Apache 2.0 licensing, multimodality, and long context.
- Gemma 3 4B is a stable smaller multimodal open-weight model.

Caution:

- The 4B class is likely too small to be the main quality upgrade from an 8B LLaMA decoder.
- Native multimodal paths create a causal confound.
- Use as a cost-down diagnostic, not as the first serious replacement.

#### External VLM baseline, not a core swap: Qwen/Qwen2.5-VL-7B-Instruct

Run this out-of-box only as a target baseline if time permits.

Purpose:

```text
How far is AnyMAL V3 robust from a strong current open VLM under the same eval bundle?
```

Do not count it as a decoder-swap result. It uses its own multimodal stack and processor.

---

## 4. Engineering requirements

### 4.1 Add a decoder-backbone abstraction

The current code is likely hardwired around LLaMA. Implement the smallest abstraction that supports at least:

```text
--llm-backbone meta-llama/Llama-3-8B-Instruct       # current/default compatibility
--llm-backbone Qwen/Qwen3-8B                        # V8 main target
```

Suggested metadata fields to write into every checkpoint:

```json
{
  "architecture": "anymal_v3",
  "vision_tower": "SigLIP2-So400m-384",
  "connector_type": "perceiver_resampler",
  "image_tokens": 128,
  "llm_backbone": "Qwen/Qwen3-8B",
  "llm_model_type": "qwen3",
  "llm_hidden_size": 4096,
  "llm_num_hidden_layers": 36,
  "llm_num_attention_heads": 32,
  "llm_num_key_value_heads": 8,
  "tokenizer_name": "Qwen/Qwen3-8B",
  "chat_template_family": "qwen3_non_thinking",
  "pad_token_id": null,
  "bos_token_id": null,
  "eos_token_id": null,
  "image_placeholder_token": "<image>",
  "image_placeholder_count": 128,
  "added_special_tokens": [],
  "padding_side_for_generation": "left",
  "stage1_connector_init": "fresh_random",
  "stage2_lora_target_modules": []
}
```

Fill real token IDs and module names at runtime.

### 4.2 Preserve the AnyMAL image interface

The model should still answer like this:

```text
image pixels
  -> frozen SigLIP2 vision encoder
  -> trainable V3 Perceiver connector
  -> 128 image embeddings in decoder hidden space
  -> splice into the prompt at 128 contiguous image placeholders
  -> frozen decoder plus LoRA adapters generates answer tokens
```

Do not use native Qwen-VL, Mistral-Vision, or Gemma-Vision tokens in the main line.

### 4.3 Tokenizer and placeholder handling

For each decoder tokenizer:

1. Verify whether `<image>` is encoded as one token.
2. If not, add a dedicated special placeholder token or use an existing sentinel token.
3. Ensure prompts contain one contiguous block of exactly 128 placeholder IDs.
4. Ensure every placeholder embedding is replaced by visual embeddings before the decoder forward pass.
5. Ensure labels are masked at prompt and image positions.
6. Ensure no placeholder token appears in supervised answer spans.
7. Record tokenizer changes in `model_meta.json`.

It is acceptable to resize the decoder token embeddings to add placeholder tokens, provided the placeholder positions are always replaced by visual embeddings and never supervised. If resizing occurs, record it.

### 4.4 Chat-template handling

For Qwen3:

- Use the model’s chat template only if it preserves the image-placeholder block exactly.
- Use non-thinking/direct-answer mode.
- Disable chain-of-thought, hidden reasoning, or explanatory answer styles.
- Validate that the decoded raw answer does not include role strings, template delimiters, or thinking tags.

The direct-answer system prompt remains:

```text
Answer with only the final answer. Do not include role labels, explanations, or the word assistant. End after the answer.
```

Do not rely on post-processing to strip role labels. Strict/raw parity is a promotion gate.

### 4.5 Generation with `inputs_embeds`

Do not assume every decoder’s `.generate()` path handles `inputs_embeds` identically.

Required smoke tests:

1. Forward pass with `input_ids` only.
2. Forward pass with `inputs_embeds` equivalent to embedded `input_ids`.
3. Forward pass with image-placeholder embeddings replaced by random visual-shaped embeddings.
4. Greedy generation with `inputs_embeds` for the first step.
5. Batched left-padded generation with different prompt lengths.
6. Stop-token behavior with direct-answer prompt.

If Hugging Face `.generate()` fails with `inputs_embeds`, implement a small custom greedy generation loop:

```text
step 0: run full prompt as inputs_embeds with attention_mask
step t>0: feed generated token IDs / embeddings with past_key_values
stop when eos or max_new_tokens
```

The custom loop must preserve left-padding semantics and record that it was used in eval metadata.

### 4.6 LoRA target modules

Do not copy LLaMA target-module names blindly. Inspect the loaded model and print trainable parameter groups.

Expected likely modules for Qwen3 and LLaMA-like decoders:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

But verify in code. If a target module does not exist, fail early instead of silently training fewer adapters.

### 4.7 Quantization and cost

The default should stay close to current QLoRA cost:

```text
base decoder loaded frozen, likely 4-bit or equivalent
connector trained in Stage 1
LoRA adapters trained in Stage 2
vision tower frozen
```

Record:

```text
precision
quantization method
GPU type
batch size
gradient accumulation
peak memory
tokens/sec or examples/sec
wall-clock
```

A decoder swap that is 2x slower or requires much larger hardware is not a success unless the quality gain is large and explicit.

---

## 5. Training plan

### 5.1 Do not reuse old trained weights across decoder spaces

The current V3 connector maps SigLIP2 features into the LLaMA-3-8B hidden space. A new decoder has a different embedding manifold. Therefore:

```text
Do not initialize Qwen connector from the LLaMA connector.
Do not reuse LLaMA LoRA adapters.
Do not warm-start from B1 LoRA.
Do not mix A1 architecture changes.
```

Start with a fresh V3 Perceiver connector for the new decoder.

### 5.2 Stage 0: integration smoke

Before training, run local or Modal smoke tests.

Required outputs:

```text
outputs/v8_llm_swap_smoke/<backbone>/tokenizer_report.json
outputs/v8_llm_swap_smoke/<backbone>/prompt_contract_report.json
outputs/v8_llm_swap_smoke/<backbone>/inputs_embeds_report.json
outputs/v8_llm_swap_smoke/<backbone>/generation_report.json
```

Smoke pass criteria:

- Tokenizer loads.
- Decoder loads with requested precision.
- Hidden size is read from config and matches connector output projection.
- Prompt contains exactly 128 contiguous placeholders.
- `inputs_embeds` forward works.
- Batched left-padded generation works.
- Direct-answer prompt does not emit template garbage in a text-only sanity check.
- LoRA target modules are found and printed.

### 5.3 Stage 1A: connector-only alignment

Goal: learn SigLIP2-to-new-decoder hidden-space alignment.

Use the same V3 connector architecture:

```text
Architecture: anymal_v3
Vision: frozen SigLIP2-So400m at 384px
Connector: perceiver_resampler
Image tokens: 128
Perceiver layers: same as V3 baseline
Heads: same as V3 baseline
Projection output width: decoder hidden_size
Decoder base: frozen
Trainable: connector only
```

Dataset:

Use the most successful V3/V6 connector-alignment dataset path in the repo. Prefer the exact Stage 1 / Stage 1B recipe family that led to V6-C1b V3 robust if it is available. Do not switch to B1 counterfactual data in Stage 1.

Suggested first run names:

```text
v8-qwen3-8b-v3-stage1a-smoke20
v8-qwen3-8b-v3-stage1a-fixed300
```

Stop rules:

- Stop if placeholder contract fails.
- Stop if W&B active alerts persist after the first stable logging window.
- Stop if gradient clipping is severe and persistent under the known stable V3 recipe.
- Stop if connector output norms explode or collapse.
- Stop if model metadata differs from planned decoder/hidden-size/token-count.

### 5.4 Stage 1B: grounding continuation

Goal: give the new connector the same grounding path that made V3 robust work.

Use the V3/V6 grounding Stage 1B recipe. Fixed checkpoint selection is required.

Recommended fixed checkpoints:

```text
Stage1A checkpoint: fixed step 300 or repo-equivalent fixed step
Stage1B checkpoint: fixed step 400 or repo-equivalent fixed step
```

Do not pick the best visible VQA checkpoint unless the selection rule was preregistered before training.

Suggested run name:

```text
v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300
```

### 5.5 Stage 2: robust semantic calibration

Goal: reproduce the V3 robust Stage 2 behavior with the new decoder.

Use:

```text
Dataset: v5_semantic_calibration_robust or exact robust dataset used by V6-C1b
Trainable: decoder LoRA only
Frozen: vision tower, connector, decoder base
Loss scale: start with the V3 robust value if known; otherwise use 0.03
Batch/accumulation: use V3 robust settings, likely batch size 4, accumulation 16
Max steps: 100
```

Suggested run name:

```text
v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003
```

If Qwen3 has verbose or thinking-style output during early eval, do not solve it by post-processing. Fix the chat-template/direct-answer path and rerun.

---

## 6. Evaluation plan

### 6.1 Required eval bundle

Every candidate that reaches Stage 2 checkpoint-100 must be evaluated against the aligned V3 robust incumbent and B1 control.

Required VQAv2 current-cache seed-42 1000-sample artifacts:

```text
clean
blank image
shuffled image
wrong image, same answer-type
mild blur
center crop 90
translate 5 percent
```

Required external/diagnostic artifacts:

```text
POPE
GQA diagnostic slice
```

Required hygiene:

```text
strict accuracy
cleaned accuracy
strict-clean gap
EOS rate
max-token-hit rate
assistant-prefix rate
answer-kind rates by ground-truth answer type
top raw answers
top cleaned answers
full prediction samples
image IDs
leakage audit
```

Required larger confirmation if seed-42 1k is promising:

```text
VQAv2 seed42 n=3000 clean
VQAv2 seed42 n=3000 blank
VQAv2 seed42 n=3000 shuffled
VQAv2 seed42 n=3000 wrong image same-answer-type
```

### 6.2 Primary metrics

The primary metric is not clean VQAv2 alone.

Use:

```text
grounding_gap_blank    = clean - blank
grounding_gap_shuffle  = clean - shuffled
grounding_gap_wrong    = clean - wrong_image_same_answer_type
perturb_mean           = mean(mild_blur, center_crop90, translate5pct, and any existing resize_up if available)
```

Compare against aligned V3 robust:

```text
clean: 62.967
blank: 39.733
shuffled: 37.367
wrong image same-answer-type: 38.900
blank gap: 23.233
shuffle gap: 25.600
wrong gap: 24.067
POPE: 77.100
GQA: 43.800
VQAv2 n=3000 clean: 63.556
```

Also compare against B1:

```text
clean: 63.267
blank: 39.433
shuffled: 36.633
wrong image same-answer-type: 37.833
blank gap: 23.833
shuffle gap: 26.633
wrong gap: 25.433
POPE: 75.000
GQA: 44.600
```

### 6.3 Success tiers

#### Tier 0: integration pass

A decoder candidate passes integration if:

- Stage 0 smoke tests pass.
- Stage 1 canary trains without active health alerts.
- Stage 2 canary can generate left-padded direct answers.
- strict-clean gap <= 1.0.
- assistant-prefix <= 0.01.
- EOS >= 0.98.
- max-token-hit <= 0.02.

#### Tier 1: viable decoder swap

A candidate is viable if checkpoint-100 satisfies:

```text
clean >= 61.5
POPE >= 76.0
GQA >= 43.0
blank gap >= 22.5
shuffle gap >= 25.0
wrong gap >= 23.5
EOS >= 0.98
max-token-hit <= 0.02
assistant-prefix <= 0.01
```

This means the swap is close enough to V3 robust to continue optimizing.

#### Tier 2: decoder improvement

A candidate is an actual improvement if:

```text
clean >= 63.5 on seed42 n=1000 or n=3000
POPE >= 77.1
GQA >= 44.6 or improves meaningfully over V3 robust
blank/shuffle/wrong corrupted-image accuracies do not increase versus V3 robust
at least two of the three grounding gaps improve by >= 1.0 point versus V3 robust
strict-clean gap <= 1.0
hygiene gates pass
leakage audit passes
```

#### Tier 3: promotion candidate

A candidate can be considered for promotion only after:

```text
clean seeds 42/43/44 on 1000-sample current-cache or a larger locked slice
n=3000 clean plus n=3000 corrupted-image controls
full perturbation suite
POPE and GQA
leakage audits for all evals and training sources
comparison to B1 no-architecture control
training health is green
metadata is complete
```

Promotion requires either:

- clear clean-score improvement without worse corrupted-image behavior, or
- equal clean score with materially better grounding gaps, POPE, or GQA.

### 6.4 Failure patterns and interpretation

| Pattern | Interpretation | Next action |
|---|---|---|
| Qwen clean improves, blank/shuffle/wrong also improve | Better answer priors/calibration, not grounding | Do not promote; add or tune counterfactual grounding only after isolating decoder effect |
| Qwen clean near V3, gaps improve, POPE/GQA hold | Good decoder-swap signal | Run larger confirmation and seeds |
| Qwen clean drops >5 points, POPE/GQA drop | Decoder or connector alignment failed | Check Stage 1, tokenizer, chat template, inputs_embeds generation |
| Qwen outputs role labels or thinking tags | Prompt/template failure | Fix non-thinking direct-answer path; do not strip in postprocess |
| Qwen Stage 1 loss improves but VQA fails | Connector internal loss not predictive | Compare hidden norms, answer distributions, and reproduce the current LLaMA-3-8B V3 robust path through the new abstraction |
| Current LLaMA-3-8B abstraction reproduction and Qwen both fail | Decoder-swap pipeline issue | Reproduce current LLaMA-3-8B V3 robust path with the old path and compare implementation differences |
| Qwen beats V3 but loses to B1 on gaps | Data/objective story remains strong | Treat as useful but not a clean decoder win |

---

## 7. Concrete execution sequence

### Step 1: close current V7 report gaps

Before or in parallel with decoder work, fetch/produce:

```text
B1 leakage audit
A1 leakage audit
W&B health table for B1 and A1
A1 checkpoint-225 explanation
full model_meta / connector_meta dumps for V3 robust, B1, and A1
```

This should be quick. Do not let it block Stage 0 decoder smoke unless it requires the same resources.

### Step 2: implement decoder abstraction and Stage 0 smoke for Qwen3

Target:

```text
Qwen/Qwen3-8B
```

Required smoke outputs are listed in Section 5.2.

Stop if:

- Qwen cannot load in the training environment.
- `inputs_embeds` generation cannot be made to work.
- prompt placeholders are not exactly contiguous.
- non-thinking/direct-answer mode cannot be enforced.

### Step 3: Qwen3 Stage 1A smoke and fixed Stage 1A

Suggested shape after adding `--llm-backbone` or equivalent:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset <V3_STAGE1A_DATASET_USED_BY_SUCCESSFUL_V3> \
  --pretrain-image-tokens 128 \
  --max-steps 20 \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1a-smoke20
```

Then fixed Stage1A:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset <V3_STAGE1A_DATASET_USED_BY_SUCCESSFUL_V3> \
  --pretrain-image-tokens 128 \
  --max-steps 300 \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1a-fixed300
```

Use the actual successful V3 Stage 1A dataset and training knobs from the repo. The command above shows the required shape, not a guarantee that flags already exist.

### Step 4: Qwen3 Stage 1B grounding continuation

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage pretrain \
  --dataset <V3_STAGE1B_GROUNDING_DATASET_USED_BY_SUCCESSFUL_V3> \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1a-fixed300/checkpoint-300 \
  --pretrain-image-tokens 128 \
  --max-steps 400 \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300
```

Again, use actual repo flags and dataset names.

### Step 5: Qwen3 robust Stage 2

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-grounding-fixed400-from-stage1a300/checkpoint-400 \
  --freeze-connector \
  --use-wandb \
  --run-name v8-qwen3-8b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003
```

If `loss_scale=0.03` is unstable with Qwen, run exactly one fallback:

```text
loss_scale = 0.01
same data
same checkpoint
same accumulation
same max steps
```

Do not tune many tiny variants before evaluating the first clean run.

### Step 6: full eval bundle

Use existing V7/V6 eval tooling. The agent should adapt commands to the exact scripts in the repo.

Minimum required candidate evals:

```text
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_clean_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_blank_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_shuffled_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_wrongimage_sameanswertype_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_mildblur_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_centercrop90_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_translate5pct_currentcache.json
pope_eval_v8_qwen3_v3_robustcal_ckpt100_currentcache.json
gqa_eval_v8_qwen3_v3_robustcal_ckpt100_currentcache.json
```

If promising:

```text
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_n3000_clean_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_n3000_blank_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_n3000_shuffled_currentcache.json
vqa_eval_v8_qwen3_v3_robustcal_ckpt100_seed42_n3000_wrongimage_sameanswertype_currentcache.json
```

### Step 7: comparison report

Produce a compact report with this table:

| Model | Clean | Blank | Shuffle | Wrong | Blank gap | Shuffle gap | Wrong gap | POPE | GQA | EOS | Max hit | Prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 robust aligned | 62.967 | 39.733 | 37.367 | 38.900 | 23.233 | 25.600 | 24.067 | 77.100 | 43.800 | 1.000 | 0.000 | 0.000 |
| B1 control | 63.267 | 39.433 | 36.633 | 37.833 | 23.833 | 26.633 | 25.433 | 75.000 | 44.600 | 1.000 | 0.000 | 0.000 |
| Qwen3 V8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Also include:

```text
training health summary
model_meta diff
answer-kind rates
raw/clean top answers
leakage audit summary
pairwise files: Qwen correct / V3 wrong, V3 correct / Qwen wrong, Qwen correct / B1 wrong, B1 correct / Qwen wrong
```

---

## 8. What not to do

Do not do any of the following in the main line:

1. Do not reuse the old LLaMA-trained V3 connector.
2. Do not reuse old LLaMA LoRA adapters.
3. Do not combine the decoder swap with A1 prompt-conditioned additive latent shifting.
4. Do not combine the decoder swap with B1 counterfactual data in the first run.
5. Do not switch vision encoder.
6. Do not change image resolution.
7. Do not change image token count from 128.
8. Do not use Qwen2.5-VL, Qwen3-VL, Gemma-Vision, or Ministral native vision as the main “core LLM swap.” Those are external VLM baselines.
9. Do not promote on clean VQAv2 alone.
10. Do not compare to old V4/V5 as the main baseline. The current baseline is V3 robust current-cache aligned.
11. Do not use post-processing to hide template, role, thinking, or assistant-prefix leakage.
12. Do not pick best checkpoints by visible VQA unless the rule is preregistered.
13. Do not ignore blank/shuffle/wrong-image increases.
14. Do not add or run the Llama 3.1 canary route for this execution.

---

## 9. Required final deliverable from the execution agent

Create a handoff file named:

```text
V8_CORE_LLM_SWAP_RESULTS.md
```

It must include:

1. **Backbone choice and rationale**
   - Which LLM was selected.
   - Why it passed selection constraints.
   - License and model-card notes.

2. **Implementation summary**
   - Files changed.
   - Decoder abstraction details.
   - Tokenizer handling.
   - Placeholder contract validation.
   - Generation path used: `.generate(inputs_embeds=...)` or custom greedy loop.
   - LoRA target module list.

3. **Training summary**
   - Stage 0 smoke results.
   - Stage 1A run name, checkpoint, health.
   - Stage 1B run name, checkpoint, health.
   - Stage 2 run name, checkpoint, health.
   - W&B alerts, clipping, loss spikes, eval losses.

4. **Metadata summary**
   - `model_meta.json` contents or key fields.
   - Decoder config fields.
   - Connector metadata.
   - Tokenizer changes.

5. **Evaluation table**
   - V3 robust aligned.
   - B1 control.
   - Qwen3 candidate.

6. **Grounding interpretation**
   - Did clean accuracy improve?
   - Did corrupted-image accuracy decrease or increase?
   - Did grounding gaps improve?
   - Did POPE/GQA improve?
   - Did answer distributions change?

7. **Leakage audit**
   - VQAv2, POPE, GQA.
   - Training sources used.
   - Any overlap policy and outcome.

8. **Decision**
   - Integration pass/fail.
   - Viable decoder swap yes/no.
   - Decoder improvement yes/no.
   - Promotion candidate yes/no.
   - Next recommended action.

---

## 10. Source notes for model selection

The following public model-card facts were checked on 2026-05-10. Recheck before launch if model repositories or licenses have changed.

- `Qwen/Qwen3-8B`: Hugging Face model card describes Qwen3 as the latest Qwen generation with thinking/non-thinking modes, instruction-following improvements, and Apache 2.0 licensing. Its config lists `hidden_size=4096`, `num_hidden_layers=36`, `num_attention_heads=32`, `num_key_value_heads=8`, and `model_type=qwen3`.
- `mistralai/Ministral-3-8B-Instruct-2512-BF16`: Hugging Face model card describes an Apache 2.0 model with an 8.4B language model plus 0.4B vision encoder, 256K context, and native vision capability.
- `google/gemma-4-E2B-it` / Gemma 4 family: Hugging Face model card describes Gemma 4 open-weight multimodal models, Apache 2.0 licensing, E2B/E4B/26B A4B/31B sizes, and long context.
- `Qwen/Qwen2.5-VL-7B-Instruct`: Hugging Face model card describes a 7B native vision-language model using its own processor/toolkit. Use as an external VLM baseline only.

---

## 11. Final decision rule

The default run is successful only if it answers this question cleanly:

> Holding AnyMAL’s V3 visual interface and robust semantic recipe constant, does a newer open-weight 8B-class decoder improve visual grounding or answer quality over the current LLaMA-3-8B V3 robust incumbent?

If Qwen3 improves clean VQA but also raises blank/shuffle/wrong-image scores, it is not a grounding win. If Qwen3 ties clean but improves grounding gaps, POPE, or GQA, it is a useful decoder improvement. If Qwen3 fails badly, reproduce the current LLaMA-3-8B V3 robust path through the new abstraction to determine whether the failure is Qwen-specific or a pipeline problem.
