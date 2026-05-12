# V9 Qwen Scale-1.05 Confirmation And Promotion Steps

Date: 2026-05-12

Audience: execution agent with access to the AnyMAL working directory, Modal jobs, checkpoints, and the V9 artifacts.

## Goal

Confirm and promote the current viable Qwen/Qwen3-8B replacement candidate without starting new architecture or Stage2 experiments.

The candidate is:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

This is the **V9 Qwen Connection V2** candidate:

```text
architecture: anymal_v3
llm_backbone: Qwen/Qwen3-8B
connector: perceiver_resampler
image tokens: 128
Stage1B anchor: checkpoint-2350 from the V8/V9 Qwen Stage1B bracket
connector_output_scale: 1.05 materialized in model_meta.json
Stage2 LoRA: none for the promoted candidate
```

Important: this materialized checkpoint is for **eval/inference**, not optimizer resume. It contains `model_meta.json` and `projector.pt`; do not treat it as a training-resume checkpoint.

## Current evidence to preserve

The scale-1.05 candidate already passed the V9 gates through the scale-override path and was materialized afterward.

Known passing metrics:

```text
clean n=3000 seed42:     66.133
clean n=3000 seed43:     66.922
clean n=3000 seed44:     65.256
blank n=3000 seed42:     38.811
shuffled n=3000 seed42:  36.900
wrong n=1000 seed42:     37.967
perturb mean:            66.856
POPE:                    79.100
GQA:                     44.000
EOS/max-hit/prefix:      clean
leakage audit:           pass
```

Incumbent gates to beat or match:

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
leakage audit pass
```

## Do not run

Do not run new Stage2 LoRA experiments.
Do not run new connector training.
Do not run V9 Batch 3C/3D/3E architecture changes.
Do not change the decoder, prompt template, dataset, or checkpoint selection.
Do not use eval-time scale overrides for the final confirmation artifacts.

The only purpose of this handoff is to verify that the **materialized** checkpoint reproduces the passing bundle and then record promotion.

## Step 1: verify materialized metadata

Inspect the materialized checkpoint and confirm:

```text
model_meta.json exists
projector.pt exists
architecture == anymal_v3
llm_backbone == Qwen/Qwen3-8B
connector_type == perceiver_resampler
image_tokens == 128
connector_output_scale == 1.05
question_conditioning == null
Stage2 LoRA is absent / not required
```

Fail closed if `connector_output_scale` is missing or not exactly `1.05`.

## Step 2: rerun core bundle from the materialized checkpoint, no override

Run the following from:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Use the normal checkpoint metadata path. Do **not** pass a connector scale override.

Required evals:

```text
VQAv2 seed42 n=1000 clean
VQAv2 seed42 n=1000 blank
VQAv2 seed42 n=1000 shuffled
VQAv2 seed42 n=1000 wrong-image same-answer-type
POPE current cache
GQA current cache
```

Use artifact names that make the no-override status explicit, for example:

```text
vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_clean_n1000.json
vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_blank_n1000.json
vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_shuffled_n1000.json
vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_wrongimage_sameanswertype_n1000.json
pope_eval_v9_qwen3_scale105_materialized_nooverride.json
gqa_eval_v9_qwen3_scale105_materialized_nooverride.json
```

Pass condition: the materialized no-override bundle must remain within normal sampling/eval tolerance of the known scale-1.05 bundle and must pass all gates.

Hard fail thresholds:

```text
clean < 62.967
blank > 39.733
shuffled > 37.367
wrong > 38.900
POPE < 77.100
GQA < 43.800
EOS < 0.98
max-token-hit > 0.02
assistant-prefix > 0.01
strict-clean gap > 1.0
```

## Step 3: run wrong-image n=3000 for symmetry

The winning report used n=3000 for clean, blank, and shuffled, but wrong-image was n=1000. Run:

```text
VQAv2 seed42 n=3000 wrong-image same-answer-type
```

Pass condition:

```text
wrong-image same-answer-type <= 38.900
```

This is not expected to fail, but it closes the only asymmetric VQAv2 control in the final record.

## Step 4: repeat GQA or expand if available

The candidate passed GQA at exactly the preferred target:

```text
GQA = 44.000
minimum gate = 43.800
preferred target = 44.000
```

Rerun GQA from the materialized no-override checkpoint. If a larger trusted GQA diagnostic slice exists locally, run the larger slice as well. Do not introduce a new untrusted harness.

Pass condition:

```text
GQA >= 43.800
```

If the rerun lands between 43.8 and 44.0, keep the candidate viable but mark GQA as narrow-pass. If it falls below 43.8, do not promote until investigated.

## Step 5: rerun leakage audit on final artifact set

Run the V9 leakage audit over the final no-override artifacts, including:

```text
clean
blank
shuffled
wrong-image
perturbations if included
POPE
GQA
```

Expected result:

```text
exact val2014 overlap: 0
raw ref overlap: 0
missing/unparseable refs: 0
missing sources: 0
numeric-only filename collisions: warning only, not failure, if exact/raw overlap is 0
```

Fail if exact or raw-reference leakage appears.

## Step 6: answer-distribution comparison

Run the answer analysis script on these artifacts:

```text
V3 robust incumbent
B1 control
Qwen Stage1B2350 scale1.00
Qwen Stage1B2350 scale1.05 materialized
Qwen Stage2 ckpt800
```

Summarize:

```text
top cleaned answers
top raw answers
answer-kind rates by ground-truth answer type
yes/no rate on non-yes/no questions
strict-clean gap
assistant-prefix rate
```

This is not a promotion blocker unless it reveals mode collapse or answer-kind pathology.

## Step 7: final promotion record

If Steps 1-6 pass, write a concise promotion note in the repo, for example:

```text
V9_QWEN_SCALE105_PROMOTION.md
```

The note must state:

```text
Promoted checkpoint:
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105

Promotion label:
V9 Qwen Connection V2 / viable Qwen3 core decoder replacement

Key distinction:
No Stage2 LoRA. This is the Stage1B2350 Qwen connector-only anchor with connector_output_scale=1.05 materialized.

Reason:
Passes clean, blank, shuffled, wrong-image, perturbation, POPE, GQA, generation hygiene, and leakage gates against the V3 robust LLaMA-based incumbent.
```

Also include paths to the final no-override artifacts and leakage audit.

## Final decision rule

Promote only if the materialized no-override checkpoint passes the gates. If the materialized checkpoint fails but the old scale-override artifacts passed, do not promote. In that case, debug materialization or checkpoint loading first rather than launching new experiments.

If promotion succeeds, make this checkpoint the current Qwen baseline for future work.
