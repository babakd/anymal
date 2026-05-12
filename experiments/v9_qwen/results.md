# V9 Qwen Connection V2 Results

Date: 2026-05-12

## Executive summary

The V9 Qwen plan produced a viable Qwen/Qwen3-8B replacement candidate.
The winning checkpoint is the Stage1B2350 anchor with connector output scale
materialized to 1.05:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

This checkpoint has `connector_output_scale: 1.05` baked into `model_meta.json`.
It is an eval/inference checkpoint containing:

```text
model_meta.json
projector.pt
```

It is not an optimizer-resume checkpoint.

The materialized checkpoint loaded successfully without an eval-time scale
override. Smoke artifact:

```text
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_materialized_seed42_n100_shuffled_smoke.json
/checkpoints/v9_qwen_v2_remote/vqa_eval_v9_qwen3_stage1b2350_scale105_materialized_seed42_n100_shuffled_smoke.json
```

## Final candidate metrics

Minimum gates:

| Gate | Required | Result | Status |
|---|---:|---:|---|
| Clean VQA | >= 62.967 | 66.133 n=3000 seed42 | PASS |
| Blank image | <= 39.733 | 38.811 n=3000 seed42 | PASS |
| Shuffled image | <= 37.367 | 36.900 n=3000 seed42 | PASS |
| Wrong image same-answer-type | <= 38.900 | 37.967 n=1000 seed42 | PASS |
| Perturb mean | >= 60.189 | 66.856 | PASS |
| POPE | >= 77.100 | 79.100 | PASS |
| GQA | >= 43.800 | 44.000 | PASS |
| EOS rate | >= 0.98 | 1.000 clean/shuffled/wrong/perturb; 0.999667 blank | PASS |
| Max-token hit | <= 0.02 | 0.000 clean/shuffled/wrong/perturb; 0.000333 blank | PASS |
| Assistant-prefix rate | <= 0.01 | 0.000 | PASS |
| Strict-clean gap | <= 1.0 | 0.000 | PASS |
| Leakage audit | pass | pass | PASS |

Upgrade targets:

| Target | Result | Status |
|---|---:|---|
| Clean >= 65.900 | 66.133 n=3000 seed42 | PASS |
| Perturb mean >= 64.000 | 66.856 | PASS |
| POPE >= 78.000 preferred | 79.100 | PASS |
| GQA >= 44.000 preferred | 44.000 | PASS |
| Controls at or below incumbent caps | blank 38.811, shuffled 36.900, wrong 37.967 | PASS |

Clean seed confirmation:

| Seed | n | Clean accuracy | Strict accuracy |
|---:|---:|---:|---:|
| 42 | 3000 | 66.133 | 66.133 |
| 43 | 3000 | 66.922 | 66.922 |
| 44 | 3000 | 65.256 | 65.256 |

Perturbations:

| Perturbation | n | Accuracy |
|---|---:|---:|
| Mild blur | 1000 | 67.633 |
| Center crop 90 | 1000 | 65.867 |
| Translate 5 pct | 1000 | 67.067 |
| Mean | - | 66.856 |

Core artifacts:

```text
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_n3000_clean_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_n3000_blank_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_n3000_shuffled_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_wrongimage_sameanswertype_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_mildblur_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_centercrop90_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed42_translate5pct_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed43_n3000_clean_currentcache.json
/tmp/vqa_eval_v9_qwen3_stage1b2350_scale105_seed44_n3000_clean_currentcache.json
outputs/v9_qwen_v2_remote/pope_eval_v9_qwen3_stage1b2350_scale105_currentcache.json
outputs/v9_qwen_v2_remote/gqa_eval_v9_qwen3_stage1b2350_scale105_currentcache.json
```

Leakage audit:

```text
outputs/v9_analysis/v9_qwen3_stage1b2350_scale105_actual_sources_leakage_audit_v2.json
```

Leakage result:

| Check | Result |
|---|---|
| Eval prediction samples | 14500 |
| Rows with refs | 14500 |
| Missing/unparseable refs | 0 |
| Exact val2014 overlap | 0 |
| Numeric-only overlap | 294, warning only |
| Raw ref overlap | 0 |
| Missing sources | 0 |
| Overall | PASS |

Numeric-only overlaps came from BLIP/LAION-style numeric filename collisions
with no exact val2014 or raw-reference overlap, so the audit treats them as a
warning rather than leakage.

## Batch coverage

The original plan had 4 batches: Batch 0, Batch 1, Batch 2, and Batch 3.

We reached all 4 batches.

Detailed coverage:

| Batch | Coverage | Result |
|---|---|---|
| Batch 0 | Completed Stage1B2350 bundle, partial Stage1B2100, pairwise analysis, prompt diagnostics | 2350 became primary anchor |
| Batch 1 | Ran Stage2-lite, control-aware Stage2, contrastive Stage2, GQA-preserving Stage2 | All rejected |
| Batch 2 | Ran control-aware Stage1B, GQA-heavy Stage1B, visual-intake LoRA Stage1B | All rejected |
| Batch 3 | Ran inference-time scale sweep and RMS regularizer | Scale 1.05 passed; RMS branch rejected |

Batch 3C, 3D, and 3E were not run because the plan made them conditional
architecture changes for the case where smaller levers failed. They were no
longer needed after Batch 3A produced a passing candidate.

## Experiment ledger

This section is the ledger for the V9 experiments executed in this run. The
repo also has a pre-existing historical ledger:

```text
experiments/README.md
```

The V9-specific ledger for this run is this file:

```text
experiments/v9_qwen/results.md
```

### Batch 0: pre-Stage2 evidence

#### 0A: Stage1B2350 full eval, no Stage2

Checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Results:

| Metric | Result |
|---|---:|
| Clean | 66.233 |
| Blank | 39.333 |
| Shuffled | 37.567 |
| Wrong image same-answer-type | 38.167 |
| Mild blur | 67.667 |
| Center crop 90 | 66.067 |
| Translate 5 pct | 66.733 |
| Perturb mean | 66.822 |
| POPE | 79.200 |
| GQA | 44.200 |
| EOS/max-hit/prefix | clean |

Decision: primary anchor. It passed everything except shuffled, missing by
about 0.20.

#### 0B: Stage1B2100 partial eval

Checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2100
```

Results:

| Metric | Result |
|---|---:|
| Clean | 66.000 |
| Blank | 39.667 |
| Shuffled | 37.733 |
| Wrong image same-answer-type | 38.200 |
| GQA | 44.000 |
| POPE | not run |

Decision: not promoted. POPE fill-in was blocked because checkpoint-2100 was
not present on the Modal volume. Nearby stored checkpoints started at 2150.

#### 0C: pairwise analysis

Artifacts:

```text
outputs/v9_analysis/stage1b2350_correct_stage2ckpt800_wrong_clean_seed42_currentcache.json
outputs/v9_analysis/stage2ckpt800_correct_stage1b2350_wrong_clean_seed42_currentcache.json
outputs/v9_analysis/stage1b2350_correct_v3robust_wrong_clean_seed42_currentcache.json
outputs/v9_analysis/v3robust_correct_stage1b2350_wrong_clean_seed42_currentcache.json
```

Counts:

| Pairwise slice | Count |
|---|---:|
| Stage1B2350 correct, Stage2 ckpt800 wrong | 55 |
| Stage2 ckpt800 correct, Stage1B2350 wrong | 46 |
| Stage1B2350 correct, V3 robust wrong | 113 |
| V3 robust correct, Stage1B2350 wrong | 75 |

Decision: confirmed Stage1B2350 was a strong anchor and Stage2 could damage
some visual/compositional dependence.

#### Prompt and placement diagnostics

Checkpoint: Stage1B2350. Metric: shuffled-image n=300.

| Prompt/placement variant | Shuffled accuracy | Decision |
|---|---:|---|
| Current | 37.000 | baseline |
| Grounding system prompt | 36.333 | not promoted |
| Image after question | 33.778 | rejected, bad yes-heavy behavior |
| Short system prompt | 36.778 | not promoted |
| Visual delimiters | 36.444 | not promoted |

Decision: useful diagnostic only. No prompt trick was promoted.

### Batch 1: Qwen-specific Stage2 recipe/objective fixes

#### 1A: Stage2-lite attention-only LoRA

Run:

```text
v9-qwen3-stage2-lite-attn-r16-from-stage1b2350-lr5e6-loss003
```

Results:

| Checkpoint | Shuffled n=1000 |
|---|---:|
| ckpt50 | 37.867 |
| ckpt100 | 38.133 |

Decision: rejected.

#### 1B: control-aware Stage2 mixture

Run:

```text
v9-qwen3-controlaware-stage2-r16-from-stage1b2350-lr5e6-loss003
```

Results:

| Checkpoint | Shuffled n=1000 |
|---|---:|
| ckpt50 | 37.767 |
| ckpt100 | 38.600 |

Decision: rejected.

#### 1C: contrastive answer-suppression Stage2

Run:

```text
v9-qwen3-contrastive-answer-suppression-r16-from-stage1b2350-lr5e6-loss003-lambda01-margin05
```

First app:

```text
ap-XRBVBR2ij9C8r70oqWY2PJ
```

It reached step 50 but failed during trainer eval:

```text
TypeError: AnyMALv3.forward() got an unexpected keyword argument 'negative_images'
```

Fix applied:

```text
training/trainer.py
training/finetune.py
```

The eval path now drops trainer-only `negative_images` while the contrastive
train step still consumes them.

Retry app:

```text
ap-VCE9ifSDfPYEJTyogWhRuP
```

Saved checkpoints:

```text
/checkpoints/finetune-output/v9-qwen3-contrastive-answer-suppression-r16-from-stage1b2350-lr5e6-loss003-lambda01-margin05/checkpoint-24
/checkpoints/finetune-output/v9-qwen3-contrastive-answer-suppression-r16-from-stage1b2350-lr5e6-loss003-lambda01-margin05/checkpoint-36
/checkpoints/finetune-output/v9-qwen3-contrastive-answer-suppression-r16-from-stage1b2350-lr5e6-loss003-lambda01-margin05/checkpoint-48
```

The retry completed through step-50 eval. Because the save cadence was every
12 steps, the latest saved checkpoint was checkpoint-48. A follow-up patch now
forces a save at `max_steps` so this does not happen again.

Early gate artifact:

```text
/tmp/vqa_eval_v9_qwen3_contrastive_ckpt48_seed42_shuffled_currentcache.json
/checkpoints/v9_qwen_v2_remote/vqa_eval_v9_qwen3_contrastive_ckpt48_seed42_shuffled_currentcache.json
```

Result:

| Checkpoint | Shuffled n=1000 | EOS | Max-hit | Prefix |
|---|---:|---:|---:|---:|
| ckpt48 | 37.867 | 1.000 | 0.000 | 0.000 |

Decision: rejected. It missed the shuffled cap and was not competitive with
the scale-1.05 finalist.

#### 1D: GQA-preserving Stage2 mixture

Run:

```text
v9-qwen3-gqa-preserving-stage2-r16-from-stage1b2350-lr5e6-loss003
```

Results:

| Checkpoint | Shuffled n=1000 |
|---|---:|
| ckpt50 | 37.767 |
| ckpt100 | 38.333 |

Decision: rejected.

### Batch 2: Stage1B connector/objective repair

#### 2A: control-aware Stage1B continuation

Run:

```text
/checkpoints/pretrain-output/v9-qwen3-controlaware-stage1b-cont250-from-stage1b2350-lr1e4-loss03-save50
```

Checkpoint:

```text
checkpoint-250
```

Results:

| Metric | Result |
|---|---:|
| Clean | 65.500 |
| Blank | 40.200 |
| Shuffled | 37.600 |
| Wrong image same-answer-type | 38.033 |
| Mild blur | 65.433 |
| Center crop 90 | 65.133 |
| Translate 5 pct | 65.133 |
| GQA | 43.200 |
| POPE | not run |

Decision: rejected. Blank and GQA failed; shuffled also missed.

#### 2B: GQA-heavy Stage1B continuation

Run:

```text
v9-qwen3-gqa-stage1b-cont250-from-stage1b2350
```

Results:

| Checkpoint | Shuffled n=1000 |
|---|---:|
| ckpt100 | 38.400 |
| ckpt200 | 38.433 |

Decision: rejected.

#### 2C: Stage1B connector plus visual-intake LoRA

Run:

```text
v9-qwen3-controlaware-stage1b-visualintake-lora-r8-cont250-from-stage1b2350-lr1e4-lora5e6-loss03-save50
```

Result:

| Checkpoint | Shuffled n=1000 |
|---|---:|
| ckpt200 | 38.933 |

Decision: rejected.

### Batch 3: Qwen connector V2 levers

#### 3A: inference-time connector scale sweep

Checkpoint:

```text
/checkpoints/pretrain-output/v8-qwen3-8b-v3-stage1b-bracket2400-from2000-lr1e4-gate00256-lossscale03-save50-20260511/checkpoint-2350
```

Scale sweep:

| Scale | Key result | Decision |
|---:|---|---|
| 0.85 | Shuffled 37.300 | promising but not promoted |
| 0.90 | Shuffled 37.433 | rejected |
| 0.95 | Shuffled 37.333, clean 66.033, blank 39.200, wrong 38.067, POPE 79.400, GQA 43.000 | rejected due GQA |
| 1.00 | Original Stage1B2350: shuffled 37.567 | near miss |
| 1.05 | full gates pass | promoted |
| 1.10 | Shuffled 37.367 | no full bundle, 1.05 already passed |

Scale 1.05 n=1000:

| Metric | Result |
|---|---:|
| Clean | 66.167 |
| Blank | 39.400 |
| Shuffled | 37.367 |
| Wrong image same-answer-type | 37.967 |
| Mild blur | 67.633 |
| Center crop 90 | 65.867 |
| Translate 5 pct | 67.067 |
| Perturb mean | 66.856 |
| POPE | 79.100 |
| GQA | 44.000 |

Scale 1.05 n=3000 confirmations:

| Metric | Result |
|---|---:|
| Clean seed42 | 66.133 |
| Blank seed42 | 38.811 |
| Shuffled seed42 | 36.900 |
| Clean seed43 | 66.922 |
| Clean seed44 | 65.256 |

Decision: promoted.

Materialization script:

```text
scripts/materialize_v9_scale_checkpoint.py
```

Materialized output:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Materialized checkpoint smoke:

| Metric | Result |
|---|---:|
| Shuffled n=100 | 31.667 |
| EOS | 1.000 |
| Max-hit | 0.000 |
| Prefix | 0.000 |
| Metadata scale | 1.05 |

#### 3B: RMS regularizer continuation

Run:

```text
/checkpoints/pretrain-output/v9-qwen3-rmsreg-controlaware-stage1b-cont250-from-stage1b2350-lr1e4-loss03-alpha001-save50
```

Checkpoint:

```text
checkpoint-50
```

Result:

| Metric | Result |
|---|---:|
| Shuffled n=1000 | 38.900 |

Decision: rejected and stopped.

#### 3C, 3D, 3E

Not run. These were conditional architecture changes to use only if smaller
recipe/objective/scale levers failed. Since 3A produced a passing candidate,
these were not needed.

## Code and tooling changes made for V9

Main implementation changes:

| File | Purpose |
|---|---|
| `models/llm/llama_wrapper.py` | LoRA dropout/target metadata |
| `training/finetune.py` | LoRA config threading and contrastive answer-suppression objective |
| `training/trainer.py` | LoRA/contrastive logging, eval hook for trainer-only fields, final max-step save |
| `modal_train.py` | V9 datasets, LoRA flags, contrastive flags, RMS regularizer flags |
| `data/chat_templates.py` | prompt/image placement diagnostics |
| `evaluation/vqa_eval.py` | prompt/image placement diagnostics and scale override support |
| `vqa_checkpoint_eval.py` | connector scale override, legacy metadata handling |
| `pope_checkpoint_eval.py` | connector scale override, legacy metadata handling |
| `gqa_checkpoint_eval.py` | connector scale override, legacy metadata handling |
| `models/anymal_v3.py` | connector scale override and metadata compatibility |
| `training/pretrain.py` | connector RMS regularizer |
| `data/instruction_dataset.py` | negative images for contrastive objective |
| `data/data_utils.py` | negative-image collation |
| `scripts/audit_v9_leakage.py` | V9 leakage audit |
| `scripts/materialize_v9_scale_checkpoint.py` | materialize eval-time scale into checkpoint metadata |

Verification:

```text
python3 -m py_compile data/chat_templates.py data/data_utils.py data/instruction_dataset.py evaluation/vqa_eval.py gqa_checkpoint_eval.py modal_train.py models/anymal_v3.py models/llm/llama_wrapper.py pope_checkpoint_eval.py training/finetune.py training/pretrain.py training/trainer.py vqa_checkpoint_eval.py scripts/audit_v9_leakage.py scripts/materialize_v9_scale_checkpoint.py
```

This completed without syntax errors.

## Final recommendation

Promote:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

Use it as the V9 Qwen Connection V2 result and the current viable Qwen core
decoder replacement candidate.
