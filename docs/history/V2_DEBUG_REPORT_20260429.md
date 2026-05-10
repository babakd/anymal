# V2 Debug Report - 2026-04-29

Systematic debug pass for the completed V2 learned-compressor baseline:

- Stage 1: `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Stage 2: `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`

Generated artifacts:

- `v2_quality_audit.json`
- `v2_quality_probe_full.json`
- `v2_compare_final_training_prompt.json`
- `v2_probe_direct_object_trainprompt_50.json`
- `v2_probe_direct_object_trainprompt_100.json`
- `v2_compare_direct_object_trainprompt_50_training.json`
- `v2_compare_direct_object_trainprompt_100_training.json`
- `v1_v2_compare_direct_object_trainprompt_50.json`
- `v1_v2_compare_direct_object_trainprompt_100.json`

## Corrected VQA Protocol Update

After the first full-run reads, the VQA measurement path itself was found to be
too loose for model selection:

- The evaluator counted padded generation tokens, making many runs look like
  they always generated exactly 32 tokens.
- Generation stopped only on the tokenizer EOS id, while training supervises
  `<|eot_id|>`.
- The VQA prompt used a legacy `Question: ... Answer:` format instead of the
  Stage 2 training chat template.

The corrected scoreboard uses `prompt_style=training_chat`, trims generations at
the first stop token, and reports `hit_max_new_tokens_rate`. Under this protocol
the V1 baseline is:

| Model | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate | Hit max |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 7.57% | 4.86% | 3.90% | 14.19% | 10.93 | 96.7% | 3.3% |
| Original V2 balanced-mix checkpoint 3000 | 0.07% | 0.23% | 0.00% | 0.10% | 4.82 | 96.0% | 4.0% |

Corrected candidate reads so far:

| Model | Checkpoint | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V2 balanced-mix-light from 256/384-expanded Stage 1 | 300 | 0.10% | 0.23% | 0.13% | 0.00% | 4.43 | 97.0% | stopped |
| V2 normalized light direct-object from 256/384-expanded Stage 1 | 1000 | 0.13% | 0.23% | 0.13% | 0.10% | 3.65 | 97.0% | stopped |
| V2 normalized direct-object from 256/384-expanded Stage 1 | 1000 | 0.37% | 0.46% | 0.52% | 0.10% | 3.71 | 97.3% | stopped |

These runs are no longer viable quality candidates. The active quality path is
now the true 384-query Stage 1 checkpoint:

`/checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000`

Two full Stage 2 branches have been launched from it and must be judged only at
scheduled checkpoints 300, 1000, 2000, and 3000 using the corrected protocol:

- `v2-stage2-384q-balanced-mix-light-3000-20260429`
- `v2-stage2-384q-light-direct-object-3000-20260429`

## Fixes Landed

- Added metadata value validation via `validate_checkpoint_metadata_values()` so
  V2 loaders can reject mismatched connector/token-budget checkpoints, not just
  mismatched architecture.
- Hardened V2 `from_pretrained()` to require `projector.pt` and
  `token_compressor.pt`, validate `vision_encoder_type`, `token_compressor_type`,
  `max_image_tokens`, and `min_image_tokens`, and avoid fragile nested PEFT
  wrapping.
- Updated V1/V2 comparison helpers to unload existing PEFT adapters before
  loading saved LoRA adapters.
- Reworked `v2_compare_inference.py` to compare the final Stage 1/Stage 2
  checkpoints by default, use the exact training system prompt by default, and
  build image placeholder blocks as explicit token IDs with an exact count check.
- Added `v2_quality_diagnostics.py` for checkpoint/data audit and image-use
  ablation probes.
- Added focused tests for V2 metadata mismatch detection and Stage 1
  256-query-to-Stage 2 384-query expansion.
- Added a V1/V2 same-prompt comparison runner so V1 ablation-F and V2 candidates
  are evaluated through a matched chat-format prompt path.
- Added Stage 2 continuation support from a full V2 fine-tune checkpoint, plus
  COCO-instance-derived direct object/count/vehicle SFT data.
- Fixed Stage 2 label construction to tokenize role segments explicitly, so
  assistant masks no longer depend on character offsets after expanding the
  image-placeholder block.
- Normalized Mix-665K direct-answer data to one user turn and one assistant
  answer turn. The previous filter selected the first short answer but preserved
  the full original multi-turn source conversation.
- Updated dataset diagnostics to decode supervised labels separately from the
  surrounding raw token window.
- Disabled the broken local `--background` path in favor of durable
  `modal run --detach` jobs.
- Fixed VQA and captioning evaluation to trim generations at the first stop token
  and to report max-token hits.
- Updated VQA checkpoint evaluation to default to the Stage 2 training chat
  prompt, including exact V2 image placeholder insertion.
- Updated LLaMA generation to stop on both tokenizer EOS and `<|eot_id|>` when
  available.
- Added checkpoint cleanup preservation for scheduled evaluation milestones
  `300`, `1000`, `2000`, and `3000`.

## Inference Path

Audit confirms the final checkpoints are structurally valid:

- Stage 1 metadata: `architecture=anymal_v2`, `token_compressor_type=learned`,
  `max_image_tokens=256`, `min_image_tokens=256`, `vision_encoder_type=siglip2`.
- Stage 2 metadata: `architecture=anymal_v2`, `token_compressor_type=learned`,
  `max_image_tokens=384`, `min_image_tokens=384`, `vision_encoder_type=siglip2`.
- Stage 1 compressor `pool_queries`: `[256, 1152]`.
- Stage 2 compressor `pool_queries`: `[384, 1152]`.
- Stage 2 LoRA adapter exists at checkpoint `llm/adapter_model.safetensors` with
  rank `64`, alpha `16`, target modules `q/k/v/o`, `gate/up/down`.

Conclusion: the final inference path is not missing the projector, compressor,
or LoRA adapter. The previous comparison script had real measurement risks
because it defaulted to canary checkpoints, did not validate token-budget
metadata, and did not assert the post-tokenization placeholder count.

## Prompt Matching

The final 24-example comparison was rerun with the exact Stage 2 training system
prompt. Stage 2 is more instruction-following than Stage 1, but remains verbose:

- Stage 1 average response length at `max_new_tokens=96`: `13.8` words.
- Stage 2 average response length at `max_new_tokens=96`: `63.8` words.
- Stage 1 exact question echo count: `2/24`.
- Stage 2 exact question echo count: `0/24`.
- Stage 2 rough long/unfinished count at the cap: `15/24`.

Prompt mismatch was a confounder for style, but not the primary cause of visual
misses.

## Image-Use Probes

Full probe grid covered:

- Correct image
- Wrong image
- Blank gray image
- Text-only
- Training prompt and strict prompt
- `max_new_tokens=32,64,128,192`

Key results:

- `000000291841.jpg`, "What vehicle is shown?"
  - Correct image: `city bus` / strict `Bus`
  - Wrong image: `train`
  - Blank/text-only: generic vehicle priors
- Same image, "What object is this?"
  - Training prompt correct image: `bicycle`
  - Strict prompt correct image: `Bus`
- `000000145989.jpg`, train color
  - Correct image: `orange and black` / strict `Orange`
  - Wrong image/text-only/blank produce different color priors
- `000000258823.jpg`, giraffe count
  - Correct image: `three` / strict `3`
  - Wrong image: `no giraffes` / strict `0`
  - Blank/text-only: low-count priors

No responses varied across the tested token caps. Conclusion: the image path is
alive, and the dominant issue is prompt/category grounding plus style, not a
dead connector or generation-length artifact.

## Stage 1 Data Reality

The live Modal volume confirms true LLaVA-Pretrain data was staged and available:

- Manifest: `/checkpoints/llava_pretrain/manifest.json`
- Zip: `/checkpoints/llava_pretrain/images.zip`
- Manifest `num_images`: `558128`
- Zip image members: `558128`
- Annotation count: `558128`

Conclusion: this final baseline should not be dismissed as the old COCO-fallback
Stage 1. It did train on true zip-backed LLaVA-Pretrain images/captions.

## Stage 2 Data Reality

Balanced mix audit:

- Cached COCO images: `81479`
- Instruct-150K raw/kept: `157712/157712`
- Mix-665K raw/kept: `665298/338470`
- Balanced epoch length: `676940`

Answer/style stats:

| Source | Avg words | P50 | P90 | Short-answer rate | Chatty phrase rate |
|---|---:|---:|---:|---:|---:|
| Instruct-150K | 78.9 | 91 | 140 | 0.002% | 14.4% |
| Filtered Mix-665K | 37.8 | 4 | 120 | 42.1% | 6.7% |

Conclusion: the mix contains many short-answer examples, but the Instruct source
is extremely long and chatty. Since balanced sampling oversamples both sources
to equal source frequency, long-form behavior remains strongly represented.

## Repair Run

The successful repair was a short continuation from the final V2 Stage 2
checkpoint:

- Base checkpoint:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`
- Selected repair checkpoint:
  `/checkpoints/finetune-output/v2-stage2-direct-object-trainprompt-600-20260429/checkpoint-100`
- Dataset mode: `balanced_mix_direct_object_trainprompt`
- Sources:
  - Instruct-150K: `157712`
  - filtered Mix-665K: `338470`
  - natural direct-answer Mix subset: `75016`
  - COCO-instance synthetic object/count/vehicle questions: `274460`
- Optimization: continued LoRA/projector/compressor training with
  `learning_rate=5e-6`, `lora_learning_rate=7e-5`, `batch_size=2`.

The run was stopped after checkpoint 100 because checkpoint 50 already repaired
the targeted object/category failure and checkpoint 100 slightly improved the
broad comparison without losing the probe gains.

## Repair Results

Targeted image-ablation probe:

| Checkpoint | Target correctness | Changed vs blank | Changed vs wrong | Changed vs text | Avg words |
|---|---:|---:|---:|---:|---:|
| Final V2 Stage 2 | 87.5% | 100.0% | 100.0% | 100.0% | 3.8 |
| Repair checkpoint 50 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0 |
| Repair checkpoint 100 | 100.0% | 100.0% | 100.0% | 100.0% | 1.6 |

Strict-prompt checkpoint-100 probe answers:

- `000000291841.jpg`, "What vehicle is shown?": `bus`
- `000000291841.jpg`, "What object is this?": `bus`
- `000000145989.jpg`, "What color is the train?": `Orange`
- `000000258823.jpg`, "How many giraffes are visible?": `3`

Broad 24-example comparison against original V2:

| Run | Avg words | Median words | Long at 96-token cap | Chatty | Bus case |
|---|---:|---:|---:|---:|---|
| Final V2 Stage 2 | 63.8 | 83.5 | 15/24 | 2/24 | city bus sentence |
| Repair checkpoint 50 | 61.3 | 83.0 | 15/24 | 3/24 | `Bus` |
| Repair checkpoint 100 | 61.1 | 82.5 | 13/24 | 2/24 | `bus` |

Same-prompt V1 versus selected V2 checkpoint:

| Run | Avg words | Median words | Long at 96-token cap | Chatty | Bus case |
|---|---:|---:|---:|---:|---|
| V1 ablation-F checkpoint 500 | 62.6 | 83.0 | 14/24 | 4/24 | long city-bus sentence |
| V2 repair checkpoint 100 | 61.1 | 82.5 | 13/24 | 2/24 | `bus` |

Qualitatively, checkpoint 100 keeps long-form answers for reasoning/caption
questions while producing short direct answers for simple object/color/category
questions. It also improves several Stage 1 failures, including the bus terminal,
banana, horse-jump, and rural-truck examples. The known spatial boat/land
example remains wrong in both V1 and the repaired V2, so spatial relation
grounding needs a separate targeted slice.

## Anti-Overfit Evaluation Protocol

The short repair probes above are canaries, not model-selection metrics. They are
useful because they isolate known failure modes, but they are too small and too
visible to optimize against. Future V2 candidates should be selected only from
predeclared checkpoints and held-out benchmark slices.

For the 2026-04-29 full V2 run, the fixed external scoreboard is:

- Dataset: VQAv2 validation questions with cached COCO val2014 images.
- Subset: deterministic 1000-sample subset, `subset_seed=42`.
- Minimum valid sample gate: 500; benchmark eval fails loudly below that.
- Reported metrics: overall accuracy, answer-type accuracy for `yes/no`,
  `number`, and `other`, average generated tokens, and EOS rate.
- Scheduled candidate checkpoints: 300, 1000, 2000, and 3000. Do not add
  unscheduled checkpoint probes just because a nearby checkpoint looks promising.
- Canary probes may be run after the held-out scoreboard, but they should explain
  behavior, not choose the winner.
- Confirmatory evaluation should use a separate deterministic VQA image cache
  after the candidate shortlist is frozen, so expanding image coverage does not
  move the primary scoreboard mid-run.

Superseded legacy held-out VQA scoreboard, 1000 samples, seed 42:

| Model | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate |
|---|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 5.03% | 2.55% | 9.10% | 0.00% | 32.0 | 0.0 |
| Original V2 balanced-mix checkpoint 3000 | 6.70% | 6.48% | 4.09% | 10.69% | 32.0 | 0.0 |

These numbers are retained only as provenance. They used the legacy VQA prompt
and padded-token generation accounting, so they must not be used for checkpoint
selection.

Scheduled full-run checkpoint reads:

| Model | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| V2 direct-object full checkpoint 300 | 7.63% | 5.79% | 12.67% | 0.87% | 32.0 | 0.0 | Diagnostic only; superseded |
| V2 normalized direct-object checkpoint 300 | 3.50% | 3.94% | 5.72% | 0.00% | 32.0 | 0.0 | Active full run; weak early read |
| V2 normalized light direct-object checkpoint 300 | 7.27% | 9.03% | 10.85% | 1.17% | 32.0 | 0.0 | Active full run; first corrected read beats V1/original V2 overall |

This establishes two useful constraints: V2 already beats V1 overall on this
held-out slice, but it regresses on `other` questions. A better V2 needs to
improve the overall score without merely trading away broad open-answer
grounding for yes/no or counting gains.

## Superseded Full V2 Training Run

The first full V2 Stage 2 run was launched from the real V2 Stage 1 checkpoint
instead of continuing from the short inspected repair checkpoint:

- Run name: `v2-stage2-direct-object-full-3000-20260429`
- W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/1mhpwbki`
- Base checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Dataset mode: `balanced_mix_direct_object_trainprompt`
- Target steps: `3000`
- Batch size: `2`
- Base learning rate: `1e-5`
- LoRA learning rate: `2e-4`
- GPU: `h100`

The run uses the object/direct-answer data hypothesis from the short repair, but
it restarts from Stage 1 so the final candidate is not a cherry-picked
continuation of the known canary-winning checkpoint. It also uses the hardened
VQA benchmark path: expanded VQA image cache, deterministic held-out subset,
answer-type breakdowns, and fail-loud missing/empty eval behavior.

In-training validation loss:

| Step | Avg eval loss | Status |
|---:|---:|---|
| 300 | 1.0627 | Healthy |
| 600 | 1.0144 | Improved; checkpoint saved |

This run and the parallel natural-direct branch were stopped after discovering
the data-path issue above. Their checkpoints should not be used for model
selection.

## Corrected Full V2 Training Runs

After the normalized direct-answer fix, a one-step remote sanity run completed
successfully:

- Run name: `v2-stage2-sanity-normalized-direct-1step-20260429`
- Supervised-only previews: answer-only strings ending in `<|eot_id|>`.
- Result: checkpoint-1 saved after one optimizer step.

Three full corrected runs are now active:

| Run | Dataset | LR | LoRA LR | Modal app | W&B |
|---|---|---:|---:|---|---|
| `v2-stage2-normalized-direct-object-full-3000-20260429` | `balanced_mix_direct_object_trainprompt` | `1e-5` | `2e-4` | `ap-42JwN1un0vfyekswO9IqFE` | `https://wandb.ai/babakdam/anymal-finetune/runs/4mi0p44q` |
| `v2-stage2-normalized-light-direct-object-full-3000-20260429` | `concat_mix_direct_object_light_trainprompt` | `5e-6` | `7e-5` | `ap-AKiUSa9ZbCGBj4QubRRm9n` | `https://wandb.ai/babakdam/anymal-finetune/runs/ozalncdz` |
| `v2-stage2-balanced-mix-light-3000-20260429` | `balanced_mix` | `5e-6` | `7e-5` | `ap-G4EThJMPjngFoCxLghhzmQ` | `https://wandb.ai/babakdam/anymal-finetune/runs/hf1i064o` |

Both start from
`/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
and keep the same scheduled external held-out VQA reads at steps 300, 1000,
2000, and 3000. The training-loop step eval is validation loss only; VQA
checkpoint reads are launched separately to keep the selection protocol explicit.
The light branch is the anti-oversampling control: it uses concat sampling and
lower LoRA pressure, so a result must generalize rather than merely memorize
short COCO-style labels. The balanced-mix light hedge removes direct-answer
oversampling entirely while preserving the corrected label-mask/data path.

## Diagnosis Matrix

| Hypothesis | Current evidence | Status |
|---|---|---|
| Inference prompt mismatch | Exact training prompt rerun improves fairness, strict prompt fixes brevity. Visual errors remain prompt-sensitive. | Confounder, not root cause |
| LoRA load bug | Final adapter exists and loads; PEFT wrapping hardened. Outputs are coherent and image-sensitive. | Unlikely after fix |
| Stage 1 weak/fallback data | Live volume confirms true 558K zip-backed data. Captions are web-caption noisy. | Fallback ruled out; caption noise remains plausible |
| Stage 2 style overfit | Long answers, cap pressure, high chatty/long-answer source stats. Strict prompt fixes style. | Supported |
| Compressor bottleneck | Learned compressor is shallow, but image ablations change outputs correctly; a 250-step Perceiver Stage 1 probe was repetitive and unusable. | Possible, not supported by current evidence |
| Eval mismatch | VQA aggregate is noisy; targeted probes are more diagnostic. | Supported |

## Recommended Next Experiments

1. Let the two corrected full runs reach their scheduled checkpoints unless a
   hard health failure appears. Do not evaluate unscheduled neighboring
   checkpoints.
2. Select only after comparing the scheduled held-out VQA metrics and answer-type
   breakdowns for both corrected branches against V1 and original V2.
3. Add a fixed targeted eval set to CI/Modal eval that reports image-ablation
   sensitivity, strict-prompt accuracy, chatty prefix rate, and average words.
4. Add a spatial-relation repair slice for boat/land, left/right, in/on/near, and
   relative-position questions; the current object/count repair does not address
   that class.
5. Revisit the Perceiver compressor only after it has a healthy Stage 1
   checkpoint; checkpoint 250 was not a meaningful connector comparison.
6. Consider a Stage 1 caption-quality ablation after connector capacity is tested:
   the true LLaVA-Pretrain path is real, but sampled captions include noisy web
   fragments and stock-photo metadata.
