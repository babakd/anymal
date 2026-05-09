# AnyMAL V3 Architecture Plan

## Goal

V3 should fix the failure modes introduced by V2 without discarding the useful
parts of the V2 investigation. The target is not "more tokens" or "bigger
encoder" by default; the target is better visual-language grounding under a
held-out, non-cherry-picked evaluation protocol.

V2 remains preserved as `anymal_v2`. V3 should be a new architecture label,
`anymal_v3`, so V1/V2/V3 checkpoints and ablations stay comparable.

## Diagnosis From V2

The V2 image path was alive, but held-out VQA accuracy stayed far below V1.
The most likely causes were:

- The learned compressor was too shallow: a single learned-query cross-attention
  pool plus MLP projection did not provide enough trainable translation between
  SigLIP2 features and LLaMA token space.
- The 384-token visual prefix made the LLM's job harder. More visual tokens
  increased sequence length and attention diffusion without enough evidence that
  the model learned to use them.
- Stage 1 and Stage 2 changed token counts in early runs, creating avoidable
  alignment instability.
- Stage 2 data overrepresented long, chatty instruction behavior relative to
  short grounded answers.
- Caption alignment loss was not a reliable proxy for VQA/object/color/count
  grounding.

## V3 Model Contract

V3 should keep the strongest V2 idea, SigLIP2 at 384px, but replace the weak
connector with a V1-style trainable resampler.

Recommended default:

```yaml
architecture: anymal_v3
vision_encoder_type: siglip2
vision_model_name: google/siglip2-so400m-patch14-384
connector_type: perceiver_resampler
num_image_tokens: 128
connector_layers: 6
connector_heads: 16
connector_ff_mult: 4
project_directly_to_llm_dim: true
use_qlora: true
lora_r: 64
lora_alpha: 16
```

Key choices:

- Use a deep residual Perceiver resampler that projects SigLIP2 patch features
  directly into LLaMA hidden space.
- Do not use the V2 learned compressor as the default.
- Do not use a separate shallow token compressor plus MLP bottleneck as the
  default connector.
- Use a fixed image-token count in Stage 1 and Stage 2. Start with 128 tokens.
- Treat 64 and 192 tokens as ablations, not defaults.
- Keep the vision encoder frozen at first. Unfreezing SigLIP2 should be a later
  high-cost ablation only after the connector/data recipe clears V1.

Why 128 tokens:

- V1's 64-token bottleneck likely helped force compact usable visual summaries.
- V2's 384-token prefix likely made attention diffuse and training expensive.
- 128 gives V3 more capacity than V1 while avoiding the worst long-prefix cost.

## Training Curriculum

V3 should use a curriculum that separates alignment from grounded instruction
behavior.

### Stage 1A: Caption Alignment

Train only the connector on true LLaVA-Pretrain images/captions.

Purpose:

- Align SigLIP2 feature geometry to LLaMA token space.
- Teach the visual prefix to support natural image descriptions.

Defaults:

```yaml
max_steps: 20000
learning_rate: 2.0e-4
train_modules: connector_only
num_image_tokens: 128
```

### Stage 1B: Grounding Alignment

Continue connector-only training on train-split short-answer data.

Sources should be train-only:

- VQAv2 train2014 direct answers
- COCO train object/category/count/color prompts
- Visual Genome or GQA train split if available
- LLaVA/Mix short-answer subsets filtered to one user turn and one assistant
  answer

Purpose:

- Teach direct object, color, count, yes/no, and spatial grounding before LoRA
  can learn language shortcuts.

Defaults:

```yaml
max_steps: 5000-10000
learning_rate: 1.0e-4
train_modules: connector_only
answer_style: direct_when_question_is_direct
```

### Stage 2: Instruction Tuning

Train connector plus LoRA, but keep the mixture explicitly grounded.

Suggested sampling weights:

```yaml
vqa_train_direct: 0.30
coco_object_count_color_direct: 0.20
short_llava_mix: 0.20
llava_instruct_general: 0.20
caption_long_form: 0.10
```

Rules:

- Direct questions should supervise direct answers.
- Caption/explain prompts should supervise longer answers.
- Do not balance by raw dataset name if that oversamples long chatty sources.
- Keep in-training held-out VQA disabled. Run scheduled external evals only.

## Evaluation Contract

The anti-overfit protocol is part of V3, not an afterthought.

Primary selection:

- VQAv2 val2014 fixed 1000-sample seed42 slice for continuity.
- Additional locked slices with different seeds only after a candidate clears V1
  on seed42.
- Scheduled checkpoint reads only: 300, 1000, 2000, 3000, final.

Secondary checks:

- VQAv2 answer-type breakdown: yes/no, number, other.
- Image-use ablations: correct, wrong, blank, text-only.
- Direct-answer style metrics: average generated tokens, EOS rate, max-token
  hit rate, chatty prefix rate.
- Caption quality on a held-out COCO-caption slice.
- A small qualitative canary set only for diagnosis, never model selection.

Promotion bar:

- V3 must beat V1 on the fixed held-out VQA slice before qualitative wins count.
- V3 must preserve image-use ablation sensitivity.
- V3 must not get there by collapsing to one-word answers for caption/explain
  prompts.

## Initial Ablations

Run these only after the V3 default has a clean Stage 1A/1B/2 result:

| Ablation | Change | Question |
|---|---|---|
| V3-64tok | 64 image tokens | Was V1's compact bottleneck the key? |
| V3-192tok | 192 image tokens | Does more capacity help after fixing connector depth? |
| V3-avg | average/interpolation compressor | Is learned resampling necessary? |
| V3-lora-only-stage2 | freeze connector in Stage 2 | Is Stage 2 damaging connector alignment? |
| V3-connector-only-longer | longer Stage 1B | Is grounding mostly pre-LoRA alignment? |

Avoid reintroducing 384 tokens as the first ablation. Test it only after 128
tokens establishes a strong baseline.

## Stable Recipe Snapshot (2026-05-08)

The first stable V3 recipe keeps the architecture fixed and moves the remaining
quality work into Stage 2 data and adapter policy:

```yaml
architecture: anymal_v3
num_image_tokens: 128
connector_type: perceiver_resampler
training.train_adapter: false
model.use_qlora: true
data.recipe: v3_direct_calibration
training.max_steps: 100
```

Why this is the default now:

- The `v3_direct_calibration` LoRA-only branch reached `9.10%` at checkpoint
  100 on the fixed VQAv2 seed-42 training-chat screen, ahead of V1's `7.57%`
  baseline and the earlier `9.03%` V3 screen.
- Freezing the connector generalized better than connector+LoRA. Connector+LoRA
  had a tempting early number score, but generation hygiene degraded and max-token
  hits rose.
- Checkpoint 100 had the best checked validation and held-out VQA result. The
  continuation to checkpoint 150 regressed on validation (`0.6071 -> 0.6229`)
  and held-out VQA (`9.10% -> 8.70%`), so the stable recipe stops at 100 steps.
- V3's main remaining gap is V1-style yes/no calibration, not connector capacity
  or longer Stage 2 training.

Use `configs/finetune_v3.yaml` locally, or on Modal:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage finetune \
  --dataset v3_direct_calibration \
  --max-steps 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --freeze-connector \
  --use-wandb
```

The `v3_direct_calibration` dataset alias freezes the connector by default on
Modal and weights short-answer data toward yes/no recovery while retaining
number/count/object grounding:

```yaml
vqa_train_yes_no: 0.45
vqa_train_number: 0.25
coco_object_count_color_direct: 0.20
vqa_train_other: 0.05
short_llava_mix: 0.05
```

Before promoting a checkpoint, run the saved-artifact guard:

```bash
python3 scripts/check_v3_promotion.py
```

The pass condition is still deliberately modest: V3 must beat V1 on the fixed
held-out VQA slice before qualitative wins count. Claims beyond that require
more seeds or a larger locked slice.

## Implementation Notes

V3 can reuse existing components:

- `SigLIP2Encoder`
- `PerceiverResampler`
- `LlamaWrapper`
- V2 strict placeholder insertion logic
- V2 metadata validation and checkpoint loader hardening
- corrected VQA/caption evaluators

Likely code shape:

- Add `models/anymal_v3.py`.
- Register `anymal_v3` in `models/factory.py` and architecture metadata.
- Use V2's strict placeholder machinery but replace `TokenCompressor` plus
  `MLPBottleneckProjector` with `PerceiverResampler`.
- Save connector weights under a V3-specific name or keep `projector.pt` if the
  checkpoint metadata clearly says `anymal_v3`.
- Add `configs/pretrain_v3_alignment.yaml` and `configs/finetune_v3.yaml`.
- Add Modal aliases for V3 Stage 1A, Stage 1B, and Stage 2 recipes.

## First V3 Run

The first full V3 candidate should be:

1. Stage 1A: SigLIP2 384px, 128-token 6-layer Perceiver, true LLaVA-Pretrain,
   connector-only.
2. Stage 1B: same checkpoint, train-only direct grounding data, connector-only.
3. Stage 2: grounded weighted mixture, connector plus LoRA, no in-training VQA
   eval.
4. External eval at scheduled checkpoints only.

If this still cannot approach V1, the issue is probably not just V2's shallow
compressor. At that point the next suspects are the SigLIP2-to-LLaMA feature
interface itself, data format/label quality, or an architectural need for a
Q-Former-style text-conditioned connector.
