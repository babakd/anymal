# AnyMAL V4 Architecture Plan

Implementation/research update: see `V4_RESEARCH_RECIPE_20260508.md` for the
current v4 code path, source-backed research notes, configs, and run commands.

## Purpose

V4 should be a new architecture label, `anymal_v4`, whose job is not merely to
beat V1. V3 already did that on the corrected fast screen. V4 should take the
V1/V2/V3 lessons and move the model closer to modern VLM behavior: higher
resolution when useful, spatially aware visual tokens, a stronger multimodal
connector, disciplined training recipes, and evaluation that catches both
accuracy and generation-pathology regressions.

Keep V1, V2, and V3 load paths intact. V4 must be comparable against the saved
artifacts, not a silent replacement.

## Current Baseline To Beat

Use the corrected VQAv2 val2014 fast screen as the continuity benchmark:

- `max_samples=1000`
- `seed=42`
- `prompt_style=training_chat`
- first-stop-token trimming
- EOS includes `<|eot_id|>`
- answer-type breakdown required
- generation hygiene required: EOS rate and max-token-hit rate

Best current checkpoint:

| Model | Overall | Number | Other | Yes/No | EOS | Max-token hits |
|---|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 21.10 | 15.74 | 17.35 | 28.96 | 0.981 | 0.019 |
| V3 direct-calibration LoRA-only checkpoint 100 | 9.40 | 6.48 | 9.10 | 11.08 | 0.999 | 0.001 |
| V4 semantic calibration Stage1B248 checkpoint 100 | 52.40 | 37.04 | 40.87 | 76.09 | 1.000 | 0.000 |

The V4 target is therefore met on this seed-42 fast screen; before calling it
stable, confirmation still needs to:

- Beat V3 overall, not just V1 overall, on at least one additional seed or a
  larger locked slice.
- Preserve V3's clean generation behavior: EOS `>= 0.98`, max-token hits
  `<= 0.02`.
- Preserve the large yes/no and `other` gains after the corrected
  post-processing path.

Current post-ablation recommendation: promote the Stage1B248
`v4_semantic_calibration` checkpoint as the evidence-backed V4 recipe candidate
under the corrected VQA evaluator. Its architecture is the spatial connector
family: SigLIP2-So400m at 384px, 64 global + 64 local visual tokens, 2D
position features, strict 128-token image placeholder contract, direct
projection to LLM dimension, and tiny output gate `0.0001`. Keep DeepStack-lite
as a wired ablation only. The lean A1 bottleneck spatial connector remains the
efficiency target, but it needs a repeat of the promoted recipe before replacing
the Stage1B248 evidence-backed checkpoint.

Artifacts to keep in view:

- `EXPERIMENTS.md`
- `V3_ARCHITECTURE_PLAN.md`
- `configs/finetune_v3.yaml`
- `scripts/check_v3_promotion.py`
- `vqa_checkpoint_eval_baselines_1000_seed42_training_chat.json`
- `vqa_eval_v3_direct_calibration_ckpt100_training_chat.json`

## Lessons From V1, V2, And V3

### Evaluation Lessons

- The old VQA numbers were not comparable. The corrected `training_chat`
  protocol changed the leaderboard. Never mix old/default-prompt rows with the
  corrected rows when choosing a checkpoint.
- Fast-screen VQA is useful for triage, not for final claims. Treat `1000`
  samples, seed `42` as the cheap gate. Stable V4 claims should use more seeds
  or a larger held-out slice.
- Accuracy alone is not enough. V2 and early V3 variants could look interesting
  while hitting max tokens or failing yes/no. Always report answer types, EOS,
  max-token hits, average generated tokens, and validation trend.
- Qualitative canaries are diagnostic only. Do not choose checkpoints from a
  handful of memorable generations.

### Architecture Lessons

- V1's compact visual prefix was surprisingly strong. It likely forced useful
  compression instead of dumping too many weak visual tokens into the LLM.
- V2's "more tokens plus shallow learned compressor" direction failed under the
  corrected protocol. The image path was alive, but the connector did not
  translate features into a usable language-space representation.
- V2's 384-token prefix made training and attention harder without proving that
  the LLM could use the extra spatial detail.
- V3's 128-token, 6-layer Perceiver-style connector is the first architecture
  after V1 that clearly beat V1 on the corrected screen.
- Connector+LoRA Stage 2 was not the clean lever in V3. It gave tempting local
  bumps, but generation hygiene degraded badly. Freeze the connector for short
  calibration unless a specific architecture ablation requires otherwise.

### Recipe Lessons

- Generic long-form LLaVA-style Stage 2 data mostly changed style. It did not
  reliably add visual capability and often increased verbosity/truncation.
- Short direct-answer data moved the real metric. V3 improved by focusing on
  answer-type-balanced direct supervision.
- Most useful V3 Stage 2 movement happened early, around 50-100 optimizer
  steps. Continuing to 150 or 300 could regress validation and held-out VQA.
- Yes/no remains the main weakness. V1 still has the strongest yes/no
  calibration, while V3 is much stronger on `other` and cleaner on EOS.
- Raw train loss is noisy for sparse direct-answer mixtures. Use validation
  trend, gradient health, held-out VQA, answer-type movement, EOS, and max-token
  behavior to babysit runs.
- Be willing to stop runs. The V3 direct-calibration 300-step candidate was
  stopped after checkpoint 150 because validation regressed from `0.6071` at
  checkpoint 100 to `0.6229`, and VQA fell from `9.10` to `8.70`.

## V4 Design Hypothesis

V4 should keep V3's proven core but add spatial awareness and high-resolution
capacity in a controlled way. The safest hypothesis is not "more visual tokens";
it is "better selected and better typed visual tokens."

Recommended V4 starting point:

```yaml
architecture: anymal_v4
vision_encoder_type: siglip2
vision_model_name: google/siglip2-so400m-patch14-384
image_resolution: 384
connector_type: spatial_perceiver_resampler
num_global_image_tokens: 64
num_local_image_tokens: 64
num_image_tokens: 128
connector_layers: 3
connector_heads: 8
connector_ff_mult: 2
connector_hidden_dim: 1024
connector_output_scale: 1.0
connector_output_gate_init: 0.0001
use_2d_position_features: true
project_directly_to_llm_dim: false
stage2_default_train_connector: false
use_qlora: true
lora_r: 64
lora_alpha: 16
```

The architectural change from V3 should be explicit:

- V3: one compact set of learned Perceiver queries over the image features.
- V4: separate global and local/spatial latents, with 2D position information
  and optional high-resolution/tiling ablations, still capped to a compact
  token budget by default.

Do not start V4 by simply raising the token count. That repeats the V2 failure
mode.

## Architecture Ablations

Run these in a sequence where each candidate has to clear the V3 baseline gate
before the next expensive variant gets more compute.

### A0: V3 Reproduction Guard

Before changing architecture, reproduce the V3 stable recipe path:

- Load the V3 Stage 1B checkpoint used by the direct-calibration run.
- Run the 100-step LoRA-only direct-calibration recipe.
- Confirm the promotion guard still passes against V1 and lands near the saved
  V3 result.

This catches data-path, prompt, checkpoint, or dependency drift before V4
experiments start.

### A1: Spatial Perceiver With Same Token Budget

Keep 128 total tokens, but split them:

- 64 global latents attend over all SigLIP2 patches.
- 64 local latents attend with 2D position features and/or region-biased query
  initialization.
- Run the Perceiver at hidden dim `1024`, then project both streams into LLaMA
  hidden space as one contiguous visual block.

The earlier 6-layer, 16-head, direct 4096-wide connector is now a failed
ablation: it trained `1.633B` connector params and hit persistent Stage 1A
clipping. The bottleneck1024 connector cut trainables to `44.4M`, but its first
canary still clipped on every logged step, so the next A1 attempt must address
gradient scale, initialization, or the caption-alignment recipe as well as
connector shape. The `connector_output_scale: 0.1` follow-up also clipped on
every W&B row, so output amplitude alone is not the stabilizer. Trainable visual
gate canaries initialized at `0.01`, `0.001`, and `0.0001` also clipped on every
W&B row; even the completed `gate00001` checkpoint ended with
`health/grad_clip_fraction=1.0`. A follow-up that masked all connector gradients
except `projector.output_proj.*` and `projector.output_gate_logit` for the first
30 optimizer steps at LR `1e-5` also failed during warmup with
`health/grad_clip_fraction=1.0`. Switching from web captions to the existing
short-answer `v4_grounding` mixture at LR `1e-5` also failed immediately with
`health/grad_clip_fraction=1.0`. Adding `pretrain_loss_scale=0.1` on that same
grounding-first recipe was the first canary to clear the immediate clipping gate
(`health/grad_clip_fraction=0.04` at step `50`, eval loss `8.0005`), but W&B
raised `recent_loss_window_mean_up_gt_25pct` at step `60` and the run was stopped
with last W&B step `70`; `checkpoint-50` is complete but yellow/red. A larger
accumulation follow-up with effective batch size `64`
(`v4-stage1a-groundingfirst-lossscale01-accum16-gate00001-lr1e5-canary100-20260508-codex`)
kept clipping controlled (`0.1`) but failed the same W&B loss-window gate by step
`20` with no checkpoint. A lower-LR follow-up at `3e-6`
(`v4-stage1a-groundingfirst-lossscale01-lr3e6-canary120-20260508-codex`) fully
controlled clipping (`0.0`) and saved `checkpoint-50`, but still failed the W&B
loss-window gate by step `80` and had worse first eval loss (`8.6358`). The next
A1 answer-focus attempt
(`v4-stage1a-answerfocus-lossscale01-gate00001-lr1e5-canary120-20260508-codex`)
tested answer-type rebalancing on the same loss-scaled recipe, but the monitoring
sub-agent stopped it before eval because W&B clipping rose from `0.30` at step
`10` to `0.50` at step `30` with active `high_grad_clip_fraction`; no checkpoint
landed. A supervised-token-target objective follow-up
(`v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr1e5-canary120-20260508-codex`)
multiplied raw token-mean loss by `supervised_tokens / 8` before the existing
`loss_scale=0.1`. Its first launch clipped cleanly and improved step-50 eval to
`7.8830`, but W&B raised `recent_loss_window_mean_up_gt_25pct` at step `60`.
A direct LR-only shrink to `5e-6`
(`v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr5e6-canary120-20260508-codex`)
was independently judged red by the monitoring sub-agent: W&B still raised
`recent_loss_window_mean_up_gt_25pct` at step `60`, and first eval worsened to
`8.2909`. The decisive follow-up was
`v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120-20260508-codex`,
which kept the LR `1e-5` token-normalized recipe but patched W&B/HealthMonitor
to log accumulated optimizer-step means across the full gradient-accumulation
window. With `train/accumulation_micro_batches=8.0` verified in W&B, the canary
completed green to step `120`: no alerts, clip fraction `0.0`, no loss/grad
spikes, eval improved from `7.8095` at step `50` to `5.1151` at step `100`, and
the final recent loss window improved (`2.1678 -> 1.7965`). Therefore the next
A1 baseline is supervised-token-target normalization plus accumulated-step W&B
health logging. The current A1 default is therefore the bottleneck1024 spatial
connector with `connector_output_scale=1.0`, `connector_output_gate_init=0.0001`,
supervised-token-target normalization, `loss_scale=0.1`, and accumulated
optimizer-step W&B health metrics. Do not go back to output-scale shrinkage,
connector handoff variants, accumulation-only scaling, LR shrinkage alone, or
answer-focus data reweighting alone. The first two `1000`-step extensions were
operational aborts rather than negative model evidence: one was stopped before
the first W&B train row, and one was stopped after a monitor incorrectly
expected `checkpoint-100` even though the implicit save cadence for
`max_steps=1000` was `250`. Use the new `--pretrain-save-steps` flag on long A1
runs and make monitors follow the printed save cadence. The corrected
`extended1000c-save100` launch then passed W&B/artifact gates through step
`300`, with eval improving `5.2072 -> 3.2008 -> 2.6205`, before being stopped at
W&B step `340` for active `recent_loss_spikes` (`1.9874` loss vs `0.9362` EMA,
`2.12x`). Treat `checkpoint-300` as the usable A1 Stage 1A candidate for
downstream checks; do not call the full 1000-step recipe green.

Stage 2A now has two `loss_scale=0.03`, LoRA-only reads. The Stage 1A300 branch
trained health-green but scored only `5.367` overall on the direct-prompt
seed-42 VQA read, below both V1 and V3. The Stage 1B248 branch trained from the
yellow-history grounding checkpoint and reached a lower internal eval loss
(`0.7956`), but it is not green-pure because W&B raised transient
`recent_loss_window_mean_up_gt_25pct` alerts at steps `7` and `92`; final W&B
cleared alerts and artifacts are complete. Its external direct-prompt VQA score
was `7.600` overall, number `4.861`, other `7.083`, yes/no `9.524`, EOS `1.0`,
and max-hit `0.0`. This proves Stage 1B grounding helps, narrowly clears the V1
overall floor, and still fails V3 (`9.100` overall) plus the yes/no recovery
goal. The checkpoint-50 read was lower (`7.433` overall, yes/no `9.038`), so
checkpoint `100` remains the best Stage 2A point for this branch. Keep the alert
history attached and do not promote. The next expensive move was moved upstream:
`v4-stage1b-clean500-from-a1ckpt300-20260509-codex` trained a cleaner Stage 1B
continuation from the accumulated-metrics A1 `checkpoint-300`. It passed W&B and
artifact gates through `checkpoint-400` with monotonic eval improvement to
`2.0128`, zero clipping, correct accumulation, and no W&B loss/grad spike
alerts. It did not complete step `500`: after `checkpoint-400`, logs showed
repeated auto-resume from `checkpoint-400`, W&B stopped at step `410` with state
`crashed`, and the parent stopped the Modal app. Treat `checkpoint-400` as the
usable clean-through-400 Stage 1B candidate for downstream Stage 2A, but label
the run yellow-history/operational-crash rather than green. The first downstream
Stage 2A retry from `checkpoint-400` reused the earlier `loss_scale=0.03`,
LoRA-only recipe and failed by W&B step `3` with active
`high_grad_clip_fraction` and clip fraction `1.0`; parent stopped it at final
W&B step `4`, with no eval rows and no checkpoints. The next accuracy read must
first reduce Stage 2 update scale for this checkpoint. The `loss_scale=0.01`
retry then completed cleanly to step `100`, with eval `1.1746`, zero clipping,
no W&B health alerts, and complete `checkpoint-100`. That checkpoint is the
current Stage1B400 Stage 2A candidate for the direct-prompt VQA screen. The
direct-prompt VQA read then failed: overall `5.400`, number `3.704`, other
`3.379`, yes/no `9.135`, EOS `0.618`, and max-token hits `0.382`. So Stage1B400
plus lower Stage 2 scale is an optimization fix, not an accuracy/hygiene fix.
The next architecture run has now moved to DeepStack-lite Stage 1A. It completed
the 500-step extension cleanly enough for downstream testing: W&B `xb0silsi`
finished with alerts `[]`, correct accumulation `8.0`, no inspector loss/grad
spikes, eval improving through `2.7120`, and complete checkpoints through
`checkpoint-500`. Keep its yellow-history attached: startup W&B sync gap,
early eval lag versus the canary, step-60 objective jump, step-100 subthreshold
loss-window rise, console-only spikes around steps `353` and `499`, and small
nonzero clipping (`0.002000` final). The next expensive move should be Stage 1B
grounding from DeepStack `checkpoint-500`, then Stage 2A direct calibration if
the Stage 1B gate is clean.

Gate: must beat or match V3 overall and hygiene while improving yes/no or
number. If it only improves `other`, keep it as an observation, not the default.

### A2: Token Budget

Compare only after A1 works mechanically:

| Variant | Global | Local | Total | Purpose |
|---|---:|---:|---:|---|
| compact | 48 | 48 | 96 | Test whether stronger selection beats capacity |
| default | 64 | 64 | 128 | V3-comparable budget |
| expanded | 64 | 128 | 192 | Test spatial capacity without V2-scale bloat |

Avoid 384 tokens until 192 clearly beats 128 without generation regressions.

### A3: Resolution And Tiling

Modern VLMs often benefit from high-resolution detail, but the AnyMAL history
says uncontrolled token growth is dangerous. Test high resolution through
selection, not raw prefix expansion.

Candidate ablations:

- single 384px image, V4 spatial connector
- single 448px or 512px image, same 128 selected output tokens
- global 384px image plus one or more local crops/tiles, still capped at 128 or
  192 output tokens
- crop selection from fixed grids before learned/dynamic crop selection

Report OCR/text-like prompts separately if introduced; do not let OCR-only gains
mask VQA regressions.

### A4: Connector Depth And Fusion

Test connector changes only after the spatial/token-budget question has a
winner:

- 4, 6, and 8 Perceiver layers
- residual cross-attention plus FFN vs cross-attention-only
- learned query initialization vs query initialization from pooled visual
  features
- gated global/local fusion before projection vs simple concat

Keep the LLM API stable: contiguous image placeholder block, explicit metadata,
and exact token-count validation.

### A5: Vision Adaptation

Keep SigLIP2 frozen until V4 beats V3 with a frozen encoder. Then test:

- last-block vision LoRA
- small vision adapter on the final feature map
- connector-only Stage 1 followed by vision-adapter Stage 1B

This is high cost and high risk. Do not combine it with new recipes in the same
first run.

## Recipe Ablations

Do recipe ablations on the strongest stable architecture at that point. Early
V4 work should use V3's direct-calibration recipe as the control.

### R1: Yes/No Recovery

Goal: recover V1 yes/no calibration while preserving V3 `other`.

Compare small LoRA-only Stage 2 runs:

| Mix | Yes/No | Number | COCO object/count/color | VQA other | Short LLaVA direct |
|---|---:|---:|---:|---:|---:|
| V3 stable | 0.45 | 0.25 | 0.20 | 0.05 | 0.05 |
| yes/no plus | 0.55 | 0.20 | 0.15 | 0.05 | 0.05 |
| balanced recovery | 0.50 | 0.25 | 0.15 | 0.05 | 0.05 |
| other guard | 0.40 | 0.25 | 0.20 | 0.10 | 0.05 |

Checkpoints: 50, 100, 150 only. Stop if validation and held-out VQA both move
against the candidate after 100.

### R2: Negative And Contrastive Yes/No

If yes/no remains weak, add explicit calibration data rather than just more
yes/no volume:

- balanced yes/no pairs by object, attribute, count, and spatial relation
- hard negatives from COCO categories not present in the image
- avoid label leakage and repeated template artifacts
- keep val2014 out of all training data

Gate: yes/no must improve without dropping `other` below the V3 baseline band.

### R3: Concision And EOS

V3 direct-calibration fixed the max-token problem. Do not lose it.

Test only if V4 starts getting verbose:

- answer-only labels for direct questions
- small concise instruction regularizer
- explicit EOS-bearing labels
- no long-form LLaVA ratio above 5-10% unless a held-out eval justifies it

### R4: Instruction Preservation

After V4 clears VQA gates, run a separate small qualitative and captioning pass
for broad instruction behavior. Do not blend generic instruction data into the
first VQA-selection runs unless the candidate is losing basic instruction
formatting.

## Training Curriculum

### Stage 1A: Caption/Description Alignment

Train connector-only on image-caption/description data. The purpose is visual
to language-space alignment, not final VQA optimization.

Defaults:

```yaml
train_modules: connector_only
vision_encoder: frozen
llm: frozen
max_steps: 3000-10000
eval: caption/description loss plus qualitative fixed set
```

### Stage 1B: Grounding Alignment

Continue connector-only on train-split direct grounding data:

- VQAv2 train direct answers
- COCO object, count, color, and attribute prompts
- GQA/Visual Genome if available and cached
- OCR/text data only if the V4 architecture includes a high-res path designed
  to use it

### Stage 2A: Short Direct Calibration

Freeze the connector. Train LoRA only. Start with the V3 stable recipe and the
R1 variants.

Defaults:

```yaml
train_modules: lora_only
max_steps: 100
checkpoints: [50, 100]
learning_rate: 1.0e-5
lora_learning_rate: 1.0e-5
loss_scale: 0.03
selection: external held-out VQA plus validation trend
```

Current result: the `loss_scale=0.03` Stage 2A recipe completed the 100-step
training run cleanly. The first external VQA reads failed at both checkpoint 50
and checkpoint 100 (`0.10` overall on the seed-42 `training_chat` screen) under
the evaluator's generic system prompt. Re-running checkpoint 100 with the
direct-calibration system prompt fixed generation hygiene (`eos_rate=1.0`,
`hit_max_new_tokens_rate=0.0`) but still scored only `5.367` overall, below the
V1 floor (`7.567`) and V3 incumbent (`9.100`). Treat the loss scale as a
stability requirement only; it does not validate the current Stage 1A300 base or
Stage 2A data recipe for promotion.

Next recipe after the DeepStack failure is `v4_semantic_calibration`. It keeps
the same frozen-connector LoRA-only setup but replaces the plain yes/no source
with a balanced canonical yes/no source and changes the mix to
`0.40` yes/no, `0.20` number, `0.15` COCO object/count/color, `0.20` VQA other,
and `0.05` short LLaVA. The gate is not merely lower loss: it must beat the best
V4 Stage1B248 branch (`7.600` overall) while preserving EOS `>=0.98`,
max-token hits `<=0.02`, and no active W&B health alerts.

First semantic-calibration launch from Stage1B248 validated startup and data
construction but was stopped before checkpoint/eval on repeated active W&B
loss-window alerts at steps `6` and `10`. Do not count it as a negative VQA
result or a green optimization result; it is a strict-monitor stop with no model
artifact.

Second semantic-calibration launch from Stage1B248 used bs4/effective batch
`32` and completed cleanly. W&B finished at step `100` with alerts `[]`, clip
`0.0`, no inspector loss/grad spikes, eval `0.8416`, and complete
`checkpoint-100`. The initial external VQA read scored only `7.333`, but raw
prediction capture showed duplicated decoded chat-role prefixes
(`assistant\n\n<answer>`) were being scored as the literal answer `assistant`.
After correcting VQA post-processing and refreshing V1/V3 comparison artifacts,
the same checkpoint scored overall `52.400`, number `37.037`, other `40.871`,
yes/no `76.093`, EOS `1.0`, max-hit `0.0`. Corrected V1 is `21.100` overall
and corrected V3 incumbent is `9.400`; the promotion guard passes all gates.
Seed-43 confirmation stayed stable at overall `51.367`, number `35.430`, other
`40.370`, yes/no `72.043`, EOS `1.0`, max-hit `0.0`.
Conclusion: the promoted recipe candidate is Stage1B248 plus frozen-connector
LoRA-only `v4_semantic_calibration`, while DeepStack remains a non-default
ablation.

### Stage 2B: Optional Instruction Regularization

Only after Stage 2A clears the V3 gates, add a tiny instruction-preservation
branch. Keep it separate from the core architecture selection so the metric
movement remains interpretable.

## Promotion Protocol

Minimum promotion gates for a V4 candidate:

- Passes the V1 guard used by `scripts/check_v3_promotion.py`.
- Beats V3 direct-calibration checkpoint 100 on the seed-42 fast screen, or
  matches it while materially improving yes/no and preserving `other`.
- EOS `>= 0.98`.
- Max-token hits `<= 0.02`.
- No validation regression like the V3 checkpoint-150 continuation.
- Confirmed on at least one additional seed or a larger held-out VQA subset
  before being called the new stable default.

Suggested future script: extend `scripts/check_v3_promotion.py` into a generic
`scripts/check_vlm_promotion.py` that accepts both a V1 floor and a V3 incumbent
baseline.

## Babysitting Rules For V4 Runs

Use `TRAINING_RUN_BABYSITTING_PLAYBOOK.md`. In addition:

- Check trainable parameter groups at startup. Stage 2A should print connector
  trainables as zero.
- Inspect supervised-label previews. Direct-answer labels should decode as
  short answers plus `<|eot_id|>`, not chat headers.
- Do not wait for planned max steps if validation and held-out reads agree that
  the run is regressing.
- Monitoring agents must lead with W&B, not log tails: latest step, active
  alerts, recent-window loss movement, loss/grad EMAs, clipping fraction, LR,
  eval loss, and checkpoint status must be in every monitor handoff.
- Treat active W&B health alerts as unresolved until W&B clears them; console
  logs alone cannot downgrade a spike or loss-window alert.
- Keep Modal app IDs, W&B URLs, checkpoint paths, and external eval artifact
  paths in `EXPERIMENTS.md` immediately after each run.
- Never promote from a run whose checkpoint metadata does not explicitly say
  `anymal_v4`.

## First V4 Experiment To Run

The first useful V4 experiment is not a giant sweep. It should be:

1. Add `anymal_v4` with V3-compatible checkpoint metadata and a
   spatial/global-local Perceiver connector.
2. Keep total visual tokens at 128.
3. Train Stage 1A/1B connector-only with the same fixed token count.
4. Run the V3 stable 100-step direct-calibration Stage 2, connector frozen.
5. Evaluate checkpoints 50, 100, and 150 on the corrected VQA screen.
6. Promote only if it beats V3 checkpoint 100 or narrows yes/no without losing
   V3's `other` and hygiene wins.

If that fails, V4 should first debug spatial-token design, not jump to 384
tokens or broad instruction data.

## DeepStack-Lite V4 Candidate

Implemented as the next connector ablation after Stage1B400 fixed optimization
but failed direct-prompt VQA hygiene. The new connector is
`deepstack_spatial_perceiver_resampler`: it keeps the fixed 128-token global /
local output contract but feeds the Perceiver context with multiple SigLIP2
hidden-state levels, defaulting to `[-3, -2, -1]`. Each level gets its own
learned level embedding and 2D positional features are applied per level before
concatenation.

Launch knobs:

```bash
modal run modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_grounding \
  --v4-connector-type deepstack_spatial_perceiver_resampler \
  --v4-deepstack-num-feature-levels 3 \
  --v4-deepstack-hidden-state-indices=-3,-2,-1 \
  --v4-connector-output-gate-init 0.0001
```

Metadata now records `vision_feature_strategy=deepstack_lite`,
`vision_feature_layers`, `deepstack_num_feature_levels`, and
`deepstack_hidden_state_indices`; Modal auto-discovery and VQA eval read these
fields to avoid mixing DeepStack checkpoints with default spatial checkpoints.
For paid runs, keep the stricter babysitting posture from the playbook: W&B
alerts and historical spikes remain yellow/red history even if later rows
recover.

Canary result on `2026-05-09`: DeepStack-lite is viable enough to become the
next V4 architecture branch. The 20-step canary
(`v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary20-20260509-codex`,
W&B `w7ktjb94`) finished with no W&B alerts, zero clipping, no inspector
loss/grad spikes, and a complete DeepStack `checkpoint-20`. The 120-step canary
(`v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120b-20260509-codex`,
W&B `kb8js5z0`) finished with alerts `[]`, accumulation `8.0`, clip `0.0`, no
loss/grad spikes, eval improving `7.7001 -> 7.1668`, final train/objective loss
`1.9442`, loss EMA `2.3480`, grad norm `0.1339`, and complete checkpoints at
`50` and `100`. Both runs carry yellow-history for startup W&B sync gaps and
empty W&B artifact listings, but they did not trip a model-health stop.

Extended Stage 1A result: the 500-step DeepStack run
(`v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex`,
W&B `xb0silsi`) finished with alerts `[]`, accumulation `8.0`, no inspector
loss/grad spikes, final clip fraction `0.002000`, eval improving
`8.2100 -> 6.9942 -> 6.1354 -> 3.6915 -> 3.3106 -> 3.0524 -> 2.8963 -> 2.8282 -> 2.7581 -> 2.7120`,
and complete checkpoints at `200`, `300`, `400`, and `500`. Downstream
validation did not hold up: Stage 1B from DeepStack `checkpoint-500` was stopped
at step `350` on active W&B `recent_loss_spikes`, with `checkpoint-300` as the
last safe artifact; Stage 2A from that artifact failed `loss_scale=0.03` on
clipping and `loss_scale=0.01` on a late active loss-window alert. External VQA
on the last clean Stage 2A checkpoint (`checkpoint-50`) was accuracy-red
(`4.233` overall, `7.580` yes/no) despite clean hygiene. Current architecture
action: keep DeepStack-lite as a wired ablation, but do not promote it as the v4
default; focus the next recipe on calibration data/answer semantics and
loss-normalization rather than deeper connector feature stacking.
