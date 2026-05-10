# V4/V5 Ablation And V6 Guidance Brief

Date: 2026-05-09

This is a self-contained brief for an agent that needs to give high-level
guidance to a future agent who will run V6 ablations and experiments. It
summarizes what the V4 and V5 architecture/recipe ablation plans did, what
actually ran, what the current evidence says, and what risks from
`v4_critique.md` and `v5_progress.md` should shape V6 planning.

The agent consuming this file should not need access to this directory. Paths,
artifact names, and commands are included as provenance and for the next
execution agent, but the decision guidance is meant to stand on its own.

## Whole Model Architecture

AnyMAL is an image-conditioned language model. It does not train a vision model
from scratch and it does not train the full language model. Instead, it learns a
bridge between a frozen vision encoder and a mostly frozen LLaMA-3-8B-Instruct
decoder.

At inference time, the model answers an image question like this:

```text
image pixels
  -> frozen vision encoder
  -> trainable multimodal connector/projector
  -> fixed number of "image token" embeddings in LLaMA hidden space
  -> splice those embeddings into the text prompt at <image> placeholders
  -> LLaMA-3-8B-Instruct generates answer tokens
```

The universal components are:

| Component | Role | Usually frozen? | What changes across versions |
|---|---|---:|---|
| Vision encoder | Converts image pixels into patch features | Yes | V1 uses CLIP ViT-L/14 at 224px; V2/V3/V4 use SigLIP2-So400m at 384px. |
| Connector/projector | Compresses/selects visual features and maps them to LLaMA hidden size | Trained in Stage 1; usually frozen in Stage 2A | This is the main architecture research surface: Perceiver, token compressor, global/local spatial Perceiver, DeepStack-lite. |
| LLM | LLaMA-3-8B-Instruct decoder that consumes text and image embeddings | Base frozen | Stage 2 trains LoRA/QLoRA adapters on the LLM. |
| Image placeholder contract | Tells the model where visual embeddings enter the prompt | n/a | V2+ require strict contiguous placeholder blocks matching the image-token count. |
| Evaluation/generation contract | Decides how prompts are padded, decoded, cleaned, and scored | n/a | V5 fixed left-padding and added strict/raw answer diagnostics. |

### How Image Tokens Enter The LLM

The LLM never sees image pixels. It sees embeddings with the same width as
LLaMA hidden states (`4096` for LLaMA-3-8B). The connector produces those
embeddings from vision features.

There are two insertion modes in the historical code:

- Prepend mode: if no image placeholder tokens are present, projected image
  tokens are prepended before the text embeddings. This was useful for early
  caption-alignment Stage 1.
- Splice mode: if the text prompt contains image placeholder tokens, the model
  replaces those placeholder token embeddings with projected image embeddings.
  V2/V3/V4 training and evaluation rely on this strict splice path.

For V2/V3/V4, the prompt must contain one contiguous block of placeholder tokens
whose length exactly equals the number of image tokens the model will produce.
This matters because otherwise eval/training can silently test a different
interface than the one the model learned.

### Forward Pass And Loss

A training/eval sample contains an image, a text prompt, and usually target
answer tokens. The model flow is:

1. Normalize and resize the image for the selected vision encoder.
2. Encode image patches with the frozen vision tower.
3. Use the connector to produce image embeddings in LLaMA hidden space.
4. Tokenize the text prompt and answer with the LLaMA tokenizer.
5. Replace the prompt's image placeholder embeddings with connector outputs.
6. Run the resulting embedding sequence through LLaMA.
7. Compute next-token loss only on supervised answer tokens; prompt and image
   positions are masked out.

This masking detail matters. Some failed V4 caption-alignment runs had label
formatting bugs that exposed special tokens such as `<|begin_of_text|>` in the
supervised span. Fixing that was required but not sufficient for stable V4
training.

During generation, the model greedily decodes short answers for VQA eval. V5-era
eval requires left-padding decoder prompts before batched generation. Right
padding caused shorter prompts to generate from pad-token positions and created
misleading raw answers such as `assistant\n\n<answer>`.

### Training Stages

The project uses staged training because the vision encoder and LLM are large
and mostly frozen:

| Stage | What trains | What is frozen | Purpose |
|---|---|---|---|
| Stage 1A | Connector/projector only | Vision encoder and LLM | Align visual features to language-space embeddings. Early versions used caption data; later V4 work found grounding-first short-answer supervision easier to stabilize. |
| Stage 1B | Connector/projector only | Vision encoder and LLM | Continue alignment on more direct visual grounding: VQA direct answers and COCO-style object/count/color prompts. |
| Stage 2A | LLM LoRA adapters only; connector should be frozen for clean calibration | Vision encoder, LLM base, usually connector | Short direct-answer calibration for VQA-style behavior. V4 semantic calibration and V5 role-clean semantic calibration are Stage 2A recipes. |
| Stage 2B | Optional small instruction regularization | Depends on experiment | Only after VQA behavior is stable; should not be mixed into first-pass architecture selection. |

Why freezing matters:

- The frozen vision encoder already knows useful visual features.
- The frozen LLaMA already knows language and answer formatting.
- The connector learns the translation between visual feature space and LLaMA
  embedding space.
- LoRA lets Stage 2 adjust the LLM's behavior cheaply without changing the full
  base model.

The common Stage 2 LoRA setup uses rank `64`, alpha `16`, dropout `0.05`, and
targets the LLaMA attention and MLP projection modules (`q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). QLoRA/4-bit loading is
used to keep LLaMA-3-8B training feasible on Modal GPUs.

### Version Lineage

| Version | Vision path | Connector / visual tokens | Core lesson |
|---|---|---|---|
| V1 | CLIP ViT-L/14 at 224px | Perceiver Resampler, 64 image tokens projected to LLaMA hidden size | Simple compact visual prefix was surprisingly strong, especially for yes/no. |
| V2 | SigLIP2-So400m at 384px | Token compressor plus MLP bottleneck, commonly 256-384 image tokens | More visual tokens through a weak connector did not help; strict placeholder/eval plumbing became important. |
| V3 | SigLIP2-So400m at 384px | Fixed 128-token Perceiver Resampler, 6 layers, 16 heads, direct projection to 4096 | First stable architecture after V1 to beat the older corrected fast screen; compact token budget and direct-answer calibration mattered. |
| V4 | SigLIP2-So400m at 384px | Spatial global/local Perceiver: usually 64 global + 64 local tokens with 2D position features; legacy direct-width and lean bottleneck variants both exist | High score came from a bundle of spatial connector checkpoint plus semantic calibration, but architecture contribution is not isolated. |
| V5 | Same `anymal_v4` model class | No new model class; stricter role-clean Stage 2 recipes and stricter eval | Cleaned up eval and evidence protocol. V5-R0 is a small recipe win, not an architecture win. |
| V6 | Not implemented yet | Should be defined only after causal controls | Should separate recipe gains from architecture gains before adding new complexity. |

### Connector Details By Version

V1 connector:

- Frozen CLIP ViT-L/14 returns about 257 visual tokens of width 1024 for a
  224px image.
- A 6-layer Perceiver Resampler uses learned latent queries to attend over
  visual features.
- It outputs 64 LLaMA-width image embeddings, `[batch, 64, 4096]`.
- Stage 1 trains this resampler; Stage 2 historically trained resampler plus
  LoRA, though later versions learned that freezing the connector in short
  calibration is often cleaner.

V2 connector:

- Frozen SigLIP2-So400m at 384px replaces CLIP.
- A `TokenCompressor` compresses/chooses a larger bounded number of visual
  tokens, with options such as learned query pooling, small Perceiver-style
  compressor, or average pooling.
- An MLP bottleneck projector maps compressed SigLIP features to LLaMA hidden
  size.
- V2 introduced strict placeholder counts and metadata checks, but the larger
  token budget did not produce strong VQA behavior.

V3 connector:

- Keeps SigLIP2 at 384px.
- Uses a fixed-size Perceiver Resampler with 128 image tokens.
- The connector projects directly to LLaMA hidden size.
- V3's practical win was not just the connector: short direct-answer Stage 2,
  frozen connector during calibration, and stopping early around checkpoint 100
  all mattered.

V4 connector:

- Keeps the 128-token budget but splits it into typed visual tokens:
  64 global summary latents and 64 local/spatial latents.
- Local tokens can receive learned 2D position features.
- The connector can be legacy/direct-width or lean/bottleneck:
  - legacy/direct-width: 6 layers, 16 heads, FF mult 4, effectively 4096-wide;
    this is what the promoted Stage1B248 semantic-calibration checkpoint used.
  - lean/bottleneck: 3 layers, 8 heads, FF mult 2, hidden dim 1024, projected
    back to LLaMA width, output gate init `0.0001`; this is the desired
    modernization target but has not reproduced the promoted score.
- DeepStack-lite is a V4 ablation that feeds multiple SigLIP2 hidden-state
  levels `[-3, -2, -1]` into the same 128-token output contract. It improved
  some internal Stage 1 losses but failed downstream VQA, so it is not default.

### Data And Metrics Vocabulary

The common data sources in the V4/V5 story are:

- VQAv2 train direct answers, split by answer type: yes/no, number, other.
- Balanced canonical yes/no source, used by V4/V5 semantic calibration.
- COCO object/count/color direct-answer prompts.
- Short LLaVA direct-answer mixture.
- LLaVA-pretrain captions, used by earlier caption-alignment Stage 1 variants.

The main cheap eval is VQAv2 val2014 on a locked 1000-sample slice, usually
seed 42 and `training_chat` prompt style. This eval reports:

- overall VQA accuracy
- accuracy by answer type: number, other, yes/no
- average generated tokens
- EOS rate
- max-token-hit rate
- strict/raw accuracy in V5-era artifacts
- assistant-role prefix rate
- predicted answer-kind rates by answer type
- top answer histograms

High-level rule: do not trust a candidate from accuracy alone. A model can look
better while getting verbose, hitting max tokens, collapsing to yes/no answers,
leaking eval images into training, or relying on post-processing.

Checkpoint metadata is part of the architecture contract. V2/V3/V4 checkpoints
record architecture, connector type, token counts, token split, hidden width,
2D-position setting, and DeepStack fields when relevant. A future agent should
never load or evaluate a checkpoint by path name alone; verify metadata before
comparing runs. Many V4 labels sound similar while pointing to different
connector widths or histories.

## Executive Summary

V4 produced a large VQAv2 fast-screen jump only as a bundle: a V4 spatial
connector checkpoint, Stage1B248 grounding history, semantic-calibration Stage
2 data, frozen connector LoRA-only finetuning, and corrected evaluation. The
best V4 score under the newer left-padded strict/raw eval is approximately
`51-52%` on VQAv2 val2014 1000-sample slices, with clean EOS/max-token behavior.

However, V4 has not proved its central architecture claim. The most important
planned architecture controls were scaffolded but not run: token-budget
variants, no-2D-position controls, V3/V1 architecture with the V4/V5 recipe,
and broader robustness/second-benchmark checks. The promoted V4 checkpoint also
uses the older legacy/direct-width V4 connector path, while the desired modern
target is the lean bottleneck spatial connector. That distinction matters.

V5 is currently a stricter recipe/evidence layer, not yet a new architecture.
V5-R0 reruns semantic calibration with a role-clean answer-only prompt from the
same Stage1B248 V4 base. It slightly beats V4 on clean seed-42/43/44 means under
left-padded eval, with `0.0` assistant-prefix rate, but loses the tiny margin on
the mild-blur robustness check. V5-R1 tries light visual robustness augmentation;
the accumulation-8 attempt stopped before checkpoint on a W&B grad-spike alert,
and the accumulation-16 R1b attempt was interrupted during step-50 validation
with no visible checkpoint. V5-A1, the first lean-connector architecture
transfer, should remain gated until R1/robustness and causal recipe controls are
cleaner.

The planner's first job is to turn this from "a high-scoring bundle" into a
causal study: architecture vs recipe vs eval vs data leakage vs robustness.

Current operational state from `v5_progress.md`: all V5 jobs started in that
thread were stopped. There are no known live `modal_train`,
`vqa_checkpoint_eval`, or W&B inspection processes from that work. V5-R1b was
interrupted by user request while step-50 validation was running; W&B had synced
train step 50, but no eval loss or checkpoint-50 artifact was visible. Treat
R1b as incomplete and not promotion-eligible.

## Current Baselines And Incumbents

All headline numbers below refer to the corrected VQAv2 val2014 fast screen
unless noted otherwise:

- `max_samples=1000`
- seed as stated, usually `42`
- `prompt_style=training_chat`
- direct-answer system prompt unless artifact says otherwise
- answer-type breakdown required
- EOS and max-token-hit hygiene required
- newer V5-era eval requires left-padded decoder prompts and full prediction
  dumps for strict/raw diagnostics

| Model/artifact | Eval mode | Overall | Number | Other | Yes/No | EOS | Max hit | Prefix rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 ablation-F ckpt 500 | postprocess-fix | 21.100 | 15.741 | 17.349 | 28.960 | 0.981 | 0.019 | n/a |
| V2 final balanced-mix ckpt 3000 | postprocess-fix | 7.800 | 5.093 | 7.018 | 10.107 | 1.000 | 0.000 | n/a |
| V3 direct-cal LoRA-only ckpt 100 | postprocess-fix | 9.400 | 6.481 | 9.097 | 11.079 | 0.999 | 0.001 | n/a |
| V4 semantic-cal Stage1B248 ckpt 100 | old right-pad plus prefix strip | 52.400 | 37.037 | 40.871 | 76.093 | 1.000 | 0.000 | not reliable |
| V4 semantic-cal Stage1B248 ckpt 100 | leftpad seed 42 | 51.267 | 38.657 | 39.961 | 73.469 | 1.000 | 0.000 | 0.000 |
| V5-R0 role-clean semantic-cal ckpt 100 | leftpad seed 42 | 51.933 | 38.194 | 40.676 | 74.538 | 1.000 | 0.000 | 0.000 |

Important interpretation:

- The old V4 `52.400` number is not the cleanest final evidence anymore. It
  came after stripping decoded `assistant\n\n<answer>` prefixes in the
  post-processor.
- V5-era evaluation left-pads decoder prompts. This eliminated assistant-prefix
  artifacts for V4 and V5 full-batch evals and makes strict accuracy equal to
  cleaned accuracy on the left-padded artifacts.
- The right-padding bug weakens the strongest version of the V4 critique's
  "model emits role prefix" claim, but it does not solve the causal-ablation
  problem. V4 still has not isolated architecture from recipe and eval changes.

## What V4 Was Designed To Test

V4 was intended as a new architecture label, `anymal_v4`, not a silent V3
replacement. The stated goal was to move AnyMAL closer to modern VLM behavior:
spatially aware visual tokens, controlled high-resolution capacity, a stronger
connector, short direct-answer calibration, and evaluation that catches
generation pathologies.

The V4 design hypothesis:

- Keep V3's compact 128-token image-prefix discipline.
- Avoid repeating V2's failure mode of dumping too many weak visual tokens into
  the LLM.
- Use SigLIP2-So400m at 384px.
- Split visual latents into global and local/spatial tokens.
- Add 2D position features to the local/spatial path.
- Preserve a strict contiguous placeholder block so model/eval plumbing remains
  comparable across versions.
- Use connector-only Stage 1/1B, then freeze connector for short LoRA-only Stage
  2 calibration.

Two V4 connector shapes are easy to confuse:

| Connector path | Shape | Status |
|---|---|---|
| Legacy/direct V4 spatial connector | 64 global + 64 local, 6 layers, 16 heads, FF mult 4, direct 4096-wide projection to LLM dim, 2D positions | Evidence-backed V4 promoted checkpoint uses this through Stage1B248 metadata. It scored best after semantic calibration. |
| Lean A1 bottleneck spatial connector | 64 global + 64 local, 3 layers, 8 heads, FF mult 2, hidden dim 1024, output gate `0.0001`, projected to LLM dim | Desired modernization target and Modal default for future V4/V5 architecture work. It has not reproduced the promoted semantic-calibration win. |

The current code supports both `spatial_perceiver_resampler` and
`deepstack_spatial_perceiver_resampler`. Modal defaults now favor the lean
bottleneck connector:

```text
V4_DEFAULT_CONNECTOR_LAYERS = 3
V4_DEFAULT_CONNECTOR_HEADS = 8
V4_DEFAULT_CONNECTOR_FF_MULT = 2
V4_DEFAULT_CONNECTOR_HIDDEN_DIM = 1024
V4_DEFAULT_CONNECTOR_OUTPUT_SCALE = 1.0
V4_DEFAULT_CONNECTOR_OUTPUT_GATE_INIT = 0.0001
V4_DEFAULT_CONNECTOR_TYPE = "spatial_perceiver_resampler"
```

Note: some local YAML configs still reflect earlier defaults or historical
settings. Always verify startup metadata, checkpoint metadata, and printed
trainable parameter groups before trusting a run label.

## V4 Planned Ablations And Actual Outcomes

| ID | Planned question | Actual status |
|---|---|---|
| A0: V3 reproduction guard | Can the V3 stable 100-step LoRA-only recipe still reproduce the incumbent before V4 spend? | V3 artifacts exist and promotion guard was validated, but the V4 work largely moved on to V4-specific branches. Keep V3 as incumbent and rerun if data/eval code changes. |
| A1: spatial Perceiver at same 128-token budget | Does 64 global / 64 local spatial typing beat or match V3 without token growth? | Mechanically explored heavily. Direct 4096-wide path had severe Stage 1 clipping. Lean 1024 path needed loss scale, token normalization, and accumulated-step W&B metrics to train. Downstream direct-cal reads did not beat V3. |
| A2: token budget | Are 96, 128, or 192 tokens best? | Configs exist (`48/48`, `64/64`, `64/128`), but the actual token-budget ablations were not run through the full Stage1A/1B/2A/eval pipeline. |
| A3: resolution/tiling | Can higher resolution or crops help without prefix growth? | Not run as architecture training. Eval perturbations were later used for robustness diagnostics. |
| A4: connector depth/fusion and spatial attribution | Do 2D positions and global/local split matter? | No-position config exists, but no completed causal run. This is a major gap. |
| A5: vision adaptation | Should frozen SigLIP2 be adapted? | Deferred. Correctly high-risk and not yet justified. |
| DeepStack-lite | Do multi-level SigLIP2 hidden features help? | Wired and trained. Stage1A internal eval improved, but downstream Stage1B/Stage2/VQA failed. Keep as ablation, not default. |
| R1/A7 semantic calibration | Does balanced answer semantics recover yes/no and accuracy? | Passed dramatically on V4 Stage1B248 under corrected eval, but as a recipe/bundle result, not an isolated architecture result. |

## V4 Run Story In More Detail

### A1 Optimization Fight

The early V4 Stage 1A work burned many canaries on apparent model pathologies:
connector width, output scale, gate init, output-only warmup, LR, accumulation,
loss scale, and data mixture. The eventual lesson from `v4_critique.md` is
important: debug the metric before debugging the model.

Key chronology:

- Initial direct-to-LLM 4096-wide connector trained about `1.633B` connector
  params and repeatedly hit Stage 1A gradient clipping.
- Shrinking to the bottleneck1024 connector reduced connector trainables to
  about `44.4M`, but did not solve clipping by itself.
- Output amplitude-only fixes failed:
  - `connector_output_scale=0.1` still clipped.
  - trainable visual gates at `0.01`, `0.001`, and `0.0001` still clipped.
  - output-projection/gate-only warmup still clipped.
- Switching from captions to `v4_grounding` alone still clipped.
- `pretrain_loss_scale=0.1` controlled clipping but tripped loss-window alerts.
- Supervised-token target normalization to 8 tokens improved Stage 1A signal.
- The decisive monitoring fix was accumulated optimizer-step W&B logging.
  Health signals had been reading noisy micro-batch rows under gradient
  accumulation. With `train/accumulation_micro_batches=8.0` logged correctly,
  the same token-normalized recipe completed canaries cleanly.

Resulting A1 baseline:

- bottleneck1024 spatial connector
- 64 global / 64 local tokens
- 2D positions enabled
- output scale `1.0`
- output gate init `0.0001`
- grounding-first short-answer objective
- supervised-token target normalization to 8
- Stage 1 backward `loss_scale=0.1`
- accumulated optimizer-step W&B health metrics

The long A1 extension
`v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex`
passed through `checkpoint-300` with eval improving `5.2072 -> 3.2008 ->
2.6205`, then stopped around W&B step `340` on active `recent_loss_spikes`.
Treat `checkpoint-300` as usable but yellow-history, not a fully green Stage 1A
completion.

### Direct-Calibration Downstream Reads

Direct calibration showed that good optimization did not automatically translate
to VQA accuracy:

| Branch | Outcome |
|---|---|
| Stage1A300 -> Stage2A direct cal `loss_scale=0.03` | Training health green, direct-prompt seed-42 VQA `5.367` overall, below V1 and V3. |
| Stage1B248 -> Stage2A direct cal `loss_scale=0.03` | Yellow-history but artifact complete. VQA `7.600` overall, EOS `1.0`, max hit `0.0`. Better than Stage1A300 and barely above old V1 floor, still below V3 and weak yes/no. |
| Stage1B400 clean-through-400 -> Stage2A `loss_scale=0.01` | Optimization clean, but VQA `5.400` overall, EOS `0.618`, max hit `0.382`. Accuracy/hygiene red. |

Conclusion: Stage1B grounding helps, but the lean bottleneck pathway has not
yet reproduced the promoted semantic-calibration result. Stage1B248 beat
Stage1B400 downstream despite worse/yellow history, which is a selection-on-
noise risk called out by the critique.

### DeepStack-lite

DeepStack-lite changes the connector input from only the final SigLIP2 feature
map to the last three hidden-state levels `[-3, -2, -1]`, with learned level
embeddings and the same fixed 128-token output contract.

What ran:

- 20-step and 120-step Stage 1A canaries passed strict W&B-first monitoring.
- 500-step Stage 1A extension completed with improving internal eval and
  complete checkpoints.
- Stage 1B from DeepStack `checkpoint-500` stopped at step `350` on active W&B
  `recent_loss_spikes`; last safe artifact was `checkpoint-300`.
- Stage 2A from that artifact failed at `loss_scale=0.03` on clipping.
- Lower `loss_scale=0.01` branch stopped on a late loss-window alert.
- Last clean Stage 2A artifact, `checkpoint-50`, scored only `4.233` overall,
  `7.580` yes/no, EOS `1.0`, max hit `0.0`.

Conclusion: DeepStack-lite is a wired ablation, not the V4 default.

### Semantic Calibration

Semantic calibration is the only V4 branch with a large accuracy win. It keeps
the connector frozen and trains LoRA-only Stage 2 on a balanced answer mixture:

- yes/no balanced canonical source: `0.40`
- number: `0.20`
- COCO object/count/color: `0.15`
- VQA other: `0.20`
- short LLaVA direct: `0.05`

The first semantic-calibration attempt from Stage1B248 validated the data path
but stopped before checkpoint/eval on W&B loss-window alerts. The second attempt
used batch size 4 / effective batch 32 and completed to `checkpoint-100`.

Important artifacts:

| Artifact | Interpretation |
|---|---|
| `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat.json` | Old scorer: `7.333` overall because raw decoded answers often looked like `assistant\n\n<answer>` under right-padded batched eval. |
| `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat_postprocessfix.json` | Prefix-stripping postprocess: `52.400` overall, `76.093` yes/no. Useful history but not the cleanest evidence. |
| `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json` | Left-padded strict/raw eval: `51.267` overall, `51.267` strict, prefix rate `0.0`, EOS `1.0`, max hit `0.0`. Treat this as the cleaner V4 incumbent artifact. |
| Seed 43/44 leftpad artifacts | V4 leftpad scores `52.400` and `51.000`, so the result is not only seed 42. |

The promoted V4 checkpoint uses the legacy-width V4 connector from Stage1B248,
not the lean A1 bottleneck connector.

## What The V4 Critique Adds

`v4_critique.md` was written after reviewing V4 plans, logs, and the V4 eval
artifacts. Its most important claims for the planner:

1. The headline V4 result depended heavily on evaluation changes.
2. Single-benchmark promotion is fragile.
3. A yes/no jump from about `11` to `76` can be a red flag unless answer
   distributions and non-yes/no question behavior are measured.
4. The Stage 1A ordeal was partly a process/metric failure: W&B aggregation was
   wrong for gradient accumulation.
5. V4's architecture claim has little direct evidence because token-budget and
   no-2D-position ablations were not run.
6. Stage1B248 selection may be noise-driven.
7. No causal experiment isolates architecture, data recipe, training recipe, and
   evaluator patch.

The critique's original assistant-prefix interpretation should be updated with
the V5 evidence:

- Right-padded batched decoder-only generation was invalid for raw-generation
  hygiene decisions. It caused shorter prompts in a batch to generate from pad
  positions.
- Left-padding fixed the prefix artifact for V4 and V5 artifacts with
  prediction dumps.
- This rescues the raw-cleanliness concern more than the critique expected, but
  it does not rescue the missing causal ablations.

The critique's recommended sanity checks still stand:

- Print marginal answer distribution and predicted answer kind by answer type.
- Verify Stage 2 train image IDs are disjoint from evaluated val2014 image IDs.
- Use cleanly generated left-padded outputs and compare strict vs cleaned
  accuracy.
- Add confusion/mode-collapse metrics to eval.
- Confirm on more seeds or a larger locked slice.
- Run a second benchmark such as POPE/MMBench/MMVet or VQAv2 test-dev.
- Run V3/V1 architecture with the V4/V5 recipe to test whether architecture
  matters at all.

## What V5 Is Doing

V5 is currently an evidence-first modernization layer on top of `anymal_v4`.
There is no separate `anymal_v5` model implementation in the current artifacts;
V5 experiments use `--architecture anymal_v4` with stricter data/eval contracts.

V5 goals:

- Do not hide behind post-processing.
- Make raw/strict generated answers clean.
- Keep compact visual-token discipline.
- Prefer the lean bottleneck spatial connector as the eventual architecture
  target.
- Keep semantic calibration but make labels/prompting answer-only and role-free.
- Do the minimum causal grid needed before claiming an architecture win.

V5 promotion gates add requirements beyond V4:

| Gate | Requirement |
|---|---|
| Clean VQAv2 seed-42 accuracy | At least promoted V4 under left-padded raw-clean eval |
| Strict/raw VQAv2 seed-42 accuracy | Within `1.0` point of cleaned accuracy |
| Assistant role prefix rate | `<= 0.01` overall and `<= 0.02` in every answer-type bucket |
| EOS rate | `>= 0.98` |
| Max-token-hit rate | `<= 0.02` |
| Yes/no predictions on non-yes/no questions | `<= 0.05` |
| Confirmation | At least two additional VQAv2 seeds or one larger locked slice |
| Robustness | Must not lose to V4 on locked resize/blur/crop/translation perturbations |
| Leakage | Train/calibration image IDs disjoint from eval val2014 image IDs |

New V5-era diagnostics in `evaluation/vqa_eval.py` and helper scripts:

- `strict_accuracy`
- `assistant_role_prefix_rate`
- per-answer-type assistant prefix rates
- `top_answers` and `top_raw_answers`
- predicted answer-kind rates by answer type
- full `prediction_samples` with image IDs for leakage audit
- left-padded VQA collate for decoder-only generation

V5 code/config changes recorded in `v5_progress.md`:

- VQA eval now records strict/raw diagnostics, image IDs, answer-kind rates,
  top raw answers, and left-pads decoder prompts.
- `scripts/analyze_vqa_predictions.py` summarizes prediction artifacts for
  prefix leakage and mode collapse.
- `scripts/audit_vqa_leakage.py` compares eval image IDs against training
  source JSONs.
- `scripts/check_v5_promotion.py` enforces complete evidence, strict/clean
  parity, low prefix rates, EOS/max-token hygiene, no yes/no collapse, and full
  prediction samples. Its clean-drop allowance was tightened to `0.0` after the
  mild-blur reversal.
- `modal_train.py` added `v5_semantic_calibration`,
  `v5_semantic_calibration_robust`, `--finetune-gradient-accumulation-steps`,
  V5 augmentation plumbing, and auto-freeze for V5 semantic datasets.
- The V5 robust recipe uses `vqa_light` augmentation: constrained random resized
  crop plus low-probability mild Gaussian blur, with no horizontal flip because
  VQA can depend on left/right semantics.

Validation recorded:

- `python3 -m py_compile` passed for touched Python files.
- `git diff --check` passed.
- Local pytest and local transform import smoke could not run because the local
  Python environment lacked `torch`; the next execution agent should rerun tests
  in a training environment with dependencies installed.

## V5 Ablation Tiers And Current Evidence

### Tier 0: No-GPU Sanity

Implemented or scaffolded:

- Left-padding in VQA eval collate.
- Strict/raw scoring and role-prefix metrics.
- `scripts/analyze_vqa_predictions.py` for answer distribution and mode-collapse
  diagnostics.
- `scripts/audit_vqa_leakage.py` for eval image IDs vs Stage 2 source image IDs.
- `scripts/check_v5_promotion.py` for stricter promotion gates.

Not yet shown as completed:

- Second benchmark result.
- Full perturbation suite beyond mild blur.

Leakage audit completed for the seed-42 V5-R0 corrected eval artifact:

- Artifact audited:
  `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`
- Unique eval image IDs: `751`
- Exact/numeric overlap with VQA train yes/no/number/other: `0`
- COCO object direct overlap: `0`
- Mix665k direct filtered overlap: `0`

Interpretation: no obvious image-ID leakage was found for the seed-42 corrected
eval sample. V6 should still require a leakage audit for every new locked eval
slice and every new calibration source.

### Tier 1: Cheap V5 Recipe Isolation

Planned runs:

| ID | Base checkpoint | Dataset | Purpose | Current status |
|---|---|---|---|---|
| V5-R0 | Stage1B248 legacy V4 | `v5_semantic_calibration` | Does stricter answer-only prompt preserve V4 score while making raw outputs clean? | Completed, slight clean mean win vs V4 on seeds 42/43/44; no prefix. Not robust enough to unlock architecture transfer by itself. |
| V5-R1 | Stage1B248 legacy V4 | `v5_semantic_calibration_robust` | Does light VQA-safe augmentation recover blur robustness? | Accumulation-8 run stopped at synced step 16 on active W&B `recent_grad_spikes`; no usable checkpoint. |
| V5-R1b | Stage1B248 legacy V4 | `v5_semantic_calibration_robust`, accumulation 16 | Does higher accumulation stabilize robust augmentation? | Interrupted by user request while step-50 eval was running. W&B train step 50 was clean, but no eval loss/checkpoint was visible. Incomplete, not promotion-eligible. |
| V5-A1 | A1 bottleneck Stage1B400 | `v5_semantic_calibration` | Does the lean connector inherit semantic-calibration gain? | Deferred. Do not launch until R0/R1/robustness and causal controls justify it. |
| V5-C1 | V3 Stage 1B | `v5_semantic_calibration` | Does the recipe alone explain the gain? | Planned/not run. This is a high-value causal control. |

V5-R0 evidence:

- Modal app: `ap-r4xWCKv5hEMULoUHbPiLvv`
- W&B run: `babakdam/anymal-finetune/jpqogd86`
- Output directory:
  `/checkpoints/finetune-output/v5-stage2a-roleclean-semanticcal-bs4-lossscale003-from-stage1b248-20260509-codex`
- Health: W&B finished green, alerts `[]`, gradient clipping `0`,
  accumulation `8`, eval losses `0.84634` and `0.84325`, complete
  `checkpoint-100`.

| Artifact | Overall | Strict | Number | Other | Yes/No | Prefix | EOS | Max hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42 leftpad | 51.933 | 51.933 | 38.194 | 40.676 | 74.538 | 0.000 | 1.000 | 0.000 |
| seed43 leftpad | 52.467 | 52.467 | 37.526 | 41.151 | 73.118 | 0.000 | 1.000 | 0.000 |
| seed44 leftpad | 51.033 | 51.033 | 38.136 | 38.228 | 72.134 | 0.000 | 1.000 | 0.000 |

V4 vs V5-R0 seed 42/43/44 leftpad mean:

- V5-R0: `51.81`
- V4 incumbent: `51.56`
- Mean margin: about `+0.26`
- This is encouraging but small.

Mild blur:

| Comparison | V4 | V5-R0 | Margin |
|---|---:|---:|---:|
| seed42 `mild_blur` leftpad | 51.600 | 51.467 | V5 loses by `0.133` |

Prompt-isolation mild-blur artifacts:

- V5-R0 with V4 prompt: `51.300`
- V4 with V5 prompt: `51.600`
- Interpretation in the V5 plan: the mild-blur gap points more to
  checkpoint/training mix than to inference prompt wording.

Answer distribution sanity on seed42 leftpad:

- V4 top answers: `no` 244, `yes` 109, `1` 68, `2` 38, then colors/common
  objects.
- V5-R0 top answers: `no` 241, `yes` 111, `1` 67, `2` 38, then colors/common
  objects.
- V4 predicted yes/no on `other`: `9/513 = 0.018`.
- V5-R0 predicted yes/no on `other`: `9/513 = 0.018`.
- Both pass the V5 non-yes/no yes/no-collapse gate on seed42.

The single-artifact V5 gate passes for V5-R0 seed42 leftpad, but the robustness
gate fails when the V4 mild-blur incumbent is used.

Robust recipe attempts:

- V5-R1, app `ap-vYcwU9fpufZN5UXNBncP3L`, W&B
  `babakdam/anymal-finetune/h4oqlm9g`: stopped at synced step `16`; active
  `recent_grad_spikes`; spike source step `8`, grad `0.024826` vs EMA
  `0.010866` (`2.28x`); no usable checkpoint.
- V5-R1b, app `ap-TvaVjB2cLqR8upiUwcb0XC`, W&B
  `babakdam/anymal-finetune/5zj9atks`: accumulation `16`; early checks at
  synced steps `40` and `50` had alerts `[]`, clip `0.0`, no loss/grad spikes,
  and stable tiny grad norms. The user stopped all jobs while step-50 validation
  was in progress. No eval loss had logged and the output directory was empty,
  so there is no visible `checkpoint-50`. Relaunch from scratch if continuing
  this line.

### Tier 2: Architecture Attribution

Only after V5-A1 is mechanically and diagnostically healthy, planned V5
architecture controls are:

| ID | Change | Question |
|---|---|---|
| V5-A2 | no 2D position features | Are spatial positions doing work? |
| V5-A3 | 48 global / 48 local tokens | Is 128 tokens necessary? |
| V5-A4 | 64 global / 128 local tokens | Does more local capacity help without hygiene loss? |
| V5-A5 | prompt-conditioned patch selector before Perceiver | Can token routing improve robustness without prefix growth? |

The planner should consider moving recipe-only controls earlier than V5-A1:
V3 architecture plus V5 recipe and V1 architecture plus V5 recipe are arguably
more informative than another V4 architecture branch.

### Tier 3: Robustness And Second Benchmark

Planned but not complete:

- VQAv2 perturbations: `resize_up`, `mild_blur`, `center_crop_90`,
  `translate_5pct`.
- POPE or equivalent hallucination/negative yes-no benchmark.
- MMBench/MMVet/public multimodal benchmark if cached or cheap.
- VQAv2 test-dev under public protocol if available.

Only `mild_blur` is currently represented in the local V4/V5 artifacts.

## High-Level Guidance For V6

V6 should be planned as a pre-registered ablation campaign, not as "try a new
architecture until one looks better." The main unsolved question after V4/V5 is
not whether AnyMAL can score around `51-52` on a 1000-sample VQAv2 slice; it
can. The unsolved question is why: architecture, answer-calibration recipe,
Stage 1 checkpoint choice, eval hygiene, or dataset composition.

Recommended V6 thesis:

- First stabilize the evidence protocol and robust recipe.
- Then run recipe-transfer controls that could falsify the architecture story.
- Only then spend on new architecture components.

### V6 Non-Negotiable Evidence Protocol

Every V6 candidate should produce:

- Left-padded VQA eval with full prediction samples.
- Clean and strict accuracy, with strict/clean gap `<= 1.0` point and ideally
  `0.0`.
- Per-answer-type accuracy, prefix rate, predicted answer-kind rates, and top
  answer histograms.
- EOS `>= 0.98` and max-token-hit rate `<= 0.02`.
- Predicted yes/no rate on non-yes/no questions `<= 0.05`.
- Leakage audit for every locked eval slice and every calibration source.
- At least seeds 42/43/44 on clean VQAv2 before architecture claims.
- Robustness suite on `resize_up`, `mild_blur`, `center_crop_90`, and
  `translate_5pct`.
- At least one independent benchmark, preferably POPE for yes/no/hallucination
  and one general multimodal benchmark such as MMBench/MMVet if staging is
  practical.

V6 should define "recipe win" and "architecture win" separately:

- Recipe win: improves V4/V5 on the same Stage1B248 legacy V4 base.
- Architecture win: improves after controlling for the recipe, using the same
  Stage 2 data, eval, seeds, robustness suite, and leakage policy, and after a
  V3/V1 recipe-transfer baseline fails to explain the gain.

### V6 Experiment Order

1. Finish the robust recipe question on the Stage1B248 legacy V4 base.
   Relaunch R1b from scratch rather than relying on the interrupted step-50
   state. Keep accumulation `16`. If it finishes cleanly, evaluate clean
   seeds 42/43/44 and the full perturbation suite. If it fails or still trails
   V4 on mild blur, try a weaker robust recipe before any architecture work:
   lower blur probability/intensity, apply augmentation only to selected
   sources, or evaluate checkpoint 50 on clean/mild blur before training to
   checkpoint 100.
2. Run recipe-transfer controls before V6 architecture claims:
   - V3 architecture plus V5 semantic calibration.
   - If feasible, V1 architecture plus V4/V5 semantic calibration. This is the
     single most decisive control from the V4 critique: if V1 plus the recipe
     reaches the V4/V5 regime, the architecture story collapses.
3. Run the lean-connector transfer only after the robust recipe and
   recipe-transfer controls are understood:
   - Base: A1 bottleneck Stage1B400 checkpoint
     `/checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400`
   - Dataset: V5 semantic calibration or the robust recipe if it passed.
   - Expected risk: Stage1B400 previously required smaller Stage 2 loss scale
     and had bad EOS/max-token behavior under direct calibration. Treat hygiene
     as a co-primary outcome, not an afterthought.
4. Only after the lean connector inherits the recipe gain, run architecture
   attribution:
   - no 2D positions
   - 96-token compact `48/48`
   - 192-token local-expanded `64/128`
   - global-only/local-only if code supports it
   - prompt-conditioned patch selector or learned token router only after the
     fixed-token controls are interpretable
5. Treat DeepStack-lite as a negative-but-useful ablation. Do not revive it for
   V6 unless the hypothesis changes from "multi-level features are better" to a
   specific fix for why Stage1B/Stage2 failed downstream.

### V6 Stop/Go Gates

Stop before architecture spend if:

- Robust recipe cannot match V4 on mild blur.
- Clean seed margins remain below noise and robustness flips the sign.
- Leakage audit fails.
- Strict/raw gap reappears.
- Answer histograms show yes/no collapse or heavy generic-answer mode collapse.

Go to architecture attribution only if:

- Recipe baseline is clean and robust enough to be a fair training target.
- V3/V1 recipe-transfer controls do not explain most of the V4/V5 score.
- Candidate checkpoints finish with W&B alerts empty, complete artifacts, and
  clean generation hygiene.

V6 should not be promoted on a single 1000-sample VQAv2 slice, even if the score
is higher than V5. A V6 promotion should mean: same or better clean accuracy,
same or better robustness, clean strict/raw behavior, no leakage, no mode
collapse, and at least one independent benchmark win or tie.

## Known Eval And Data Landmines

1. Right-padded decoder-only batched generation is invalid for raw hygiene. Use
   left-padded eval artifacts or rerun eval.
2. Always ask for full `prediction_samples=1000` for promotion candidates so
   image IDs, strict/raw answers, answer kinds, and prefix metrics are present.
3. Cleaned accuracy alone is not sufficient. Strict/raw parity is co-primary.
4. Answer distributions are load-bearing. Large yes/no gains need predicted
   answer-kind by ground-truth answer type.
5. Leakage was clean for the V5-R0 seed-42 leftpadded artifact, but that does
   not generalize automatically. Run `scripts/audit_vqa_leakage.py` for every
   new candidate slice/source before architecture claims.
6. W&B active alerts win over console optimism. Yellow-history runs can be used
   for diagnosis, but should not be treated as green promotion candidates.
7. Local YAML names and run labels can hide connector differences. Check
   `model_meta.json` and startup logs for connector type, layers, heads, hidden
   dim, token split, 2D positions, and direct/bottleneck projection.

## Commands And Run Shapes Already Defined

V5-R0:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-bs4-lossscale003-from-stage1b248-20260509-codex
```

V5-R1/R1b:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-stage1b248-20260509-codex
```

V5-A1 candidate shape:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.01 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400 \
  --freeze-connector \
  --use-wandb \
  --run-name v5-stage2a-roleclean-semanticcal-bs4-lossscale001-from-a1stage1b400-20260509-codex
```

Canonical V5-era eval:

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/<run-name>/checkpoint-100 \
  --candidate-label <label> \
  --candidate-architecture anymal_v4 \
  --no-include-baselines \
  --max-samples 1000 \
  --seed 42 \
  --prompt-style training_chat \
  --system-prompt 'Answer with only the final answer. Do not include role labels, explanations, or the word assistant. End after the answer.' \
  --prediction-samples 1000 \
  --output <artifact>.json
```

V5 promotion check:

```bash
python3 scripts/check_v5_promotion.py \
  --v4-incumbent vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json \
  --candidates <candidate-artifact>.json
```

Prediction diagnostics:

```bash
python3 scripts/analyze_vqa_predictions.py <artifact>.json
```

Leakage audit scaffold:

```bash
modal run scripts/audit_vqa_leakage.py \
  --eval-artifacts <artifact-with-prediction-samples>.json
```

V4 token-budget and no-position launch knobs:

```bash
# 96-token compact: 48 global + 48 local
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_caption_alignment \
  --pretrain-image-tokens 96 \
  --gpu-type h100 \
  --run-name v4-stage1a-compact96-YYYYMMDD \
  --use-wandb

# 192-token expanded local: 64 global + 128 local
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_caption_alignment \
  --pretrain-image-tokens 192 \
  --v4-global-image-tokens 64 \
  --v4-local-image-tokens 128 \
  --gpu-type h100 \
  --run-name v4-stage1a-local128-YYYYMMDD \
  --use-wandb

# no-2D-position control at default 64/64 split
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_caption_alignment \
  --pretrain-image-tokens 128 \
  --no-v4-use-2d-position-features \
  --gpu-type h100 \
  --run-name v4-stage1a-no2dpos-YYYYMMDD \
  --use-wandb
```

DeepStack-lite launch knobs:

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

## What The Next Planner Should Produce

The next agent should produce a detailed ablation plan, not another broad
research memo. It should include:

1. A ranked sequence of experiments with stop/go gates.
2. The causal question each run answers.
3. Exact base checkpoints and whether their history is green/yellow/red.
4. Exact dataset mixture and prompt contract.
5. Connector metadata expectations for each run.
6. Evaluation artifacts required before promotion.
7. Robustness and second-benchmark requirements.
8. Leakage-audit requirement and failure policy.
9. Compute/cost estimates and early-stop criteria.
10. A promotion rubric that distinguishes recipe win, architecture win, and eval
    artifact.

Recommended priorities from this brief:

- Finish Tier 0 evidence: repeat leakage audits for new slices/sources, keep
  full answer histograms and strict/raw parity, and left-pad re-evaluate any
  artifact that is still right-padded.
- Relaunch V5-R1b from scratch as the next cheap recipe check if continuing the
  robustness line.
- Run a recipe-only architecture control before expensive V5-A1 claims:
  V3 Stage 1B + V5 semantic calibration at minimum; V1 architecture + V5/V4
  semantic recipe is the most decisive but may require more plumbing.
- Do not claim V5 architecture improvement from V5-R0. It uses the same legacy
  Stage1B248 V4 base as promoted V4.
- Do not claim V4's global/local 2D-position architecture matters until no-2D,
  token-budget, and recipe-transfer controls exist.
- Require at least one robustness suite and one second benchmark before calling
  any V4/V5 checkpoint the new stable default.

## Source Artifacts Read

Core plans and critiques:

- `V4_ARCHITECTURE_PLAN.md`
- `V4_RESEARCH_RECIPE_20260508.md`
- `V5_RESEARCH_PLAN_20260509.md`
- `v5_progress.md`
- `.claude/worktrees/tender-lewin-e1e6f0/v4_critique.md`
- `TRAINING_RUN_BABYSITTING_PLAYBOOK.md`
- `EXPERIMENTS.md`
- `CLAUDE.md`
- `README.md`

Configs:

- `configs/pretrain_v4_alignment.yaml`
- `configs/pretrain_v4_grounding.yaml`
- `configs/pretrain_v4_alignment_compact96.yaml`
- `configs/pretrain_v4_alignment_local128.yaml`
- `configs/pretrain_v4_alignment_no2dpos.yaml`
- `configs/finetune_v4.yaml`
- `configs/finetune_v4_semantic_calibration.yaml`
- `configs/finetune_v5_semantic_calibration.yaml`
- `configs/finetune_v5_semantic_calibration_robust.yaml`

Implementation:

- `models/anymal_v4.py`
- `models/anymal.py`
- `models/anymal_v2.py`
- `models/anymal_v3.py`
- `models/projectors/perceiver_resampler.py`
- `models/projectors/token_compressor.py`
- `models/projectors/spatial_perceiver_resampler.py`
- `models/projectors/deepstack_spatial_perceiver_resampler.py`
- `modal_train.py`
- `evaluation/vqa_eval.py`
- `data/data_utils.py`
- `scripts/analyze_vqa_predictions.py`
- `scripts/audit_vqa_leakage.py`
- `scripts/check_v5_promotion.py`
- `scripts/check_vlm_promotion.py`

Key eval artifacts:

- `vqa_checkpoint_eval_baselines_1000_seed42_training_chat_directprompt_postprocessfix.json`
- `vqa_eval_v3_direct_calibration_ckpt100_training_chat_postprocessfix.json`
- `vqa_eval_v4_stage2a_stage1b248_lossscale003_ckpt100_directprompt_training_chat.json`
- `vqa_eval_v4_stage2a_stage1b400_lossscale001_ckpt100_directprompt_training_chat.json`
- `vqa_eval_v4_stage2a_deepstack_stage1b300_lossscale001_ckpt50_directprompt_training_chat.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat_postprocessfix.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed43_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed44_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_mildblur_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed43_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed44_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_mildblur_leftpad.json`
- `vqa_eval_v5_stage2a_roleclean_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_v4prompt_mildblur_leftpad.json`
- `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_v5prompt_mildblur_leftpad.json`
