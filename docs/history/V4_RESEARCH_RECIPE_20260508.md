# AnyMAL V4 Research Recipe - 2026-05-08

## Current Incumbent

The corrected continuity benchmark is VQAv2 val2014, `1000` samples, seed `42`,
`training_chat`, first stop-token trimming, with EOS and max-token-hit hygiene.

| Model | Overall | Number | Other | Yes/No | EOS | Max hit |
|---|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 21.10 | 15.74 | 17.35 | 28.96 | 0.981 | 0.019 |
| V2 final balanced-mix checkpoint 3000 | 7.80 | 5.09 | 7.02 | 10.11 | 1.000 | 0.000 |
| V3 direct-calibration LoRA-only checkpoint 100 | 9.40 | 6.48 | 9.10 | 11.08 | 0.999 | 0.001 |
| V4 semantic calibration Stage1B248 checkpoint 100 | 52.40 | 37.04 | 40.87 | 76.09 | 1.000 | 0.000 |

The table uses the corrected 2026-05-09 VQA post-processing path, which strips
a duplicated decoded `assistant` role prefix before answer scoring. V4 must beat
V3 overall, or match it while improving yes/no without losing `other` accuracy
or generation hygiene.

## Current Recommendation After 2026-05-09 Ablations

Promote the tested V4 semantic-calibration checkpoint as the current recipe
candidate under the corrected VQA protocol. The evidence-backed architecture is
the spatial connector family, not DeepStack-lite:

- The promoted checkpoint uses `anymal_v4` with SigLIP2-So400m at 384px,
  `64` global + `64` local visual tokens, `spatial_perceiver_resampler`,
  `6` connector layers, `16` heads, FF mult `4`, direct projection to the LLM
  dimension, 2D position features, strict contiguous 128-token placeholders,
  `connector_output_scale=1`, and `connector_output_gate_init=0.0001`.
- The lean A1 bottleneck variant (`1024` connector width, fewer layers/heads)
  remains the desired efficiency modernization, but it must repeat the promoted
  semantic-calibration recipe before replacing the evidence-backed checkpoint.
- Keep DeepStack-lite implemented and metadata-gated as an ablation only. It
  improved internal Stage 1A/1B eval but failed the downstream Stage 2A/VQA gate
  (`4.233` overall from the last clean checkpoint), so it should not become the
  V4 default without a new grounding and calibration recipe.
- The key accuracy correction was evaluator-side, not another training run:
  raw prediction dumps showed many answers decoded as `assistant\n\n<answer>`,
  and the old post-processor scored only the first line. After stripping that
  duplicated role prefix, the same health-clean `v4_semantic_calibration`
  checkpoint scores overall `52.400`, number `37.037`, other `40.871`,
  yes/no `76.093`, EOS `1.0`, max-hit `0.0`. Under the corrected protocol it
  clears V1, V3, yes/no recovery, and hygiene gates.

Promotion remains V3-relative: beat corrected V3 `9.400` overall, or materially improve
yes/no while preserving V3's `other` score and generation hygiene.

## What Modern VLMs Suggest

Recent high-performing VLMs point in the same direction: preserve spatial
structure, route/select visual detail intelligently, expose useful multi-level
visual features, and avoid dumping raw token volume into the LLM. The numbers
below are not directly comparable to our VQAv2 fast screen, but they show where
open VLMs are getting their gains.

- [Qwen3-VL](https://arxiv.org/abs/2511.21631) reports dense and MoE variants
  with 256K interleaved context. Its main architectural upgrades are
  interleaved-MRoPE, DeepStack multi-level ViT feature fusion, and text-based
  timestamp alignment for video. The [model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
  highlights stronger 2D/3D spatial perception and DeepStack for finer
  image-text alignment.
- [InternVL3.5](https://arxiv.org/abs/2508.18265) keeps the practical
  `ViT-MLP-LLM` paradigm, then adds a Visual Resolution Router and Cascade RL.
  The paper reports up to `+16.0%` reasoning improvement and `4.05x` inference
  speedup over InternVL3, with `73.4` MMMU for the 8B model and `77.7` for the
  241B-A28B model in its summary.
- [Kimi-VL](https://arxiv.org/abs/2504.07491) uses a native-resolution MoonViT,
  pixel-shuffle/MLP projection, and an MoE decoder. Its thinking variant reports
  `64.0` MMMU, `46.3` MMMU-Pro, `56.9` MathVision, and `80.1` MathVista with
  only `2.8B` activated decoder parameters; the architecture section explicitly
  adds 2D RoPE for high-resolution positional detail.
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) emphasizes native dynamic
  resolution, window attention, object localization, document parsing, and
  structured table/form understanding. This argues for resolution adaptivity,
  but not for uncontrolled fixed-prefix growth.
- [Gemma 3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
  reports multimodal IT scores of `48.8/59.6/64.9` MMMU for 4B/12B/27B,
  `75.8/87.1/86.6` DocVQA, and `62.4/71.6/71.0` VQAv2. This is a reminder that
  stronger language backbones help, but not every visual metric scales
  monotonically with model size.
- [PaliGemma 2](https://arxiv.org/abs/2412.03555) combines SigLIP-So400m with
  Gemma 2 at 224/448/896px and trains in multiple stages. This supports a later
  resolution ablation, especially for OCR/document-like tasks.
- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) shows that a single model
  can transfer across single-image, multi-image, and video scenarios when data,
  model, and visual representation choices are aligned.
- [VLM-RobustBench](https://arxiv.org/abs/2603.06148) is a useful warning for
  evaluation: current VLMs can be semantically strong but spatially fragile,
  with resampling and geometric distortions causing the largest drops. V4 should
  not promote on clean VQAv2 alone if the claimed gain is spatial.
- [Penguin-VL](https://arxiv.org/abs/2603.06569), refreshed on
  `2026-03-14`, argues that encoder quality and objective match can matter more
  than simply scaling the VLM. Its 2B/8B design uses an LLM-initialized vision
  encoder with bidirectional attention and 2D-RoPE, and reports competitive
  image/video performance against larger VLMs. This is not a drop-in change for
  the current SigLIP2 path, but it makes encoder-side ablations more valuable
  than more LLM-side LoRA churn.
- [FastVLM](https://huggingface.co/docs/transformers/v5.0.0rc2/en/model_doc/fast_vlm)
  reports that high-resolution VLMs can recover speed and accuracy by designing
  the vision encoder to emit fewer tokens. Its Hugging Face documentation cites
  better SeedBench/MMMU/DocVQA than LLaVA-OneVision at `1152x1152` with the same
  `0.5B` LLM, while using `85x` faster TTFT and a smaller vision encoder.
- [EvoComp](https://arxiv.org/abs/2604.17087), submitted `2026-04-18`, is a
  learned visual-token compression method. It selects informative,
  non-redundant visual tokens with visual and textual context and reports `99.3%`
  accuracy retention under `3x` compression plus up to `1.6x` mobile speedup.
  This strengthens the case for a learned selector after the fixed
  global/local-latent default is measured.
- [ForestPrune](https://arxiv.org/abs/2603.22911), revised `2026-04-12`, shows
  that spatial/semantic/temporal token pruning can preserve most video accuracy
  at very high compression ratios. AnyMAL does not need video now, but the paper
  supports treating token routing as a first-class architecture knob rather than
  a post-hoc efficiency trick.

Implications for AnyMAL:

- Keep the 64 global / 64 local token contract as the running default, but the
  original direct 4096-wide connector is now a failed ablation. The active V4
  default uses a bottlenecked 1024-wide spatial connector with
  `connector_output_scale=1.0` and `connector_output_gate_init=0.0001`. Output
  amplitude-only fixes were not enough: `0.1` output scale, trainable visual
  gates at `0.01`, `0.001`, and `0.0001`, and output-projection warmup all
  failed Stage 1A health gates before the supervised-token-normalized,
  accumulated-metrics recipe cleared the canary. Do not revert the default to
  output-scale shrinkage or staged connector warmup without new evidence.
- Add a `DeepStack-lite` connector ablation after A1 has a completed Stage 2A
  read: expose the last 2-3 SigLIP2 hidden-state levels to the
  `SpatialPerceiverResampler`, with learned per-level embeddings and the same
  output token budget.
- Add an adaptive-resolution/token-routing ablation only after the default 384px
  model earns a V3-comparable score: compare 384px, 448/512px, and fixed-grid
  crop packs while keeping final visual tokens at `128` or `192`.
- Add a learned selector/compressor ablation after A1/A2 identify a useful token
  budget: preselect or softly weight SigLIP2 patch tokens before the shared
  Perceiver using image features plus optional prompt text features. Keep the
  output placeholder budget unchanged so gains cannot come from longer prefixes.
- Add a small robustness slice to the promotion suite for V4 spatial claims:
  resized/upsampled, mild blur, crop/translation, and aspect-ratio perturbations
  on the locked VQA subset.

2026-05-08 update: the corrected A1 extension
`v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex`
passed strict W&B/artifact gates through `checkpoint-300`, with validation loss
improving `5.2072 -> 3.2008 -> 2.6205`, then stopped at W&B step `340` for an
active `recent_loss_spikes` alert (`1.9874` objective loss vs `0.9362` EMA,
`2.12x`). `checkpoint-300` is the current A1 Stage 1A candidate for downstream
Stage 1B/Stage 2 reads; the 1000-step extension is not green.

## V4 Architecture Decision

Default V4 is now implemented as `anymal_v4`:

```yaml
architecture: anymal_v4
vision_encoder_type: siglip2
vision_model_name: google/siglip2-so400m-patch14-384
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
stage2_train_adapter: false
```

The connector keeps V3's compact 128-token discipline but splits capacity into
global summary latents and local spatial latents. Local latents receive learned
2D position features, then global and local typed queries pass through a shared
Perceiver tower over position-augmented SigLIP2 patch context. The active default
runs that tower at hidden dim `1024` and projects back to the LLM embedding size,
which cuts Stage 1 connector trainables from `1.633B` to `44.4M`. This shrink is
necessary for memory and iteration speed but was not sufficient by itself; the
current stable A1 recipe keeps output scale at `1.0`, initializes the trainable
output gate at `0.0001`, and stabilizes optimization through supervised-token
target normalization plus accumulated optimizer-step W&B health logging. The
output remains a strict contiguous placeholder block, so V2/V3 data and eval
plumbing stay comparable.

What is deliberately not in the default: new native ViT pretraining, MoE LLMs,
RL, 448/896px training, video, or 384 image tokens. Those are later ablations
after the spatial connector earns its keep.

## Training Recipe

The original caption-first outline is no longer the recommended next run. The
validated optimization path is grounding-first, token-normalized, and monitored
with accumulated optimizer-step health metrics.

1. Stage 1A grounding-first alignment: connector-only, frozen SigLIP2 and
   frozen LLM, VQAv2 direct answers plus COCO object/count/color and short-answer
   data, supervised-token target normalization to `8` tokens, pretrain backward
   `loss_scale=0.1`, gradient accumulation `8`, explicit save/eval cadence, and
   W&B gates on active alerts, clipping, recent-window movement, and artifacts.
2. Stage 1B grounding continuation: connector-only continuation from the best
   safe Stage 1A artifact. Stop on W&B alerts even when internal eval improves;
   downstream evidence says the lower internal eval Stage1B400 checkpoint
   over-calibrated into poor EOS behavior, while the earlier Stage1B248 branch
   gave the best V4 external VQA read (`7.600` overall, EOS `1.0`).
3. Stage 2A direct calibration control: freeze connector, train LoRA only for
   `100` steps, checkpoints at `50` and `100`, explicit LoRA LR `1e-5`, and
   direct-prompt VQA after each checkpoint. `loss_scale=0.03` is the control
   when clipping stays clean; lower `0.01` is an optimization rescue only and
   must clear EOS/max-token gates before it can be trusted.
4. Stage 2A semantic calibration ablation: `v4_semantic_calibration` from
   Stage1B248 completed cleanly at bs4/effective batch `32`. It keeps the
   direct-answer prompt but uses a balanced canonical yes/no source, weights
   `yes/no:number:coco:other:short` as `0.40:0.20:0.15:0.20:0.05`, and freezes
   the connector by default. The old scorer reported `7.333` because it counted
   duplicated decoded `assistant` role prefixes as answers. The corrected
   scorer reports `52.400` overall with EOS `1.0` and max-hit `0.0`, clearing
   the corrected V1 and V3 gates.
5. Promote only after corrected VQA seed-42 beats V3 or materially improves
   yes/no while preserving V3's `other`, EOS `>=0.98`, and max-hit `<=0.02`.
   Then confirm on at least one extra seed or a larger locked VQA slice.

## First Ablations

| ID | Change | Purpose | Gate |
|---|---|---|---|
| A0 | Re-run V3 stable 100-step LoRA-only recipe | Detect drift before spending on V4 | Near saved 9.10 result |
| A1 | V4 64 global / 64 local, 128 total, bottleneck1024 connector, gated or zero-init output projection | Test spatial token typing at same budget with a stable-size connector and controlled initial gradients | Beat/match V3 with better yes/no or number |
| A2a | V4 48 global / 48 local, 96 total | Test stronger selection vs capacity | No hygiene loss; near A1 |
| A2b | V4 64 global / 128 local, 192 total | Test local spatial capacity | Improvement without max-hit regression |
| A3 | 448/512px into same 128 tokens | Test resolution without prefix growth | Improves detail/OCR slice without VQA loss |
| A4 | No 2D positions / local-only / global-only | Attribute wins to spatial design | Keep only if interpretable |
| A5 | Learned patch selector before the shared Perceiver | Test modern compression/routing without longer prefixes | Maintains clean VQA and improves perturbation/OCR/detail slices |
| A6 | Alternative vision encoder or multi-level encoder features | Test whether SigLIP2 objective/level choice is the bottleneck | Only after A1/A2 show connector recipe is stable |
| A7 | Semantic calibration Stage 2 (`v4_semantic_calibration`) | Test answer-label balance and EOS semantics after DeepStack failed | PASS under corrected scorer: `52.400` overall, yes/no `76.093`, EOS `1.0`, max-hit `0.0` |

## Run Commands

Local configs:

```bash
python3 scripts/train_pretrain.py --config configs/pretrain_v4_alignment.yaml
python3 scripts/train_pretrain.py --config configs/pretrain_v4_grounding.yaml
python3 scripts/train_finetune.py --config configs/finetune_v4.yaml
python3 scripts/train_finetune.py --config configs/finetune_v4_semantic_calibration.yaml
```

Local Stage 1A ablation configs:

```bash
python3 scripts/train_pretrain.py --config configs/pretrain_v4_alignment_compact96.yaml
python3 scripts/train_pretrain.py --config configs/pretrain_v4_alignment_local128.yaml
python3 scripts/train_pretrain.py --config configs/pretrain_v4_alignment_no2dpos.yaml
```

Modal equivalents:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_caption_alignment \
  --max-steps 20000 \
  --pretrain-image-tokens 128 \
  --gpu-type h100 \
  --use-wandb

modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_grounding \
  --max-steps 10000 \
  --pretrain-image-tokens 128 \
  --pretrain-checkpoint /checkpoints/pretrain-output/<v4-stage1a>/checkpoint-20000 \
  --gpu-type h100 \
  --use-wandb

modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v4_direct_calibration \
  --max-steps 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/<v4-stage1b>/checkpoint-10000 \
  --freeze-connector \
  --use-wandb

modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v4_semantic_calibration \
  --max-steps 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --pretrain-checkpoint /checkpoints/pretrain-output/<v4-stage1b>/checkpoint-248 \
  --freeze-connector \
  --use-wandb
```

External VQA reads after Stage 2A checkpoints:

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/<v4-stage2a>/checkpoint-100 \
  --candidate-label v4-stage2a-ckpt100 \
  --candidate-architecture anymal_v4 \
  --no-include-baselines \
  --max-samples 1000 \
  --seed 42 \
  --prompt-style training_chat \
  --system-prompt 'Answer the image question directly and briefly. End after the answer.' \
  --image-perturbation none \
  --output vqa_eval_v4_stage2a_ckpt100_directprompt_training_chat.json

python3 scripts/check_vlm_promotion.py \
  --candidate-arch anymal_v4 \
  --candidates vqa_eval_v4_stage2a_ckpt100_directprompt_training_chat.json
```

Spatial robustness follow-up uses the same command with
`--image-perturbation resize_up`, `mild_blur`, `center_crop_90`, or
`translate_5pct`. Keep those artifacts separate from the clean promotion read;
they are regressions/diagnostics unless we explicitly add robustness gates.

V4 token-budget and spatial-attribution ablation launch knobs:

```bash
# A2a compact 96-token run: 48 global + 48 local
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage pretrain \
  --dataset v4_caption_alignment \
  --pretrain-image-tokens 96 \
  --gpu-type h100 \
  --run-name v4-stage1a-compact96-YYYYMMDD \
  --use-wandb

# A2b expanded 192-token run: 64 global + 128 local
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

# A4 no-position control at the default 64/64 split
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

## Implementation Status

- Added `SpatialPerceiverResampler`.
- Added `AnyMALv4`.
- Registered `anymal_v4` in factory and checkpoint metadata.
- Added configs: `pretrain_v4_alignment.yaml`, `pretrain_v4_grounding.yaml`,
  `finetune_v4.yaml`.
- Added Stage 1A ablation configs:
  `pretrain_v4_alignment_compact96.yaml`,
  `pretrain_v4_alignment_local128.yaml`, and
  `pretrain_v4_alignment_no2dpos.yaml`.
- Updated local and Modal train paths to treat V4 as SigLIP2/strict-placeholder.
- Added focused connector, factory, and metadata tests.
- Added `scripts/check_vlm_promotion.py` to compare future V4 eval artifacts
  against both the V1 floor and V3 incumbent with EOS/max-token hygiene gates.
- Updated `vqa_checkpoint_eval.py` to load `anymal_v4` checkpoints and to run
  optional clean-slice perturbations (`resize_up`, `mild_blur`,
  `center_crop_90`, `translate_5pct`) for spatial robustness diagnostics.
- Added Modal V4 ablation knobs for global/local token split and 2D position
  features, while preserving the default `64/64` split for the active A1 run.
- Stage 2 and VQA eval now infer V4 token split and 2D-position settings from
  checkpoint metadata when available, so non-default ablation checkpoints do not
  require hand-matched evaluator/model kwargs.
- Stage 2 pretrain-checkpoint auto-discovery now runs before model construction
  and filters V4 candidates by requested token split/2D-position metadata, which
  prevents non-default ablations from silently loading into a default connector.

Validation so far:

- `python3 -m compileall models model_metadata.py scripts/train_pretrain.py scripts/train_finetune.py training/finetune.py modal_train.py` passes.
- `git diff --check` passes.
- `python3 scripts/check_vlm_promotion.py --candidate-arch anymal_v3 --candidates vqa_eval_v3_direct_calibration_ckpt100_training_chat.json --incumbent vqa_eval_v3_direct_calibration_ckpt100_training_chat.json`
  passes, validating the generic checker against the saved V3 incumbent.
- `python3 -m py_compile modal_train.py vqa_checkpoint_eval.py scripts/check_vlm_promotion.py training/finetune.py`
  passes after adding V4 eval and ablation knobs.
- `python3 -m py_compile scripts/train_pretrain.py` passes after making its YAML
  `defaults` loader recursive.
- Standalone YAML-load smoke verified the V4 Stage 1A ablation configs resolve
  expected token splits (`48/48`, `64/128`, `64/64` no-position) while retaining
  inherited training/data defaults.
- Local helper smoke verified V4 checkpoint compatibility filtering accepts a
  matching `64/128` no-position metadata fixture and rejects split/position
  mismatches; the same smoke also verified V2 compressor mismatch rejection.
- `modal run modal_train.py --help` exposes `--v4-global-image-tokens`,
  `--v4-local-image-tokens`, `--pretrain-image-tokens`, and
  `--v4-use-2d-position-features / --no-v4-use-2d-position-features`.
- Focused pytest could not run in this local Python because `torch` is not
  installed. Re-run in the training environment before launching paid jobs.
- First Modal smoke (`ap-jgsTiijSOk64EpIUAtz8aV`) reached real-data training
  but exposed an OOM in the initial two-tower connector (`3.24B` trainable
  params on A100-80GB). V4 was revised to use shared Perceiver layers for the
  global/local queries.
- Revised Modal smoke (`ap-BrHJUzGWawtihedGJbvJ9r`) completed `2` distributed
  Stage 1A optimizer steps on real LLaVA-Pretrain data and saved
  `/checkpoints/pretrain-output/v4-stage1a-smoke-shared-20260508-codex/checkpoint-2`.
  The shared-tower connector reports `1.63B` trainable params.
- Full Stage 1A run launched detached: `ap-hTW04HKrFzV2fbDo6SVTBP`,
  run name `v4-stage1a-shared-20000-20260508-codex`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/6ffd2qwa`, output
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex`.
  Checkpoints through `checkpoint-4500` are verified complete with
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. W&B step `4160`
  showed train loss `2.01`, loss EMA `2.58`, grad norm `0.80`, grad norm EMA
  `0.88`, grad clip fraction `0.162`, LR `1.82e-4`, and active
  `recent_loss_spikes` plus `recent_grad_spikes`; a later W&B step `4360`
  showed train loss `1.8235`, loss EMA `2.8327`, grad norm `1.2129`, grad norm
  EMA `1.5173`, grad clip fraction `0.1732`, LR `1.8040e-4`, and the same
  alerts. W&B step `4550` showed train loss `3.0209`, loss EMA `2.6327`, grad
  norm `1.2151`, grad norm EMA `1.7192`, grad clip fraction `0.1892`, LR
  `1.7869e-4`, and the same alerts. Eval/loss improved from
  `2.9393` at step `2000` to `2.8393` at step `4000`, and the recent 220-row
  train-loss window was not rising (`2.9019` first half to `2.7948`
  second half), but this is still yellow-watch, not green. W&B/log history
  includes grad spikes at steps `220`, `2540`, and `3948`, loss spikes at steps
  `960`, `2860`, `3996`, and `4140`, and a step-`3998` loss EMA plateau
  warning. Clip fraction is now close to the `0.20` escalation line, so this is
  escalated yellow/orange until W&B shows the drift reversing. W&B then crossed
  the line: step `4730` showed clip fraction `0.2093`, and post-stop step
  `4750` showed clip fraction `0.2112` with active `high_grad_clip_fraction`.
  The app was stopped by babysitting policy; do not smooth over these as
  harmless. `checkpoint-4500` is the last verified complete checkpoint.
  `checkpoint-4750` exists but is partial/unverified because `trainer_state.pt`
  stayed at `1.0 GiB` instead of the expected `12.2 GiB`; do not resume from it.
  Two recovery attempts failed early: a checkpoint-4500 recovery with fresh
  optimizer and LR `8e-5` hit `high_grad_clip_fraction` by step `30`/`40`
  (`0.80`/`0.825`), and a checkpoint-4000 diagnostic canary with LR `5e-5` hit
  `high_grad_clip_fraction` by step `40` (`0.35`). Both were stopped before any
  checkpoint/eval gate. This points away from late optimizer-state-only failure
  and toward connector state, data formatting, or gradient scaling. Logs from
  those runs also showed Stage 1A supervised caption previews included
  `<|begin_of_text|>`, so caption label masking was patched and verified in
  `v4-stage1a-fixedlabels-ckpt4000-lr5e5-canary100-20260508-codex`
  (`https://wandb.ai/babakdam/anymal-pretrain/runs/froy06eg`): supervised
  previews then started with real caption text, but W&B still showed
  `high_grad_clip_fraction` at step `10` with clip fraction `0.60`, and the
  post-stop step `40` clip fraction remained `0.60`. Therefore special-token
  masking is required but not sufficient; do not relaunch the unchanged direct
  V4 Stage 1A recipe.
  A from-scratch bottleneck1024 canary
  (`v4-stage1a-bottleneck1024-canary100-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/dsv4mlbs`) then verified the
  new smaller connector shape, real data, and fixed-label previews: trainable V4
  connector params dropped to `44,374,016`. It still failed the W&B clipping gate
  immediately: step `10` had `high_grad_clip_fraction`, clip fraction `1.0`,
  train loss `5.0610`, loss EMA `9.5638`, grad norm `8.3869`, grad EMA
  `90.8336`, and LR `1e-4`; post-stop step `20` still had clip fraction `1.0`.
  Therefore connector shrinkage is useful but insufficient by itself. The
  immediate follow-up added `connector_output_scale: 0.1` as a gradient-scale
  intervention before any longer run. That canary
  (`v4-stage1a-bottleneck1024-scale01-canary100-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/wm6y5cvh`) also failed the
  W&B gate: step `10` had active `high_grad_clip_fraction`, clip fraction `1.0`,
  train loss `6.7240`, loss EMA `10.1241`, grad norm `48.4955`, grad EMA
  `132.7394`, and LR `1e-4`; post-stop step `20` still had clip fraction `1.0`
  even though train loss fell to `3.9441`. Therefore output-scale-only
  stabilization is insufficient.
  Trainable visual-gate follow-ups at `0.01`, `0.001`, and `0.0001`
  (`cdtr4gp6`, `us0ls2wu`, `2qdqg86p`) also failed the W&B clipping gate. The
  `0.0001` canary completed and saved `checkpoint-20`, but final W&B still
  showed active `high_grad_clip_fraction`, clip fraction `1.0`, train loss
  `4.8986`, loss EMA `6.0846`, grad norm `1.4890`, and grad EMA `78.9965`.
  Therefore gate amplitude alone is not the missing stabilizer.
  The output-projection/gate-only warmup follow-up
  (`v4-stage1a-outputwarmup30-gate00001-lr1e5-canary60-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/iip0i3e3`) also failed before
  full-connector unfreeze: W&B step `20` still had active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.0813`, loss EMA
  `6.5835`, grad norm `17.0686`, and grad EMA `31.0523`. The code path is useful
  for future recipes, but it does not rescue noisy caption Stage 1A. The next A1
  canary changed the supervision to `v4_grounding`
  (`v4-stage1a-groundingfirst-gate00001-lr1e5-canary80-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/udsb1fr9`), but W&B step `10`
  still had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss
  `10.8670`, loss EMA `8.0383`, grad norm `7.5836`, and grad EMA `7.2998`.
  A follow-up with `pretrain_loss_scale=0.1`
  (`v4-stage1a-groundingfirst-lossscale01-gate00001-lr1e5-canary80-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/7lizvbnm`) was the first
  canary to clear the immediate clipping failure: step `50` had clip fraction
  `0.04`, grad norm `0.4288`, and eval loss `8.0005`. It still failed the
  stricter monitor because W&B raised `recent_loss_window_mean_up_gt_25pct` at
  step `60`; the app was stopped with last W&B step `70`, clip fraction `0.0714`,
  train loss `7.4674`, and `checkpoint-50` complete. Therefore the remaining
  likely lever is effective update/gradient semantics plus loss smoothness. A
  direct larger-accumulation follow-up
  (`v4-stage1a-groundingfirst-lossscale01-accum16-gate00001-lr1e5-canary100-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/a530s1tz`) kept clipping
  controlled (`0.1`) but failed faster on the same W&B loss-window alert by step
  `20` and saved no checkpoint, so effective batch scaling alone is not the next
  primary fix. A lower-LR canary at `3e-6`
  (`v4-stage1a-groundingfirst-lossscale01-lr3e6-canary120-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/o72woiib`) also kept clipping
  fully controlled (`0.0` through W&B step `80`) and saved `checkpoint-50`, but
  still failed the same loss-window alert (`7.3366 -> 11.5327`) with first eval
  loss `8.6358`, worse than the `1e-5` loss-scale canary. Next should be
  supervised-token-aware loss normalization, objective/data balancing, or an
  actually gentler schedule shape rather than only lowering LR or increasing
  effective batch. An answer-type-focused Stage 1 mixture
  (`v4-stage1a-answerfocus-lossscale01-gate00001-lr1e5-canary120-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/sjrrgvm6`) then regressed the
  clipping gate: W&B showed active `high_grad_clip_fraction` with clip fraction
  `0.30` at step `10`, `0.25` at step `20`, and `0.50` at step `30`; the
  monitoring sub-agent stopped the app before eval/checkpoint, and no checkpoint
  landed. Therefore answer-type rebalancing alone is not the next sufficient
  fix. A supervised-token-target objective follow-up
  (`v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr1e5-canary120-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/jwd997c5`) then normalized raw
  token-mean loss by `supervised_tokens / 8` before applying
  `pretrain_loss_scale=0.1`. W&B step `50` was the best Stage 1A signal so far:
  no alerts, clip fraction `0.0`, train/objective loss `2.9637`, raw loss
  `11.8549`, supervised tokens `2`, multiplier `0.25`, grad norm `0.1417`,
  grad EMA `0.1296`, eval loss `7.8830`, and complete `checkpoint-50`.
  However, W&B step `60` raised `recent_loss_window_mean_up_gt_25pct` with clip
  fraction still `0.0`; the monitor and parent kept the strict rule and did not
  call it healthy. A direct lower-LR follow-up
  (`v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr5e6-canary120-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/1isusl46`) confirmed LR-only
  shrink is not enough: step `50` was alert-free with clip `0.0`, but eval was
  worse (`8.2909`), and step `60` again raised
  `recent_loss_window_mean_up_gt_25pct` with window `1.5663 -> 2.8931`. The
  monitoring sub-agent judged it red. The decisive follow-up
  (`v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120-20260508-codex`,
  `https://wandb.ai/babakdam/anymal-pretrain/runs/mqebqqc0`) patched trainer
  logging and HealthMonitor inputs to aggregate the full gradient-accumulation
  window before W&B logging. With `train/accumulation_micro_batches=8.0` present
  and correct, the same LR `1e-5` token-normalized recipe completed green to
  step `120`: no alerts, no loss/grad spikes, clip fraction `0.0`, eval improved
  from `7.8095` at step `50` to `5.1151` at step `100`, and final W&B showed
  train/objective loss `1.2308`, raw loss `4.1529`, loss EMA `1.7561`, grad norm
  `0.2211`, grad EMA `0.2826`, and a decreasing recent loss window
  (`2.1678 -> 1.7965`). Treat accumulated-step W&B metrics as mandatory for all
  future gradient-accumulated sparse-label runs; the next long run should extend
  this exact baseline rather than restart with LR-only or connector-only changes.
  The first two `1000`-step extensions were operational aborts, not model-health
  failures: one monitor stopped before the first W&B train row, and the next
  monitor stopped after assuming step `100` was a checkpoint boundary. For
  `max_steps=1000`, the implicit `_checkpoint_save_interval(max_steps)` is
  `250`, while eval cadence is `100`; future long runs should set
  `--pretrain-save-steps` explicitly and the monitoring ledger must record that
  cadence before making artifact stop decisions. For the earlier 20k-step run,
  `_checkpoint_save_interval(max_steps)` also saved every `250` optimizer steps.
- Stage 1B early grounding canary (`ap-A8Op30esUTAMBs2ZCokzU7`, run name
  `v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/ssoqebwc`) loaded
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-2000`,
  not the similarly named path without the date. Logs verified `v4_grounding`,
  `anymal_v4`, and V4 connector-only trainables (`1.63B`). The canary completed
  cleanly and the app stopped. Eval improved monotonically from `1.1841` at step
  `50` to final `0.9570` at step `250`; checkpoints `62`, `124`, `186`, and
  `248` are complete with `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Final W&B at step `250` showed train loss `0.415`, loss EMA
  `0.910`, grad norm `0.571`, grad EMA `0.678`, clip fraction `0.152`, LR
  `2.0e-5`, no grad spikes, and active `recent_loss_spikes` from loss spikes at
  steps `80` and `170`. Treat this as a successful wiring/grounding canary with
  yellow spike history, not final model quality.
- Stage 1B grounding smoke (`ap-Sgt4cPkIPPKUYc6AtPbWNS`, run name
  `v4-stage1b-grounding-smoke-20260508-codex`) loaded
  `/checkpoints/pretrain-output/v4-stage1a-smoke-shared-20260508-codex/checkpoint-2`,
  completed `2` optimizer steps on the V4 grounding mixture, saved
  `/checkpoints/pretrain-output/v4-stage1b-grounding-smoke-20260508-codex/checkpoint-2`,
  and was stopped after completion. The checkpoint contains the full V4
  projector, trainer state, and metadata.
- Stage 2A direct-calibration smoke (`ap-q3IwxSS9IPdAyu0ONA1JwD`, run name
  `v4-stage2a-direct-calibration-smoke-20260508-codex`) loaded the Stage 1B
  smoke checkpoint, froze the connector, trained LoRA-only for `2` optimizer
  steps, and saved
  `/checkpoints/finetune-output/v4-stage2a-direct-calibration-smoke-20260508-codex/checkpoint-2`.
  The trainer verified trainables as `adapter: 0`, `lora: 167,772,160`,
  `other: 0`.
- Stage 2A direct calibration from the Stage 1A `checkpoint-300` candidate
  needed explicit backward loss scaling. The unscaled run
  (`cv70gfmy`) clipped at `1.0`; the explicit LoRA-LR-only run (`7iz3gqye`)
  also clipped at `1.0`; and `loss_scale=0.05` (`ivvpnfop`) still had active
  `high_grad_clip_fraction` with clip fraction `0.4286`. The first clean run
  was `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1a300-20260508-codex`
  (`ap-zvc4ZdZ0C1YHBSpyWvukVr`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/fmrgmyy8`): frozen connector,
  LoRA-only trainables (`167,772,160`), projector LR `1e-5`, LoRA LR `1e-5`,
  and Stage 2 `loss_scale=0.03`. Final W&B step `100` was green: alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, train/raw loss `1.1870`, backward
  loss `0.0356`, loss EMA `1.1151`, grad norm `0.1255`, grad EMA `0.1138`, LR
  `1e-6`, no loss/grad spikes, and eval improved from `1.2776` at step `50` to
  `1.1988` at step `100`. `checkpoint-100` contains `trainer_state.pt`,
  `model_meta.json`, `llm/`, and `projector.pt`. This is the current Stage 2A
  candidate for external held-out VQA, not a promotion by itself. The first
  external VQA reads failed badly despite clean optimization: both checkpoint
  `50` and checkpoint `100` scored `0.10` overall on the 1000-sample seed-42
  `training_chat` screen with the evaluator's generic system prompt, with `0.0`
  yes/no and number accuracy; a 300-sample `legacy_qa` diagnostic rose only to
  `0.667` overall and hit max tokens on `99.7%` of samples. A direct-calibration
  system prompt fixed generation hygiene but not accuracy: checkpoint `100`
  scored `5.367` overall, number `3.704`, other `3.509`, yes/no `8.844`, EOS
  `1.0`, and max-token hit `0.0` on the full 1000-sample read. The promotion
  guard still fails against V1 (`7.567`) and the V3 incumbent (`9.100`).
  Therefore `loss_scale=0.03` is a training-stability fix and direct prompting
  is an evaluation-contract fix, not an accuracy recipe. Do not promote this
  Stage 2A branch; the next ablation should isolate whether the failure is the
  Stage 1A300 base, V4 connector capacity/training, or the direct-calibration
  data mixture.
- The Stage 1B248 isolation ablation
  `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b248-20260508-codex`
  (`ap-lOnsPRzER2eCaTdpEpxEHu`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/z7c3o6ug`) completed the same
  Stage 2A recipe from the Stage 1B grounding `checkpoint-248`. Treat it as
  artifact-good/yellow-history, not green-pure: Lorentz flagged active
  `recent_loss_window_mean_up_gt_25pct` at W&B step `7`, parent W&B checks at
  steps `9` and `12` cleared the alert with accumulation `8.0`, clip `0.0`, and
  no pointwise spikes, and a later W&B read near step `92` again showed active
  `recent_loss_window_mean_up_gt_25pct` while the run had already entered final
  eval/checkpoint. Final W&B step `100` cleared alerts and showed eval `0.7956`,
  clip `0.0`, train/raw loss `0.8841`, backward loss `0.0265`, loss EMA
  `0.6793`, grad norm `0.0144`, grad EMA `0.0156`, LR `1e-6`, no inspector
  loss/grad spikes, and complete `checkpoint-100`. The direct-prompt seed-42
  VQA read scored `7.600` overall, number `4.861`, other `7.083`, yes/no
  `9.524`, EOS `1.0`, max-hit `0.0`, and avg generated tokens `6.446`. This
  narrowly clears the V1 floor overall (`7.600` vs `7.567`) and improves sharply
  over the Stage 1A300 branch (`5.367`), so Stage 1B grounding is useful. It is
  still not promotable because it fails the V3 incumbent (`9.100` overall) and
  does not recover V1 yes/no (`9.524` vs `14.189`). A checkpoint-50 read was
  lower (`7.433` overall, yes/no `9.038`) with the same hygiene, so checkpoint
  `100` remains the best Stage 2A read for this branch. Next action should not
  be more Stage 2A steps on this branch; spend the next expensive run on either
  a cleaner/longer Stage 1B continuation from the accumulated-metrics A1
  checkpoint or the planned DeepStack-lite connector ablation, then rerun the
  same 100-step direct-calibration gate.
- Cleaner Stage 1B continuation from the accumulated-metrics A1 checkpoint is
  now live as `v4-stage1b-clean500-from-a1ckpt300-20260509-codex`
  (`ap-iU9sh2Zc43bqt7W4qcttjm`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/dsuz75tp`). This run loads
  `/checkpoints/pretrain-output/v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex/checkpoint-300`,
  uses `v4_grounding`, connector-only trainables (`44,374,017`), Stage 1 loss
  scale `0.1`, supervised-token-target normalization target `8`, accumulation
  `8`, and explicit save/eval cadence at `100` optimizer steps. W&B is the
  source of truth. Startup had a delegated-monitor W&B sync delay with zero
  history rows while logs had started, preserved as startup yellow-history but
  not a model-health alert. Through W&B step `130`, active alerts are `[]`,
  accumulation is `8.0`, clip fraction is `0.0`, no inspector loss/grad spikes
  are present, eval improved from `2.5661` to `2.3389`, and `checkpoint-100`
  contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Carver's
  W&B step `140` check stayed clean on active alerts but moved to
  yellow-history/watchlist because the recent loss window rose `1.6190 -> 1.9696`
  (`21.6%`, below the W&B alert threshold). Latest step `140` metrics:
  train/objective loss `2.2612`, raw loss `1.9379`, backward loss `0.2261`,
  loss EMA `1.6980`, grad norm `0.1818`, grad EMA `0.3226`, LR `9.1406e-6`.
  W&B logged artifacts were still empty, so volume checkpoint files are the
  current artifact evidence. Step `200` subsequently passed the W&B/eval/artifact
  gate: alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, eval improved again to `2.1605`
  (`2.5661 -> 2.3389 -> 2.2426 -> 2.1605`), train/objective loss `1.3378`, raw
  loss `2.2441`, backward loss `0.1338`, loss EMA `1.5344`, grad norm `0.2582`,
  grad EMA `0.2349`, LR `7.75e-6`, and recent window `1.6604 -> 1.8989`.
  `checkpoint-200` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Step `300` also passed by delegated monitor: W&B alerts
  `[]`, accumulation `8.0`, clip fraction `0.0`, no W&B loss/grad spikes,
  train/objective loss `1.1607`, raw loss `1.9990`, backward loss `0.1161`,
  loss EMA `1.5307`, grad norm `0.1923`, grad EMA `0.2195`, LR `4.7186e-6`,
  recent window `1.8337 -> 1.5904`, and eval trajectory
  `2.3389 -> 2.2426 -> 2.1605 -> 2.0969 -> 2.0531`. `checkpoint-300` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`; metadata matches
  V4 shape (`64` global / `64` local, hidden `1024`, output scale `1.0`, gate
  init `0.0001`, 2D positions enabled). Parent W&B step `330` remained clean
  with alerts `[]`, clip `0.0`, accumulation `8.0`, no inspector spikes, eval
  `2.0531`, and recent window `1.7791 -> 1.6317`. Step `400` gate passed:
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad
  spikes, eval improved to `2.0128`, train/objective loss `2.5717`, raw loss
  `2.2020`, backward loss `0.2572`, loss EMA `1.4973`, grad norm `0.1209`, grad
  EMA `0.1950`, LR `2.0528e-6`, and recent window `1.7796 -> 1.5779`.
  `checkpoint-400` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. The run did not complete to step `500`: post-400 logs
  showed repeated auto-resume from `checkpoint-400`, including
  `Skipping 3200 micro-batches to resume position...`, while W&B stopped
  advancing after step `410` and reported state `crashed`. Final W&B step `410`
  had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, train/objective loss `1.3611`, raw loss `1.6318`, backward
  loss `0.1361`, loss EMA `1.5296`, grad norm `0.1832`, grad EMA `0.1969`, LR
  `1.8594e-6`, eval `2.0128`, and recent window `1.7796 -> 1.5676`. Parent
  stopped the Modal app; final app state was `stopped` with zero tasks. Use
  `checkpoint-400` as the clean-through-400 Stage 1B candidate, but label the
  run yellow-history/operational-crash rather than completed green. There is no
  `checkpoint-500`.
- Downstream Stage 2A from Stage1B400 with the previous stable settings failed
  immediately on clipping. `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b400clean-20260509-codex`
  (`ap-ukMv087X6GXABYwoH5Pet2`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/k00wp8wx`) loaded the
  Stage1B400 checkpoint and correctly ended up LoRA-only after Stage 2 adapter
  disabling (`adapter/projector=0`, LoRA `167,772,160`, other `0`), with frozen
  connector, LR/LoRA LR `1e-5`, accumulation `8`, and `loss_scale=0.03`.
  W&B step `3` already had active `high_grad_clip_fraction`, clip fraction
  `1.0`, grad norm `1.8728`, grad EMA `1.2073`, raw/train loss `3.7292`,
  backward loss `0.1119`, LR `3e-6`, and accumulation `8.0`; parent stopped the
  app. Final W&B step `4` stayed red with clip fraction `1.0`, no eval rows, and
  no checkpoints. This means the cleaner Stage 1B400 checkpoint changes the
  Stage 2 scale requirement: retry with a lower Stage 2 update scale before any
  accuracy read, rather than running the `0.03` recipe longer.
- The lower-scale retry
  `v4-stage2a-directcal-lossscale001-lora1e5-from-stage1b400clean-20260509-codex`
  (`ap-NIhQMJHaMAc1eEZbJvlBdJ`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/qx0mcvwr`) completed cleanly.
  It kept the same frozen-connector LoRA-only contract, but used
  `loss_scale=0.01`. W&B step `2` had alerts `[]`, clip `0.0`, grad norm
  `0.5715`, and accumulation `8.0`, clearing the early clipping failure. Step
  `50` passed with eval `1.2514` and complete `checkpoint-50`. Final W&B step
  `100` finished with alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector
  loss/grad spikes, eval `1.1746`, raw/train loss `1.1429`, backward loss
  `0.0114`, loss EMA `1.0822`, grad norm `0.0310`, grad EMA `0.0404`, LR
  `1e-6`, and complete `checkpoint-100`. This is now the Stage1B400 Stage 2A
  checkpoint to evaluate on the direct-prompt VQA screen. That read failed:
  `vqa_eval_v4_stage2a_stage1b400_lossscale001_ckpt100_directprompt_training_chat.json`
  scored overall `5.400`, number `3.704`, other `3.379`, yes/no `9.135`, EOS
  `0.618`, max-hit `0.382`, and avg generated tokens `14.831`. Optimization was
  clean, but generation hygiene regressed badly versus the Stage1B248 branch
  (`7.600` overall, EOS `1.0`, max-hit `0.0`) and the V3 incumbent. Do not
  promote Stage1B400. The next meaningful move is not more Stage 2A on this
  checkpoint; it is a connector/recipe change that preserves grounding without
  making the LoRA calibration brittle, e.g. DeepStack-lite or a Stage 1B recipe
  that does not require shrinking Stage 2 gradients to the point where EOS
  behavior degrades.
- DeepStack-lite is now wired for the next V4 architecture ablation. It keeps
  the stabilized V4 spatial global/local token budget, but changes the
  connector input from only the final SigLIP2 feature map to a small stack of
  hidden-state levels. Default launch should use
  `--v4-connector-type deepstack_spatial_perceiver_resampler`,
  `--v4-deepstack-num-feature-levels 3`,
  `--v4-deepstack-hidden-state-indices=-3,-2,-1`, and explicitly preserve the
  stabilized `--v4-connector-output-gate-init 0.0001` instead of Modal's legacy
  default. Checkpoint metadata and VQA eval now understand the DeepStack fields;
  do not rely on auto-discovery unless the connector type is passed, because the
  default spatial and DeepStack branches are not interchangeable.
- DeepStack-lite has now cleared two paid Stage 1A canaries under the strict
  W&B-first babysitting policy. The 20-step run
  `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary20-20260509-codex`
  (`w7ktjb94`) finished with alerts `[]`, accumulation `8.0`, clip `0.0`, no
  inspector loss/grad spikes, train/objective loss `2.3092`, raw loss
  `7.4254`, loss EMA `2.7728`, grad norm `0.1844`, and a complete
  `checkpoint-20` with DeepStack metadata. The 120-step run
  `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120b-20260509-codex`
  (`kb8js5z0`) finished with alerts `[]`, accumulation `8.0`, clip `0.0`, no
  loss/grad spikes, eval improving `7.7001 -> 7.1668`, train/objective loss
  `1.9442`, raw loss `6.7862`, loss EMA `2.3480`, grad norm `0.1339`, and a
  complete `checkpoint-100` whose metadata confirms
  `deepstack_spatial_perceiver_resampler` over layers `[-3, -2, -1]`. Both runs
  keep yellow-history for startup W&B sync gaps and empty W&B artifacts, but
  neither shows a model-health stop condition.
- The longer DeepStack Stage 1A extension has now completed to step `500`.
  Run
  `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex`
  (`xb0silsi`) finished with W&B state `finished`, alerts `[]`, train rows
  `50`, accumulation `8.0`, no inspector loss/grad spikes, final clip fraction
  `0.002000`, train/objective loss `1.9103`, raw loss `3.9376`, loss EMA
  `1.0196`, grad norm `0.2567`, grad EMA `0.1705`, and eval improving
  `8.2100 -> 6.9942 -> 6.1354 -> 3.6915 -> 3.3106 -> 3.0524 -> 2.8963 -> 2.8282 -> 2.7581 -> 2.7120`.
  Checkpoints `200`, `300`, `400`, and `500` contain `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`. Keep yellow-history attached for
  startup W&B sync, early eval lag, the step-60 objective jump, the step-100
  subthreshold loss-window rise, console-only spikes around steps `353` and
  `499`, and nonzero clipping.
- Recommendation update after downstream validation: DeepStack-lite is no longer
  the default V4 architecture branch. Stage 1A improved internal eval, but Stage
  1B had to be stopped at step `350` on active W&B `recent_loss_spikes`, leaving
  `checkpoint-300` as the last safe artifact. Stage 2A from that artifact failed
  at `loss_scale=0.03` on clipping, and the lower `loss_scale=0.01` branch was
  stopped at step `95` on active W&B `recent_loss_window_mean_up_gt_25pct`.
  The last clean Stage 2A artifact (`checkpoint-50`) passed generation hygiene
  but failed held-out VQA: overall `4.233`, number `2.546`, other `2.469`,
  yes/no `7.580`, EOS `1.0`, max-hit `0.0`. Promotion guard fails V1 floor,
  incumbent, and yes/no recovery.
- Architecture direction: keep the DeepStack code path and metadata support as a
  useful ablation, but do not make it v4 default without a new grounding and
  calibration recipe. The next v4 recipe should prioritize Stage 2 calibration
  data/answer semantics and supervised-token/loss normalization over more
  DeepStack connector training; the best observed downstream branch remains the
  earlier Stage1B248 Stage2A result (`7.600` overall), still below incumbent and
  yes/no recovery.
- The first `v4_semantic_calibration` isolation attempt from Stage1B248
  (`muyq9oz9`) proved the new data path works but did not validate the recipe.
  It built the balanced canonical yes/no source and reached LoRA-only training
  with accumulation `8.0` and clipping `0.0`, then parent stopped at W&B step
  `10` after repeated active `recent_loss_window_mean_up_gt_25pct` alerts. Final
  W&B step `11` cleared after the stop signal, but there was no eval/checkpoint;
  keep this as STOP/yellow-red history. A retry needs a predeclared early-window
  policy or a recipe change that avoids the early loss-window trip without
  weakening EOS learning.
- The bs4/effective-batch-32 semantic retry (`oxuky3ss`) is the first promoted
  V4 candidate under the corrected evaluator. W&B finished cleanly at step
  `100`, artifacts are complete, the corrected seed-42 direct-prompt VQA score
  is `52.400` overall, and the refreshed promotion guard passes against the
  corrected V1 floor (`21.100`) and corrected V3 incumbent (`9.400`). Preserve
  the old `7.333` read as evaluator-bug history: it came from scoring decoded
  `assistant\n\n<answer>` prefixes as the answer.
- Seed-43 confirmation on the same checkpoint also stayed in the same regime:
  overall `51.367`, number `35.430`, other `40.370`, yes/no `72.043`, EOS
  `1.0`, max-hit `0.0`. This clears the "not a single seed-42 accident" bar,
  though larger-slice confirmation is still useful before a public benchmark
  claim.
