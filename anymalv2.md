# AnyMAL v2 Status

`anymal_v2` is the alternate multimodal architecture in this repo. It is no longer just a design note: the model, training entry points, evaluation code, checkpoint metadata, and Modal inference path now all understand it as a first-class architecture separate from `anymal_v1`.

## What `v2` Is

Current `v2` stack:

- Vision encoder: `google/siglip2-so400m-patch14-384`
- Visual bridge:
  - learned token compressor
  - MLP bottleneck projector
- LLM: LLaMA-3-8B-Instruct
- Fusion: strict placeholder splice only

High-level path:

```text
image -> SigLIP2 -> visual tokens -> token compressor -> MLP projector -> LLaMA-3
```

## What Is Implemented

### 1. Explicit architecture contract

`models/anymal_v2.py` now exposes the same contract surface as `v1`, but with `v2`-specific values:

- `architecture = "anymal_v2"`
- `preprocessing_family = "siglip2"`
- `fixed_image_token_count = max_image_tokens`
- `get_visual_bridge_modules() -> {"token_compressor", "projector"}`

This contract is what training, eval, and inference now consume.

### 2. Architecture-specific preprocessing

`v2` no longer rides on the CLIP transform path.

The repo now derives transforms from the instantiated model:

- `v1` -> CLIP resize/crop/normalize
- `v2` -> Hugging Face SigLIP2 image processor

This is wired through:

- `InstructionDataset`
- `LaionDataset`
- `EvalRunner`
- `VQADataset`
- `COCOCaptionDataset`
- `modal_train.py`
- `modal_inference.py`

Current `v2` configs use native `384` preprocessing.

### 3. Visual bridge warmup and optimizer grouping

Training code no longer assumes the only trainable visual module is named `projector`.

The trainer now:

- groups optimizer params by architecture-defined bridge modules
- logs learning rates and gradient norms by those groups
- freezes and unfreezes the full `v2` visual bridge during Stage 2 warmup

For `v2`, that means both:

- `token_compressor`
- `projector`

### 4. Shared multimodal prompt building

Prompt and placeholder handling is centralized in `data/multimodal_inputs.py`.

Repo-owned train/eval/inference paths now share the same helper logic for:

- resolving the image placeholder token ID
- building fixed placeholder blocks
- building single-turn multimodal chat prompts
- replacing the training-time sentinel with real placeholder tokens

`v1` still keeps prepend fallback inside the model for backward compatibility. `v2` remains strict and requires placeholders.

### 5. Architecture-aware checkpoint loading and inference

Checkpoints now save `model_meta.json` with architecture metadata.

That metadata is used to:

- reject cross-architecture loads
- distinguish legacy `v1` checkpoints from explicit `v2` checkpoints
- let `modal_inference.py` discover and group checkpoint progressions by architecture

`modal_inference.py` no longer hard-codes `AnyMAL`/`v1`.

## What Is Still Intentionally Simple

This hardening pass did not try to make `v2` frontier-like. The following are still future work:

- variable per-sample image token counts
- dynamic or native aspect-ratio packing
- late partial SigLIP2 unfreezing
- richer multimodal continuation-pretrain data mixtures
- larger ablation grids on token budget and resolution

The point of this pass was separation and correctness, not a full training recipe overhaul.

## `v1` vs `v2`

| Area | `anymal_v1` | `anymal_v2` |
|---|---|---|
| Vision encoder | CLIP ViT-L/14 | SigLIP2 So400m |
| Bridge | Perceiver Resampler | Token compressor + MLP projector |
| Preprocessing | CLIP transform | SigLIP2 processor |
| Placeholder behavior | splice or prepend fallback | strict splice |
| Stage 1 trainable modules | projector | token compressor + projector |
| Stage 2 warmup target | projector | full visual bridge |
| Checkpoint compatibility | legacy default | explicit metadata required |

## Current Training Entry Points

Local:

- `configs/pretrain_v2_alignment.yaml`
- `configs/finetune_v2.yaml`

Modal:

- `modal run modal_train.py --stage pretrain --architecture anymal_v2`
- `modal run modal_train.py --stage finetune --architecture anymal_v2`

Evaluation and inference now pick transforms and prompt construction from the active model instead of assuming `v1`.

## Why Keep `v2` Separate Instead Of Folding It Into `v1`

- The preprocessing contract is genuinely different.
- The trainable bridge is genuinely different.
- The placeholder semantics are stricter.
- Checkpoint interchangeability would be misleading and unsafe.

Keeping the split explicit makes the repo easier to understand, easier to debug, and easier to extend.

## Immediate Next Steps

The highest-value follow-on work is now training-focused rather than plumbing-focused:

1. Run a clean `v2` alignment baseline on a few H100s.
2. Improve Stage 1 data quality before adding more architectural complexity.
3. Add one text-rich/OCR evaluation bucket alongside VQAv2 and captioning.
4. Ablate token budget and bridge LR after the baseline is stable.
