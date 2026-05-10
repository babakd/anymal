# AnyMAL V2 Quality Plan

Last updated: 2026-04-30

## Purpose

This is the execution plan for improving AnyMAL V2 quality after the V2 hardening pass. V2 is now the active architecture:

`SigLIP2 -> learned token compressor -> MLP bottleneck projector -> LLaMA-3-8B-Instruct`

The goal is to move measured quality, not just training loss. Future agents should treat this document as the priority stack for V2 experiments unless a newer plan supersedes it.

## Current Baseline

- V2 smoke runs passed for Stage 1 and Stage 2 on Modal.
- V2 uses SigLIP2 preprocessing at 384px and strict placeholder/token matching.
- Stage 1 currently trains `token_compressor + projector`, with vision and LLM frozen.
- Stage 2 trains `token_compressor + projector + LoRA`.
- The active 2026-04-29 branch uses 384 image tokens in both Stage 1 and Stage 2.
  Older 256-query Stage 1 / 384-query Stage 2 expansion runs are diagnostic only
  and must not be used for model selection.
- VQA eval is V2-placeholder-aware, uses deterministic held-out subsets, reports
  answer-type breakdowns, and fails loudly when valid coverage is too low.
- Existing V1 Stage 2 experiments showed dataset choice mattered far more than LR sweeps, and SFT tended to increase verbosity and confident hallucination.
- The corrected 2026-04-29 held-out VQA protocol uses the Stage 2 training chat
  prompt, trims generations at the first stop token, and stops on `<|eot_id|>`.
  Under this protocol, V1 ablation-F is the current baseline at `7.57%` overall
  on the 1000-sample seed-42 slice.
- The earlier VQA scoreboard with V1 at `5.03%` and original V2 at `6.70%` is
  superseded. It used a legacy prompt and counted padded generation tokens.
- The corrected full branches from the 256-query Stage 1 / 384-query Stage 2
  expansion collapsed on scheduled reads and were stopped:
  `v2-stage2-balanced-mix-light-3000-20260429` checkpoint 300 scored `0.10%`,
  `v2-stage2-normalized-light-direct-object-full-3000-20260429` checkpoint 1000
  scored `0.13%`, and
  `v2-stage2-normalized-direct-object-full-3000-20260429` checkpoint 1000 scored
  `0.37%`.
- A true 384-query Stage 1 pretrain completed on real LLaVA-Pretrain images:
  `/checkpoints/pretrain-output/v2-stage1-learned-384q-3000-20260429/checkpoint-3000`.
  Three full Stage 2 runs from this checkpoint are active:
  `v2-stage2-384q-balanced-mix-light-3000-20260429`,
  `v2-stage2-384q-light-direct-object-3000-20260429`, and
  `v2-stage2-384q-vqa-direct-object-balanced-noeval-3000-20260429`.
- The VQA-enhanced Stage 2 branch adds VQAv2 train2014 direct-answer
  supervision from a deterministic COCO train2014 image subset. It does not add
  held-out VQAv2 val2014 examples to training.
- A full 384-query Perceiver Stage 1 connector ablation is active:
  `v2-stage1-perceiver-384q-3000-20260429`.

## Implementation Status

The first parallel implementation batch has landed locally:

- Eval hardening:
  - COCO captioning eval now supports V2 contiguous image placeholder blocks.
  - Captioning eval uses V2 SigLIP2 transforms when evaluating AnyMALv2.
  - Captioning eval filters/skips missing or corrupt images and reports `num_samples`, `avg_generated_tokens`, and `eos_rate`.
  - VQA eval defensively skips all-invalid batches.
- Data plumbing:
  - Added `LlavaPretrainCaptionDataset` for LLaVA-Pretrain/BLIP-style caption JSON with real-image filtering, caption length filtering, and caption de-duplication.
  - Added `InstructionMixtureDataset` and factory support for balanced/concat Stage 2 mixtures.
  - V2 pretrain config now points at LLaVA-Pretrain style data knobs; V2 finetune config includes inactive mixture knobs.
  - Local training scripts consume the new config keys.
- Architecture:
  - Added config-gated `token_compressor_type="perceiver"` / `"perceiver2"` variants.
  - The new compressor is a 2-layer residual cross-attention + FFN resampler.
  - Existing `token_compressor_type="learned"` parameter names and behavior are preserved for checkpoint compatibility.
- Modal integration:
  - Stage 1 Modal pretrain prefers true LLaVA-Pretrain captions when `/checkpoints/llava_pretrain/images` is staged, otherwise falls back to the existing COCO/instruction-derived caption path.
  - Stage 2 Modal finetune supports `--dataset balanced_mix`, combining `instruct_150k` and filtered `mix_665k`.

Validation completed:

- `pytest tests -q` -> 118 passed, 1 skipped.
- `git diff --check` passed.
- Modal V2 Stage 1 smoke: `v2-pretrain-integration-smoke-20260427`, app run `ap-krX0b1NJwdAD0WGO4VdJAD`, completed 2 optimizer steps.
- Modal V2 Stage 2 `balanced_mix` smoke: `v2-finetune-balanced-mix-smoke-20260427`, app run `ap-jEdIuwx1tvvHe4TiSdlE1v`, completed 2 optimizer steps.

Known caveat:

- The true LLaVA-Pretrain annotation JSON is cached on Modal, but its corresponding image payload is not staged at `/checkpoints/llava_pretrain/images`. Until those images are staged, Stage 1 Modal runs intentionally fall back to the existing COCO-backed caption extraction path.

## Priority 0: Make Evaluation Trustworthy

Do this before judging architecture changes.

### Anti-Overfit Rules

- Predeclare checkpoint evaluation points before looking at generations.
- Use the fixed held-out VQA slice (`max_eval_samples=1000`, `subset_seed=42`,
  `min_eval_samples=500`) as the first scoreboard for Stage 2 candidates.
- Keep known failure examples as canaries only. They can diagnose whether a
  capability exists, but they must not select the final checkpoint.
- Report answer-type breakdowns so a gain on `yes/no` or `number` does not hide
  a regression on open-answer `other` questions.
- Do not add new eval examples from inspected failures to the selection set for
  the same run. Put them in a future canary/probe suite and evaluate a later run.
- Treat train loss as a health signal, not a capability signal.
- Use a separate deterministic confirmatory VQA image cache after checkpoint
  shortlisting; do not expand or reshuffle the primary scoreboard during a run.
- Do not select from the old direct-object or direct-natural runs. They are
  systems diagnostics only because they were launched before the normalized
  direct-answer data fix.
- Do not select from the stopped 256/384-expanded Stage 1 branches above. Their
  corrected VQA scores are below V1 by a wide margin.
- For the 384-query Stage 1 branches, evaluate only checkpoints `300`, `1000`,
  `2000`, and `3000` before shortlisting.
- Keep train-loop VQA benchmarking disabled for future full training runs.
  Held-out VQA reads should be launched as explicit external jobs at the
  predeclared checkpoint milestones.

### Tasks

- Fix/expand VQA image coverage so VQA scores are not dominated by missing images.
- Make captioning eval V2-compatible by inserting the correct V2 placeholder block.
- Add a stable qualitative eval set: around 100 fixed images, fixed prompts, saved generations, and side-by-side checkpoint comparison.
- Add benchmark coverage for:
  - VQAv2: short-answer visual QA.
  - GQA: compositional and spatial grounding.
  - TextVQA or OCR-style eval: text in images.
  - POPE/MMHal-style hallucination checks.
  - COCO captioning or similar caption metrics.
- Track generation health:
  - generated token count,
  - EOS/clean-stop rate,
  - repetition rate,
  - concise-answer compliance,
  - unsupported-detail or hallucination rate on curated examples.

### Relevant Files

- `evaluation/vqa_eval.py`
- `evaluation/eval_runner.py`
- `evaluation/captioning_eval.py`
- `modal_train.py`
- `three_way_inference.py`
- `modal_viewer.py`

## Priority 1: Improve Data Before Recipe Tuning

The previous ablations strongly suggest data moves quality more than small hyperparameter sweeps.

### Stage 1 Data

Current Modal Stage 1 extracts first GPT responses from `llava_instruct_150k.json` via `load_llava_pretrain_dataset()`. That is not ideal alignment data.

Replace or augment it with real pretraining image-caption data:

- Use the downloaded LLaVA-Pretrain `blip_laion_cc_sbu_558k.json` instead of instruction-derived captions.
- Add CC3M/CC12M or another higher-quality caption corpus if available.
- Keep real images only. Do not reintroduce dummy images for real experiments.
- Add caption-quality filtering and de-duplication.

### Stage 2 Data

Build a balanced instruction mixture rather than training only on long GPT-style conversations.

Include:

- short-answer VQA,
- object/attribute/spatial grounding,
- OCR and document/chart examples,
- concise captioning,
- detailed captioning,
- false-premise and unanswerable examples,
- multi-turn visual chat,
- hallucination contrast examples.

The model should learn both "describe in detail" and "answer in one word" behavior. The prior Stage 2 issue was stylistic overfitting toward long, confident answers.

### Relevant Files

- `modal_train.py`
- `data/instruction_dataset.py`
- `data/laion_dataset.py`
- `configs/pretrain_v2_alignment.yaml`
- `configs/finetune_v2.yaml`

## Priority 2: Dynamic and High-Resolution Visual Tokens

Fixed token counts are a likely quality bottleneck. Easy images waste tokens, while detail-heavy images are underrepresented.

### Target Behavior

- 128 tokens for simple images.
- 256-384 tokens as the default.
- 576-768 tokens for dense scenes, OCR, charts, documents, and small-object questions.
- Preserve native aspect ratio where possible.
- Add a global low-resolution view plus high-resolution tiles for detail-heavy images.

### Implementation Direction

- Extend the dataset to choose token budget per sample or per prompt type.
- Extend eval/inference prompt builders to insert matching placeholder counts.
- Use the existing V2 strict splice checks to catch mismatches.
- Add config knobs for max token budget and policy.

### Relevant Files

- `models/anymal_v2.py`
- `models/projectors/token_compressor.py`
- `data/instruction_dataset.py`
- `data/laion_dataset.py`
- `evaluation/vqa_eval.py`
- `modal_train.py`

## Priority 3: Replace the One-Layer Compressor

The current `TokenCompressor` is a single learned-query cross-attention pooling layer. It is probably too shallow for strong spatial reasoning and OCR/detail preservation.

### Experiments

- Add a 2-layer Perceiver-style resampler with residuals, LayerNorm, cross-attention, and FFN.
- Add a 4-layer variant if the 2-layer variant helps.
- Add explicit 2D positional embeddings before compression.
- Add row/newline embeddings for packed image grids.
- Try concatenating multiple SigLIP2 feature layers, not only final hidden states.
- Preserve a few salience/detail tokens alongside global compressed tokens.

### Success Criteria

Look for gains on GQA, TextVQA/OCR, curated small-object prompts, hallucination rate, and qualitative detail grounding. Do not accept a connector change based only on train loss.

### Relevant Files

- `models/projectors/token_compressor.py`
- `models/projectors/perceiver_resampler.py`
- `models/anymal_v2.py`
- `tests/test_model.py`

## Priority 4: Query-Aware Token Selection

V2 currently compresses the image the same way regardless of the question. That can discard the exact region needed for a prompt.

### Experiments

- Encode the text prompt before image compression.
- Use text-conditioned queries for cross-attention into image features.
- Keep global image tokens plus prompt-focused detail tokens.
- Evaluate especially on questions about small objects, text, counts, color, and spatial relations.

This is more invasive than a deeper compressor, so run it after the baseline dynamic-token and Perceiver-style connector experiments.

## Priority 5: Light Vision-Tower Adaptation

SigLIP2 is frozen in V2. After a strong adapter baseline exists, test whether small vision adaptation helps.

### Experiments

- Add LoRA to the last few SigLIP2 attention/MLP blocks.
- Or unfreeze only the last 2-4 SigLIP2 blocks with a much lower LR than the connector.
- Train on high-quality grounding, OCR, and VQA data.

### Risks

- Higher overfitting risk.
- Higher memory and training cost.
- Harder checkpoint compatibility.

Gate this on evals, not loss.

## Priority 6: Add Hallucination and Verbosity Alignment

SFT alone previously made responses longer and sometimes more confidently wrong. Add an alignment phase after SFT once the SFT model is reasonably grounded.

### Experiments

- DPO/ORPO-style preference tuning.
- Chosen responses: concise, grounded, correct, and uncertainty-aware.
- Rejected responses:
  - verbose hallucinations,
  - unsupported breed/object/color claims,
  - false-premise answers that fail to correct the premise,
  - long answers when a short answer was requested.

Track POPE/MMHal-style hallucination metrics and curated false-premise prompts.

## Priority 7: Consider the Base LLM

If the goal is maximum quality rather than preserving the LLaMA-3-8B stack, test a stronger or more VLM-friendly text backbone.

Options:

- Larger LLaMA-family model if compute allows.
- Qwen-family model if OCR, multilingual, structured output, or document understanding matter.
- Warm-start from an existing open VLM, then adapt AnyMAL-specific data.

This is a larger product decision. Do it after the V2.1 data/eval/connector work unless there is a strong reason to switch earlier.

## Recommended Experiment Sequence

1. Build the stable eval suite and curated qualitative set.
2. Run a real V2 baseline: real Stage 1 checkpoint, then Stage 2 from that checkpoint.
3. Replace Stage 1 data with true pretraining captions and compare against the current instruction-derived caption path.
4. Compare Stage 2 mixtures:
   - `instruct_150k`,
   - `mix_665k`,
   - short-answer-balanced mix,
   - OCR/grounding-enriched mix.
5. Ablate token budget:
   - fixed 256,
   - fixed 384,
   - dynamic 128-768.
6. Ablate connector:
   - current learned compressor,
   - 2-layer Perceiver-style resampler,
   - 4-layer Perceiver-style resampler.
7. Add global-plus-local high-resolution tiling.
8. Add hallucination/verbosity DPO.
9. Test light SigLIP2 adaptation.
10. Only then consider LLM backbone changes.

## First Concrete Execution Target

For the next real run, do not start with another LR sweep.

Prepared run commands, canary run IDs, and gates live in `V2_TRAINING_RUNBOOK.md`.
The 2026-04-27 true LLaVA-Pretrain image staging attempt failed on the existing
Modal volume because the exploded JPEG tree exhausted the volume inode/device
budget; use COCO-backed Stage 1 canaries only as systems checks until image
storage is moved to tar/WebDataset shards, object storage, or a larger/different
volume.

Target:

1. Fix VQA/captioning eval coverage enough to produce comparable V2 metrics.
2. Add or wire a real LLaVA-Pretrain/CC3M Stage 1 dataset path.
3. Train V2 Stage 1 for a meaningful checkpoint.
4. Train V2 Stage 2 from that checkpoint on `mix_665k`.
5. Compare against:
   - Stage 1-only baseline,
   - Stage 2 `instruct_150k`,
   - Stage 2 `mix_665k`.

Required reporting:

- checkpoint paths,
- exact data mixture,
- token budget policy,
- train/eval losses,
- VQA/GQA/OCR/caption/hallucination metrics where available,
- generated-token and EOS rates,
- qualitative viewer link or saved JSON.

## Design Principle

Prefer changes that improve grounded visual information reaching the LLM and changes that make evaluation more truthful. Avoid judging V2 progress from train loss alone.
