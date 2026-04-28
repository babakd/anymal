# Fix Training: Why AnyMAL Produces No Multimodal Output

## Problem Statement

After 500 steps of Stage 2 finetuning on real COCO images, the model produces zero multimodal output. Predictions are purely language-dependent — at step 500, 16/20 examples produce the identical system prompt echo regardless of input image. The model actively learned to **ignore** image tokens during training.

## Root Cause

**We skipped Stage 1 (alignment pretraining).** The Perceiver Resampler is randomly initialized and has never been trained to project CLIP features into LLaMA's embedding space. When Stage 2 trains LoRA + the random perceiver simultaneously, LoRA finds a shortcut: suppress the perceiver noise and predict from text context alone.

## Evidence

### 1. Prediction collapse across training steps

Generated via `modal run modal_inference.py --num-examples 20`. Results in `predictions.json`, viewable in `prediction_viewer_output.html`.

| Step | Unique outputs (of 20) | Dominant pattern |
|------|----------------------|------------------|
| 0 | 18 | Random gibberish — each image produces different noise (`istaista...`, `HexString...`, `pora...`) |
| 250 | 18 | Echoes question or system prompt fragments |
| 375 | 17 | Loops question text or role tokens (`assistant assistant assistant...`) |
| 500 | 5 | 16/20 identical: system prompt echo. Image-independent |

**Key insight:** Step 0 has MORE diversity than step 500. The random perceiver creates different noise per image, which perturbs LLaMA differently. Training actively removes this sensitivity — LoRA learns to suppress the noise. The model becomes LESS multimodal over training.

### 2. Loss drops but only reflects text learning

Training loss went 7.5 → 1.4. This looks like learning but is explained entirely by text pattern acquisition:

- Starting loss 7.5 is elevated above LLaMA's natural text loss (~1-2) because 64 random perceiver tokens disrupt attention
- LoRA learns to suppress these noise tokens → loss recovers to near-baseline LLaMA performance
- Loss 1.4 = LLaMA predicting text well while ignoring image tokens. No multimodal contribution needed
- The training data has highly predictable text patterns: 35% of responses start with generic phrases ("The image shows...", "There are two..."). Average first response is only 81 chars

### 3. No Stage 1 checkpoint loaded

In `training/finetune.py:98`:
```python
if config.pretrain_checkpoint:
    self._load_pretrain_checkpoint(model, config.pretrain_checkpoint)
```
`pretrain_checkpoint` is `None`. The perceiver starts with random weights.

### 4. LoRA alpha is 4x too strong

The paper uses `r=64, alpha=16` (scaling ratio 0.25). Our code uses `r=64, alpha=64` (ratio 1.0) — set in `modal_train.py:587`. This amplifies LoRA's capacity to override the base model, making it even easier for LoRA to learn the "ignore images" shortcut.

### 5. Inference script has a separate bug

The inference script (`modal_inference.py`) hits the PREPEND fallback path instead of SPLICE:
```
[AnyMAL] _splice_image_tokens: PREPEND (fallback), placeholder_id=128002, found=0
```
This means the inference prompt doesn't contain placeholder tokens. The training dataset's `_encode_with_response_masking` handles `<image>` → sentinel → placeholder replacement, but the inference script constructs the prompt manually without this. Fix this when fixing inference, but it's secondary — the model wouldn't produce meaningful output even with correct splicing because the perceiver is untrained.

## What the Paper Says

Source: `2309.16058v1.pdf` (extracted to `/tmp/anymal_paper.txt` for grep/search)

### Stage 1: Alignment Pretraining (Section 3.1)
- Train ONLY the projection module (Perceiver Resampler)
- LLM is **completely frozen** — no LoRA, no adapters
- Data: paired image-caption (LAION-2B cleaned subset, 200M images)
- Objective: next-token prediction on captions given image tokens
- **100K steps, batch size 2048, LR 2e-4** (Table 12, line 1340)
- 6-layer Perceiver Resampler, 64 modality embeddings
- Uses 4-bit quantization of frozen LLM to fit on single A100-80GB

### Stage 2: Instruction Fine-tuning (Section 3.2)
- Train Perceiver Resampler + LoRA adapters
- Data: MM-IT (60K human + 150K synthetic instruction pairs)
- **3K steps, batch size 128, LR 1e-5** (Appendix B.3, line 1405)
- LoRA r=64, alpha=16, on all linear layers
- Paper ablates: "(1) training projection layers without altering LLM, or (2) using LoRA" (line 210-212)

### Why Stage 1 must come first (my reasoning)
When you train LoRA + a random perceiver simultaneously:
- Perceiver outputs random 4096-dim vectors (noise to LLaMA)
- LoRA has two paths to reduce loss: (a) interpret image tokens, or (b) ignore them and predict from text
- Path (b) is far easier — small perturbation to existing LLaMA text capabilities
- Path (a) is a moving-target problem — perceiver representations change every step while LoRA tries to interpret them
- Result: LoRA always finds the shortcut

Stage 1 fixes this by giving the perceiver a stable learning target (frozen LLM). The perceiver converges to produce embeddings the LLM already understands. Then Stage 2 adds LoRA to fine-tune against already-meaningful image tokens — no shortcut available.

## Implementation Plan

### Step 1: Fix LoRA alpha (quick, do first)

In `modal_train.py:587`, change:
```python
lora_alpha=64,  # WRONG: ratio 1.0
```
to:
```python
lora_alpha=16,  # Paper: ratio 0.25
```

### Step 2: Implement Stage 1 pretraining on Modal

The Stage 1 trainer already exists (`training/pretrain.py`), and `set_training_stage(1)` correctly freezes LLM + unfreezes projector only. The `modal_train.py` already has `--stage pretrain` support (see the `pretrain` method around line 700+). What's needed:

1. **Data**: Stage 1 needs image-caption pairs, not instruction data. Options:
   - Use LLaVA-Pretrain (CC3M subset, 558K pairs) — JSON already cached on volume at `/checkpoints/llava_data/blip_laion_cc_sbu_558k.json`
   - Or use the LLaVA-Instruct data but only train on first-turn responses (simpler, no new data needed)

2. **Training config**: Match paper where possible:
   - LR: 2e-4
   - Steps: Start with 5K-10K (we can't do 100K at paper's batch size, but our dataset is much smaller)
   - Batch size: as large as fits (2-4 on single A100-80GB with QLoRA)
   - Gradient accumulation: increase to simulate larger effective batch
   - Only projector trainable, LLM completely frozen (no LoRA)

3. **Save perceiver checkpoint**: Save `projector.pt` at the end of Stage 1

4. **Cost estimate**: 10K steps at ~3.5 it/s ≈ 48 min ≈ $4 on A100-80GB

### Step 3: Run Stage 2 with pretrained perceiver

1. Pass `--pretrain-checkpoint` pointing to the Stage 1 output
2. This loads `projector.pt` via `FinetuneTrainer._load_pretrain_checkpoint()` (line 115-119)
3. Use corrected LoRA alpha=16
4. Run 3K steps (paper's recommendation) instead of 500

### Step 4: Re-run inference to verify multimodal learning

```bash
modal run modal_inference.py --num-examples 20
```

**What to look for:**
- Predictions should vary meaningfully across different images at the same step
- Unique output count should INCREASE during training (not collapse like before)
- Responses should reference image-specific content (colors, objects, counts, spatial relationships)
- Step 0 (base model after Stage 1 only) should already produce somewhat relevant captions
- Later steps should show instruction-following improvement

### Step 5: Fix inference placeholder bug

The inference prompt in `modal_inference.py:_run_inference()` constructs the prompt as a raw string with `<image>` but doesn't replace it with placeholder token IDs. The tokenizer doesn't know `<image>` is special — it tokenizes it as normal text tokens, so `_splice_image_tokens` can't find placeholders and falls back to PREPEND.

Fix: either replicate the sentinel → placeholder replacement logic from `InstructionDataset._encode_with_response_masking`, or insert `num_image_tokens` copies of `model.image_placeholder_token_id` directly into `input_ids` where `<image>` would go.

## How to Verify Each Fix

### Verify Stage 1 is working
- Loss should start near ln(vocab_size) ≈ 11.76 and decrease
- After training, load the perceiver checkpoint, run inference on a few images with caption prompts ("Describe this image")
- Output should be rough but image-relevant captions (not gibberish or system prompt echoes)

### Verify LoRA alpha fix
- Compare grad norms before/after — with alpha=16, LoRA updates should be ~4x smaller
- Less aggressive LoRA = less ability to shortcut

### Verify Stage 2 multimodal learning
- Run the prediction viewer pipeline: `modal run modal_inference.py --num-examples 20`
- In `prediction_viewer_output.html`, use the step slider to verify predictions become more image-specific over training (not less)
- Quantitative check: count unique predictions per step. Should increase or stay high, not collapse

### Verify inference placeholder fix
- The `_splice_image_tokens` log line should say `SPLICE (positional insertion)`, not `PREPEND (fallback)`
- Placeholder count should be 64 (= `num_image_tokens`)

## Files to Modify

| File | Change |
|------|--------|
| `modal_train.py:587` | `lora_alpha=64` → `lora_alpha=16` |
| `modal_train.py` (pretrain method) | Verify Stage 1 works end-to-end on Modal, saves projector checkpoint |
| `modal_train.py` (finetune method) | Add `--pretrain-checkpoint` CLI arg, wire to `FinetuneConfig.pretrain_checkpoint` |
| `modal_inference.py:_run_inference()` | Fix placeholder token insertion in prompt |
| `training/finetune.py:115-119` | Verify `_load_pretrain_checkpoint` loads projector correctly |

## Key Files for Understanding

| File | What's there |
|------|-------------|
| `2309.16058v1.pdf` | Original paper. Section 3.1 = pretraining, 3.2 = finetuning, Appendix B.3 = hyperparameters |
| `models/anymal.py:516-558` | `set_training_stage()` — what's frozen/unfrozen per stage |
| `models/anymal.py:240-340` | `_splice_image_tokens()` — how image tokens enter the sequence |
| `models/anymal.py:610-652` | `from_pretrained()` — checkpoint loading |
| `training/finetune.py:90-120` | `FinetuneTrainer.__init__` — stage 2 setup, pretrain checkpoint loading |
| `training/pretrain.py` | Stage 1 trainer |
| `data/instruction_dataset.py:318-418` | `_encode_with_response_masking()` — sentinel → placeholder logic |
| `data/dataset_splitter.py` | Deterministic train/val split (seed=42, 5% val) |
| `predictions.json` | Inference results showing the collapse (20 examples × 4 checkpoints) |
| `prediction_viewer_output.html` | Visual evidence of the problem |
