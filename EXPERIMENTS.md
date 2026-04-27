# AnyMAL Experiments & Ablations

Running log of what we've tried, what we found, and what's open. Newest entries at the bottom of each section.

---

## Reference: Stage 1 Pretrain

See `CLAUDE.md` for the canonical Stage 1 run. Final artifact:

- **Checkpoint**: `/checkpoints/pretrain-output/checkpoint-2500/projector.pt`
- Trained 2500 steps on 4× A100-80GB, loss 12 → 1.5, wall time ~1.5 hr, cost ~$22
- Perceiver Resampler only (LLM fully frozen, no LoRA)
- Dataset: 157K COCO caption samples from 81K images
- Used as the starting point for every Stage 2 run below

---

## Stage 2 Ablation Batch 1 — hyperparameter sweep (2026-04-15)

**Goal:** sweep LR, LoRA LR, dataset, and step count to see which hyperparameters matter most. All runs: 1× A100-80GB, 500 steps (unless noted), batch_size 2, `lora_r=64`, `lora_alpha=16`, same Stage-1 projector, same 95/5 deterministic val split.

| Run | Base LR | LoRA LR | Dataset | Steps | train_loss | VQA (%) |
|-----|---------|---------|---------|-------|-----------|---------|
| A (baseline) | 2e-5 | 2e-4 | instruct_150k | 500 | 1.2898 | 0.40 |
| B | 5e-5 | 2e-4 | instruct_150k | 500 | 1.2905 | 0.27 |
| C | 1e-5 | 2e-4 | instruct_150k | 500 | 1.2964 | 0.27 |
| D | 2e-5 | 1e-4 | instruct_150k | 500 | 1.2964 | 0.87 |
| E | 2e-5 | 5e-4 | instruct_150k | 500 | 1.2905 | 0.20 |
| **F** | 2e-5 | 2e-4 | **mix_665k** | 500 | **1.1898** | **4.80** |
| G | 2e-5 | 2e-4 | instruct_150k | **1000** | 1.2706 | 0.40 |

**Findings**
- **Dataset >> hyperparameters.** F (mix_665k) is the only run that moved the needle on either metric. The six LR/step variants all clustered around train_loss ≈ 1.29.
- **Learning-rate tuning is a wash** at this scale. Base LR 1e-5 → 5e-5 and LoRA LR 1e-4 → 5e-4 all produce near-identical train loss.
- **Longer training helps marginally.** G (1000 steps) improved train_loss by 0.02 over A. Not enough to justify 2× the compute.
- **VQA numbers are noisy.** All below 5%, likely dominated by the known VQA image-coverage issue (many val2014 images aren't cached) — treat as a rough qualitative signal, not a benchmark.

**Cost:** 6 × ~$2.50 + 1 × $5 ≈ $20.

---

## Bug discovered: race condition on Modal Volume dir creation (2026-04-18)

**Symptom:** after batch 1 finished, only 4 checkpoint directories existed (`run-0010` … `run-0013`) for 7 parallel runs. Three runs' checkpoints were overwritten.

**Root cause:** `_create_versioned_run_dir()` in `modal_train.py` used `os.listdir` → `os.mkdir` with a `FileExistsError` retry loop. On a POSIX filesystem this is atomic and race-free. On a **Modal Volume**, cross-container writes are eventually consistent — each container's `os.mkdir("run-0010")` appeared to succeed locally because it couldn't see the sibling containers' writes yet. When the volume synced, last-writer-wins clobbered earlier checkpoints.

**Fix** (commit pending in `modal_train.py`):
- New `_generate_unique_run_name()` → `{prefix}-YYYYMMDD-HHMMSS-XXXX` (datetime + 4 hex chars)
- `main()` generates the name locally **before** any `.remote()` call, then passes it down to every container
- Each container writes to its pre-assigned path; no cross-container coordination needed
- `--run-name` CLI flag added for explicit naming (used below for the clean re-run)

---

## Stage 2 Ablation Batch 2 — clean A vs F re-run (2026-04-18)

Parallel re-run with explicit `--run-name ablation-A` / `--run-name ablation-F`. Verified the fix: both dirs exist cleanly on the volume, no overwrites.

| Run | Base LR | LoRA LR | Dataset | Steps | train_loss | Wall |
|-----|---------|---------|---------|-------|-----------|------|
| ablation-A | 2e-5 | 2e-4 | instruct_150k | 500 | — (similar to batch 1 A) | 1.57 hr |
| ablation-F | 2e-5 | 2e-4 | mix_665k | 500 | — (similar to batch 1 F) | 1.55 hr |

Checkpoints: `/checkpoints/finetune-output/ablation-A/checkpoint-500`, `.../ablation-F/checkpoint-500`.

---

## Three-way inference comparison: Stage-1 baseline vs A vs F (2026-04-18)

**Goal:** the ablation batches only compared Stage-2 configurations against each other. But we never checked the obvious baseline: **does Stage 2 even help, compared to just using the Stage-1 pretrained perceiver + base LLaMA-3-8B-Instruct with no LoRA?**

Ran `three_way_inference.py` on 20 stride-sampled val examples, three models generating answers to the same LLaVA-style Q&A prompts at `max_new_tokens=384`:

- **Baseline**: Stage-1 projector + LLaMA-3-8B-Instruct, `use_qlora=False` (no LoRA at all)
- **A**: Baseline + Stage-2 instruct_150k finetune
- **F**: Baseline + Stage-2 mix_665k finetune

Viewer: https://babakd--anymal-viewer-web.modal.run (predictions persisted at `/checkpoints/three_way_predictions.json` in the volume).

### Summary stats (20 examples)

| Model | Avg chars | Ends cleanly (.,!,?) |
|-------|-----------|----------------------|
| Baseline (Stage-1 only) | **1284** | **13/20 (65%)** |
| Stage-2 A (instruct_150k) | 1749 | 7/20 (35%) |
| Stage-2 F (mix_665k) | 1617 | 9/20 (45%) |

### Qualitative findings

1. **Stage 2 makes responses verbose and truncation-prone.** At the same max_new_tokens cap, baseline terminates cleanly 65% of the time; Stage-2 models terminate cleanly only 35-45%. The LLaVA training distribution has long GPT-4-generated answers, and the model is learning that length as style.

2. **Much of "F's visual grounding wins" over A was already in the baseline.** Examples where F beat A on specific-object identification (police officers, named kite colors, carrots in the meal, "holding" vs "standing with" skateboard) — the Stage-1 baseline already got those right. The mix_665k dataset isn't teaching perception; it's mostly adjusting phrasing.

3. **Stage 2 can increase confident hallucinations.** Example 4 (a golden retriever): baseline says "brown dog" (vague, safe); A confidently says "specifically a collie" (wrong). Finetuning gave the model more confidence to name breeds it can't actually distinguish.

4. **Baseline occasionally fails on open-ended prompts.** Example 7 ("Analyze the image in a comprehensive and detailed manner") — baseline echoed the instruction back instead of answering. Stage-2 models don't do this.

5. **Stage 2 helps with format consistency.** Stage-2 models more reliably produce numbered-list answers for "why" / "what factors" questions.

### Implication

At 500 steps, **Stage 2 is a stylistic adjustment, not a capability gain.** The visual grounding is already there from Stage 1. To see Stage 2 actually add capability we'd need either:
- Many more steps (maybe 5-10×) to let the model internalize the instruction distribution
- A training signal that penalizes verbosity / hallucination (we have neither)
- A different dataset that doesn't bias toward long verbose answers

---

## Open questions / TODO

- [ ] Measure Stage 2 at 2000-3000 steps with mix_665k — does the verbosity issue resolve, or does the model just learn longer answers?
- [ ] Train Stage 2 on a shorter-response dataset (e.g., VQAv2 rationales, ShareGPT4V short captions) and compare.
- [ ] Fix the VQA eval image-coverage issue so the VQA accuracy metric becomes meaningful as a comparison signal.
- [ ] Add length/concision metrics to training logs (ratio of generated tokens to GT tokens on val).
- [ ] Consider whether a length penalty during generation (repetition penalty, length penalty in beam search) would fix the verbosity cheaply at inference time.
- [ ] Run a full-scale Stage 2 (e.g., 3000 steps on mix_665k) with clean naming and track it carefully.
- [ ] Investigate the eval loss logging bug from Stage 1 (always shows 0.0000 in logs even though eval loop runs).

---

## V2 hardening pass — SigLIP2 preprocessing, strict eval, and Modal smokes (2026-04-27)

**Goal:** make V2 the active training path and remove silent V1 assumptions before starting longer V2 experiments.

### Code changes

- Added SigLIP/SigLIP2 image preprocessing via `get_vision_transform()` and routed V2 local scripts, Modal datasets, and VQA eval through it.
- Updated V2 configs and Modal V2 runs to use 384px images.
- Made VQA eval insert V2 image placeholders, so strict splice is exercised consistently at eval time.
- Treated `token_compressor` as part of the multimodal adapter in optimizer groups and gradient diagnostics.
- Disabled Stage 2 V2 adapter warmup by default (`projector_warmup_steps=0`) and removed the old optimizer/scheduler rebuild behavior from warmup.
- Made pretrain checkpoint auto-discovery architecture-aware so V2 refuses legacy V1 checkpoints.
- Added supervised-token, active-token, placeholder-token, generated-token, and EOS-rate metrics.
- Fixed Modal runtime dependencies and COCO cache detection.

### Validation

- Local tests: `python3 -m pytest tests/test_model.py tests/test_training.py tests/test_health_monitor.py -q` -> 110 passed, 1 skipped.
- Static checks: `python3 -m compileall ...` and `git diff --check` passed.
- Modal V2 Stage 2 smoke: `v2-smoke-20260427-codex-2`, run `ap-SLiGhwZe4E5UQlcXm4cE5i`, 2 optimizer steps completed.
- Modal V2 Stage 1 smoke: `v2-pretrain-smoke-20260427-codex`, run `ap-WK9FApkcZviH9IgaWYfGHZ`, 2 optimizer steps completed on 4x A100.

### Notes for future agents

- The old `/checkpoints/pretrain-output/checkpoint-2500` checkpoint is V1/legacy. V2 must train or load a V2 checkpoint with metadata.
- V2 Stage 2 can run without a V2 pretrain checkpoint for smoke testing, but real experiments should start from a V2 Stage 1 checkpoint.
- VQA metrics are now V2-path compatible, but image coverage is still partial, so use them as smoke/relative signals until the VQA cache is expanded.
- The forward-looking V2 quality roadmap lives in `V2_QUALITY_PLAN.md`; use it as the execution plan for V2.1 experiments.

---

## V2 quality-plan implementation batch 1 — eval, data, compressor, and Modal smokes (2026-04-27)

**Goal:** execute the first parallelizable slice of `V2_QUALITY_PLAN.md`: make eval more trustworthy, add real-pretrain/mixture data plumbing, and add a stronger config-gated connector candidate.

### Code changes

- Added `V2_QUALITY_PLAN.md` and linked it from `CLAUDE.md` and this experiment log.
- Hardened captioning eval for AnyMALv2:
  - inserts contiguous V2 image placeholder blocks,
  - uses SigLIP2/V2 image preprocessing,
  - filters/skips missing or corrupt images,
  - reports `num_samples`, `avg_generated_tokens`, and `eos_rate`.
- Hardened VQA eval to skip all-invalid batches from collate.
- Added no-download eval tests in `tests/test_evaluation.py`.
- Added `LlavaPretrainCaptionDataset` with JSON caption loading, real-image filtering, caption minimum length, and caption de-duplication.
- Added `InstructionMixtureDataset` and `create_instruction_dataset(..., mixture_config=...)` for balanced/concat Stage 2 mixtures.
- Updated V2 configs with LLaVA-Pretrain style Stage 1 data knobs and inactive Stage 2 mixture knobs.
- Added `token_compressor_type="perceiver"` / `"perceiver2"` as a 2-layer residual cross-attention + FFN compressor while preserving existing `learned` behavior and state-dict surface.
- Wired local scripts and Modal:
  - local Stage 1 consumes `dataset_type`, `image_dir`, and filtering knobs,
  - local Stage 2 consumes `mixture` and `filter_to_available_images`,
  - Modal Stage 1 prefers true LLaVA-Pretrain captions when images are staged and otherwise falls back to the existing COCO-backed caption extraction,
  - Modal Stage 2 supports `--dataset balanced_mix`.

### Validation

- `python3 -m py_compile ...` on touched integration files passed.
- Focused tests: `pytest tests/test_evaluation.py tests/test_model.py::TestAnyMALv2CoreModules tests/test_training.py -q` -> 29 passed.
- Full tests: `pytest tests -q` -> 118 passed, 1 skipped.
- Static check: `git diff --check` passed.

### Modal smoke runs

- V2 Stage 1 pretrain smoke:
  - Run name: `v2-pretrain-integration-smoke-20260427`
  - App run: `ap-krX0b1NJwdAD0WGO4VdJAD`
  - Command shape: `--stage pretrain --architecture anymal_v2 --max-steps 2 --batch-size 1`
  - Result: completed 2 optimizer steps.
  - Confirmed 256 V2 image placeholders, 384px SigLIP2 preprocessing, and Stage 1 trainables limited to `token_compressor + projector`.
  - Data path: true LLaVA-Pretrain images were not staged, so the loader fell back to existing COCO-backed instruction-caption extraction (`157,712` samples from `81,479` images).
- V2 Stage 2 balanced-mixture smoke:
  - Run name: `v2-finetune-balanced-mix-smoke-20260427`
  - App run: `ap-jEdIuwx1tvvHe4TiSdlE1v`
  - Command shape: `--stage finetune --architecture anymal_v2 --dataset balanced_mix --max-steps 2 --batch-size 1 --no-run-eval-benchmarks --no-track-per-layer-grad-norms`
  - Result: completed 2 optimizer steps.
  - Mix-665K filtered to `338,470 / 665,298` samples with cached COCO images.
  - Instruct-150K path kept `157,712 / 157,712` samples.
  - Balanced mixture length: `676,940`.
  - Confirmed 384 V2 image placeholders and Stage 2 trainables as adapter + LoRA with `other=0`.

### Caveats and next work

- True LLaVA-Pretrain images still need to be staged under `/checkpoints/llava_pretrain/images`; only the annotation JSON is currently cached.
- The 2-step Stage 1 smoke did not save a checkpoint because `save_steps=250`, so the Stage 2 smoke intentionally ran without a V2 pretrain checkpoint. It validates mechanics, not quality.
- `balanced_mix` currently balances `instruct_150k` and cached-COCO-filtered `mix_665k`; broader OCR/GQA sources still need image caches before they can contribute.
- In-training `EvalRunner` still runs VQA only. Captioning eval is now V2-compatible but not yet wired into the training runner.

---

## Infra notes

- **Monitoring**: `Monitor` tool (tails Modal logs via `perl alarm` snapshots) worked better than `CronCreate` for training runs — crons only fire when REPL is idle, which fails when actively chatting.
- **Log streaming**: `modal app logs <id>` streams indefinitely. On macOS (no `timeout`), use `perl -e 'alarm(N); exec @ARGV' -- modal app logs <id>` for snapshots.
- **Viewer**: FastAPI ASGI app on Modal (`modal_viewer.py`), `min_containers=1` for snappy loads. Reads predictions JSON from the shared volume on each request — can update data without redeploying.
