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

## V2 debug, anti-overfit policy, and full Stage 2 run (2026-04-29)

**Goal:** turn the V2 debug pass into a real generalization experiment instead
of stopping at short targeted repairs.

### What was ruled out

- Final V2 checkpoints contain the expected metadata, projector, learned token
  compressor, and LoRA adapter.
- V2 inference now validates token-budget metadata and exact contiguous image
  placeholder count.
- LoRA loading was hardened to avoid nested/masked PEFT adapters.
- True LLaVA-Pretrain data was present for Stage 1: 558,128 annotations and
  558,128 zip-backed images.
- Image-use probes show the connector is alive: correct, wrong, blank, and
  text-only conditions produce different answers.

### Held-out scoreboard

The fixed scoreboard for the current full run is VQAv2 validation, deterministic
1000-sample subset, seed 42, with answer-type breakdowns. Known failure examples
are canaries only and must not be used to choose checkpoints.

A larger confirmatory VQA cache is supported in `vqa_checkpoint_eval.py`, but it
should be run only after the checkpoint shortlist is fixed. The primary
checkpoint trend must stay on the same seed-42 scoreboard below.

| Model | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate |
|---|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 5.03% | 2.55% | 9.10% | 0.00% | 32.0 | 0.0 |
| Original V2 balanced-mix checkpoint 3000 | 6.70% | 6.48% | 4.09% | 10.69% | 32.0 | 0.0 |

Scheduled full-run checkpoint reads:

| Model | Overall | Number | Other | Yes/No | Avg generated tokens | EOS rate | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| V2 direct-object full checkpoint 300 | 7.63% | 5.79% | 12.67% | 0.87% | 32.0 | 0.0 | Superseded by data-path fix below |
| V2 normalized direct-object checkpoint 300 | 3.50% | 3.94% | 5.72% | 0.00% | 32.0 | 0.0 | Active full run; weak early read, do not stop on 300 alone |
| V2 normalized light direct-object checkpoint 300 | 7.27% | 9.03% | 10.85% | 1.17% | 32.0 | 0.0 | Active full run; first corrected read above V1/original V2 overall |

### 2026-04-29 data-path correction

The first full direct-object and direct-natural runs were stopped before they
could become selection candidates. The direct-answer filter selected examples by
the first short answer, but wrote the entire original multi-turn conversation
back to disk. That made the run a mixed multi-turn SFT branch, not the intended
direct-answer intervention. The label-mask diagnostic also decoded a raw token
window after the first supervised token, which made masked user headers look
like supervised content.

Fixes now in place:

- Stage 2 labels are built by segment-wise tokenization rather than character
  offsets over the image-placeholder expansion.
- Dataset diagnostics decode supervised labels separately from the surrounding
  raw token window.
- `mix665k_direct_answer_filtered.json` is normalized to one human turn and one
  GPT answer turn, preserving only the selected short answer.
- Direct-answer examples containing reserved LLaMA chat markers are skipped.

Remote sanity run:

- Run name: `v2-stage2-sanity-normalized-direct-1step-20260429`
- Result: completed 1 optimizer step from the V2 Stage 1 checkpoint.
- Direct-answer supervised previews are now answer-only, e.g. `Black and
  white<|eot_id|>`, `D<|eot_id|>`, and `1<|eot_id|>`.

### Full runs in progress

- Primary run name: `v2-stage2-normalized-direct-object-full-3000-20260429`
- Primary dataset: `balanced_mix_direct_object_trainprompt`
- Parallel branch: `v2-stage2-normalized-light-direct-object-full-3000-20260429`
- Branch dataset: `concat_mix_direct_object_light_trainprompt`
- Generalization hedge: `v2-stage2-balanced-mix-light-3000-20260429`
- Hedge dataset: `balanced_mix`
- Shared base checkpoint:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Shared target steps: `3000`
- Shared scheduled held-out checkpoint evals: external VQA reads at `300`,
  `1000`, `2000`, `3000`; the in-training step eval is validation loss only.
- Primary Modal app: `ap-42JwN1un0vfyekswO9IqFE`
- Primary W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/4mi0p44q`
- Parallel Modal app: `ap-AKiUSa9ZbCGBj4QubRRm9n`
- Parallel branch W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/ozalncdz`
- Hedge Modal app: `ap-G4EThJMPjngFoCxLghhzmQ`
- Hedge W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/hf1i064o`

In-training validation loss so far:

| Run | Step | Avg eval loss | Status |
|---|---:|---:|---|
| normalized direct-object | 300 | 1.0423 | Running; external VQA checkpoint-300 = 3.50% |
| normalized light direct-object | 300 | -- | Running; external VQA checkpoint-300 = 7.27% |
| balanced-mix light | startup | -- | Running; no direct-answer oversampling hedge |

Both runs intentionally start from Stage 1 rather than the short 100-step repair
checkpoint. The direct-object run tests whether COCO object/count supervision
improves broad grounding. The light branch keeps the same ingredients but uses
concat sampling plus lower LoRA pressure so a direct-answer boost cannot win only
by oversampling short COCO-style labels. Both runs must earn selection on the
held-out scoreboard.

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
- The forward-looking V2 quality roadmap lives in `docs/history/V2_QUALITY_PLAN.md`; use it as the execution plan for V2.1 experiments.

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

## V3 recipe exploration — answer-type focus with frozen connector (2026-05-05)

**Goal:** distill recipe and architecture direction from the V1/V2/V3 evolution. Treat these results as guidance for the next experiment, not as checkpoint selection.

### Result snapshot

All numbers below are VQAv2 val2014 held-out samples (`1000`, seed `42`) using the `training_chat` prompt path. Do not overfit future decisions to this single seed; use it as a fast screen and confirm real candidates on more seeds or a larger eval.

| Branch | Overall | Number | Other | Yes/No | EOS rate | Max-token hits |
|---|---:|---:|---:|---:|---:|---:|
| V1 ablation-F checkpoint 500 | 7.57 | 4.86 | 3.90 | **14.19** | **0.967** | **0.033** |
| V2 final | 0.07 | 0.23 | 0.00 | 0.10 | 0.960 | 0.040 |
| V3 grounded checkpoint 1200 | 8.33 | 2.55 | 8.38 | 10.69 | 0.842 | 0.158 |
| V3 low-LR continuation checkpoint 250 | 8.20 | 2.78 | 8.25 | 10.40 | 0.963 | 0.037 |
| V3 answer-type LoRA-only checkpoint 100 | **9.03** | 5.09 | 8.77 | 11.08 | 0.851 | 0.149 |
| V3 answer-type LoRA-only checkpoint 300 | **9.03** | 5.32 | **8.84** | 10.88 | 0.836 | 0.164 |
| V3 answer-type connector+LoRA checkpoint 50 | 8.90 | **5.56** | 8.38 | 11.08 | 0.788 | 0.212 |
| V3 answer-type connector+LoRA checkpoint 100 | 8.77 | 5.09 | 8.84 | 10.20 | 0.734 | 0.266 |
| V3 direct-calibration LoRA-only checkpoint 50 | 8.60 | **6.48** | 8.38 | 9.82 | **0.993** | **0.007** |
| V3 direct-calibration LoRA-only checkpoint 100 | **9.10** | 6.25 | 8.38 | 11.37 | **0.997** | **0.003** |
| V3 direct-calibration LoRA-only checkpoint 150 | 8.70 | 5.79 | 8.25 | 10.59 | **0.994** | **0.006** |

### What happened

- V3's architecture is viable. Without changing the architecture, an answer-type-balanced recipe moved V3 from `8.33` to `9.03` overall, beating the V1 fast-screen overall while preserving V3's much stronger `other` accuracy.
- V1 still has the best yes/no calibration. The direct-calibration run recovered
  generation discipline beyond V1's EOS/max-token screen, but still trails V1 on
  yes/no, so the next recipe should improve yes/no without sacrificing V3's
  `other` gains.
- LoRA-only was the clean branch. It had stable enough grad norms, improved held-out accuracy early, and plateaued rather than collapsed from checkpoint 100 to 300.
- Connector+LoRA is not the next lever. It produced a tempting early number score, but grad norms were spikier and generation hygiene degraded badly by checkpoint 100. Drop connector finetuning for now unless there is a new grounding dataset or a very controlled low-LR connector-only ablation.
- Raw per-step training loss is expected to be jagged here because direct-answer supervision has sparse answer tokens and mixes heterogeneous answer types. Judge runs by validation loss trend, grad norms, EOS/max-token behavior, and held-out VQA, not by cosmetic smoothness.

### Recommendation for future agents

- Fix the V3 architecture for the next round and explore the recipe. The architecture has enough capacity to beat the V1 overall screen; recipe quality is currently the bottleneck.
- Pursue short LoRA-only direct-answer runs. Most useful movement happened by 50-100 steps; the 100-to-300 continuation mostly traded number/other gains for slightly worse yes/no and generation hygiene.
- Increase yes/no calibration pressure relative to the last `v3_yn_number_focus` mix, retain number/count supervision, and add explicit concise-answer/EOS pressure. The target is not "match V1"; it is to keep V3's broader `other` strength while recovering V1-style answer discipline.
- Deprioritize generic instruction/LLaVA-style mixture as a source of more accuracy for this question. Use it only as a small regularizer unless a held-out eval says otherwise.
- Predeclare evaluation seeds and selection rules before running longer sweeps. Fast-screen on `1000` examples is useful for triage, but final recipe claims need multiple seeds or a larger held-out slice.

**One-line lesson:** keep V3, freeze the connector, and spend the next iteration on a cleaner short LoRA-only answer recipe that restores yes/no and EOS behavior while preserving V3's stronger `other` accuracy.

---

## V3 stable recipe codification (2026-05-08)

**Goal:** turn the May 5 V3 result into a reproducible recipe instead of an
experiment note.

### Stable default

- Architecture stays fixed: SigLIP2 384px, 128-token 6-layer Perceiver connector.
- Stage 2 is LoRA-only by default for the direct-calibration recipe; the connector
  is frozen after Stage 1B.
- `configs/finetune_v3.yaml` now represents the short direct-calibration recipe:
  100 steps, checkpoint every 50 steps, `train_adapter: false`.
- Modal dataset alias `v3_direct_calibration` freezes the connector by default
  and shifts the answer-type mix toward yes/no recovery:
  yes/no `0.45`, number `0.25`, COCO object/count/color `0.20`, VQA other `0.05`,
  short LLaVA direct `0.05`.
- `scripts/check_v3_promotion.py` checks saved VQA eval artifacts against the V1
  held-out baseline before a candidate is promoted.

### Current guard result

Using the saved 1000-sample seed-42 `training_chat` artifacts, the best V3
direct-calibration checkpoint clears V1 overall (`9.10` vs `7.57`) and improves
number, other, EOS, and max-token behavior, while still trailing V1 on yes/no.
The live 300-step candidate was stopped after checkpoint 150 because validation
regressed from `0.6071` at checkpoint 100 to `0.6229`, and checkpoint-150 VQA
fell to `8.70`. Treat checkpoint 100 as the stable fast-screen win, not a final
generalization claim across seeds.

### V4 handoff

The architecture/recipe lessons from V1, V2, and V3 have been consolidated in
`docs/V4_ARCHITECTURE_PLAN.md`. Future V4 work should use that file as the planning
entry point. The intended direction is a new `anymal_v4` label that keeps V3's
compact 128-token discipline while testing modern VLM-style spatial selection:
global/local visual latents, 2D position features, optional high-resolution or
tiled inputs, and strict promotion gates against the V3 checkpoint-100
incumbent. Do not restart from broad instruction tuning or raw token-count
expansion; those were the V2 and early Stage 2 failure modes.

---

## V4 architecture and recipe execution log (2026-05-08)

**Goal:** modernize the stack beyond V3 while preserving the corrected VQA
promotion discipline. The design and rationale live in
`docs/history/V4_RESEARCH_RECIPE_20260508.md`.

### Architecture selected

- `anymal_v4`: SigLIP2 So400m at 384px, strict V2/V3 placeholder splice, and a
  `SpatialPerceiverResampler` connector.
- Visual-token budget stays at 128: 64 global summary latents plus 64 local
  spatial latents.
- The first two-tower connector smoke OOM'd on A100-80GB at `3.24B` trainable
  connector params, so V4 was revised to use shared Perceiver layers over typed
  global/local queries. The shared-tower connector has `1.63B` trainable params.

### Modal validation runs

| Run | App | Stage | Result |
|---|---|---|---|
| `v4-stage1a-smoke-shared-20260508-codex` | `ap-BrHJUzGWawtihedGJbvJ9r` | Stage 1A smoke | Completed 2 distributed optimizer steps on real LLaVA-Pretrain data; saved `/checkpoints/pretrain-output/v4-stage1a-smoke-shared-20260508-codex/checkpoint-2`. |
| `v4-stage1a-shared-20000-20260508-codex` | `ap-hTW04HKrFzV2fbDo6SVTBP` | Full Stage 1A | Stopped on escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/6ffd2qwa`. Checkpoint cadence was every 250 optimizer steps; `checkpoint-4500` is the last verified complete checkpoint and contains `trainer_state.pt` `12.2 GiB`, `projector.pt` `6.1 GiB`, and `model_meta.json` `485 B`, all modified `13:53 EDT`. Eval/loss improved from `2.9393` at step `2000` to `2.8393` at step `4000`; no later eval landed before stop. W&B was checked explicitly with `scripts/inspect_wandb_run.py --recent-window 220`: step `4550` had clip fraction `0.1892`, step `4640` hit `0.2000` with `high_grad_clip_fraction`, and step `4680` rose to `0.204059829` with train loss `3.9157`, loss EMA `2.9647`, grad norm `0.7353`, grad EMA `2.4286`, LR `1.7748e-4`, active `recent_loss_spikes`, `recent_grad_spikes`, and `high_grad_clip_fraction`; the 220-row loss mean still did not trigger the `+25%` rule (`2.9101 -> 2.7615`). Parent stopped app `ap-hTW04HKrFzV2fbDo6SVTBP` after W&B step `4730` crossed the playbook escalation line with clip fraction `0.2093` and active spike/high-clip alerts. Post-stop app status was `stopped` with zero tasks. `checkpoint-4750` exists but is partial/unverified: two Modal volume listings about one minute apart showed all three filenames, but `trainer_state.pt` stayed at `1.0 GiB` (modified `14:03 EDT`) rather than the expected `12.2 GiB`; do not resume from it. Yellow events remain part of run history: step-220 grad spike, step-960 loss spike, step-2540 grad spike, step-2860 loss spike, step-3948 grad spike, step-3996 loss spike, and post-4000 W&B loss spike at step `4140` (`6.3670` vs EMA `2.8839`). |
| `v4-stage1a-recovery4500-lr8e5-3000-20260508-codex` | `ap-BR97ZXuooRK292Ja0Kgflv` | Stage 1A recovery | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/g607yqek` (`graceful-dawn-24`). This intentionally used `--pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4500`, not `--resume-checkpoint`, so it loaded connector weights but started a fresh optimizer/scheduler. LR was reduced to `8e-5` from the failed run's `~1.8e-4`, but the lower-LR recovery did not clear clipping: parent stopped after W&B step `30` showed `health/grad_clip_fraction` about `0.80` at LR `2.4e-5` during warmup. Post-stop W&B was checked explicitly with `scripts/inspect_wandb_run.py --recent-window 220`: run state `finished`, latest step `40`, active alert `high_grad_clip_fraction`, train loss `2.8424`, loss EMA `2.8923`, grad norm `2.0451`, grad EMA `2.3047`, clip fraction `0.825`, LR `3.2e-5`, no evals, no recent loss/grad spikes, and four-row loss window `3.4226 -> 2.2750`. Modal listed app `stopped` with zero tasks; logs showed real LLaVA-Pretrain data, Stage 1 connector load from `checkpoint-4500`, a step-8 grad spike warning, step `30`/`40` progress, and `Stopping app - user stopped from CLI`. No `checkpoint-50` or `checkpoint-250` directory was found, so no recovery checkpoint/eval gate landed. |
| `v4-stage1a-recovery4000-lr5e5-canary300-20260508-codex` | `ap-E3Ke98Cu4xVqc7Aal55iSm` | Stage 1A diagnostic canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/r4hx5xrs` (`sage-cherry-25`). This used `--pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4000`, not `--resume-checkpoint`, with fresh optimizer/scheduler, LR `5e-5`, max steps `300`, warmup `30`, eval every `50`. Purpose was to test whether the clipping pathology was already baked into `checkpoint-4000` or emerged later. Logs verified Stage 1 connector load from `checkpoint-4000`, real LLaVA-Pretrain captions/images, trainable V4 connector only (`1.633B`), and W&B run `r4hx5xrs`. Parent stopped it after early W&B check at step `40`: active `high_grad_clip_fraction`, clip fraction `0.35`, train loss `2.6687`, loss EMA `2.9513`, grad norm `1.0645`, grad EMA `1.0090`, LR `4.98e-5`, no evals yet, and four-row loss window `3.3475 -> 2.2407`. Post-stop W&B was checked explicitly with `scripts/inspect_wandb_run.py --recent-window 220`: run state `finished`, latest W&B step `50`, active `high_grad_clip_fraction`, clip fraction `0.30`, train loss `3.3043`, loss EMA `2.8749`, grad norm `1.0543`, grad EMA `0.9180`, LR `4.939e-5`, no evals, no recent loss/grad spikes, and five-row loss window `3.3475 -> 2.5952`. Modal app listed `stopped` with zero tasks; logs showed `Stopping app - user stopped from CLI` as the first eval was just starting. The Modal volume run directory exists but is empty, and `checkpoint-50`/`checkpoint-75` are absent, so no checkpoint/eval gate landed. Dataset diagnostics also show supervised caption previews include `<|begin_of_text|>`, e.g. `'<|begin_of_text|>madonna tour...'`; investigate this label formatting because it may contribute to unstable connector gradients. |
| `v4-stage1a-fixedlabels-ckpt4000-lr5e5-canary100-20260508-codex` | `ap-Du9jNqbTRtxsNAzf3X8ZVB` | Stage 1A patched-label canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/froy06eg` (`visionary-gorge-26`). This used `--pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4000`, not `--resume-checkpoint`, with fresh optimizer/scheduler, LR `5e-5`, max steps `100`, warmup `10`, eval every `50`, H100. Logs verified Stage 1 connector load from `checkpoint-4000`, real LLaVA-Pretrain captions/images, trainable V4 connector only (`1.633B`), and fixed-label diagnostics: supervised previews no longer included `<|begin_of_text|>` and began `madonna tour...`, `the bmw m3...`, and `the wordpress...`. W&B was checked explicitly early: at step `10`, run state `running`, active `high_grad_clip_fraction`, clip fraction `0.60`, train loss `3.7993`, loss EMA `4.2794`, grad norm `0.8121`, grad EMA `1.2927`, LR `5.0e-5`, no evals, and no recent loss/grad spikes. Parent stopped the app because clipping was already far above the `0.20` escalation line. Post-stop W&B was checked explicitly with `scripts/inspect_wandb_run.py --recent-window 100`: run state `finished`, latest W&B step `40`, active `high_grad_clip_fraction`, clip fraction `0.60`, train loss `2.9152`, loss EMA `3.2533`, grad norm `0.7119`, grad EMA `1.3856`, LR `3.875e-5`, no eval rows, no recent loss/grad spikes, and four-row loss window `3.6531 -> 2.5407`. Modal listed the app `stopped` with zero tasks; logs showed step `40` then `Stopping app - user stopped from CLI`. The Modal volume run directory exists but is empty, so no `checkpoint-50` and no eval/checkpoint gate landed. Judgment: fixed-label masking is insufficient by itself to solve the Stage 1A clipping pathology. |
| `v4-stage1a-bottleneck1024-canary100-20260508-codex` | `ap-76TjvdhPvFm3LdHXdHJ6SC` | Stage 1A bottleneck connector canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/dsv4mlbs` (`cool-gorge-27`). This was a from-scratch A100 canary using the new bottlenecked V4 connector defaults: 64 global + 64 local tokens, connector layers `3`, heads `8`, FF mult `2`, hidden dim `1024`, 2D position features enabled, LR `1e-4`, max steps `100`, warmup `10`, eval every `50`, batch size `1`. Logs verified the intended shape and real data: V4 connector-only trainables dropped to `44,374,016`, real LLaVA-Pretrain captions/images loaded (`555,258` deduped samples), and fixed-label supervised previews started with real caption text (`madonna tour...`, `the bmw m3...`, `the wordpress...`) with no `<|begin_of_text|>`. W&B was checked explicitly at step `10`: active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `5.0610`, loss EMA `9.5638`, grad norm `8.3869`, grad EMA `90.8336`, LR `1e-4`, no evals, and no loss/grad spike alerts. Parent stopped the app immediately. Post-stop W&B was checked explicitly: run state `finished`, latest W&B step `20`, active `high_grad_clip_fraction`, clip fraction still `1.0`, train loss `4.2093`, loss EMA `7.8934`, grad norm `3.2777`, grad EMA `58.6991`, LR `9.7286e-5`, no eval rows, no recent loss/grad spikes, and two-row loss window `5.0610 -> 4.2093`. Modal listed the app `stopped` with zero tasks; the Modal volume run directory listing was empty (`[]`), so no `checkpoint-50` or eval/checkpoint gate landed. Judgment: shrinking the connector from `1.633B` to `44.4M` is not sufficient by itself; the next Stage 1A fix should target gradient scale/initialization/loss recipe before another longer run. |
| `v4-stage1a-bottleneck1024-scale01-canary100-20260508-codex` | `ap-TOyngwQQ017RjE8vDGZbjo` | Stage 1A scaled-output bottleneck canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/wm6y5cvh` (`dry-river-28`). This reused the bottlenecked V4 connector defaults and added `connector_output_scale=0.1` to test whether smaller random visual-token amplitude would control Stage 1A gradients. Logs verified the intended shape and real data: 64 global + 64 local tokens, connector layers `3`, heads `8`, FF mult `2`, hidden dim `1024`, output scale `0.1`, connector-only trainables `44,374,016`, real LLaVA-Pretrain captions/images, and fixed-label supervised previews without `<|begin_of_text|>`. W&B was checked explicitly before trusting the local stream: at step `10`, active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.7240`, loss EMA `10.1241`, grad norm `48.4955`, grad EMA `132.7394`, LR `1e-4`, no eval rows, and no loss/grad spike alerts. Parent stopped the app immediately. Post-stop W&B was checked explicitly: run state `finished`, latest W&B step `20`, active `high_grad_clip_fraction`, clip fraction still `1.0`, train loss `3.9441`, loss EMA `8.0904`, grad norm `2.5491`, grad EMA `81.1468`, LR `9.7286e-5`, no eval rows, no recent loss/grad spikes, and two-row loss window `6.7240 -> 3.9441`. Modal listed the app `stopped` with zero tasks; the Modal volume run directory listing was empty, so no `checkpoint-50` or eval/checkpoint gate landed. Judgment: falling loss is not enough when W&B clipping is pegged; output scaling at `0.1` did not solve Stage 1A stability, so the next canary must change initialization/gating or the alignment objective before any long run. |
| `v4-stage1a-bottleneck1024-gate001-canary100-20260508-codex` | `ap-h9cCm08DcrIDEqyIcJy1Sw` | Stage 1A gated-output canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/cdtr4gp6` (`clean-wildflower-29`). This used the bottleneck1024 connector with `connector_output_scale=1.0` and trainable `connector_output_gate_init=0.01`; logs verified real LLaVA-Pretrain data, fixed-label previews, and `44,374,017` trainables. W&B was checked before trusting logs: step `10` already had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `5.8066`, loss EMA `5.7100`, grad norm `3.8188`, grad EMA `31.0132`, LR `1e-4`, and no eval rows. Parent stopped the app immediately. Post-stop W&B reached step `30` with active `high_grad_clip_fraction` and `recent_loss_window_mean_up_gt_25pct`, clip fraction still `1.0`, train loss `9.0393`, loss EMA `8.4019`, grad norm `5.1590`, grad EMA `17.0959`, LR `8.9472e-5`, no eval rows. Modal listed the app `stopped` with zero tasks and the output directory was empty. Judgment: a `0.01` trainable visual gate does not clear Stage 1A clipping and produced a loss-window alert. |
| `v4-stage1a-bottleneck1024-gate0001-canary50-20260508-codex` | `ap-753TNbms8Pv2wNLeL0qrut` | Stage 1A gated-output canary | Stopped on early W&B clipping escalation. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/us0ls2wu` (`toasty-paper-30`). This repeated the bottleneck1024 gated handoff with `connector_output_gate_init=0.001`, real LLaVA-Pretrain data, fixed labels, and `44,374,017` trainables. W&B step `10` showed active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `5.4899`, loss EMA `6.0910`, grad norm `2.3484`, grad EMA `62.3095`, LR `9.7286e-5`, and no eval rows. Parent stopped the app. Post-stop W&B was `finished` at step `20` with active `high_grad_clip_fraction`, clip fraction still `1.0`, train loss `4.7958`, loss EMA `5.9601`, grad norm `21.4131`, grad EMA `39.3029`, LR `7.75e-5`, no eval rows, and no loss/grad spike alerts. Modal listed the app `stopped` with zero tasks and the output directory was empty. Judgment: lower gate amplitude improved early loss movement but did not matter for the stop criterion; falling loss is not a pass when clipping is pegged. |
| `v4-stage1a-bottleneck1024-gate00001-canary20-20260508-codex` | `ap-FNaPaB3I8yEpFaF7fmLefB` | Stage 1A tiny-gated-output canary | Failed W&B clipping gate despite completing and saving a checkpoint. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/2qdqg86p` (`clear-durian-31`). This used `connector_output_gate_init=0.0001`; logs verified real LLaVA-Pretrain data, fixed labels, and `44,374,017` trainables. W&B step `10` already had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.2635`, loss EMA `6.2312`, grad norm `3.4968`, grad EMA `130.6474`, LR `6.2814e-5`, and no eval rows. The run finished around the stop request and saved `/checkpoints/pretrain-output/v4-stage1a-bottleneck1024-gate00001-canary20-20260508-codex/checkpoint-20`; Modal app then listed `stopped` with zero tasks, and the checkpoint contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Final W&B was `finished` at step `20` with active `high_grad_clip_fraction`, clip fraction still `1.0`, train loss `4.8986`, loss EMA `6.0846`, grad norm `1.4890`, grad EMA `78.9965`, LR `1e-5`, no eval rows, no loss/grad spike alerts, and two-row loss window `6.2635 -> 4.8986`. Judgment: the checkpoint exists but is not a healthy candidate; a tiny trainable gate alone does not clear clipping. |
| `v4-stage1a-outputwarmup30-gate00001-lr1e5-canary60-20260508-codex` | `ap-6PUomRff3C9w5V8LMkNegC` | Stage 1A output-only warmup canary | Stopped on W&B clipping during the warmup, before full-connector unfreeze. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/iip0i3e3` (`flowing-water-32`). This used the 4x A100 distributed Stage 1 path, bottleneck1024 V4, real LLaVA-Pretrain caption alignment, fixed labels, `connector_output_gate_init=0.0001`, LR `1e-5`, and new `connector_warmup_steps=30`, which masked all connector gradients except `projector.output_proj.*` and `projector.output_gate_logit`; logs verified `3` active warmup tensors and `71` masked tensors. W&B was checked at step `20` and was already red: active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.0813`, loss EMA `6.5835`, grad norm `17.0686`, grad EMA `31.0523`, LR `8.5881e-6`, no eval rows, and two-row loss window `7.6394 -> 6.0813`. Parent stopped the app immediately; Modal listed it `stopped` with zero tasks and the output directory listing was empty. Judgment: output-projection/gate-only warmup does not rescue noisy caption Stage 1A; the next canary should change supervision/objective, not keep searching connector handoff variants. |
| `v4-stage1a-groundingfirst-gate00001-lr1e5-canary80-20260508-codex` | `ap-eE4Hll6DpCL0A6rqvc1fni` | Stage 1A grounding-first canary | Stopped on immediate W&B clipping. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/udsb1fr9` (`gentle-sun-33`). This used the 4x A100 distributed Stage 1 path, bottleneck1024 V4, dataset `v4_grounding`, `connector_output_gate_init=0.0001`, output scale `1.0`, and LR `1e-5`, without connector warmup. Logs verified the grounding mixture rather than web captions: VQAv2 train direct, COCO object/count/color direct, and short LLaVA direct-answer samples; supervised examples were short answers such as `C<|eot_id|>`, `phone<|eot_id|>`, and `no<|eot_id|>`. W&B was checked at step `10` and already had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `10.8670`, loss EMA `8.0383`, grad norm `7.5836`, grad EMA `7.2998`, LR `9.9829e-6`, no eval rows, and no loss/grad spike alerts. Parent stopped before eval/checkpoint. Final W&B state was `finished`; Modal listed the app `stopped` with zero tasks and the run directory had no checkpoint files. Judgment: switching to the existing grounding-first mixture alone does not solve immediate V4 Stage 1 clipping. |
| `v4-stage1a-groundingfirst-lossscale01-gate00001-lr1e5-canary80-20260508-codex` | `ap-5eFp7HMB0KnQkoscsSxy6y` | Stage 1A grounding-first loss-scale canary | Stopped on W&B loss-window alert under the stricter non-charitable monitoring rule. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/7lizvbnm` (`morning-paper-34`). This repeated the grounding-first Stage 1A canary with bottleneck1024 V4, dataset `v4_grounding`, `connector_output_gate_init=0.0001`, output scale `1.0`, LR `1e-5`, and new `pretrain_loss_scale=0.1`, which multiplies the backward loss while logging the raw HF loss. Logs verified the 4x A100 distributed Stage 1 path, `Stage 1 loss scale: 0.1`, and the real short-answer grounding mixture. W&B looked materially better on clipping at step `50`: no alerts, clip fraction `0.04`, grad norm `0.4288`, grad EMA `0.4147`, train loss `12.1285`, and eval loss `8.0005`. By step `60`, W&B raised `recent_loss_window_mean_up_gt_25pct` despite controlled clipping; the monitor stopped the app. Final W&B inspection after stop had last step `70`, active `recent_loss_window_mean_up_gt_25pct`, clip fraction `0.0714`, train loss `7.4674`, loss EMA `8.5904`, grad norm `0.6900`, grad EMA `0.6184`, LR `1.4216e-6`, one eval row (`8.0005`), no grad spikes, and no pointwise loss spikes. Modal listed the app `stopped` with zero tasks. `checkpoint-50` exists and contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`, but this is yellow/red rather than green because the W&B loss-window alert was real and the app was stopped before the requested 80 steps. Judgment: `loss_scale=0.1` fixes the immediate clipping pathology, but the loss trajectory is still not stable enough for a long run without additional recipe changes. |
| `v4-stage1a-groundingfirst-lossscale01-accum16-gate00001-lr1e5-canary100-20260508-codex` | `ap-c1HAaQCQ5vroHYRlZ023ac` | Stage 1A grounding-first accumulation canary | Stopped on W&B loss-window alert. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/a530s1tz` (`scarlet-yogurt-35`). This kept bottleneck1024 V4, dataset `v4_grounding`, `connector_output_gate_init=0.0001`, output scale `1.0`, LR `1e-5`, and `pretrain_loss_scale=0.1`, while increasing Stage 1 gradient accumulation to `16` for effective batch size `64`. Logs verified the 4x A100 distributed path, real short-answer grounding mixture, fixed dataset label (`Loaded 27446000 v4_grounding instruction samples`), and `Stage 1 loss scale: multiplying backward loss by 0.1`. W&B was checked directly by the monitoring agent and remained red: latest step `20`, active `recent_loss_window_mean_up_gt_25pct`, train loss `11.5021`, loss EMA `8.0981`, grad norm `0.6147`, grad norm EMA `0.6188`, clip fraction `0.1`, LR `9.7286e-6`, no eval rows, no inspector grad spikes, and no inspector pointwise loss spikes; the recent loss window rose from `5.2983` to `11.5021`. Logs also showed a HealthMonitor loss spike at step `8` (`16.6235 > 2.0x EMA 8.2828`), but the stop decision was W&B-driven. Modal listed the app `stopped` with zero tasks. The volume run directory exists but is empty, so no checkpoint landed. Judgment: red. Accumulation alone did not solve loss-window health; next Stage 1A should move to lower peak LR/gentler schedule or supervised-token-aware/objective-balancing changes rather than simply increasing effective batch. |
| `v4-stage1a-groundingfirst-lossscale01-lr3e6-canary120-20260508-codex` | `ap-P2rKyVW0yk5S7IfmSo1I7K` | Stage 1A grounding-first lower-LR canary | Stopped on W&B loss-window alert by the monitoring sub-agent and parent. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/o72woiib` (`sandy-capybara-36`). This kept bottleneck1024 V4, dataset `v4_grounding`, `connector_output_gate_init=0.0001`, output scale `1.0`, `pretrain_loss_scale=0.1`, and Stage 1 gradient accumulation `8`, while lowering peak LR to `3e-6`. Logs verified the 4x A100 distributed path, real short-answer grounding mixture, fixed dataset label, effective batch size `32`, eval every `50`, and V4 connector-only trainables (`44,374,017`). W&B step `20` was encouraging but not final: no alerts, clip fraction `0.0`, train loss `5.8749`, loss EMA `8.6810`, grad norm `0.4596`, grad EMA `0.5964`, LR `2.9636e-6`. W&B step `50` was still health-clean on clipping/window (`0.0` clip, no alerts, eval loss `8.6358`, train loss `12.4370`, loss EMA `9.4071`, grad norm `0.3117`, grad EMA `0.4865`, window `8.4460 -> 8.2972`) and saved `checkpoint-50`. By W&B step `80`, the run had active `recent_loss_window_mean_up_gt_25pct`; final snapshot after stop was train loss `11.6854`, loss EMA `9.3434`, grad norm `0.6954`, grad EMA `0.4057`, clip fraction `0.0`, LR `1.1153e-6`, eval loss still `8.6358`, no W&B loss/grad spikes, and recent loss window `7.3366 -> 11.5327`. Modal listed the app `stopped` with zero tasks. `checkpoint-50` is complete with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment: red/yellow. Lower LR plus loss scaling fully controls clipping, but does not clear the loss-window pathology and gives a worse first eval than the LR `1e-5` loss-scale canary; next fix should change objective/data normalization/schedule semantics rather than only shrinking update magnitude. |
| `v4-stage1a-answerfocus-lossscale01-gate00001-lr1e5-canary120-20260508-codex` | `ap-3HDCBEK0StEV7mn2T1GkXx` | Stage 1A answer-focus canary | Stopped by the monitoring sub-agent on W&B clipping, not console loss. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/sjrrgvm6` (`fine-field-37`). This kept bottleneck1024 V4, dataset `v4_answer_type_focus`, `connector_output_gate_init=0.0001`, output scale `1.0`, LR `1e-5`, `pretrain_loss_scale=0.1`, and Stage 1 gradient accumulation `8` for effective batch size `32`. Logs verified 4-GPU distributed Stage 1 V4 connector-only training and `44,374,017` trainables. The monitor was explicitly instructed to be non-charitable on spikes/clipping and to use W&B before logs. W&B showed persistent `high_grad_clip_fraction`: step `10` clip fraction `0.30`, step `20` `0.25`, and latest synced step `30` `0.50`; final snapshot had train loss `1.2553`, loss EMA `8.8399`, grad norm `4.4440`, grad norm EMA `1.5574`, LR `9.397e-6`, no eval rows, no inspector loss/grad spikes, and active alert `high_grad_clip_fraction`. The app was stopped before the step-50 eval/checkpoint boundary; Modal listed it `stopped` with zero tasks, logs showed `Stopping app - user stopped from CLI`, and the checkpoint volume listing for this run returned empty. Judgment: red. Answer-type rebalancing alone regressed the clipping gate compared with the plain `v4_grounding` + loss-scale run; do not spend a long run here without objective/normalization changes. |
| `v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr1e5-canary120-20260508-codex` | `ap-mj06o5UUMhVj7IrOaW0amS` | Stage 1A supervised-token-normalized canary | Stopped/rejected on W&B loss-window alert under non-charitable monitoring. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/jwd997c5` (`dandy-bush-38`; display name was random because `wandb_run_name` was not wired through Modal distributed pretrain at launch). This kept bottleneck1024 V4, dataset `v4_grounding`, `connector_output_gate_init=0.0001`, output scale `1.0`, LR `1e-5`, `pretrain_loss_scale=0.1`, and Stage 1 gradient accumulation `8`, and changed the objective to `loss_normalization=supervised_token_target` with target `8` supervised tokens and clamp `[0.05, 4.0]`. Logs verified the 4x A100 path, real VQAv2/COCO/LLaVA short-answer grounding mixture, effective batch size `32`, eval every `50`, and V4 connector-only trainables (`44,374,017`). W&B step `50` passed the first gate: no alerts, clip fraction `0.0`, train/objective loss `2.9637`, raw loss `11.8549`, supervised tokens `2`, normalization multiplier `0.25`, backward loss `0.2964`, loss EMA `2.5360`, grad norm `0.1417`, grad EMA `0.1296`, LR `7.5196e-6`, eval loss `7.8830`, and complete `checkpoint-50` with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. At W&B step `60`, active `recent_loss_window_mean_up_gt_25pct` appeared and persisted: train/objective loss `2.6404`, raw loss `10.5617`, supervised tokens `2`, multiplier `0.25`, backward loss `0.2640`, loss EMA `2.5327`, grad norm `0.1086`, grad EMA `0.1496`, clip fraction `0.0`, LR `6.2814e-6`, eval loss still `7.8830`, and no pointwise loss/grad spikes. Modal listed the app `stopped` with zero tasks and W&B state `finished`. Judgment: red/yellow. This is the best stabilizer so far for clipping and first eval, but it still fails the W&B loss-window gate and should feed the next recipe rather than be resumed as-is. |
| `v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr5e6-canary120-20260508-codex` | `ap-6NhnUkn2iXHsNwhdLqvGYh` | Stage 1A supervised-token-normalized lower-LR canary | Stopped on W&B loss-window alert. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/1isusl46` (display name correctly matched the requested run name after the Modal `wandb_run_name` wiring patch). This repeated the `supervised_token_target` objective with target `8`, `pretrain_loss_scale=0.1`, dataset `v4_grounding`, gate init `0.0001`, accumulation `8`, but lowered LR from `1e-5` to `5e-6`. Logs verified the 4x A100 path, real v4 grounding mixture, W&B URL, explicit run name, and V4 connector-only trainables (`44,374,017`). W&B was clean through step `50`: no alerts, clip fraction `0.0`, train/objective loss `3.1060`, raw loss `12.4238`, supervised tokens `2`, multiplier `0.25`, backward loss `0.3106`, loss EMA `2.6217`, grad norm `0.4639`, grad EMA `0.2449`, LR `3.7598e-6`, and flat five-row loss window (`2.1089 -> 2.1093`). Eval at step `50` was `8.2909`, worse than the LR `1e-5` token-normalized canary. `checkpoint-50` exists with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. At W&B step `60`, active `recent_loss_window_mean_up_gt_25pct` appeared: train/objective loss `2.8325`, raw loss `11.3299`, supervised tokens `2`, multiplier `0.25`, backward loss `0.2832`, loss EMA `2.6547`, grad norm `0.3481`, grad EMA `0.3792`, clip fraction `0.0167`, LR `3.1407e-6`, eval loss `8.2909`, no inspector loss/grad spikes, and recent window `1.5663 -> 2.8931`. Parent stopped the app immediately; W&B state finished and checkpoint-50 was verified. The monitoring sub-agent Poincare independently finalized this as red, not green or soft-yellow, because a finished/stopped run does not rescue an active W&B loss-window alert. Judgment: red. Lowering LR on top of token normalization does not clear the loss-window pathology and worsens first eval, so the next lever should be objective scheduling/balancing, not another LR-only shrink. |
| `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120-20260508-codex` | `ap-se9J3ORGBtFruPUtZKIfZF` | Stage 1A accumulated-metrics canary | Completed naturally to max step `120` and passed strict W&B-first monitoring. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/mqebqqc0`. This repeated the best LR `1e-5` supervised-token-normalized recipe, but patched trainer logging and HealthMonitor inputs to aggregate `train/loss` and numeric `_last_batch_metrics` over the full gradient-accumulation window instead of reporting only the last microbatch. The monitor was explicitly told to be non-charitable on W&B spikes/alerts and to reject missing or incorrect `train/accumulation_micro_batches`. W&B showed `train/accumulation_micro_batches=8.0` throughout, no active alerts, no inspector loss/grad spikes, and `health/grad_clip_fraction=0.0` at every checked window. Step `50` eval was `7.8095`; step `100` eval improved to `5.1151`. Final W&B step `120`: train/objective loss `1.2308`, raw loss `4.1529`, backward loss `0.1231`, supervised tokens `3.125`, normalization multiplier `0.390625`, loss EMA `1.7561`, grad norm `0.2211`, grad EMA `0.2826`, LR `1e-6`, recent window `2.1678 -> 1.7965`, and state `finished`. Modal app ended `stopped` with zero tasks. `checkpoint-50` and `checkpoint-100` both contain `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment: green. Earlier tokennorm loss-window failures were very likely a W&B/HealthMonitor last-microbatch artifact; accumulated optimizer-step metrics are now a recipe requirement, not optional telemetry. |
| `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000-20260508-codex` | `ap-0CIzlBmd5f34AzScvBDoti` | Stage 1A extended-run attempt | Operational abort by monitoring sub-agent before the first intended checkpoint/eval boundary. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/7nh4z51u`. Recipe matched the green accumulated-metrics canary, but max steps was raised to `1000` with eval/checkpoint every `100`. The monitor stopped the app after an over-strict pre-first-row W&B read showed no train history yet; final W&B had synced step `10` and was actually health-clean: no alerts, `train/accumulation_micro_batches=8.0`, train/objective loss `2.3653`, raw loss `7.8291`, backward loss `0.2365`, supervised tokens `2.5`, multiplier `0.3125`, loss EMA `2.6372`, grad norm `0.1649`, grad EMA `0.2095`, clip fraction `0.0`, LR `1.0e-6`, no eval rows, no loss/grad spikes. Modal listed the app `stopped` with zero tasks; no checkpoint boundary was reached and the output directory was empty. Judgment: operational abort, not model failure. Playbook refined afterward: zero W&B train rows before the first logging boundary is pending; missing or wrong `train/accumulation_micro_batches` on an existing W&B train row is red. Relaunch the extended run with that clarified rule. |
| `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000b-20260508-codex` | `ap-jdJIdfkNWydB3SsjEs68b6` | Stage 1A extended-run attempt | Operational abort by monitoring sub-agent on an incorrect checkpoint expectation, not a model-health failure. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/qjsw0606`. The run was health-clean through the stop: final W&B step `150`, state `finished`, no active alerts, `train/accumulation_micro_batches=8.0`, train/objective loss `1.8965`, raw loss `6.3185`, backward loss `0.1897`, supervised tokens `2.375`, normalization multiplier `0.296875`, loss EMA `2.3580`, grad norm `0.1074`, grad EMA `0.2540`, clip fraction `0.04`, LR `9.9316e-6`, eval loss `7.8015`, no inspector loss/grad spikes, and recent window `2.5072 -> 2.3553`. The monitor stopped after W&B advanced past step `140` and no `checkpoint-100` existed, but code inspection showed this launch's implicit save cadence was `_checkpoint_save_interval(1000)=250`, while eval cadence was `100`; `checkpoint-100` was not actually due. The Modal app ended `stopped` with zero tasks and the run directory was empty. Judgment: operational abort due monitor/cadence mismatch, not model failure. Patch added `--pretrain-save-steps` so future long runs can make the checkpoint contract explicit. |
| `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex` | `ap-gDtuS4QbsdA6Trg5BJ7GAc` | Stage 1A extended-run attempt | Active run, clean through the step-330 W&B soft check and step-300 artifact gate. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/npuz3sp8`. This relaunches the green accumulated-metrics A1 recipe with explicit `--pretrain-save-steps 100`, so eval and checkpoint boundaries are both every `100` optimizer steps. Step `100`: W&B alerts `[]`, `train/accumulation_micro_batches=8.0`, clip fraction `0.0`, train/objective loss `1.6548`, raw loss `5.8442`, backward loss `0.1655`, supervised tokens `2.25`, multiplier `0.28125`, loss EMA `2.1994`, grad norm `0.2551`, grad EMA `0.3166`, eval loss `5.2072`, no loss/grad spikes, and complete `checkpoint-100`. Step `200` gate also passed: W&B latest inspected at step `210` had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1918`, raw loss `2.8130`, backward loss `0.1192`, supervised tokens `2.875`, multiplier `0.359375`, loss EMA `1.2096`, grad norm `0.1375`, grad EMA `0.1835`, LR `9.6723e-6`, no loss/grad spikes, eval improved to `3.2008`, and recent window improved `2.3684 -> 1.3782`. `checkpoint-200` contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Step `250`: delegated monitor reported W&B state `running`, alerts `[]`, all `25` existing train rows with `train/accumulation_micro_batches=8.0`, clip fraction `0.0`, train/objective loss `1.1543`, raw loss `2.5884`, backward loss `0.1154`, supervised tokens `4.625`, multiplier `0.578125`, loss EMA `1.1451`, grad norm `0.0730`, grad EMA `0.1448`, LR `9.3971e-6`, no loss/grad spikes, evals improving from `5.2072` to `3.2008`, and recent loss window improving `2.1851 -> 1.3496`. Step `260`: parent re-check still had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `0.9854`, raw loss `3.0599`, backward loss `0.0985`, supervised tokens `2.5`, multiplier `0.3125`, loss EMA `1.1354`, grad norm `0.2356`, grad EMA `0.1553`, LR `9.3162e-6`, no loss/grad spikes, and recent window improving `2.1746 -> 1.2679`. Step `270`: W&B remained clean with alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `0.8212`, raw loss `2.5004`, backward loss `0.0821`, supervised tokens `2.5`, multiplier `0.3125`, loss EMA `1.0335`, grad norm `0.3569`, grad EMA `0.1999`, LR `9.2307e-6`, no loss/grad spikes, and recent window improving `2.1746 -> 1.2360`. Step `280`: W&B still had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1318`, raw loss `2.1013`, backward loss `0.1132`, supervised tokens `4.875`, multiplier `0.609375`, loss EMA `1.0287`, grad norm `0.1368`, grad EMA `0.1865`, LR `9.1406e-6`, no loss/grad spikes, and recent window improving `2.0897 -> 1.2464`. Step `290`: W&B still had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1997`, raw loss `2.4610`, backward loss `0.1200`, supervised tokens `4.75`, multiplier `0.59375`, loss EMA `0.9697`, grad norm `0.0716`, grad EMA `0.1587`, LR `9.0460e-6`, no loss/grad spikes, and recent window improving `2.0897 -> 1.2433`. Step `300`: W&B state `running`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1873`, raw loss `2.2154`, backward loss `0.1187`, supervised tokens `3.5`, multiplier `0.4375`, loss EMA `0.9823`, grad norm `0.1439`, grad EMA `0.1544`, LR `8.9472e-6`, no loss/grad spikes, eval improved again to `2.6205` (`5.2072 -> 3.2008 -> 2.6205`), and recent window improving `2.0225 -> 1.2503`. Independent monitor Russell verified all `30` W&B train rows from steps `10` through `300` had `train/accumulation_micro_batches=8.0` with missing `[]` and wrong `[]`. `checkpoint-300` contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`; app remains `ephemeral` with one active task. Step `320`: W&B state `running`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `0.5585`, raw loss `1.9328`, backward loss `0.0559`, supervised tokens `2.875`, multiplier `0.359375`, loss EMA `0.9289`, grad norm `0.0723`, grad EMA `0.1503`, LR `8.7370e-6`, no loss/grad spikes, and recent window improving `1.9974 -> 1.2017`. Step `330`: W&B state `running`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `0.9805`, raw loss `1.9945`, backward loss `0.0981`, supervised tokens `4.625`, multiplier `0.578125`, loss EMA `0.8524`, grad norm `0.3951`, grad EMA `0.1747`, LR `8.6260e-6`, no loss/grad spikes, and recent window improving `1.9974 -> 1.1887`; grad norm was above EMA but below the playbook spike rule and not paired with clipping or W&B alerts. Console showed slower microbatch stretches after step `250`; watch throughput, but do not stop on console slowdown while W&B health is clean. Continue monitoring; do not call final green until the requested run completes or a planned stop condition fires. |
| `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex` | `ap-gDtuS4QbsdA6Trg5BJ7GAc` | Stage 1A extended-run final stop | Supersedes the active note above. Stopped on strict W&B spike policy at W&B step `340`; app is now `stopped` with zero tasks. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/npuz3sp8`. Stop trigger was active alert `recent_loss_spikes`: train/objective loss `1.9874` vs loss EMA `0.9362` (`2.12x`), raw loss `3.0522`, backward loss `0.1987`, supervised tokens `6.0`, multiplier `0.75`, accumulation `8.0`, grad norm `0.0903`, grad EMA `0.1469`, clip fraction `0.0`, LR `8.5111e-6`, no grad spikes, and recent window still improving `1.9940 -> 1.1915`. Console progressed to about step `345` during termination, but no later W&B row or checkpoint boundary landed. Verified checkpoints are `checkpoint-100`, `checkpoint-200`, and `checkpoint-300`; `checkpoint-300` contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`, and `checkpoint-400` is absent. Judgment: stopped after a clean step-300 candidate, not a clipping or eval-regression failure. Use `checkpoint-300` as the current A1 Stage 1A candidate for downstream Stage 1B/Stage 2 reads; do not call the 1000-step extension green. |
| `v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex` | `ap-A8Op30esUTAMBs2ZCokzU7` | Stage 1B early canary | Completed A100 canary from Stage 1A `checkpoint-2000`; app stopped cleanly. W&B: `https://wandb.ai/babakdam/anymal-pretrain/runs/ssoqebwc` (`likely-thunder-23`). Loaded the intended checkpoint, dataset is `v4_grounding`, trainables are V4 connector-only (`1.63B`). Eval improved monotonically from `1.1841` at step `50` to final `0.9570` at step `250`; checkpoints `62`, `124`, `186`, and `248` are complete with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Final W&B read at step `250`: train loss `0.415`, loss EMA `0.910`, grad norm `0.571`, grad EMA `0.678`, clip fraction `0.152`, LR `2.0e-5`, active alert `recent_loss_spikes`. This is a useful wiring/grounding canary, but yellow-history rather than green because W&B recorded loss spikes at steps `80` and `170`; clipping improved from `1.0` at step `10` to below `0.20` by completion and no grad spikes were recorded. |
| `v4-stage1b-grounding-smoke-20260508-codex` | `ap-Sgt4cPkIPPKUYc6AtPbWNS` | Stage 1B smoke | Loaded the Stage 1A smoke checkpoint, completed 2 optimizer steps on the grounding mixture, saved `/checkpoints/pretrain-output/v4-stage1b-grounding-smoke-20260508-codex/checkpoint-2`, then stopped. |
| `v4-stage2a-direct-calibration-smoke-20260508-codex` | `ap-q3IwxSS9IPdAyu0ONA1JwD` | Stage 2A smoke | Loaded the Stage 1B smoke checkpoint, froze the connector, trained LoRA-only for 2 optimizer steps, and saved `/checkpoints/finetune-output/v4-stage2a-direct-calibration-smoke-20260508-codex/checkpoint-2`. Trainer verified `adapter: 0`, `lora: 167,772,160`, `other: 0`. |
| `v4-stage2a-directcal-from-stage1a300-20260508-codex` | `ap-FwB5yLePPpMtMLRmPjVwFf` | Stage 2A direct calibration | Stopped on strict W&B clipping. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/cv70gfmy` finished at step `8` with active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `1.3366`, loss EMA `4.2684`, grad norm `17.6211`, grad EMA `17.0653`, LR `1.6e-4`, accumulation `8.0`, no eval rows, and no checkpoint artifacts. |
| `v4-stage2a-directcal-lora1e5-from-stage1a300-20260508-codex` | `ap-eXLMYJmlxBssgDMR5XHse0` | Stage 2A direct calibration | Stopped on strict W&B clipping even with explicit LoRA LR `1e-5`. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/7iz3gqye` finished at step `5` with active `high_grad_clip_fraction`, clip fraction `1.0`, grad norm `18.9571`, grad EMA `20.0704`, train loss `4.7371`, loss EMA `4.8712`, LR `5e-6`, accumulation `8.0`, and no artifacts. |
| `v4-stage2a-directcal-lossscale005-lora1e5-from-stage1a300-20260508-codex` | `ap-xDC5VyUHd6sA0JmwUxYrtG` | Stage 2A loss-scale canary | Stopped on strict W&B clipping history. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/ivvpnfop` finished at step `7` with active `high_grad_clip_fraction`, cumulative clip fraction `0.4286`, latest grad norm `0.8014`, grad EMA `0.9160`, raw/train loss `4.5885`, backward loss `0.2294`, LR `7e-6`, accumulation `8.0`, and no artifacts. Loss scale `0.05` helped grad magnitude but did not clear early clipped rows. |
| `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1a300-20260508-codex` | `ap-zvc4ZdZ0C1YHBSpyWvukVr` | Stage 2A loss-scale canary | Completed cleanly under W&B-first monitoring. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/fmrgmyy8` finished at step `100`; Modal app completed normally. Logs verified Stage 1A `checkpoint-300` projector load, dataset `v4_direct_calibration`, frozen connector, loss scale `0.03`, explicit projector/LoRA LR `1e-5`, and LoRA-only trainables: adapter `0`, LoRA `167,772,160`, other `0`. Step `50` passed with alerts `[]`, accumulation `8.0`, clip fraction `0.0`, raw/train loss `1.3340`, backward loss `0.0400`, loss EMA `1.9015`, grad norm `0.1807`, grad EMA `0.2761`, LR `6.2814e-6`, eval loss `1.2776`, no loss/grad spikes, and complete `checkpoint-50` (`trainer_state.pt`, `model_meta.json`, `llm/`, `projector.pt`). Final W&B step `100`: alerts `[]`, accumulation `8.0`, clip fraction `0.0`, raw/train loss `1.1870`, backward loss `0.0356`, loss EMA `1.1151`, grad norm `0.1255`, grad EMA `0.1138`, LR `1.0e-6`, eval loss `1.1988`, no loss/grad spikes, and recent window `1.0693 -> 1.0453`. `checkpoint-100` is complete with `trainer_state.pt`, `model_meta.json`, `llm/`, and `projector.pt`. Judgment: green; loss scale `0.03` cleared the strict Stage 2A clipping/spike gates where unscaled, LoRA-LR-only, and `loss_scale=0.05` canaries failed. Use as the current Stage 2A candidate for external held-out VQA, not promotion by itself. |
| `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b248-20260508-codex` | `ap-lOnsPRzER2eCaTdpEpxEHu`, VQA `ap-yht7BQyZq4c2JuNbPw7Gaa`/`ap-wr2Y4OgIYoJFjaVOzLqnpD` | Stage 2A Stage1B248 ablation | Completed to step `100` from Stage 1B grounding `checkpoint-248`, with frozen connector, LoRA-only trainables (`167,772,160`), projector/LoRA LR `1e-5`, and Stage 2 `loss_scale=0.03`. W&B: `https://wandb.ai/babakdam/anymal-finetune/runs/z7c3o6ug`. Lorentz monitored with the stricter W&B-first rule and reported active `recent_loss_window_mean_up_gt_25pct` at step `7`; parent W&B checks at steps `9` and `12` cleared the alert with `train/accumulation_micro_batches=8.0`, clip fraction `0.0`, and no pointwise loss/grad spikes, so the run continued as yellow-history. Step `50` passed with alerts `[]`, accumulation `8.0`, clip fraction `0.0`, raw/train loss `0.5935`, backward loss `0.0178`, loss EMA `0.7012`, grad norm `0.0122`, grad EMA `0.0142`, LR `6.2814e-6`, eval loss `0.7979`, no spikes, recent window `0.7271 -> 0.6651`, and complete `checkpoint-50`. A W&B read near step `92` again showed active `recent_loss_window_mean_up_gt_25pct` (`0.5605 -> 0.8240`) but arrived after the app had reached final eval/checkpoint; final W&B step `100` cleared alerts with eval `0.7956`, clip `0.0`, raw/train loss `0.8841`, backward loss `0.0265`, loss EMA `0.6793`, grad norm `0.0144`, grad EMA `0.0156`, LR `1e-6`, no loss/grad spikes, and recent window `0.7837 -> 0.6510`. `checkpoint-100` is complete with `trainer_state.pt`, `model_meta.json`, `llm/`, and `projector.pt`. Direct-prompt seed-42 VQA improved over the Stage1A300 branch and narrowly cleared the V1 floor at checkpoint `100`: overall `7.600`, number `4.861`, other `7.083`, yes/no `9.524`, EOS `1.0`, max-hit `0.0`, avg tokens `6.446`. Checkpoint `50` was worse: overall `7.433`, same number/other, yes/no `9.038`, EOS `1.0`, max-hit `0.0`, avg tokens `6.442`; it fails the V1 floor. Promotion guard for checkpoint `100` still fails against V3 (`7.600` vs `9.100` overall) and fails yes/no recovery (`9.524` vs V1 `14.189`). Judgment: artifact-good/yellow-history and accuracy-improved, but not promotable; transient W&B alerts at steps `7` and `92` stay attached to this run. |
| `v4-stage1b-clean500-from-a1ckpt300-20260509-codex` | `ap-iU9sh2Zc43bqt7W4qcttjm` | Stage 1B clean continuation | Stopped by parent after W&B reported `crashed` post-400; app is now `stopped` with zero tasks. W&B `https://wandb.ai/babakdam/anymal-pretrain/runs/dsuz75tp`. Carver was assigned as strict W&B-first monitor and was re-tasked to avoid charitable readings of spikes/clipping/loss-window alerts and to preserve historical alerts. Startup sync issue: initial delegated check saw W&B return zero rows while logs had started; recorded as startup yellow-history, not a model-health alert after W&B populated. Step `140` had a watched recent-window rise (`1.6190 -> 1.9696`, `21.6%`) but no W&B alert; step `150` came close to the loss-window threshold (`1.6190 -> 2.0215`, about `24.9%`) and stayed recorded rather than erased. Step `200` and step `300` gates passed cleanly; checkpoint-300 metadata matches V4 shape (`64` global / `64` local, hidden `1024`, output scale `1.0`, gate init `0.0001`, 2D positions enabled). Step `400` gate also passed with alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes, eval `2.0128`, and complete checkpoint. The run did not reach step `500`: post-400 logs showed repeated auto-resume from `checkpoint-400`, including `Skipping 3200 micro-batches to resume position...`, while W&B stopped advancing after step `410` and state became `crashed`. Final W&B step `410`: alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector spikes, train/objective loss `1.3611`, raw loss `1.6318`, backward loss `0.1361`, loss EMA `1.5296`, grad norm `0.1832`, grad EMA `0.1969`, LR `1.8594e-6`, eval `2.0128`, recent loss window `1.7796 -> 1.5676`. Verified checkpoints are `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, and `checkpoint-400`; no `checkpoint-500`. Judgment: useful clean-through-400 Stage 1B checkpoint, but yellow-history/operational-crash rather than completed green. |
| `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b400clean-20260509-codex` | `ap-ukMv087X6GXABYwoH5Pet2` | Stage 2A Stage1B400 loss-scale canary | Stopped on strict W&B clipping. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/k00wp8wx` (`classic-terrain-61`) finished at step `4` after parent stop. Startup loaded Stage1B400 `checkpoint-400`; after Stage 2 adapter disabling the actual trainables were adapter/projector `0`, LoRA `167,772,160`, other `0`, with frozen connector, LR/LoRA LR `1e-5`, accumulation `8`, and Stage 2 `loss_scale=0.03`. W&B step `3` already had active `high_grad_clip_fraction`, clip fraction `1.0`, grad norm `1.8728`, grad EMA `1.2073`, raw/train loss `3.7292`, backward loss `0.1119`, LR `3e-6`, and accumulation `8.0`. Final W&B step `4` still had active `high_grad_clip_fraction`, clip fraction `1.0`, grad norm `1.4606`, grad EMA `1.2200`, raw/train loss `3.9609`, backward loss `0.1188`, loss EMA `3.7435`, LR `4e-6`, no evals, no inspector spikes, and no checkpoints (`checkpoint-50` absent). Judgment: red optimization failure for this Stage1B400 + `loss_scale=0.03` Stage 2 recipe; do not evaluate or promote. |
| `v4-stage2a-directcal-lossscale001-lora1e5-from-stage1b400clean-20260509-codex` | `ap-NIhQMJHaMAc1eEZbJvlBdJ`, VQA `ap-5bLA7FbwA8gvEWUSLAkeOT` | Stage 2A Stage1B400 loss-scale canary | Completed cleanly but failed external VQA/hygiene. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/qx0mcvwr` (`light-grass-62`) finished at step `100`; Modal app completed and stopped. Startup loaded Stage1B400 `checkpoint-400`; after Stage 2 adapter disabling the actual trainables were adapter/projector `0`, LoRA `167,772,160`, other `0`, with frozen connector, LR/LoRA LR `1e-5`, accumulation `8`, and Stage 2 `loss_scale=0.01`. W&B step `2` cleared the early clipping gate where `loss_scale=0.03` failed: alerts `[]`, clip `0.0`, grad norm `0.5715`, backward loss `0.0382`, accumulation `8.0`. Step `50` gate passed with alerts `[]`, clip `0.0`, no inspector spikes, eval `1.2514`, grad norm `0.0367`, grad EMA `0.1055`, raw/train loss `1.1271`, backward loss `0.0113`, loss EMA `1.5724`, and complete `checkpoint-50`. Final W&B step `100`: alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector loss/grad spikes, eval improved to `1.1746`, raw/train loss `1.1429`, backward loss `0.0114`, loss EMA `1.0822`, grad norm `0.0310`, grad EMA `0.0404`, LR `1e-6`, recent window `1.0384 -> 1.0452`. `checkpoint-100` contains `trainer_state.pt`, `model_meta.json`, `llm/`, and `projector.pt`. Direct-prompt seed-42 VQA file `vqa_eval_v4_stage2a_stage1b400_lossscale001_ckpt100_directprompt_training_chat.json`: overall `5.400`, number `3.704`, other `3.379`, yes/no `9.135`, EOS `0.618`, max-hit `0.382`, avg generated tokens `14.831`. Judgment: optimization/artifacts green, accuracy/hygiene red; do not promote. Stage1B400 plus lower Stage 2 scale reduces clipping but does not preserve generation hygiene or beat the Stage1B248 branch. |
| `v4-stage1a-deepstacklite-background-launch-20260509-codex` | `ap-Wu3TPSPQYTOhr3HWPZHHSu` | DeepStack-lite launch attempt | Operational failed launch, not model evidence. The first DeepStack-lite Modal command used `--background`; the local entrypoint rejected it with `--background is disabled ... Use modal run --detach modal_train.py ... instead.` No GPU training, checkpoint, or W&B run was created. Judgment: launcher failure only; keep in the ledger so absence of W&B data is not read as a healthy experiment. |
| `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary20-20260509-codex` | `ap-PGBfgosRiESzTdPAwUKPua` | DeepStack-lite Stage 1A canary | Completed cleanly to step `20`. W&B `https://wandb.ai/babakdam/anymal-pretrain/runs/w7ktjb94` finished with alerts `[]`, train rows `2`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes, train/objective loss `2.3092`, raw loss `7.4254`, backward loss `0.2309`, loss EMA `2.7728`, supervised tokens `3.375`, multiplier `0.421875`, grad norm `0.1844`, grad EMA `0.1474`, and LR `1e-6`; no eval was due. `checkpoint-20` is complete with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Metadata confirms `anymal_v4`, `connector_type=deepstack_spatial_perceiver_resampler`, `vision_feature_strategy=deepstack_lite`, `vision_feature_layers=[-3,-2,-1]`, `deepstack_num_feature_levels=3`, `64` global / `64` local tokens, hidden `1024`, output scale `1.0`, gate init `0.0001`, and 2D position features. Judgment: pass/continue canary, yellow-history for startup W&B sync gap and empty W&B artifacts. |
| `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120-20260509-codex` | `ap-JLfSYAeQlCyjpqIGIvYA4B` | DeepStack-lite launch attempt | Operational failed detached launch. Modal showed zero tasks and no logs, no W&B run appeared, and the local detached process was killed. No training evidence was produced. Judgment: launcher failure only; ignore as model-quality evidence but keep recorded for babysitting history. |
| `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120b-20260509-codex` | `ap-aBJHyxraKrFQJiWcIOtK9n` | DeepStack-lite Stage 1A canary | Completed cleanly to step `120`. W&B `https://wandb.ai/babakdam/anymal-pretrain/runs/kb8js5z0` finished with alerts `[]`, train rows `12`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes, and eval improving `7.7001 -> 7.1668`. Step `50` gate passed with train/objective loss `1.8737`, raw loss `6.2265`, backward loss `0.1874`, loss EMA `2.5355`, grad norm `0.1271`, grad EMA `0.1281`, LR `7.5196e-6`, recent window `2.2928 -> 1.9261`, eval `7.7001`, and complete `checkpoint-50`. Step `100` saved a complete `checkpoint-100`. Final W&B step `120`: train/objective loss `1.9442`, raw loss `6.7862`, backward loss `0.1944`, loss EMA `2.3480`, supervised tokens `3.125`, multiplier `0.390625`, grad norm `0.1339`, grad EMA `0.1412`, LR `1e-6`, recent window `2.1911 -> 2.3420` across `12` rows, below alert threshold. `checkpoint-100` metadata confirms DeepStack-lite with layers `[-3,-2,-1]`, `deepstack_num_feature_levels=3`, `64/64` token split, gate `0.0001`, and 2D positions. Modal app stopped with zero tasks. Judgment: pass/continue architecture canary, not a promotion result; yellow-history for startup W&B sync gap and empty W&B artifact table. |
| `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex` | `ap-B8Hvvx5yEIsO1sFO7LFJgU` | DeepStack-lite Stage 1A extension | Completed to step `500`, artifact-good/yellow-history. W&B `https://wandb.ai/babakdam/anymal-pretrain/runs/xb0silsi` finished with latest checked step `500`, train rows `50`, alerts `[]`, accumulation `8.0`, clip fraction `0.002000`, no inspector loss/grad spikes, train/objective loss `1.9103`, raw loss `3.9376`, backward loss `0.1910`, loss EMA `1.0196`, grad norm `0.2567`, grad EMA `0.1705`, LR `1.0e-6`, and eval improving `8.2100 -> 6.9942 -> 6.1354 -> 3.6915 -> 3.3106 -> 3.0524 -> 2.8963 -> 2.8282 -> 2.7581 -> 2.7120`. Checkpoints `200`, `300`, `400`, and `500` contain `trainer_state.pt`, `projector.pt`, and `model_meta.json`; checkpoint-200 metadata matches DeepStack-lite V4 (`deepstack_spatial_perceiver_resampler`, layers `[-3,-2,-1]`, 3 feature levels, `64/64` tokens, gate `0.0001`, 2D positions). Preserved yellow history: startup W&B sync gap, step-50 eval worse than the 120b first eval, step-60 objective jump, step-100 loss-window rise below threshold, console-only suspicious microbatch spike around step 353, late console microbatch loss `5.8067` around step 499, and small nonzero clipping at later gates. Judgment: usable DeepStack Stage 1A checkpoint for downstream Stage 1B/Stage 2A testing, not standalone accuracy promotion. |
| `v4-stage1b-deepstacklite-from-stage1a500-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex` | `ap-MEQM1e6xO2qip772hSqoHr` | DeepStack-lite Stage 1B extension | Live run from Stage 1A DeepStack `checkpoint-500` using `--pretrain-checkpoint` with fresh optimizer/scheduler state. W&B `https://wandb.ai/babakdam/anymal-pretrain/runs/2wiqwnto` is running. Startup verified DeepStack connector load, `v4_grounding`, connector-only trainables `44,377,089`, and clean short-answer dataset diagnostics. First W&B gate at step `20`: train rows `2`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes, train/objective loss `0.7141`, raw loss `1.9351`, backward loss `0.0714`, loss EMA `0.8854`, grad norm `0.2802`, grad EMA `0.1625`, LR `4.0e-6`, no eval due yet. Step `40` strict-monitor check had active `recent_loss_window_mean_up_gt_25pct` (`0.7099 -> 1.0576`) and is preserved as yellow-history even though later checks cleared. Step `90` W&B check had alerts `[]`, clip `0.0`, no spikes, eval `2.6056`, train/objective loss `1.0392`, grad norm `0.3390`, and rising recent window `0.8837 -> 1.0383`, so it stays yellow-watch. Step `100` boundary passed with alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector loss/grad spikes, eval improved to `2.4619`, train/objective loss `0.8114`, raw loss `2.8371`, backward loss `0.0811`, loss EMA `0.8870`, grad norm `0.2719`, grad EMA `0.2651`, LR `9.7286e-6`, recent window `0.8730 -> 1.0345`, and complete `checkpoint-100` (`trainer_state.pt`, `projector.pt`, `model_meta.json`). Step `130` remained W&B-clean with alerts `[]`, clip `0.0`, no spikes, train/objective loss `1.3580`, grad norm `0.2510`, LR `9.3162e-6`, latest eval `2.4619`, and rising recent window `0.8506 -> 1.0112`; console-only high microbatch losses around this area stay yellow-watch. Step `150` was clean with alerts `[]`, clip `0.0`, no spikes, train/objective loss `0.7022`, grad norm `0.1485`, eval improved to `2.2104`, and recent window flattened `0.8970 -> 0.8756`. Step `170` also stayed W&B-clean with alerts `[]`, clip `0.0`, no spikes, train/objective loss `1.4308`, grad norm `0.2820`, latest eval `2.2104`, and recent window `0.9609 -> 0.8984`; a console-only `4.7524` microbatch near step `165` remains yellow-history. Step `200` produced a console HealthMonitor warning (`loss EMA 0.5253 -> 0.8337`) but W&B stayed clean: alerts `[]`, clip `0.0`, no spikes, train/objective loss `0.4807`, grad norm `0.3981`, eval improved to `2.1062`, recent window `0.9538 -> 0.8641`, and complete `checkpoint-200` (`trainer_state.pt`, `projector.pt`, `model_meta.json`). Step `230` stayed W&B-clean with alerts `[]`, clip `0.0`, no spikes, train/objective loss `0.7526`, grad norm `0.2766`, latest eval `2.1062`, and recent window `0.9732 -> 0.8606`. Step `250` eval gate stayed clean with alerts `[]`, clip `0.0`, no spikes, train/objective loss `0.8676`, grad norm `0.3728`, eval improved to `2.0273`, and recent window nearly flat `0.8919 -> 0.9030`; no checkpoint was due. Step `260` stayed GO/yellow-history with alerts `[]`, accumulation `8.0`, clip `0.0`, no spikes, train/objective loss `0.8278`, raw loss `2.5899`, loss EMA `0.8543`, grad norm `0.1434`, grad EMA `0.1809`, LR `5.9704e-6`, and recent window `0.9371 -> 0.8430`. Step `300` hard gate passed by W&B and artifacts: state `running`, train rows `30`, history rows `36`, alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector loss/grad spikes, train/objective loss `0.9274`, raw loss `1.6377`, backward loss `0.0927`, loss EMA `0.7789`, grad norm `0.2612`, grad EMA `0.1736`, LR `4.7186e-6`, eval improved to `1.9358`, recent window `0.8641 -> 0.8489`, and complete `checkpoint-300` (`trainer_state.pt`, `projector.pt`, `model_meta.json`). Judgment: continue under strict W&B-first monitoring; Rawls assigned as monitor; run remains artifact-good/yellow-history because of the step-40 active W&B loss-window alert, console-only spikes, step-200 HealthMonitor warning, and transient checkpoint-200 visibility lag. |
| `v4-stage1b-deepstacklite-from-stage1a500...` final override | `ap-MEQM1e6xO2qip772hSqoHr` | DeepStack-lite Stage 1B closeout | The row above was superseded after the step-350 W&B inspection. Parent stopped the app because W&B had active `recent_loss_spikes`; final W&B state was `finished` at step `350`, with step `340` loss `1.6254` vs loss EMA `0.7558` (`2.15x`). Eval improved to `1.9128`, accumulation stayed `8.0`, clip stayed `0.0`, and no grad spikes were present, but the active W&B spike wins. Last verified artifact is `checkpoint-300`; `checkpoint-400` and `checkpoint-500` are absent. Judgment: STOP/yellow-red history, not green. |
| `v4-stage2a-directcal-lossscale003-lora1e5-from-deepstack-stage1b300-fix1-20260509-codex` | `ap-ktE7hOXSbNmpHZHEievcyN` | DeepStack Stage 2A loss-scale canary | Stopped on strict W&B clipping. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/3jp174rm` finished after parent stop at step `4` with active `high_grad_clip_fraction`, clip fraction `0.75`, raw/train loss `5.8650`, backward loss `0.1760`, grad norm `0.7287`, grad EMA `1.7741`, LR `4e-6`, accumulation `8.0`, no loss/grad spikes, no eval, and no checkpoint. Startup had verified the DeepStack Stage 1B `checkpoint-300`, frozen connector, LoRA-only trainables, and Stage 2 `loss_scale=0.03`. Judgment: STOP/red; do not rerun this branch unchanged. |
| `v4-stage2a-directcal-lossscale001-lora1e5-from-deepstack-stage1b300-20260509-codex` | `ap-9RS32uYnaDaBjDFWSdWEiZ`, VQA `ap-9pUJJqEd4FpkaNUIwc7Mxs` | DeepStack Stage 2A lower loss-scale validation | Stopped on strict W&B loss-window alert and failed external VQA from the last clean artifact. W&B `https://wandb.ai/babakdam/anymal-finetune/runs/zq2qg4em` initially passed early gates at steps `26`, `33`, and `42` with alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector loss/grad spikes, and improving recent-window means. Step `50`/`55` hard gate passed with eval `1.3578`, raw/train loss `0.7100`, backward loss `0.0071`, loss EMA `1.9915`, grad norm `0.0354`, grad EMA `0.1278`, LR `5.5e-6`, recent window `1.4241 -> 1.1655`, alerts `[]`, and complete `checkpoint-50` (`trainer_state.pt`, `model_meta.json`, `llm/`, `projector.pt`). A later W&B check at step `92` had active `recent_loss_window_mean_up_gt_25pct` (`0.9461 -> 1.2977`), so parent stopped immediately. Final W&B state is `finished` at step `95` with active `recent_loss_window_mean_up_gt_25pct`, clip `0.0`, no loss/grad spikes, and final window `1.0201 -> 1.2794`; `checkpoint-100` is absent. DeepStack VQA required a hook fallback in `models/encoders/siglip2_encoder.py` because the installed HF SigLIP path exposes only `last_hidden_state`. Direct-prompt seed-42 VQA on checkpoint `50` scored overall `4.233`, number `2.546`, other `2.469`, yes/no `7.580`, EOS `1.0`, max-hit `0.0`, avg tokens `3.582`; promotion guard fails V1 floor, incumbent, and yes/no recovery. Judgment: STOP/yellow-red history and accuracy-red; do not promote. |
| `v4-stage2a-lossscale003 external VQA reads` | `ap-EPqZkyI4OoAjdO24E9ympe`, `ap-sjYqaz4dGhrnpuPySS1eQO`, `ap-XlopKMmy69g61X3F2BNF5N`, `ap-IVOvhl5lQJXs6gtz0vD3mX` | Held-out VQA | Accuracy-red despite health-green training. Seed-42 `training_chat` reads with the evaluator's generic system prompt on 1000 samples: checkpoint-50 `0.10` overall, number `0.00`, other `0.195`, yes/no `0.00`, EOS `0.896`, max-hit `0.104`; checkpoint-100 `0.10` overall, number `0.00`, other `0.195`, yes/no `0.00`, EOS `0.899`, max-hit `0.101`. A 300-sample `legacy_qa` diagnostic on checkpoint-100 did not rescue the result: `0.667` overall, number `1.887`, other `0.699`, yes/no `0.00`, EOS `0.003`, max-hit `0.997`. The direct-calibration system prompt (`Answer the image question directly and briefly. End after the answer.`) fixed the generation contract but not accuracy: 20-sample diagnostic was `15.0` overall with `2.05` avg tokens, then the full 1000-sample checkpoint-100 read was `5.367` overall, number `3.704`, other `3.509`, yes/no `8.844`, EOS `1.0`, max-hit `0.0`. `scripts/check_vlm_promotion.py --no-fail` still rejects checkpoint-100 against both the V1 floor and V3 incumbent (`5.367` vs V1 `7.567` and V3 `9.100`). Judgment: do not promote. The Stage 2A loss-scale recipe fixed optimization health and the direct prompt fixed hygiene, but the Stage 1A300 base and/or V4 data/architecture is still below the incumbent on held-out VQA. |

### Active babysitting state

- The full Stage 1A H100 run was stopped after W&B clipping crossed the
  playbook escalation line. The Stage 1B early grounding canary completed
  cleanly, but remains yellow-history due W&B loss spikes.
- A lower-LR Stage 1A recovery from the last safe checkpoint was also stopped:
  `v4-stage1a-recovery4500-lr8e5-3000-20260508-codex`,
  app `ap-BR97ZXuooRK292Ja0Kgflv`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/g607yqek`. W&B was checked
  explicitly after the stop; final state was `finished` at step `40` with
  `health/grad_clip_fraction` `0.825` and active `high_grad_clip_fraction`,
  confirming that LR `8e-5` with warmup did not solve the Stage 1A clipping
  issue.
- A diagnostic Stage 1A canary from `checkpoint-4000` was also stopped:
  `v4-stage1a-recovery4000-lr5e5-canary300-20260508-codex`, app
  `ap-E3Ke98Cu4xVqc7Aal55iSm`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/r4hx5xrs`. Parent stopped it
  after the early W&B check at step `40` showed `health/grad_clip_fraction`
  `0.35` at LR about `4.98e-5`. Post-stop W&B was checked explicitly; final
  state was `finished` at step `50` with clip fraction `0.30`, active
  `high_grad_clip_fraction`, no eval rows, and no completed checkpoint. Modal
  listed the app `stopped` with zero tasks; the run directory exists on the
  Modal volume but is empty, and `checkpoint-50`/`checkpoint-75` are absent.
  This points away from late optimizer-state-only failure and toward connector
  state, data formatting, or gradient scaling.
- The patched-label Stage 1A canary from `checkpoint-4000` was also stopped:
  `v4-stage1a-fixedlabels-ckpt4000-lr5e5-canary100-20260508-codex`, app
  `ap-Du9jNqbTRtxsNAzf3X8ZVB`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/froy06eg`. Logs confirmed the
  intended data-format fix: supervised caption previews no longer contained
  `<|begin_of_text|>` and started with real caption spans such as `madonna
  tour...`, `the bmw m3...`, and `the wordpress...`. W&B was checked explicitly
  before stopping and again after stopping. The early W&B read already showed
  `high_grad_clip_fraction` at step `10` with clip fraction `0.60`; post-stop
  W&B ended `finished` at step `40` with clip fraction still `0.60`, train loss
  `2.9152`, loss EMA `3.2533`, grad norm `0.7119`, grad EMA `1.3856`, LR
  `3.875e-5`, no eval rows, and no recent loss/grad spikes. Modal listed the app
  `stopped` with zero tasks; the run directory exists on the Modal volume but is
  empty, so no `checkpoint-50` or eval/checkpoint gate landed. This confirms
  fixed-label masking is insufficient by itself to solve Stage 1A clipping.
- The from-scratch bottleneck1024 Stage 1A canary was also stopped:
  `v4-stage1a-bottleneck1024-canary100-20260508-codex`, app
  `ap-76TjvdhPvFm3LdHXdHJ6SC`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/dsv4mlbs`. Logs confirmed the
  intended architecture change: V4 connector-only trainables were only
  `44,374,016`, real LLaVA-Pretrain data loaded, and supervised captions still
  excluded `<|begin_of_text|>`. W&B tripped the stop gate immediately: step `10`
  had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss `5.0610`,
  loss EMA `9.5638`, grad norm `8.3869`, grad EMA `90.8336`, and LR `1e-4`.
  Post-stop W&B ended `finished` at step `20` with clip fraction still `1.0`,
  train loss `4.2093`, loss EMA `7.8934`, grad norm `3.2777`, grad EMA
  `58.6991`, LR `9.7286e-5`, no eval rows, and no loss/grad spike alerts.
  Modal listed the app `stopped` with zero tasks; the output directory listing
  was empty, so no checkpoint/eval gate landed. Connector shrinkage alone is not
  enough; next Stage 1A work should change gradient scale/initialization/loss
  recipe before another long run.
- The `connector_output_scale=0.1` bottleneck canary was also stopped:
  `v4-stage1a-bottleneck1024-scale01-canary100-20260508-codex`, app
  `ap-TOyngwQQ017RjE8vDGZbjo`, W&B
  `https://wandb.ai/babakdam/anymal-pretrain/runs/wm6y5cvh`. W&B was checked
  before relying on logs: step `10` already had active `high_grad_clip_fraction`,
  clip fraction `1.0`, train loss `6.7240`, loss EMA `10.1241`, grad norm
  `48.4955`, grad EMA `132.7394`, and LR `1e-4`. Parent stopped the app.
  Post-stop W&B ended `finished` at step `20` with clip fraction still `1.0`,
  train loss `3.9441`, loss EMA `8.0904`, grad norm `2.5491`, grad EMA
  `81.1468`, LR `9.7286e-5`, no eval rows, and no recent loss/grad spikes.
  Modal listed the app `stopped` with zero tasks; the output directory listing
  was empty. This is a clear negative result for output-scale-only stabilization:
  loss improved, but W&B clipping stayed pegged, so the next Stage 1A canary
  should change initialization/gating or the caption-alignment objective.
- Three trainable-gate bottleneck canaries were also stopped or rejected on W&B
  clipping, despite smaller gate initialization:
  `v4-stage1a-bottleneck1024-gate001-canary100-20260508-codex` (`cdtr4gp6`,
  app `ap-h9cCm08DcrIDEqyIcJy1Sw`) failed at step `30` with clip fraction
  `1.0`, train loss `9.0393`, loss EMA `8.4019`, grad norm `5.1590`, grad EMA
  `17.0959`, and active `high_grad_clip_fraction` plus
  `recent_loss_window_mean_up_gt_25pct`; `gate0001` (`us0ls2wu`, app
  `ap-753TNbms8Pv2wNLeL0qrut`) failed at step `20` with clip fraction `1.0`,
  train loss `4.7958`, loss EMA `5.9601`, grad norm `21.4131`, and grad EMA
  `39.3029`; `gate00001` (`2qdqg86p`, app
  `ap-FNaPaB3I8yEpFaF7fmLefB`) completed and saved `checkpoint-20`, but final
  W&B still had active `high_grad_clip_fraction`, clip fraction `1.0`, train
  loss `4.8986`, loss EMA `6.0846`, grad norm `1.4890`, grad EMA `78.9965`, and
  no eval rows. The `gate00001` checkpoint contains `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`, but it is a failed artifact, not a
  resume candidate. Judgment: trainable visual gating at `0.01`, `0.001`, or
  `0.0001` is not sufficient by itself; the next canary needs a recipe/schedule
  change, not another amplitude-only handoff.
- An output-projection/gate-only warmup canary was also stopped before unfreeze:
  `v4-stage1a-outputwarmup30-gate00001-lr1e5-canary60-20260508-codex`
  (`iip0i3e3`, app `ap-6PUomRff3C9w5V8LMkNegC`). Logs verified the new
  `connector_warmup_steps=30` behavior: only `projector.output_proj.*` and
  `projector.output_gate_logit` were allowed to update during warmup (`3` active
  tensors, `71` masked tensors), with real LLaVA-Pretrain data and fixed labels.
  W&B step `20` was already red during warmup: active `high_grad_clip_fraction`,
  clip fraction `1.0`, train loss `6.0813`, loss EMA `6.5835`, grad norm
  `17.0686`, grad EMA `31.0523`, LR `8.5881e-6`, and no eval rows. Modal listed
  the app `stopped` with zero tasks and the output directory was empty. Judgment:
  staged output-only warmup does not solve the caption-alignment Stage 1A
  clipping; next work should test grounding-first or otherwise cleaner
  supervision rather than another connector handoff variant.
- The grounding-first canary also failed immediately:
  `v4-stage1a-groundingfirst-gate00001-lr1e5-canary80-20260508-codex`
  (`udsb1fr9`, app `ap-eE4Hll6DpCL0A6rqvc1fni`) used the 4x A100 distributed
  Stage 1 path, dataset `v4_grounding`, LR `1e-5`, and
  `connector_output_gate_init=0.0001`. Logs verified the objective changed to
  the grounding mixture, with direct short-answer labels such as
  `C<|eot_id|>`, `phone<|eot_id|>`, and `no<|eot_id|>`. W&B step `10` already
  had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss
  `10.8670`, loss EMA `8.0383`, grad norm `7.5836`, grad EMA `7.2998`, LR
  `9.9829e-6`, and no eval rows. Parent stopped before eval/checkpoint; final
  W&B state was `finished`, Modal listed `stopped` with zero tasks, and the run
  directory had no checkpoint files. Judgment: changing from web captions to the
  current short-answer grounding mixture is also insufficient by itself. Next
  lever should reduce effective update scale or gradient semantics, e.g. larger
  accumulation / smaller LR / loss normalization by supervised-token count,
  before any more architecture-token ablations.
- Stage 1A last verified checkpoint:
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4500`.
- Stage 1A attempted but unverified/partial checkpoint:
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4750`.
- Stage 1B canary last verified checkpoint:
  `/checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248`.
- W&B must be checked alongside logs for every subsequent monitoring update.
  Use `scripts/inspect_wandb_run.py`; do not explain away spikes without W&B
  loss/EMA, grad/EMA, clipping, LR, and eval evidence.
- Stage 1A final state is stopped on clipping escalation, not green. The
  `checkpoint-4500` gate landed and was checked with Modal volume listings twice:
  the second listing showed `trainer_state.pt` `12.2 GiB`, `projector.pt`
  `6.1 GiB`, and `model_meta.json` `485 B`, all modified `13:53 EDT`. Modal
  logs showed training resumed past the gate to about step `4517`, and the app
  remained live with one task; no traceback, OOM, NaN, inf, or checkpoint failure
  was observed in the sampled logs. W&B was checked explicitly with
  `scripts/inspect_wandb_run.py --recent-window 220`: latest step `4550`, train
  loss `3.0209`, loss EMA `2.6327`, grad norm `1.2151`, grad EMA `1.7192`, clip
  fraction `0.1892`, LR `1.7869e-4`, active `recent_loss_spikes` and
  `recent_grad_spikes`. Eval/loss is still the improved step-4000 value
  `2.8393`; next eval is expected around step `6000`. The recent 220-row loss
  mean was not rising (`2.9019 -> 2.7948`), but clipping continued upward
  from `0.161` at step `4000` to `0.1640` at `4250`, `0.1732` at `4360`,
  `0.1775` at `4450`, `0.1863` at `4520`, `0.1892` at `4550`, `0.2000` at
  `4640`, `0.204059829` at `4680`, `0.2093` at parent-observed step `4730`,
  and `0.2112` at post-stop W&B step `4750`. Parent stopped the app at that point. Post-stop verification showed app
  `ap-hTW04HKrFzV2fbDo6SVTBP` stopped with zero tasks; `checkpoint-4750` exists
  but is partial/unverified because `trainer_state.pt` remained `1.0 GiB` after a
  second listing, so the last safe checkpoint is still `checkpoint-4500`.
- Stage 1A would have next evaluated around step `6000`, but the run was stopped
  before that because clipping crossed the playbook escalation line.
- Stage 1A recovery never reached its first checkpoint or eval gate. Parent
  stopped it after W&B step `30` already showed clipping around `0.80` at LR
  `2.4e-5`; post-stop W&B reached step `40` with clip fraction `0.825`, LR
  `3.2e-5`, train loss `2.8424`, loss EMA `2.8923`, grad norm `2.0451`, grad EMA
  `2.3047`, no evals, and alert `high_grad_clip_fraction`. Modal app
  `ap-BR97ZXuooRK292Ja0Kgflv` was stopped with zero tasks, and no
  `checkpoint-50` or `checkpoint-250` directory was present in the recovery
  output. Do not restart this recovery recipe unchanged.
- Stage 1B canary completed with a positive eval trajectory and complete
  checkpoints, but do not call it clean green: W&B recorded repeated loss spikes
  at steps `80` and `170`.
- Do not launch the full Stage 1B continuation until a sufficiently trained
  Stage 1A checkpoint is selected; the early Stage 1B canary gives wiring and
  early downstream signal, not final model quality.
- V4 DeepStack-lite connector ablation passed the 20-step and 120-step Stage 1A
  canaries under strict W&B-first monitoring, but downstream validation reversed
  the recommendation. Stage 1B stopped on an active W&B loss spike, Stage 2A
  failed `loss_scale=0.03` on clipping and `loss_scale=0.01` on a late
  loss-window alert, and checkpoint-50 VQA scored only `4.233` overall. Keep
  DeepStack-lite as a wired ablation, not the default.
- `v4-stage1b-deepstacklite-from-stage1a500-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex`
  (`babakdam/anymal-pretrain/2wiqwnto`, app `ap-MEQM1e6xO2qip772hSqoHr`) is
  STOP/yellow-red history, not green. Parent stopped the app after final W&B
  state `finished` at step `350` with active alert `recent_loss_spikes`; the
  recorded spike was step `340`, loss `1.6254` versus loss EMA `0.7558`
  (`2.15x`). Eval improved to `1.9128`, accumulation stayed `8.0`, clip
  fraction stayed `0.0`, and no grad spikes were reported, but active W&B spike
  wins. Last verified artifact is `checkpoint-300`; `checkpoint-400` and
  `checkpoint-500` are absent. Preserve earlier yellow history from the step
  `40`/`50` loss-window alert, step `90` rising-window watch, step `200`
  HealthMonitor warning, transient `checkpoint-200` visibility lag, and
  console-only step `307`-`342` spike candidates.
- Added the next concrete recipe ablation: `v4_semantic_calibration`. It keeps a
  frozen-connector LoRA-only Stage 2A and replaces the plain yes/no source with
  balanced canonical yes/no labels, using weights `0.40/0.20/0.15/0.20/0.05`
  for yes-no/number/COCO/other/short. Run it first as a Stage1B248 isolation
  branch, then repeat on the A1 bottleneck spatial connector only if it clears
  the gate. The intended gate is to beat the Stage1B248 branch (`7.600` overall)
  without losing EOS/max-token hygiene or tripping W&B.
- `v4-stage2a-semanticcal-lossscale003-lora1e5-from-stage1b248-20260509-codex`
  (`ap-JP6wrEiBQPEjVWxJ3eAQuE`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/muyq9oz9`) validated the new
  semantic data path but was stopped before checkpoint/eval under the strict
  W&B-first policy. Startup loaded Stage1B248, inferred the legacy V4 connector
  metadata (`6` layers, `16` heads, `ff_mult=4`, direct 4096-wide connector),
  built `60,040` balanced yes/no samples (`yes=30,549`, `no=29,491` after the
  available-image cap), and confirmed clean direct labels such as
  `dog<|eot_id|>`, `yes<|eot_id|>`, and `10<|eot_id|>`. After adapter disabling,
  trainables were LoRA-only (`167,772,160`), accumulation was `8.0`, and clipping
  stayed `0.0`. Parent W&B inspections at steps `6` and `10` both reported
  active `recent_loss_window_mean_up_gt_25pct`; step `10` had raw/train loss
  `0.9719`, backward loss `0.0292`, grad norm `0.0162`, grad EMA `0.0145`, loss
  EMA `0.8037`, LR `1e-5`, no point loss/grad spikes, and recent-window movement
  `0.7242 -> 0.9392`. Parent stopped the app immediately. Final W&B step `11`
  cleared active alerts after the stop signal, but there was no eval and no
  checkpoint, so this remains STOP/yellow-red history, not recipe validation.
- `v4-stage2a-semanticcal-bs4-lossscale003-lora1e5-from-stage1b248-20260509-codex`
  (`ap-0nOSPSoi635kK2vkRmuRpU`, W&B
  `https://wandb.ai/babakdam/anymal-finetune/runs/oxuky3ss`, VQA
  `ap-rIBg8GZSFF76vUeVDPPzFC`) completed the higher-effective-batch semantic
  calibration isolation branch from Stage1B248. Startup again inferred the
  legacy V4 connector metadata (`6` layers, `16` heads, `ff_mult=4`, direct
  4096-wide connector), built the same balanced yes/no source (`60,040`
  examples), confirmed clean direct labels, froze the connector, and trained
  LoRA only (`167,772,160`) with accumulation `8.0` and Stage 2
  `loss_scale=0.03`. W&B was monitored through the run rather than relying on
  logs: all explicit gates from step `4` through step `100` had alerts `[]`,
  clip fraction `0.0`, no inspector loss/grad spikes, and accumulation `8.0`.
  Yellow-watch history remains for console-only microbatch losses up to about
  `3.84` and a small non-alerting W&B rise at step `79` (`0.7903 -> 0.8280`).
  Final W&B state was `finished` at step `100`; eval improved from `0.8464` at
  step `50` to `0.8416`, recent-window movement was `0.8753 -> 0.6960`, and
  `checkpoint-100` contains `trainer_state.pt`, `model_meta.json`, `llm/`, and
  `projector.pt`. Initial direct-prompt seed-42 VQA file
  `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat.json`
  reported only overall `7.333`, number `4.861`, other `6.888`, yes/no
  `9.038`, EOS `1.0`, max-hit `0.0`, avg generated tokens `6.443`, but the
  prediction dump showed the evaluator was scoring duplicated decoded chat-role
  prefixes such as `assistant\n\nyes` as the literal answer `assistant`. After
  patching `evaluation/vqa_eval.py` to strip that prefix, the corrected file
  `vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat_postprocessfix.json`
  scored overall `52.400`, number `37.037`, other `40.871`, yes/no `76.093`,
  EOS `1.0`, max-hit `0.0`, avg generated tokens `6.443`. Corrected baselines:
  V1 direct-prompt/postprocess-fix overall `21.100`, yes/no `28.960`; V3
  direct-calibration checkpoint-100 postprocess-fix overall `9.400`, yes/no
  `11.079`. The updated promotion guard passes V1 floor, incumbent, yes/no
  recovery, and hygiene. Judgment: optimization/artifacts green with recorded
  yellow-watch history; corrected VQA promotion PASS for this Stage1B248
  semantic calibration checkpoint. Seed-43 confirmation on the same checkpoint
  was also strong: overall `51.367`, number `35.430`, other `40.370`, yes/no
  `72.043`, EOS `1.0`, max-hit `0.0`.

---

## Infra notes

- **Monitoring**: `Monitor` tool (tails Modal logs via `perl alarm` snapshots) worked better than `CronCreate` for training runs — crons only fire when REPL is idle, which fails when actively chatting.
- **Log streaming**: `modal app logs <id>` streams indefinitely. On macOS (no `timeout`), use `perl -e 'alarm(N); exec @ARGV' -- modal app logs <id>` for snapshots.
- **Viewer**: FastAPI ASGI app on Modal (`modal_viewer.py`), `min_containers=1` for snappy loads. Reads predictions JSON from the shared volume on each request — can update data without redeploying.
