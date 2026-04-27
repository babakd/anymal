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

## Infra notes

- **Monitoring**: `Monitor` tool (tails Modal logs via `perl alarm` snapshots) worked better than `CronCreate` for training runs — crons only fire when REPL is idle, which fails when actively chatting.
- **Log streaming**: `modal app logs <id>` streams indefinitely. On macOS (no `timeout`), use `perl -e 'alarm(N); exec @ARGV' -- modal app logs <id>` for snapshots.
- **Viewer**: FastAPI ASGI app on Modal (`modal_viewer.py`), `min_containers=1` for snappy loads. Reads predictions JSON from the shared volume on each request — can update data without redeploying.
