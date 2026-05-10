# AnyMAL Stage 2 Fine-Tuning Handoff (Codex)

Last updated: 2026-02-11
Authoring context: analysis of latest Stage 2 Modal run + inference comparison artifacts.

## 2026-04-28 Supersession Note

This file describes older V1-era/early Stage 2 behavior and should not be used
as the current V2 training status. The current V2 learned-compressor baseline
completed both meaningful stages on 2026-04-28. See:

- `V2_FULL_TRAINING_ARTIFACT_20260428.md`
- `V2_TRAINING_RUNBOOK.md`
- `CLAUDE.md`

Key current checkpoints:

- Stage 1:
  `/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500`
- Stage 2:
  `/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000`

## Scope

This note captures the current understanding of why Stage 2 fine-tuning has not delivered clear gains (and has some regressions), based on:

- W&B run: `babakdam/anymal-finetune/auhd0zkz` ([wise-smoke-15](https://wandb.ai/babakdam/anymal-finetune/runs/auhd0zkz))
- Inference artifact: `/Users/babakd/anymal/predictions_20260209_063338.json`
- Training/inference code paths in this repo.

## Key Artifacts and Paths

- Primary run report: `auhd0zkz` (500-step Stage 2 run from pretrain checkpoint-2500)
- Inference compare script: `/Users/babakd/anymal/modal_inference.py`
- Training entrypoint: `/Users/babakd/anymal/modal_train.py`
- Stage switching logic: `/Users/babakd/anymal/models/anymal.py`
- Optimizer/scheduler logic: `/Users/babakd/anymal/training/trainer.py`
- VQA eval loader/evaluator:
  - `/Users/babakd/anymal/evaluation/vqa_eval.py`
  - `/Users/babakd/anymal/evaluation/eval_runner.py`

## W&B Access Notes

- Local environment did not have `WANDB_API_KEY` set, so local `wandb.Api()` failed.
- Modal secret `wandb` is present and usable.
- Run metadata + history were read successfully via a short Modal function using the `wandb` secret.

## What Happened in Run `auhd0zkz`

### Config (from run config/summary)

- `max_steps = 500`
- `learning_rate = 1e-5`
- `warmup_steps = 50`
- `lr_scheduler_type = cosine`
- `weight_decay = 0.01`
- `lora_r = 64`
- `lora_alpha = 16`
- `gradient_accumulation_steps = 8`
- `save_steps = 125`
- `eval_steps = 50`
- `pretrain_checkpoint = /checkpoints/pretrain-output/checkpoint-2500`

### Train/Eval Curves (W&B history)

- `train/loss`: first `1.3516`, last `1.2194`, mean `1.3325`
- `eval/loss`: first `1.3858`, last `1.3176` (small drop, then plateau)
- `train/lr`: cosine decay to `0` by step 500
- `health/grad_clip_fraction`: always `0` (no clipping events)
- Late training appears low-yield (after ~step 250 improvements are weak).

### Final Eval Failure

Run log tail shows training finished, checkpoint-500 saved, then final VQA eval failed:

- `FileNotFoundError: /checkpoints/coco_val2014/COCO_val2014_000000145369.jpg`
- `Final VQA metrics: {}`

Implication: no trustworthy final benchmark metric was produced for this run.

## 20-Example Inference Comparison (Pretrain End vs Finetune End)

Source: `/Users/babakd/anymal/predictions_20260209_063338.json`

- Selection: 20 evenly spaced validation examples from deterministic split.
- Compared step `2500` (pretrain) vs step `500` (finetune).

Lexical token-F1 (vs ground truth):

- Better: `11`
- Worse: `5`
- Same: `4`
- Mean F1: `0.5955 -> 0.6233` (small net increase)

Worse indices: `0, 1, 10, 12, 14`

Important caveat: lexical F1 misses semantic correctness failures. Example:

- `idx=3`, question: dog type
- Pretrain answer: golden retriever (matches GT)
- Finetune answer: small fluffy white dog (semantic regression)
- Token overlap still scores slightly higher for finetune due to shared generic tokens.

Conclusion: aggregate token-F1 slightly improves, but factual grounding regressions exist.

## Likely Causes

1. **Benchmark blind spot**

- Final VQA can silently collapse to `{}` because missing files raise inside eval and are caught.
- This obscures whether run quality truly improved.

2. **Inference comparison confound**

- Pretrain checkpoints are evaluated with `use_qlora=False`.
- Finetune checkpoints are evaluated with `use_qlora=True`.
- This makes strict apples-to-apples interpretation harder.

3. **Stage 1/2 data overlap**

- Stage 1 captions are extracted from the same LLaVA instruction JSON used for Stage 2.
- Stage transition may have lower novelty signal than expected.

4. **Stage 2 optimization shape**

- 500 steps with cosine-to-zero means very low LR by late steps.
- Observed plateau aligns with low effective update near end.

5. **Objective mismatch**

- Train/eval loss improves modestly, but that objective may not track factual visual grounding quality on small spot-check sets.

## Code Locations Relevant to Fixes

- Inference phase split + QLoRA mode:
  - `/Users/babakd/anymal/modal_inference.py` (phase 1 uses `use_qlora=False`, phase 2 uses `use_qlora=True`)
- Stage 2 uses latest pretrain checkpoint by default:
  - `/Users/babakd/anymal/modal_train.py`
- Stage 2 trains projector + LoRA together:
  - `/Users/babakd/anymal/models/anymal.py`
- Optional LoRA-specific LR and cosine scheduler:
  - `/Users/babakd/anymal/training/trainer.py`
- VQA data caching (subset) + file-missing risk:
  - `/Users/babakd/anymal/modal_train.py`
- VQA dataset image loading + eval exception handling:
  - `/Users/babakd/anymal/evaluation/vqa_eval.py`
  - `/Users/babakd/anymal/evaluation/eval_runner.py`

## Recommended Next Actions (Concrete)

1. **Make final eval fail-loud and complete**

- Ensure VQA image subset includes all referenced image IDs for selected question set, or filter dataset to existing files before DataLoader workers run.
- Treat empty final metrics as a run failure condition.

2. **Remove inference confound**

- Evaluate both pretrain and finetune with identical runtime mode (`use_qlora` setting) for fair comparison.

3. **Add grounding-focused regression checks**

- Keep the 20-example qualitative set but add entity-sensitive checks (object type, count, color, breed).
- Do not rely only on lexical token overlap.

4. **Adjust Stage 2 training policy**

- Consider freezing projector for initial Stage 2 and tuning LoRA first.
- Consider non-zero LR floor or longer schedule so optimization does not effectively stop by step 500.

5. **Checkpoint selection policy**

- Prefer best checkpoint by held-out metric rather than latest step.

## Open Questions to Resolve Before Next Run

- Which failure mode is priority: semantic grounding accuracy vs fluency/detail?
- Should Stage 2 continue training projector or lock projector after Stage 1?
- Is the 20-example set fixed as regression gate, or should it be expanded and versioned?
