# AnyMAL V6 Causal Falsification Campaign

Date: 2026-05-09

## Position

V6 starts as a preregistered ablation suite, not as a new connector class.
V4/V5 show that AnyMAL can reach the low 50s on the locked VQAv2 1k slice, but
they do not isolate the cause. The live causal hypotheses are:

1. semantic-calibration recipe,
2. the specific legacy V4 Stage1B248 checkpoint,
3. corrected left-padded evaluation and prompt hygiene,
4. answer priors and yes/no calibration,
5. the V4 spatial connector architecture,
6. robustness augmentation.

The first V6 objective is to falsify the architecture story with the fewest
runs possible. Only after that should V6 mean a new architecture platform.

## Local Evidence Read

Primary local handoffs reviewed:

- `V4_RESEARCH_RECIPE_20260508.md`
- `V4_ARCHITECTURE_PLAN.md`
- `V5_RESEARCH_PLAN_20260509.md`
- `V4_V5_ABLATION_AGENT_BRIEF.md`
- `v5_progress.md`
- current V4/V5 left-padded VQAv2 artifacts

Re-run incumbent facts under the V6 prompt/eval contract:

| Candidate | Seed 42 | Seed 43 | Seed 44 | Mean | Notes |
|---|---:|---:|---:|---:|---|
| V4 semantic-cal Stage1B248 | 52.133 | 52.933 | 51.133 | 52.067 | left-padded, strict=clean, prefix 0 |
| V5-R0 role-clean semantic-cal Stage1B248 | 51.933 | 52.467 | 51.033 | 51.811 | left-padded, strict=clean, prefix 0 |

Incumbent seed-42 robustness and image-use facts:

| Metric | V4 | V5-R0 | Interpretation |
|---|---:|---:|---|
| perturbation mean | 52.083 | 51.875 | V5-R0 loses robustness to V4 |
| mild blur | 51.600 | 51.467 | V5-R0 clean margin reverses |
| blank image | 37.233 | 37.100 | both pass conservative image-use gate |
| shuffled image | 36.300 | 36.433 | both pass conservative image-use gate |
| wrong-image same answer type | 36.433 | 36.333 | both pass conservative image-use gate |
| POPE adversarial | 69.500 | 69.100 | V4 remains the stronger incumbent |
| GQA diagnostic 500 | 35.000 | 34.800 | V4 remains slightly stronger |

Conclusion: V5-R0 is a small recipe/eval artifact win, not a robust recipe win
and not an architecture win.

## Live Campaign Results

### V6-R1b Robust Recipe Win

Run:

- `v6-r1b-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-legacy-v4-stage1b248`
- checkpoint:
  `/checkpoints/finetune-output/v6-r1b-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-legacy-v4-stage1b248/checkpoint-100`
- W&B: `babakdam/anymal-finetune/8vxp4jpu`
- health: no W&B alerts, no loss/grad spike flags, grad clipping `0.0`,
  eval loss improved from `0.843548` at step 50 to `0.840265` at step 100.

| VQAv2 eval | Accuracy |
|---|---:|
| clean seed 42 | 52.533 |
| clean seed 43 | 51.433 |
| clean seed 44 | 53.500 |
| clean mean | 52.489 |
| resize up | 52.500 |
| mild blur | 51.600 |
| center crop 90 | 51.967 |
| translate 5 pct | 52.267 |
| perturbation mean | 52.083 |
| blank image | 38.400 |
| shuffled image | 35.667 |
| wrong-image same answer type | 36.467 |

Generation hygiene is clean on every R1b VQAv2 artifact:
`strict=clean`, EOS `1.0`, max-token-hit `0.0`, assistant-prefix `0.0`.
The VQAv2 leakage audit passed with `1571` eval image IDs and zero exact or
numeric overlap against the default V5 calibration sources.

Second benchmark checks:

| Benchmark | Result | Notes |
|---|---:|---|
| POPE adversarial 1k | 69.900 accuracy / 0.715 F1 | beats V4/V5-R0 accuracy |
| GQA diagnostic 500 | 37.200 exact match | beats V4 diagnostic baseline 35.000 |

Decision: R1b is a V6 recipe win. The robust semantic-calibration recipe is the
active control recipe for downstream transfers.

### Recipe-Transfer Controls

| Experiment | Base | Health | Clean mean | Perturb mean | Image controls | Second benchmark |
|---|---|---|---:|---:|---|---|
| V6-C1 V3 + semantic recipe | V3 checkpoint-1500 | yellow: early step-8 loss spike | 61.056 | 59.875 | blank 37.300, shuffled 34.233, wrong 36.400 | POPE 76.100 / GQA 44.200 |
| V6-C2 V1 + semantic recipe | V1 checkpoint-2500 | green | 52.633 | 51.917 | blank 39.433, shuffled 36.333, wrong 36.433 | POPE 66.700 / GQA 36.600 |

Decision: V3 already strongly falsifies a simple "legacy V4 spatial connector
caused the low-50s result" story, even though its training health marks it
diagnostic-only. V1 also reaches the incumbent clean regime with the modern
recipe, but has a POPE yes-bias weakness.

Robust-control follow-ups completed after R1b passed:

| Experiment | Health | Clean seeds 42/43/44 | Clean mean | Perturb mean | Image controls | Second benchmark | Gate |
|---|---|---:|---:|---:|---|---|---|
| V6-C1b V3 + robust recipe | green | 59.667 / 61.900 / 61.667 | 61.078 | 60.108 | blank 37.400, shuffled 33.600, wrong 36.800 | POPE 77.100 / GQA 43.800 | PASS |
| V6-C2b V1 + robust recipe | green | 52.467 / 53.200 / 53.200 | 52.956 | 51.933 | blank 38.700, shuffled 34.800, wrong 36.267 | POPE 69.200 / GQA 36.800 | FAIL: perturb mean -0.150 vs V4 |

V3 robust hygiene is perfect on VQAv2: `strict=clean`, EOS `1.0`,
max-token-hit `0.0`, assistant-prefix `0.0`, and predicted yes/no on
non-yes/no questions stays below the V6 limit. It passes blank, shuffled,
wrong-image, perturbation-mean, and single-perturbation gates. The VQAv2
leakage audit passed with `1571` eval image IDs and zero exact or numeric
overlap against the audited training sources.

V1 robust passes generation hygiene and image-use gates, but is not a robust
promotion because its seed-42 perturbation mean is `51.933` versus the V4
incumbent `52.083`. It still strengthens the causal recipe story: even the old
V1 line reaches the V4/V5 clean regime under the modern semantic recipe. Its
VQAv2 leakage audit also passed with zero exact or numeric overlap.

The combined POPE/GQA second-benchmark leakage audit over fetched V6 artifacts
passed with `718` unique image IDs and zero exact or numeric overlap against the
audited training sources.

Decision: V3 robust is the decisive V6 recipe-transfer control. It beats V4/V5
on clean VQAv2, perturbation mean, POPE, and GQA while showing large drops on
blank/shuffled/wrong images. The simple claim that the low-50s score was caused
by the legacy V4 spatial connector is falsified.

### Stage1B248 Selection Control

Identical semantic-calibration Stage 2 from neighboring legacy V4 checkpoints:

| Stage1B checkpoint | Clean seed-42 accuracy | Health |
|---|---:|---|
| checkpoint-62 | 47.967 | green |
| checkpoint-124 | 50.767 | green |
| checkpoint-186 | 50.733 | green |
| checkpoint-248 | 52.533 under R1b robust recipe | green R1b, yellow historical selection risk |

Decision: neighboring checkpoints work, but quality varies by more than two
points. Stage1B248 selection risk remains live and architecture claims must not
rest on one selected Stage 1 checkpoint.

### Lean Transfer

The lean V4 transfer completed with the robust recipe:

- `v6-a1-lean-v4-roleclean-semanticcal-robust-bs4-lossscale001-from-a1stage1b400`
- checkpoint:
  `/checkpoints/finetune-output/v6-a1-lean-v4-roleclean-semanticcal-robust-bs4-lossscale001-from-a1stage1b400/checkpoint-100`
- W&B: `babakdam/anymal-finetune/wwq8a91o`
- base:
  `/checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400`
- metadata: lean `spatial_perceiver_resampler`, 64 global + 64 local tokens,
  3 layers, 8 heads, FF mult 2, hidden dim 1024, output gate init `0.0001`,
  2D position features on, projected through the bottleneck path.
- health: no W&B alerts, no loss/grad spike flags, eval loss improved from
  `1.334049` to `1.269860`, generation hygiene was clean enough for diagnosis.

| VQAv2 eval | Accuracy |
|---|---:|
| clean seed 42 | 38.667 |
| clean seed 43 | 38.400 |
| clean seed 44 | 36.767 |
| clean mean | 37.944 |
| resize up | 38.533 |
| mild blur | 38.200 |
| center crop 90 | 38.333 |
| translate 5 pct | 38.333 |
| perturbation mean | 38.350 |
| blank image | 37.967 |
| shuffled image | 37.600 |
| wrong-image same answer type | 38.467 |

Second benchmark checks:

| Benchmark | Result | Notes |
|---|---:|---|
| POPE adversarial 1k | 49.400 accuracy / 0.106 F1 | severe no-bias, yes ratio 0.053 |
| GQA diagnostic 500 | 29.600 exact match | below all V4/V5/V3/V1 controls |

Gate result: FAIL. Blank, shuffled, and wrong-image controls are all within
`1.1` points of clean accuracy, and perturbation mean is `13.733` points below
the V4 incumbent. The VQAv2 leakage audit passed, so this is a model/control
failure rather than a contamination finding.

Decision: lean V4 does not inherit the recipe gain and is not the V6
architecture platform. Do not run the V6-A2/A3/A4 no-2D/token-budget grid,
because its preregistered precondition failed.

## Causal Conclusion

V6 should remain a falsification campaign, not a new connector class. The
current evidence supports a recipe/evaluation/control story, not a legacy V4
spatial-architecture win:

- R1b establishes a robust recipe win on the legacy V4 Stage1B248 base.
- V3 plus the same recipe exceeds V4/V5 by a large margin on clean VQAv2,
  perturbations, POPE, and GQA.
- V1 plus the same recipe reaches the incumbent clean regime, even with the V1
  interface caveat.
- Stage1B248 remains a checkpoint-selection risk because neighboring legacy V4
  checkpoints vary by more than two points.
- lean V4 fails both score and image-use gates, so the lean architecture grid is
  skipped.

Next work should use V3 robust as the strongest diagnostic platform and focus on
grounding, counterfactual/image-use training, POPE/GQA expansion, and
multi-seed Stage 1 selection, rather than promoting a V4-family architecture
claim.

## Non-Negotiable Measurement Contract

Every V6 promotion-eligible checkpoint must produce the same artifact bundle:

- VQAv2 val2014 locked 1000-sample clean eval on seeds `42`, `43`, `44`.
- `prompt_style=training_chat`.
- left-padded decoder generation.
- `prediction_samples=1000`.
- strict accuracy and cleaned accuracy.
- raw answer and cleaned answer in prediction samples.
- per-answer-type accuracy for `number`, `other`, and `yes/no`.
- EOS rate and max-token-hit rate.
- assistant-role prefix rate overall and by answer type.
- predicted answer-kind rates by ground-truth answer type.
- top raw answers and top cleaned answers.
- original `image_id` and, for controls, `source_image_id`.

Seed-42 image-use and robustness controls:

- clean,
- `resize_up`,
- `mild_blur`,
- `center_crop_90`,
- `translate_5pct`,
- `blank_image`,
- `shuffled_image`,
- `wrong_image_same_answer_type`.

Artifact schema fields added for V6:

- `eval_schema_version = "v6"`
- `padding_side = "left"`
- `generation_mode = "decoder_leftpad_greedy"`
- `system_prompt`
- `prompt_style`
- `seed`
- `max_samples`
- `prediction_samples`
- `candidate_checkpoint`
- `candidate_architecture`
- `model_meta`
- `connector_meta`
- `dataset_meta`
- `train_source_meta`
- `image_transform_meta`

Leakage audits are promotion blockers. Run `scripts/audit_vqa_leakage.py` for
every VQAv2 eval artifact with full prediction samples. POPE and GQA require
their own COCO / Visual Genome / GQA overlap checks before promotion.

## Gates

Generation hygiene gates:

- strict-clean gap `<= 1.0` point, ideally `0.0`,
- assistant-prefix rate `<= 0.01` overall,
- assistant-prefix rate `<= 0.02` in every answer-type bucket,
- EOS `>= 0.98`,
- max-token-hit rate `<= 0.02`,
- predicted yes/no on non-yes/no questions `<= 0.05`,
- full prediction samples exist.

Image-use gates:

- blank-image accuracy must be at least `8` points below correct-image accuracy,
- shuffled-image accuracy must be at least `8` points below correct-image accuracy,
- wrong-image same-answer-type should also show a material drop and is mandatory
  context for answer-prior interpretation.

Robustness gates:

- clean mean across seeds `42/43/44` must meet or beat the V4/V5 incumbent,
- seed-42 perturbation mean must meet or beat V4,
- no single perturbation may lose by more than `1.0` point unless offset by a
  trusted second-benchmark win,
- mild blur must not regress.

Training health gates:

- active W&B alerts mean no promotion,
- yellow-history checkpoints are diagnostic only,
- recent loss spikes, recent grad spikes, missing checkpoint artifacts, or
  missing validation logs make a run incomplete,
- checkpoint selection rule must be declared before training.

## Ranked Experiment Sequence

### V6-E0: Lock Measurement And Image-Use Controls

Evaluate V4 and V5-R0 with the V6 artifact schema:

- clean VQAv2 seeds `42`, `43`, `44`,
- seed-42 perturbations `resize_up`, `mild_blur`, `center_crop_90`,
  `translate_5pct`,
- seed-42 image controls `blank_image`, `shuffled_image`,
  `wrong_image_same_answer_type`,
- leakage audit for every artifact,
- POPE disjoint subset and GQA diagnostic slice once harness paths are ready.

Decision:

- If blank/shuffled accuracy is within `8` points of clean, V6 pivots to
  grounding and counterfactual training before connector work.
- If V5-R0 loses perturbation mean to V4, it is not the recipe control for
  architecture promotion.

### V6-R1b: Robust Recipe On Legacy V4 Stage1B248

Relaunch from scratch:

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
  --run-name v6-r1b-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-legacy-v4-stage1b248
```

At step 50, require empty W&B alerts, near-zero clipping, no eval-loss
regression worse than V5-R0 by more than `0.05` unless VQA smoke improves, and a
250-sample clean plus mild-blur smoke with EOS/max-token hygiene intact.

If R1b fails, run exactly one fallback:

```bash
modal run --detach modal_train.py \
  --architecture anymal_v4 \
  --stage finetune \
  --dataset v5_semantic_calibration_robust \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.01 \
  --finetune-gradient-accumulation-steps 16 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex/checkpoint-248 \
  --freeze-connector \
  --use-wandb \
  --run-name v6-r1c-roleclean-semanticcal-robust-acc16-bs4-lossscale001-from-legacy-v4-stage1b248
```

### V6-C1-V3: V3 Plus V5 Semantic Recipe

This is the highest-value causal control after R1b. Locate a clean
connector-only V3 Stage 1 or Stage 1B checkpoint first. Do not use a Stage 2
LoRA checkpoint unless the run is explicitly marked contaminated.

Resolved Modal-volume candidate:

- `/checkpoints/pretrain-output/v3-stage1b-grounding-128tok-3000-20260430/checkpoint-1500`
- Metadata confirms `architecture=anymal_v3`, `connector_type=perceiver_resampler`,
  `num_image_tokens=128`, `connector_layers=6`, `connector_heads=16`,
  `connector_ff_mult=4`, `project_directly_to_llm_dim=true`, and no LLM
  checkpoint payload. Treat as the preferred V3 connector-only control base
  unless W&B history later marks it red.

```bash
modal run --detach modal_train.py \
  --architecture anymal_v3 \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 8 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v3-stage1b-grounding-128tok-3000-20260430/checkpoint-1500 \
  --freeze-connector \
  --use-wandb \
  --run-name v6-c1-v3-roleclean-semanticcal-bs4-lossscale003
```

If V3 reaches within `1.0` point of the V4/V5 clean mean and does not lose
robustness, the V4 spatial-architecture story is largely falsified.

### V6-C2-V1: V1 Plus V5 Semantic Recipe

Locate a connector-only V1 Stage 1 checkpoint. The current V1 path historically
uses prepend mode; a comparable V6 control should use a strict 64-placeholder
splice interface if plumbing permits. If not, mark the control contaminated.

Resolved Modal-volume candidate:

- `/checkpoints/pretrain-output/checkpoint-2500`
- This is the legacy Stage 1 projector checkpoint used by the original V1 line.
  It has no `model_meta.json`, so the current metadata helper treats it as
  `anymal_v1`. Because the old V1 path is prepend-mode by default, this is a
  valid recipe-transfer diagnostic but not a clean architecture ablation unless
  strict 64-placeholder splice plumbing is added.

```bash
modal run --detach modal_train.py \
  --architecture anymal \
  --stage finetune \
  --dataset v5_semantic_calibration \
  --max-steps 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-learning-rate 1e-5 \
  --finetune-loss-scale 0.03 \
  --finetune-gradient-accumulation-steps 8 \
  --pretrain-checkpoint /checkpoints/pretrain-output/checkpoint-2500 \
  --freeze-connector \
  --use-wandb \
  --run-name v6-c2-v1-roleclean-semanticcal-bs4-lossscale003
```

### V6-C3-STAGE1SEL: Stage1B248 Selection Control

Run identical Stage 2 from available neighboring legacy V4 checkpoints,
preferably `checkpoint-200`, `checkpoint-248`, and `checkpoint-300`.

If only `checkpoint-248` works, V6 must require multiple Stage 1 seeds before
claiming architecture. If neighboring checkpoints vary by more than `2` points,
architecture comparisons must repeat across Stage 1 seeds.

### V6-A1-LEAN: Lean V4 Transfer

Only after the recipe-transfer controls are interpretable:

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
  --finetune-gradient-accumulation-steps 8 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v4-stage1b-clean500-from-a1ckpt300-20260509-codex/checkpoint-400 \
  --freeze-connector \
  --use-wandb \
  --run-name v6-a1-lean-v4-roleclean-semanticcal-bs4-lossscale001-from-a1stage1b400
```

If generation hygiene fails, run only one stabilization variant at LR `5e-6`,
loss scale `0.005`, and accumulation `16`.

### V6-A2/A3/A4: Lean Architecture Attribution

Run only if lean V4 inherits the recipe gain and passes V6 gates:

- `NO2D`: 64 global + 64 local, 2D positions off,
- `COMPACT96`: 48 global + 48 local, 96 tokens,
- `LOCAL192`: 64 global + 128 local, 192 tokens.

Use fixed Stage1A/Stage1B/Stage2 checkpoint budgets across variants. Do not
select different "best" internal checkpoints per architecture.

## Canonical V6 Eval Command

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/<run-name>/checkpoint-100 \
  --candidate-label <label> \
  --candidate-architecture <anymal_v1_or_v3_or_v4> \
  --no-include-baselines \
  --max-samples 1000 \
  --seed 42 \
  --prompt-style training_chat \
  --system-prompt 'Answer with only the final answer. Do not include role labels, explanations, or the word assistant. End after the answer.' \
  --prediction-samples 1000 \
  --eval-schema-version v6 \
  --output <artifact>.json
```

For controls add one of:

```bash
--image-perturbation resize_up
--image-perturbation mild_blur
--image-perturbation center_crop_90
--image-perturbation translate_5pct
--image-perturbation blank_image
--image-perturbation shuffled_image
--image-perturbation wrong_image_same_answer_type
```

Run the V6 gate checker after artifacts exist:

```bash
python3 scripts/check_v6_campaign.py \
  --clean vqa_eval_<candidate>_seed42_clean_leftpad.json \
  --blank vqa_eval_<candidate>_seed42_blankimage_leftpad.json \
  --shuffled vqa_eval_<candidate>_seed42_shuffleimage_leftpad.json \
  --wrong-image vqa_eval_<candidate>_seed42_wrongimage_sameanswertype_leftpad.json
```

## Resolved Setup Notes

The Modal volume contains the exact V3 and V1 candidate bases needed for the
first recipe-transfer controls. The remaining blocker is not path discovery; it
is V1 interface comparability. Do not call V1 prepend-mode vs V4 splice-mode a
clean architecture result.
