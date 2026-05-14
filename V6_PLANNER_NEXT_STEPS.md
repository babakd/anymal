# V6 Planner Handoff: Next Steps

Date: 2026-05-10

Audience: the planning agent that proposed the V6 causal falsification campaign.

## Bottom Line

The V6 goal was met. The campaign was executed through the preregistered gates,
and the simple "legacy V4 spatial connector caused the low-50s VQAv2 result"
story is falsified.

The decisive result is V3 with the same robust semantic-calibration recipe:

| Candidate | Clean mean | Perturb mean | POPE | GQA | Gate |
|---|---:|---:|---:|---:|---|
| V4 incumbent | 52.067 | 52.083 | 69.5 | 35.0 | incumbent |
| V6-R1b legacy V4 robust recipe | 52.489 | 52.083 | 69.9 | 37.2 | recipe win |
| V6-C1b V3 robust recipe | 61.078 | 60.108 | 77.1 | 43.8 | pass |
| V6-C2b V1 robust recipe | 52.956 | 51.933 | 69.2 | 36.8 | fails perturb mean by 0.150 |
| V6-A1 lean V4 robust transfer | 37.944 | 38.350 | 49.4 | 29.6 | fail |

V3 robust also passed the image-use gates:

- clean seed42: 59.667
- blank image: 37.400
- shuffled image: 33.600
- wrong-image same-answer-type: 36.800

Leakage checks passed for VQAv2 and the fetched POPE/GQA artifacts against the
audited training sources.

## What This Means

Do not promote a V4-family architecture claim from the prior V4/V5 result. The
score was not uniquely caused by the V4 spatial connector. The strongest current
causal read is:

1. recipe and evaluation hygiene matter a lot,
2. V3 was under-tested under the modern recipe,
3. Stage1B248 remains a checkpoint-selection risk,
4. lean V4 does not inherit the recipe gain,
5. architecture work should pause until grounding and checkpoint-selection
   questions are answered.

## Completed Campaign Artifacts

Primary campaign doc:

- `V6_CAUSAL_FALSIFICATION_PLAN.md`

Fetched local artifacts:

- `outputs/v6_remote/`

Useful scripts added or updated:

- `scripts/launch_v6_eval_bundle.py`
- `scripts/fetch_v6_eval_bundle.py`
- `scripts/summarize_v6_artifacts.py`
- `scripts/check_v6_campaign.py`
- `scripts/audit_vqa_leakage.py`

Important completed runs:

- R1b: `v6-r1b-roleclean-semanticcal-robust-acc16-bs4-lossscale003-from-legacy-v4-stage1b248`
- V3 robust: `v6-c1b-v3-roleclean-semanticcal-robust-acc16-bs4-lossscale003`
- V1 robust: `v6-c2b-v1-roleclean-semanticcal-robust-acc16-bs4-lossscale003`
- lean V4 robust: `v6-a1-lean-v4-roleclean-semanticcal-robust-bs4-lossscale001-from-a1stage1b400`

## Recommended Next Steps

### 1. Promote V3 Robust To Diagnostic Platform

Use V3 robust as the main diagnostic platform for the next phase. It is not yet
a final product claim, but it is the best current controlled model.

Next evals:

- rerun V3 robust on a larger VQAv2 slice,
- run all clean seeds and perturbations on the larger slice,
- expand POPE beyond the current adversarial 1k sample,
- run a larger GQA balanced diagnostic slice,
- add MME or MMBench only if the local harness is trusted.

### 2. Explain Why V3 Wins

The next causal question is no longer "does V4 spatial architecture cause the
gain?" It is "why did V3 with the same recipe jump so far?"

Suggested probes:

- compare V3 vs V4 prediction samples by answer type,
- inspect V3 gains on other/number questions, not just yes/no,
- compare top raw answers and answer-kind rates,
- analyze examples where V3 is correct and V4/R1b is wrong,
- test whether V3 is more visually grounded or just better calibrated.

### 3. Tighten Stage 1 Selection

Stage1B248 is still a live confound. Neighboring legacy V4 checkpoints varied
by more than two points under the same downstream recipe.

Next action:

- run at least two fresh Stage 1 seeds for the strongest candidate family,
- select checkpoints by preregistered fixed step, not best visible VQA,
- rerun the same Stage 2 recipe on those fixed checkpoints.

Do not make architecture claims from a single selected Stage 1 checkpoint.

### 4. Build Grounding And Counterfactual Training

The successful models pass blank/shuffle gates, but controls still score in the
mid/high 30s. That is enough to pass the conservative V6 gate, but not enough to
declare solved grounding.

Next data work:

- add explicit wrong-image and hard-negative image controls to training,
- include answer-type-balanced counterfactual examples,
- add POPE-style yes/no object probes to calibration,
- track blank/shuffle accuracy as a training metric, not just an eval afterthought.

### 5. Revisit V1 Only As A Diagnostic

V1 robust reached the incumbent clean band, which supports the recipe story, but
it missed perturbation mean by 0.150 and has interface comparability caveats.

Only revisit V1 if the goal is to answer one of these:

- does strict 64-placeholder splice close the robustness gap?
- is V1's residual weakness purely interface/plumbing?
- can V1 help isolate answer-prior behavior?

Do not treat V1 prepend-mode versus V4 splice-mode as a clean architecture
ablation.

## What Not To Do Next

- Do not run the lean V4 no-2D/compact96/local192 grid. Its precondition failed.
- Do not call V5-R0 or lean V4 an architecture win.
- Do not revive DeepStack-lite as the default path.
- Do not introduce vision-encoder finetuning before the current recipe,
  checkpoint, and grounding questions are closed.
- Do not tune many tiny robust-augmentation variants before expanding the V3
  diagnostic evidence.

## Suggested Next Planning Output

The next planner should produce a compact V7-style plan with this shape:

1. V3 robust confirmation on larger VQAv2, POPE, and GQA.
2. V3-vs-V4 error and answer-prior analysis.
3. multi-seed Stage 1 selection control.
4. counterfactual grounding data recipe.
5. only then, a fresh architecture proposal if the above still leaves an
   architecture-shaped gap.

The planner should explicitly retire the old question "is legacy V4 spatial
connector responsible for low-50s VQAv2?" and replace it with "what recipe,
checkpoint, and grounding choices explain V3 robust's jump to the low 60s?"
