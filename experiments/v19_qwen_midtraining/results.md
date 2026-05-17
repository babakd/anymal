# V19 Qwen3 Mid-Training Results

Last updated: 2026-05-17

## Objective

V19 tests whether the V18 + supplement 846k mid-training mixture can move the
Qwen3 V11 frontier checkpoint into a new Pareto region. The run starts from V11,
uses the V18 supplement mixture, applies elevated retention KL against the V11
teacher cache, preserves dense checkpoints, and evaluates candidates against the
V17 corrected harness with paired bootstrap.

## Current Verdict

Final outcome: **Class 4 regression**. Phase 0 preconditions passed and the
single 6000-step run completed, but the best candidate checkpoint (step 4000)
and the endpoint checkpoint (step 6000) both show large significant VQAv2
retention regressions without a statistically meaningful ChartQA/TextVQA
capability gain. V19 is operationally interpretable but is not promotion-grade
and does not replace V11.

## Phase 0 Preconditions

| Check | Outcome |
| --- | --- |
| V11 VQAv2 normalizer anomaly | Resolved as image-cache/sample-frame drift, not a normalizer regression |
| V18 supplement commit/tag | `517aaee` tagged `v18-supplement` |
| V18 supplement smoke checkpoint | `/checkpoints/v18_qwen/smoke_1step_supplement/v18-midtraining-supplement-smoke-1step/checkpoint-1` |
| Phase 0C committed-state smoke | Passed |

Phase 0A conclusion: V17/V11 VQAv2 clean n3000 must use the current V17 frame
baseline `63.8667`. The historical `66.xx` result used a different available
image/question frame after the COCO val2014 cache changed.

Phase 0C smoke:

| Field | Value |
| --- | --- |
| Modal app | `https://modal.com/apps/babakd/main/ap-GN2ArVMWKDJuHcnAEoiZXO` |
| W&B run | `https://wandb.ai/babakdam/anymal-pretrain/runs/shz7cx81` |
| Output checkpoint | `/checkpoints/v19_qwen/phase0c_smoke/v19-phase0c-v18sup-v11kl-cachev2-smoke-1step/checkpoint-1` |
| Teacher cache entries | `348796` |
| License posture | `aggregate_commercial_use_allowed=true` |

## Phase 1 Run

| Field | Value |
| --- | --- |
| Run name | `v19-qwen3-midtraining-v18sup-v11kl-w2-6000` |
| Modal app | `https://modal.com/apps/babakd/main/ap-o41VmL9i2cFG6N7OQdFa2P` |
| W&B runs | initial `https://wandb.ai/babakdam/anymal-pretrain/runs/wr5x9nl2`; resumed after preemption `https://wandb.ai/babakdam/anymal-pretrain/runs/x4unsvz1` |
| Output dir | `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000` |
| Start checkpoint | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` |
| Dataset | `v18_qwen_midtraining_stage1b` |
| Teacher cache | `/checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_v2.pt` |
| KL weight | `2.0` |
| Max steps | `6000` |
| Save steps | `500,1000,1500,2000,3000,4000,4500,6000` |

Observed training note: the worker was preempted after checkpoint 3000 and
resumed from that checkpoint. W&B split into the initial run `wr5x9nl2`
(`state=finished`, last W&B step 3600) and resumed run `x4unsvz1`
(`state=finished`, final W&B step 6000). Resume skipped the remainder of the
saved epoch before optimizer steps resumed in the next epoch; this was a
wall-time/data-order tax, not checkpoint corruption. Final resumed-run health:
`eval_loss=2.2178`, `grad_clip_fraction=0.235`, and
`placeholder_contract_valid=1.0`.

Final checkpoint metadata:

| Field | Value |
| --- | --- |
| Checkpoint | `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-6000` |
| Architecture | `anymal_v3` |
| LLM backbone | `Qwen/Qwen3-8B` |
| Vision tower | `SigLIP2-So400m-384` |
| Image tokens | `128` |
| Connector output scale | `1.125` |
| Patch-position feature scale | `0.0` |
| Metadata file | `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-6000/model_meta.json` |

## V17 Baselines

| Slice | Metric | V11 |
| --- | --- | ---: |
| GQA search n1000 | accuracy | 44.9000 |
| GQA confirm n3000 offset1000 | accuracy | 42.6000 |
| ChartQA val full | relaxed / exact | 14.6875 / 6.8229 |
| TextVQA validation full | exact / soft | 30.9400 / 28.4000 |
| POPE adversarial n1000 | accuracy | 80.3000 |
| POPE popular n1000 | accuracy | 81.8000 |
| VQAv2 clean n3000 seed42 | VQA accuracy | 63.8667 |
| VQAv2 blank n3000 seed42 | VQA accuracy | 37.9222 |
| VQAv2 shuffled n3000 seed42 | VQA accuracy | 36.1778 |
| VQAv2 wrong-image n3000 seed42 | VQA accuracy | 35.5222 |

## Phase 2A Cheap Screens

V11 n200 references: GQA `46.0`, ChartQA relaxed/exact `13.0/5.5`,
TextVQA exact/soft `30.0/29.0`, VQAv2 clean `65.8333`.

| Step | GQA n200 | ChartQA relaxed/exact | TextVQA exact/soft | VQAv2 clean | POPE adv/pop | Drift MSE | Drift cosine | V11 agree |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 44.5 | 14.0 / 5.5 | 31.5 / 30.0 | 38.5 | 82.5 / 82.0 | 8.2399e-06 | 0.996326 | 0.8125 |
| 1000 | 43.0 | 14.5 / 6.5 | 31.5 / 29.8333 | 37.3333 | 81.5 / 81.0 | 1.2981e-05 | 0.994216 | 0.8125 |
| 1500 | 43.5 | 14.5 / 6.5 | 31.5 / 29.8333 | 37.8333 | 81.0 / 81.0 | 1.6106e-05 | 0.992827 | 0.796875 |
| 2000 | 42.0 | 14.0 / 6.0 | 31.0 / 29.3333 | 40.1667 | 81.0 / 80.5 | 1.8885e-05 | 0.991592 | 0.78125 |
| 3000 | 43.0 | 14.5 / 6.5 | 31.5 / 29.8333 | 45.5 | 81.0 / 80.5 | 2.2799e-05 | 0.989854 | 0.78125 |
| 4000 | 44.0 | 14.5 / 6.5 | 32.0 / 29.8333 | 47.6667 | 81.0 / 80.5 | 2.4855e-05 | 0.988942 | 0.78125 |
| 4500 | 43.5 | 14.0 / 6.0 | 31.5 / 29.5 | 47.8333 | 80.5 / 80.5 | 2.5526e-05 | 0.988645 | 0.796875 |
| 6000 | 43.5 | 14.0 / 6.0 | 32.0 / 30.0 | 49.8333 | 81.5 / 81.0 | 2.6564e-05 | 0.988184 | 0.78125 |

Candidate rule status:

- Candidate A: maximum ChartQA relaxed n200. Final max is `14.5`, tied by
  steps 1000, 1500, 3000, and 4000.
- Candidate B: maximum `GQA + VQAv2 clean` among checkpoints with ChartQA
  relaxed n200 above `14.0`. Step 4000 is the best eligible
  checkpoint (`44.0 + 47.6667`).
- Endpoint: step 6000 was evaluated regardless of candidate status. It is not
  a candidate because ChartQA relaxed n200 returned to `14.0`.

## Phase 2C Confirmation

| Step | GQA search | GQA confirm | ChartQA relaxed/exact | TextVQA exact/soft | POPE adv/pop | VQAv2 clean | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3000 | 41.4000 | 42.1000 | 14.8958 / 6.8750 | 31.5200 / 28.7533 | 80.1000 / 81.1000 | 45.7556 | Provisional screen; no longer a top candidate |
| 4000 | 41.1000 | 41.6333 | 14.8438 / 6.8229 | 31.4600 / 28.6400 | 80.0000 / 81.4000 | 47.9667 | Final candidate A/B |
| 6000 | 41.1000 | 41.6667 | 14.7917 / 6.8229 | 31.4000 / 28.6600 | 80.2000 / 81.3000 | 50.0556 | Endpoint; not a candidate |

VQAv2 control confirmation:

| Step | Clean | Blank | Shuffled | Wrong image | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3000 | 45.7556 | skipped | skipped | skipped | Controls skipped after step 3000 fell out of top candidate status |
| 4000 | 47.9667 | 17.6444 | 31.8000 | 31.5333 | Controls confirm broad VQAv2 retention loss |
| 6000 | 50.0556 | 18.2333 | 32.5222 | 32.0111 | Endpoint improves over step 4000 but remains far below V11 |

## Phase 2D Paired Bootstrap

Step 4000 paired-bootstrap comparisons against V11 are complete. Values below
come from `scripts/v17_paired_artifact_compare.py` with 10,000 paired
resamples. The VQAv2 wrong-image bootstrap recomputes per-row accuracy from
prediction samples and therefore differs slightly from the artifact-level
summary metric in Phase 2C.

| Step | Slice | Metric | V19 | V11 | Delta | 95% CI | Significant |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 4000 | GQA search | accuracy | 41.1000 | 44.9000 | -3.8000 | [-6.0000, -1.6000] | yes |
| 4000 | GQA confirm | accuracy | 41.6333 | 42.6000 | -0.9667 | [-2.1000, +0.1667] | no |
| 4000 | ChartQA full | relaxed | 14.8438 | 14.6875 | +0.1563 | [-0.6250, +0.9375] | no |
| 4000 | ChartQA full | exact | 6.8229 | 6.8229 | +0.0000 | [-0.6771, +0.6771] | no |
| 4000 | TextVQA full | exact | 31.4600 | 30.9400 | +0.5200 | [-0.2200, +1.2400] | no |
| 4000 | TextVQA full | soft | 28.6400 | 28.4000 | +0.2400 | [-0.4467, +0.9202] | no |
| 4000 | POPE adversarial | accuracy | 80.0000 | 80.3000 | -0.3000 | [-1.9000, +1.3000] | no |
| 4000 | POPE popular | accuracy | 81.4000 | 81.8000 | -0.4000 | [-1.9000, +1.1000] | no |
| 4000 | VQAv2 clean | VQA accuracy | 47.9667 | 63.8667 | -15.9000 | [-17.4111, -14.3778] | yes |
| 4000 | VQAv2 blank | VQA accuracy | 17.6444 | 37.9222 | -20.2778 | [-21.8000, -18.7667] | yes |
| 4000 | VQAv2 shuffled | VQA accuracy | 31.8000 | 36.1778 | -4.3778 | [-5.3556, -3.4111] | yes |
| 4000 | VQAv2 wrong image | VQA accuracy | 31.5111 | 35.5222 | -4.0111 | [-5.0111, -3.0222] | yes |
| 6000 | GQA search | accuracy | 41.1000 | 44.9000 | -3.8000 | [-6.0000, -1.6000] | yes |
| 6000 | GQA confirm | accuracy | 41.6667 | 42.6000 | -0.9333 | [-2.0667, +0.2000] | no |
| 6000 | ChartQA full | relaxed | 14.7917 | 14.6875 | +0.1042 | [-0.6771, +0.8333] | no |
| 6000 | ChartQA full | exact | 6.8229 | 6.8229 | +0.0000 | [-0.6771, +0.6771] | no |
| 6000 | TextVQA full | exact | 31.4000 | 30.9400 | +0.4600 | [-0.2600, +1.2000] | no |
| 6000 | TextVQA full | soft | 28.6600 | 28.4000 | +0.2600 | [-0.4267, +0.9400] | no |
| 6000 | POPE adversarial | accuracy | 80.2000 | 80.3000 | -0.1000 | [-1.7000, +1.5000] | no |
| 6000 | POPE popular | accuracy | 81.3000 | 81.8000 | -0.5000 | [-2.0000, +0.9000] | no |
| 6000 | VQAv2 clean | VQA accuracy | 50.0556 | 63.8667 | -13.8111 | [-15.2669, -12.3444] | yes |
| 6000 | VQAv2 blank | VQA accuracy | 18.2333 | 37.9222 | -19.6889 | [-21.1781, -18.1778] | yes |
| 6000 | VQAv2 shuffled | VQA accuracy | 32.5222 | 36.1778 | -3.6556 | [-4.6114, -2.7111] | yes |
| 6000 | VQAv2 wrong image | VQA accuracy | 31.9889 | 35.5222 | -3.5333 | [-4.5111, -2.5556] | yes |

## Phase 2E Pairwise Taxonomy

Diagnostic taxonomy artifacts are written under
`experiments/v19_qwen_midtraining/taxonomy_step4000/` and
`experiments/v19_qwen_midtraining/taxonomy_step6000/`.

GQA row-delta counts:

| Step | Slice | V19 correct / V11 wrong | V11 correct / V19 wrong | Main pattern |
| ---: | --- | ---: | ---: | --- |
| 4000 | GQA search | 44 | 82 | V11 recovers more spatial-relation and left/right rows |
| 4000 | GQA confirm | 142 | 171 | Same direction, smaller net gap |
| 6000 | GQA search | 43 | 81 | V11 recovers more spatial-relation and left/right rows |
| 6000 | GQA confirm | 140 | 168 | Same direction, smaller net gap |

VQAv2 row-browser delta counts:

| Step | Slice | V19 correct / V11 wrong | V11 correct / V19 wrong |
| ---: | --- | ---: | ---: |
| 4000 | clean | 93 | 640 |
| 4000 | blank | 50 | 748 |
| 4000 | shuffled | 57 | 223 |
| 4000 | wrong-image | 76 | 227 |
| 6000 | clean | 100 | 579 |
| 6000 | blank | 55 | 731 |
| 6000 | shuffled | 59 | 202 |
| 6000 | wrong-image | 74 | 206 |

## Outcome Classification

**Class 4: regression.** Step 4000 is the final candidate A/B checkpoint and
step 6000 is the required endpoint. Neither has a significant capability gain:
ChartQA relaxed/exact and TextVQA exact/soft all have paired-bootstrap CIs that
include zero. Both have significant retention losses: GQA search regresses and
all VQAv2 clean/control slices regress significantly. POPE stays statistically
neutral, but it is not enough to offset the VQAv2 and GQA losses.

Failure-mode read:

- The connector drift trajectory grows monotonically but remains numerically
  modest: drift MSE rises from `8.2399e-06` at step 500 to `2.6564e-05` at step
  6000, cosine falls from `0.996326` to `0.988184`, and V11 probe answer
  agreement stays around `0.78`.
- The severe VQAv2 regression appears early and only partially recovers by the
  endpoint (`47.9667` at step 4000, `50.0556` at step 6000, versus V11
  `63.8667`).
- Generation hygiene is not the primary failure: EOS rates are acceptable on
  the confirmed slices, with no max-token collapse. The failure is a retention
  and answer-distribution shift, not a basic decoding failure.

V20 recommendation: follow the Class 4 debug path before launching another full
mid-training run. First inspect V19 endpoint answer distributions and VQAv2 row
losses, then relate the monotonic connector drift to the regression trajectory.
If a follow-up training diagnostic is still needed, run the plan's reduced
`KL=1.0` first-1500-step probe only as a controlled debug comparison, not as a
promotion attempt.

No `docs/STATUS.md` update is made because V11 remains the project frontier.

## Artifact Index

| Artifact | Path |
| --- | --- |
| Phase 0A diagnostic | `/checkpoints/v19_qwen/phase0a_vqa_normalizer_diagnostic.json` |
| Phase 0C smoke checkpoint | `/checkpoints/v19_qwen/phase0c_smoke/v19-phase0c-v18sup-v11kl-cachev2-smoke-1step/checkpoint-1` |
| Primary checkpoints | `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-*` |
| Cheap screens | `/checkpoints/v19_qwen/cheap_screens/step*/` |
| V17 confirmation screens | `/checkpoints/v19_qwen/v17_fixed_harness/step*/` |
| V17 V11 baselines | `/checkpoints/v17_fixed_harness/v11/` |
| Step 4000 paired bootstrap | `experiments/v19_qwen_midtraining/paired_step4000/` |
| Step 4000 taxonomy | `experiments/v19_qwen_midtraining/taxonomy_step4000/` |
| Step 6000 paired bootstrap | `experiments/v19_qwen_midtraining/paired_step6000/` |
| Step 6000 taxonomy | `experiments/v19_qwen_midtraining/taxonomy_step6000/` |
| V18 all-source leakage audit | `/checkpoints/v18_qwen/audits/all_sources_vs_v17_v11_evals.json` |
| V18 supplement leakage audit | `/checkpoints/v18_qwen/audits/supplement_lcs_vg_vs_v17_v11_evals_official_script.json` |
| V18 final mixture weights | `/checkpoints/v18_qwen/final_mixture_weights.json` |
| V18 source manifest/license summary | `/checkpoints/v18_qwen/source_manifest.json` |
