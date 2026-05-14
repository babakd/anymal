# V14 Qwen3 Focused Substrate Attempt

Date: 2026-05-14

## Phase 0 Decision Memo

primary substrate selected:

```text
spatial-preserving compressed grid
```

backup substrate selected:

```text
AnyRes / MLP pass-through, only if the primary substrate either cannot imitate
V11 after the minimum exposure or fails for a clearly substrate-specific reason.
```

why this substrate is likely to escape V11:

```text
V11's 128-token Perceiver is the best current Qwen basin, but it compresses
SigLIP2 patches into learned latents that can blur fine spatial layout. The
spatial-grid connector keeps a fixed 16x16 visual grid and raises the visual
budget to 256 tokens while preserving the V11 scale contract. If it can first
imitate V11, it should have a plausible path to stronger compositional and
fine-grained reasoning than the 128-token Perceiver.
```

what specifically was wrong or incomplete about prior related attempts:

```text
The V13 spatial-grid and AnyRes artifacts were partial and shallow. They did
not use a V14 teacher-distribution cache, did not record full imitation health
metrics, and visible runs stopped at 500 steps despite 1500-step run names.
They are useful negative context but do not satisfy V14 validity requirements.
The V13 visual-cross-attention branch imitated V11 on small GQA slices but did
not beat matched search/confirm.
```

what changes in V14:

```text
V14 adds cached V11 top-k + remainder answer-token distributions, carries
sample metadata through the instruction dataset/collator, logs cache-specific
imitation metrics, and uses a replay-style Phase 1 dataset with VQA/GQA/COCO/
short-answer/POPE examples. The primary run is fresh V14 evidence, not a reused
V13 artifact.
```

architecture changes relative to V11:

```text
V11: AnyMALv3, frozen SigLIP2 384, 128-token Perceiver projector, Qwen/Qwen3-8B,
connector output scale 1.125, patch-position feature scale 0.0.

V14 primary: AnyMALv3, frozen SigLIP2 384, spatial_grid_projector, 256 image
tokens, Qwen/Qwen3-8B, connector output scale 1.125, 2D coord-MLP patch
position features. The LLM remains frozen during Phase 1.
```

expected token count:

```text
student visual tokens: 256
teacher visual tokens for cached V11 KL: 128
```

placeholder contract changes needed:

```text
The student prompt uses 256 image placeholders. Cached teacher rows are built
with the V11 128-placeholder contract. The supervised answer token sequence is
checked at training time before applying cached KL, so placeholder-count
differences cannot silently misalign the answer-token loss.
```

teacher KL cache status:

```text
Implemented schema v14_teacher_kl_topk_v1 with top-k token ids/probabilities
plus teacher remainder probability mass per supervised answer position.
Two-entry Modal smoke succeeded:
/checkpoints/v14_qwen/smoke_v14_v11_teacher_topk8_train.pt

Full top-k=128 train-cache build completed after adding source-local indices
to mixture sample IDs:
/checkpoints/v14_qwen/v14_v11_teacher_topk128_train_unique.pt

entries: 6650
answer tokens: 23794
top-k: 128

Do not use the earlier non-unique full cache path for V14 training; it produced
6648 entries because two mixture sample IDs collided.
```

data sources used:

```text
v14_qwen_imitation_replay_stage1b, concat mixture:
- VQAv2 train direct-answer replay, 2000 samples
- GQA train_balanced direct-answer replay, 2000 samples
- COCO object direct-answer replay, 1200 samples
- LLaVA short direct-answer replay, 600 samples
- POPE-style COCO presence replay, 600 samples
- POPE-style COCO absence replay, 600 samples

The training split is the deterministic 95/5 split used by the Modal pretrain
path: 6650 train examples and 350 validation examples.
```

leakage risks:

```text
GQA train_balanced examples are present in the imitation replay mixture. V14
claims on GQA testdev/search/confirm must include a leakage audit over eval
image IDs, question IDs, and raw refs against the V14 GQA replay source.
```

minimum imitation exposure:

```text
1500 optimizer steps with at least three evaluated checkpoints. Initial save
cadence is 250 steps so the run can be diagnosed before spending the full
budget if KL or generation hygiene is clearly broken.
```

minimum compositional exposure:

```text
Phase 2 is not allowed until Phase 1 health gates pass: KL decline across three
eval checkpoints, improving V11 agreement, GQA search within 1 point of V11,
no VQA collapse, and clean generation hygiene. If those gates pass, run a
focused compositional continuation rather than a broad branch sweep.
```

How this differs from V12/V13:

```text
V12/V13 mostly explored local edits, short continuations, and partial new
substrates without a cached teacher-distribution contract or a full imitation
health gate. V14 treats imitation as the first-order validity criterion: the
new substrate must reproduce V11's answer-token distribution before any claim
about exceeding V11 is meaningful.
```

## Phase 0 Tooling

- Added batch `sample_id`, image, question, answer, and mixture-source metadata
  to instruction datasets and collator outputs.
- Added cached V11 teacher-KL loading and top-k + remainder KL in
  `training/pretrain.py`.
- Added cache metadata to AnyMALv3 checkpoints.
- Added V14 replay dataset `v14_qwen_imitation_replay_stage1b`.
- Added Modal cache builder `scripts/modal/v14_teacher_cache.py`.

Local checks:

```bash
python3 -m compileall -q models training evaluation data scripts
git diff --check
python3 scripts/repo_health_check.py
```

Status: passed.

Modal cached-KL smoke:

```text
run: v14-qwen3-spatialgrid256-cachekl-smoke1
url: https://modal.com/apps/babakd/main/ap-ko3HFXe3tiokSudbj0BAl4
checkpoint: /checkpoints/pretrain-output/v14-qwen3-spatialgrid256-cachekl-smoke1/checkpoint-1
result: loaded cache entries=6650/top_k=128, label alignment passed, checkpoint saved.
```

Startup patch:

```text
The first smoke exposed a DDP startup tax in the supervised-sample guard: it
called ds[idx], which loaded and transformed every image on every rank. The
guard now resolves mixture indices and tokenizes conversation text directly,
preserving the no-empty-label check without image I/O.
```

Modal cached-KL smoke after startup patch:

```text
run: v14-qwen3-spatialgrid256-cachekl-smoke2
url: https://modal.com/apps/babakd/main/ap-wdLPETNHesMPJHAsRnRcoV
checkpoint: /checkpoints/pretrain-output/v14-qwen3-spatialgrid256-cachekl-smoke2/checkpoint-1
result: text-only filter kept 7000/7000 samples, train split 6650, cache
entries=6650/top_k=128, initial loss=8.1513, checkpoint saved.
```

## Phase 1 Primary Imitation Run

Run:

```text
name: v14-qwen3-spatialgrid256-imitation-cachekl-topk128-1500
url: https://modal.com/apps/babakd/main/ap-1cK26nOKaQVuz3OVJR1dpg
checkpoint dir: /checkpoints/pretrain-output/v14-qwen3-spatialgrid256-imitation-cachekl-topk128-1500
steps: 1500
gpu: H100:4
dataset: v14_qwen_imitation_replay_stage1b
student: AnyMALv3 + spatial_grid_projector, 256 image tokens
teacher cache: /checkpoints/v14_qwen/v14_v11_teacher_topk128_train_unique.pt
teacher: /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
teacher KL weight: 1.0
learning rate: 2e-5
```

Saved checkpoints:

```text
checkpoint-250
checkpoint-500
checkpoint-750
checkpoint-1000
checkpoint-1250
checkpoint-1500
```

Checkpoint metadata verified on `checkpoint-1500`:

```text
connector_type: spatial_grid_projector
image tokens: 256
spatial grid: 16x16
connector_output_scale: 1.125
patch_position_feature_type: coord_mlp
patch_position_feature_scale: 1.0
teacher KL cache entries: 6650
teacher KL cache top_k: 128
trainable modules: connector only; frozen vision encoder and frozen Qwen/Qwen3-8B
```

Training health:

```text
step 150 eval avg_loss: 0.8593
step 450 eval avg_loss: 0.7090
step 600 eval avg_loss: 0.6941
step 750 eval avg_loss: 0.6809
step 900 eval avg_loss: 0.6420
step 1200 eval avg_loss: 0.6515
step 1500 eval avg_loss: 0.6526
final train_loss: about 1.37-1.41 across ranks
generation/KL cache plumbing: cache loaded, sidecar projector skipped, label alignment passed
```

Interpretation: optimization was healthy enough to complete the required
minimum exposure, and generation hygiene stayed clean in downstream probes. The
failure is downstream imitation/capability, not a startup, checkpoint, or cache
loading failure.

## V11 Reconfirmation

V11 checkpoint:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

GQA:

| Eval | Artifact | URL | Accuracy | Correct / Total | Yes/No | Other | Hygiene |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| search n1000 | `/checkpoints/v14_qwen/v11_gqa_search1000.json` | https://modal.com/apps/babakd/main/ap-AdFDMswlkWAx1lJDrSZsvf | 44.900 | 449 / 1000 | 65.527 | 33.744 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |
| confirm n3000 offset1000 | `/checkpoints/v14_qwen/v11_gqa_confirm3000_offset1000.json` | https://modal.com/apps/babakd/main/ap-zKmJWRWihMp857lk1ygghl | 42.600 | 1278 / 3000 | 60.758 | 32.751 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |

ChartQA:

| Eval | Artifact | URL | Accuracy | Correct / Total | Number | Other | Yes/No | Hygiene |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| val n200 | `/checkpoints/v14_qwen/v11_chartqa_val_n200.json` | https://modal.com/apps/babakd/main/ap-SyhVDD3Wyp5Y6BiFFwDxuO | 6.000 | 12 / 200 | 2.548 | 18.919 | 16.667 | EOS 0.940, max-hit 0.060, role/thinking 0.000 |

TextVQA was not rerun: the repo does not currently contain a ready Modal
TextVQA checkpoint-eval harness analogous to GQA, VQA, POPE, or ChartQA. The
available `evaluation/vqa_eval.py` mentions TextVQA in documentation only.

## Primary Checkpoint Screens

GQA `testdev_balanced`, seed 42, offset 0, early n200 probes:

| Checkpoint | Artifact | URL | Accuracy | Correct / Total | Yes/No | Other | Hygiene |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 250 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt250_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-jyBe4rXDb00PvyKRZORZDg | 29.500 | 59 / 200 | 50.000 | 15.833 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |
| 500 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt500_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-kz7heBmb47t1vzaG74VPZD | 28.000 | 56 / 200 | 45.000 | 16.667 | EOS 0.985, max-hit 0.015, role/thinking 0.000 |
| 750 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt750_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-VhpAGHh4tRnDxSQVB6KYrc | 29.500 | 59 / 200 | 48.750 | 16.667 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |
| 1000 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1000_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-tvLDsA6LR11gRGJRv51psx | 30.500 | 61 / 200 | 47.500 | 19.167 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |
| 1250 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1250_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-0pNuwiNWT9qv5bUKG3YmC2 | 29.000 | 58 / 200 | 41.250 | 20.833 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |
| 1500 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1500_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-B7PDK41oZPZQ0JYYImlHAn | 28.000 | 56 / 200 | 38.750 | 20.833 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |

The best primary checkpoint by n200 was `checkpoint-1000`; it received a larger
GQA search probe:

| Eval | Artifact | URL | Accuracy | Correct / Total | Yes/No | Other | Hygiene |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| checkpoint-1000 search n1000 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1000_gqa_search1000.json` | https://modal.com/apps/babakd/main/ap-gtv9t6a458oKjPyGPvH5al | 30.400 | 304 / 1000 | 52.422 | 18.490 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |

This is 14.5 points below the reconfirmed V11 search score on the same slice
and is not inside the V14 imitation gate.

Additional primary `checkpoint-1500` probes:

| Eval | Artifact | URL | Accuracy | Correct / Total | Key breakdown | Hygiene |
| --- | --- | --- | ---: | ---: | --- | --- |
| ChartQA val n200 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1500_chartqa_val_n200.json` | https://modal.com/apps/babakd/main/ap-xlpHrfHpx0e9YIUANAlw7v | 3.500 | 7 / 200 | number 2.548, other 0.000, yes/no 50.000 | EOS 0.980, max-hit 0.020, role/thinking 0.000 |
| VQA clean n1000 | `/checkpoints/v14_qwen/v14_sg256_cachekl_ckpt1500_vqa_clean_n1000.json` | https://modal.com/apps/babakd/main/ap-kskqb0gLTYLpHMs3lYGZbc | 42.433 | 424.333 / 1000 soft | yes/no 66.749, number 25.581, other 26.053 | EOS 0.998, max-hit 0.002, role 0.000, thinking 0.001 |

VQA clean collapsed relative to the V11 status baseline
(`docs/STATUS.md`: V11 clean VQA n3000 mean 65.922).

## Serious Adjustment

Because the primary run was far below V11, V14 required at least one serious
adjustment before declaring the substrate failed. The adjustment reused the best
primary checkpoint weights instead of restarting from scratch.

Run:

```text
name: v14-qwen3-spatialgrid256-cachekl-kl5-lr5e6-warm1000-500
url: https://modal.com/apps/babakd/main/ap-uD1PPHtbUzTLdta1GxjE8F
checkpoint dir: /checkpoints/pretrain-output/v14-qwen3-spatialgrid256-cachekl-kl5-lr5e6-warm1000-500
warm start: /checkpoints/pretrain-output/v14-qwen3-spatialgrid256-imitation-cachekl-topk128-1500/checkpoint-1000
steps: 500
teacher KL weight: 5.0
learning rate: 5e-6
teacher cache: /checkpoints/v14_qwen/v14_v11_teacher_topk128_train_unique.pt
```

Training health:

```text
step 50 eval avg_loss: 0.6554
step 100 eval avg_loss: 0.6555
step 200 eval avg_loss: 0.6463
step 250 eval avg_loss: 0.6442; checkpoint-250 saved
step 400 eval avg_loss: 0.6491
step 450 eval avg_loss: 0.6411
step 500 eval avg_loss: 0.6452; checkpoint-500 saved
final train_loss: about 1.78-1.86 across ranks
```

Adjusted checkpoint screen:

| Eval | Artifact | URL | Accuracy | Correct / Total | Yes/No | Other | Hygiene |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| checkpoint-500 GQA n200 | `/checkpoints/v14_qwen/v14_sg256_cachekl_kl5_warm1000_ckpt500_gqa_n200.json` | https://modal.com/apps/babakd/main/ap-Yp0xU954ZqgaV2KXZvedU3 | 31.000 | 62 / 200 | 45.000 | 21.667 | EOS 1.000, max-hit 0.000, role/thinking 0.000 |

The adjustment improved validation loss slightly and recovered only 0.5 points
over the best primary n200 GQA probe. It remains far below V11 and does not
justify Phase 2 compositional training.

## Diagnostics And Audits

GQA leakage audit:

```text
script: scripts/audit_v9_leakage.py
change: eval refs now include question_id and gqa_question_id in addition to image refs.
final eval artifact: /tmp/anymal_v14_artifacts/v14_sg256_cachekl_kl5_warm1000_ckpt500_gqa_n200.json
train source: /checkpoints/gqa_data/v9_qwen_gqa_train_balanced_10000.json
audit output: /tmp/anymal_v14_artifacts/v14_adjusted_ckpt500_gqa_leakage_audit.json
url: https://modal.com/apps/babakd/main/ap-TQSm0OTFKzOoKqda30gjjR
```

Audit result:

```text
eval prediction samples: 50
rows with refs: 50
raw refs indexed: 93
missing sources: 0
exact val2014 overlap: 0
numeric ID overlap: 0
raw ref overlap: 0
overall gate: PASS
```

No-branch C1 auxiliary-branch diagnostic context:

```text
historical no-branch C1 checkpoint:
/checkpoints/pretrain-output/v13-qwen3-nobranch-c1diag-scale105-lr5e5-200/checkpoint-200

historical no-branch C1 result:
GQA search n1000: 42.300, 423 / 1000
GQA confirm n3000: 41.033, 1231 / 3000

V11 branch-present, disabled-at-eval reconfirmation:
GQA search n1000: 44.900, 449 / 1000
GQA confirm n3000: 42.600, 1278 / 3000
```

Interpretation: the no-branch C1 diagnostic still supports the V14 premise that
the auxiliary training-time branch changed the common projector optimization
path. The V14 spatial-grid substrate did not reproduce that basin even with
cached V11 answer-token KL and a stronger-KL warm-start adjustment.

## Final V14 V2 Judgment

Outcome:

```text
completed negative: primary substrate failed Phase 1 imitation.
```

The spatial-preserving 256-token grid trained cleanly, saved compatible
checkpoints, used the cached V11 teacher-distribution contract, and generated
clean answers. It did not imitate V11:

```text
V11 GQA search n1000: 44.900
best V14 primary GQA search n1000: 30.400
best V14 adjusted GQA n200: 31.000
V11 clean VQA baseline from docs/STATUS.md: 65.922
V14 primary clean VQA n1000: 42.433
```

Failure classification:

```text
The primary failure is substrate/optimization mismatch, not generation hygiene
or artifact plumbing. The 16x16 spatial_grid_projector can minimize replay
loss modestly but does not land in the V11 answer-distribution basin under the
tested frozen-LLM, connector-only setup.
```

Next recommendation:

```text
Do not promote this V14 primary substrate and do not enter Phase 2 from it.
If continuing V14-style work, prefer a training-only auxiliary-branch scaffold
or a connector that preserves the V11 Perceiver path while adding spatial
evidence, rather than replacing the Perceiver with a pure spatial grid.
```
