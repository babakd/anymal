# GQA Pairwise Taxonomy Summary

Taxonomy labels are heuristic question-text labels for experiment steering, not publication-grade GQA metadata.

## Artifacts

### v19_step4000

- GQA accuracy: `41.633` sample / `41.63333333333333` reported
- Samples: `3000`
- Checkpoint: `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-4000`

| Taxonomy | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| attribute | 78 | 210 | 37.14 |
| color | 88 | 244 | 36.07 |
| comparison | 50 | 83 | 60.24 |
| left_right | 140 | 351 | 39.89 |
| logical_compositional | 56 | 128 | 43.75 |
| object_identity | 131 | 491 | 26.68 |
| other | 63 | 217 | 29.03 |
| spatial_relation | 284 | 683 | 41.58 |
| yes_no_object_presence | 359 | 593 | 60.54 |

### v11

- GQA accuracy: `42.600` sample / `42.6` reported
- Samples: `3000`
- Checkpoint: `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125`

| Taxonomy | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| attribute | 76 | 210 | 36.19 |
| color | 87 | 244 | 35.66 |
| comparison | 48 | 83 | 57.83 |
| left_right | 154 | 351 | 43.87 |
| logical_compositional | 57 | 128 | 44.53 |
| object_identity | 133 | 491 | 27.09 |
| other | 62 | 217 | 28.57 |
| spatial_relation | 290 | 683 | 42.46 |
| yes_no_object_presence | 371 | 593 | 62.56 |

## Pairwise Deltas

| Comparison | Rows | Top Primary Buckets | Answer Kinds |
| --- | ---: | --- | --- |
| v19_step4000_correct_v11_wrong | 142 | yes_no_object_presence=43, spatial_relation=34, left_right=20, object_identity=15, color=13, attribute=7 | yes_no=82, other=44, color=8, direction=8 |
| v11_correct_v19_step4000_wrong | 171 | yes_no_object_presence=55, spatial_relation=40, left_right=34, object_identity=17, color=12, attribute=5 | yes_no=111, other=45, color=8, direction=7 |
