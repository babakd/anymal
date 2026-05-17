# GQA Pairwise Taxonomy Summary

Taxonomy labels are heuristic question-text labels for experiment steering, not publication-grade GQA metadata.

## Artifacts

### v19_step6000

- GQA accuracy: `41.667` sample / `41.666666666666664` reported
- Samples: `3000`
- Checkpoint: `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-6000`

| Taxonomy | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| attribute | 80 | 210 | 38.10 |
| color | 86 | 244 | 35.25 |
| comparison | 50 | 83 | 60.24 |
| left_right | 141 | 351 | 40.17 |
| logical_compositional | 57 | 128 | 44.53 |
| object_identity | 131 | 491 | 26.68 |
| other | 62 | 217 | 28.57 |
| spatial_relation | 280 | 683 | 41.00 |
| yes_no_object_presence | 363 | 593 | 61.21 |

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
| v19_step6000_correct_v11_wrong | 140 | yes_no_object_presence=47, spatial_relation=30, left_right=19, object_identity=15, color=11, attribute=9 | yes_no=86, other=43, color=6, direction=5 |
| v11_correct_v19_step6000_wrong | 168 | yes_no_object_presence=55, spatial_relation=40, left_right=32, object_identity=17, color=12, attribute=5 | yes_no=112, other=42, color=8, direction=6 |
