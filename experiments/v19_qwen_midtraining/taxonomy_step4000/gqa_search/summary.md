# GQA Pairwise Taxonomy Summary

Taxonomy labels are heuristic question-text labels for experiment steering, not publication-grade GQA metadata.

## Artifacts

### v19_step4000

- GQA accuracy: `41.100` sample / `41.1` reported
- Samples: `1000`
- Checkpoint: `/checkpoints/pretrain-output/v19-qwen3-midtraining-v18sup-v11kl-w2-6000/checkpoint-4000`

| Taxonomy | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| attribute | 22 | 65 | 33.85 |
| color | 32 | 83 | 38.55 |
| comparison | 15 | 26 | 57.69 |
| left_right | 46 | 116 | 39.66 |
| logical_compositional | 15 | 42 | 35.71 |
| object_identity | 56 | 170 | 32.94 |
| other | 19 | 81 | 23.46 |
| spatial_relation | 90 | 238 | 37.82 |
| yes_no_object_presence | 116 | 179 | 64.80 |

### v11

- GQA accuracy: `44.900` sample / `44.9` reported
- Samples: `1000`
- Checkpoint: `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125`

| Taxonomy | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| attribute | 21 | 65 | 32.31 |
| color | 34 | 83 | 40.96 |
| comparison | 15 | 26 | 57.69 |
| left_right | 52 | 116 | 44.83 |
| logical_compositional | 18 | 42 | 42.86 |
| object_identity | 57 | 170 | 33.53 |
| other | 29 | 81 | 35.80 |
| spatial_relation | 105 | 238 | 44.12 |
| yes_no_object_presence | 118 | 179 | 65.92 |

## Pairwise Deltas

| Comparison | Rows | Top Primary Buckets | Answer Kinds |
| --- | ---: | --- | --- |
| v19_step4000_correct_v11_wrong | 44 | yes_no_object_presence=13, spatial_relation=11, left_right=9, object_identity=4, attribute=3, color=2 | yes_no=28, other=10, direction=6 |
| v11_correct_v19_step4000_wrong | 82 | spatial_relation=26, left_right=15, yes_no_object_presence=15, other=11, object_identity=5, color=4 | yes_no=46, other=27, direction=7, color=2 |
