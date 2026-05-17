# V19 Step4000 vs V11 Paired Bootstrap Summary

Candidate artifacts were fetched from `/v19_qwen/v17_fixed_harness/step4000/`.
V11 baselines were fetched from `/v17_fixed_harness/v11/`.

All paired bootstrap and available taxonomy/row-delta commands completed without command failures.

| Slice | Metric | N | V19 step4000 | V11 | Delta | 95% CI | p | Significant |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| GQA search | gqa_accuracy | 1000 | 41.100 | 44.900 | -3.800 | [-6.000, -1.600] | 0.0004 | yes |
| GQA confirm | gqa_accuracy | 3000 | 41.633 | 42.600 | -0.967 | [-2.100, +0.167] | 0.0499 | no |
| ChartQA full | chartqa_relaxed_match | 1920 | 14.844 | 14.688 | +0.156 | [-0.625, +0.938] | 0.3690 | no |
| ChartQA full | chartqa_exact_match | 1920 | 6.823 | 6.823 | +0.000 | [-0.677, +0.677] | 1.0000 | no |
| TextVQA full | textvqa_exact_match | 5000 | 31.460 | 30.940 | +0.520 | [-0.220, +1.240] | 0.0788 | no |
| TextVQA full | textvqa_soft_accuracy | 5000 | 28.640 | 28.400 | +0.240 | [-0.447, +0.920] | 0.2455 | no |
| POPE adversarial | pope_accuracy | 1000 | 80.000 | 80.300 | -0.300 | [-1.900, +1.300] | 0.3615 | no |
| POPE popular | pope_accuracy | 1000 | 81.400 | 81.800 | -0.400 | [-1.900, +1.100] | 0.3136 | no |
| VQAv2 clean | vqa_accuracy | 3000 | 47.967 | 63.867 | -15.900 | [-17.411, -14.378] | 0.0000 | yes |
| VQAv2 blank | vqa_accuracy | 3000 | 17.644 | 37.922 | -20.278 | [-21.800, -18.767] | 0.0000 | yes |
| VQAv2 shuffled | vqa_accuracy | 3000 | 31.800 | 36.178 | -4.378 | [-5.356, -3.411] | 0.0000 | yes |
| VQAv2 wrong-image | vqa_accuracy | 3000 | 31.511 | 35.522 | -4.011 | [-5.011, -3.022] | 0.0000 | yes |

## Diagnostic Counts

GQA search row deltas:

| Direction | Rows | Top primary buckets |
| --- | ---: | --- |
| V19 correct, V11 wrong | 44 | yes_no_object_presence=13, spatial_relation=11, left_right=9 |
| V11 correct, V19 wrong | 82 | spatial_relation=26, left_right=15, yes_no_object_presence=15 |

GQA confirm row deltas:

| Direction | Rows | Top primary buckets |
| --- | ---: | --- |
| V19 correct, V11 wrong | 142 | yes_no_object_presence=43, spatial_relation=34, left_right=20 |
| V11 correct, V19 wrong | 171 | yes_no_object_presence=55, spatial_relation=40, left_right=34 |

VQAv2 row-browser delta counts:

| Slice | V19 correct, V11 wrong | V11 correct, V19 wrong |
| --- | ---: | ---: |
| clean | 93 | 640 |
| blank | 50 | 748 |
| shuffled | 57 | 223 |
| wrong-image | 76 | 227 |
