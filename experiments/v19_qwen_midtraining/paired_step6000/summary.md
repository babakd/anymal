# V19 Step6000 vs V11 Paired Bootstrap Summary

Candidate artifacts were fetched from `/v19_qwen/v17_fixed_harness/step6000/`.
V11 baselines were fetched from `/v17_fixed_harness/v11/`.

All paired bootstrap and available taxonomy/row-delta commands completed without analysis failures.

| Slice | Metric | N | V19 step6000 | V11 | Delta | 95% CI | p | Significant |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| GQA search | gqa_accuracy | 1000 | 41.100 | 44.900 | -3.800 | [-6.000, -1.600] | 0.0005 | yes |
| GQA confirm | gqa_accuracy | 3000 | 41.667 | 42.600 | -0.933 | [-2.067, +0.200] | 0.0569 | no |
| ChartQA full | chartqa_relaxed_match | 1920 | 14.792 | 14.688 | +0.104 | [-0.677, +0.833] | 0.4252 | no |
| ChartQA full | chartqa_exact_match | 1920 | 6.823 | 6.823 | +0.000 | [-0.677, +0.677] | 1.0000 | no |
| TextVQA full | textvqa_exact_match | 5000 | 31.400 | 30.940 | +0.460 | [-0.260, +1.200] | 0.1070 | no |
| TextVQA full | textvqa_soft_accuracy | 5000 | 28.660 | 28.400 | +0.260 | [-0.427, +0.940] | 0.2288 | no |
| POPE adversarial | pope_accuracy | 1000 | 80.200 | 80.300 | -0.100 | [-1.700, +1.500] | 0.4687 | no |
| POPE popular | pope_accuracy | 1000 | 81.300 | 81.800 | -0.500 | [-2.000, +0.900] | 0.2643 | no |
| VQAv2 clean | vqa_accuracy | 3000 | 50.056 | 63.867 | -13.811 | [-15.267, -12.344] | 0.0000 | yes |
| VQAv2 blank | vqa_accuracy | 3000 | 18.233 | 37.922 | -19.689 | [-21.178, -18.178] | 0.0000 | yes |
| VQAv2 shuffled | vqa_accuracy | 3000 | 32.522 | 36.178 | -3.656 | [-4.611, -2.711] | 0.0000 | yes |
| VQAv2 wrong-image | vqa_accuracy | 3000 | 31.989 | 35.522 | -3.533 | [-4.511, -2.556] | 0.0000 | yes |

## Diagnostic Counts

GQA search row deltas:

| Direction | Rows | Top primary buckets |
| --- | ---: | --- |
| V19 correct, V11 wrong | 43 | yes_no_object_presence=13, spatial_relation=10, left_right=9 |
| V11 correct, V19 wrong | 81 | spatial_relation=24, left_right=15, yes_no_object_presence=14 |

GQA confirm row deltas:

| Direction | Rows | Top primary buckets |
| --- | ---: | --- |
| V19 correct, V11 wrong | 140 | yes_no_object_presence=47, spatial_relation=30, left_right=19 |
| V11 correct, V19 wrong | 168 | yes_no_object_presence=55, spatial_relation=40, left_right=32 |

VQAv2 row-browser delta counts:

| Slice | V19 correct, V11 wrong | V11 correct, V19 wrong |
| --- | ---: | ---: |
| clean | 100 | 579 |
| blank | 55 | 731 |
| shuffled | 59 | 202 |
| wrong-image | 74 | 206 |
