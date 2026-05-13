# V9 Qwen Scale-1.05 Materialized Confirmation Results

Date: 2026-05-12

## Decision

Hold promotion pending GQA-slice decision.

The materialized no-override checkpoint reproduced the canonical confirmation
bundle and passed the recorded gates on:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

However, the larger trusted GQA diagnostic slice exposed a follow-up concern:
the canonical first 500 GQA samples reproduced the recorded pass at `44.0`, but
the 1000-sample expansion landed at `43.1`. An incumbent diagnostic on the same
1000-sample GQA slice landed at `43.7`, also below the old `43.8` gate but still
above Qwen. Because `confirm.md` says not to promote after a below-gate GQA
surprise until investigated, no promotion note was written.

## Checkpoint Metadata

Remote files:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105/model_meta.json
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105/projector.pt
```

Verified metadata:

| Field | Value |
| --- | --- |
| architecture | `anymal_v3` |
| llm_backbone | `Qwen/Qwen3-8B` |
| connector_type | `perceiver_resampler` |
| num_image_tokens / image_tokens | `128` / `128` |
| connector_output_scale | `1.05` |
| question_conditioning | `null` |
| Stage2 LoRA | absent / not required |

The confirmation artifacts have `connector_output_scale_override: null`, no
`eval_connector_output_scale_override`, and connector metadata scale `1.05`.

## Confirmation Metrics

| Eval | n | Result | Gate | Status |
| --- | ---: | ---: | ---: | --- |
| VQAv2 clean | 1000 | `66.167` | `>= 62.967` | pass |
| VQAv2 blank | 1000 | `39.400` | `<= 39.733` | pass |
| VQAv2 shuffled | 1000 | `37.367` | `<= 37.367` | pass, edge |
| VQAv2 wrong image same-answer-type | 1000 | `37.967` | `<= 38.900` | pass |
| VQAv2 wrong image same-answer-type | 3000 | `36.756` | `<= 38.900` | pass |
| POPE adversarial | 1000 | `79.100` | `>= 77.100` | pass |
| GQA testdev_balanced canonical | 500 | `44.000` | `>= 43.800` | pass |
| GQA testdev_balanced expanded | 1000 | `43.100` | diagnostic | concern |
| V3 robust GQA expanded diagnostic | 1000 | `43.700` | diagnostic | reference |

Generation hygiene was clean on all V9 confirmation artifacts:

```text
eos_rate = 1.0
hit_max_new_tokens_rate = 0.0
assistant_role_prefix_rate = 0.0
strict_accuracy == clean accuracy
```

GQA slice breakdown:

| Model | first 500 | second 500 | all 1000 |
| --- | ---: | ---: | ---: |
| Qwen scale-1.05 materialized | `44.000` | `42.200` | `43.100` |
| V3 robust incumbent diagnostic | `43.800` | `43.600` | `43.700` |

## Leakage Audit

Artifact:

```text
outputs/v9_analysis/v9_qwen3_scale105_materialized_nooverride_leakage_audit.json
```

Summary:

| Check | Result |
| --- | ---: |
| Eval prediction samples | `9500` |
| Rows with refs | `9500` |
| Missing/unparseable refs | `0` |
| Exact val2014 overlap | `0` |
| Numeric-only overlap | `294`, warning only |
| Raw ref overlap | `0` |
| Missing sources | `0` |
| Overall | pass |

The numeric-only overlaps came from BLIP/LAION-style numeric filename collisions.
They remain warning-only because exact val2014 and raw-reference overlap are
both zero.

## Answer Distribution

Artifact:

```text
outputs/v9_analysis/v9_qwen3_scale105_materialized_nooverride_answer_analysis.json
```

Clean VQAv2 answer-kind comparison:

| Model | Clean | Strict | Prefix | yes/no on yes/no | number on number | other on other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 robust incumbent | `62.967` | `62.967` | `0.000` | `1.000` | `0.946` | `0.994` |
| B1 control | `63.267` | `63.267` | `0.000` | `1.000` | `0.938` | `0.994` |
| Qwen Stage1B2350 scale1.00 | `66.233` | `66.233` | `0.000` | `1.000` | `0.992` | `0.987` |
| Qwen Stage1B2350 scale1.05 materialized | `66.167` | `66.167` | `0.000` | `1.000` | `0.984` | `0.987` |
| Qwen Stage2 ckpt800 | `65.900` | `65.900` | `0.000` | `1.000` | `0.969` | `0.987` |

No mode-collapse or answer-kind pathology was observed in the clean answer
distribution comparison.

## Artifacts

Final no-override artifacts:

```text
outputs/v9_confirmation/vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_clean_n1000.json
outputs/v9_confirmation/vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_blank_n1000.json
outputs/v9_confirmation/vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_shuffled_n1000.json
outputs/v9_confirmation/vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_wrongimage_sameanswertype_n1000.json
outputs/v9_confirmation/vqa_eval_v9_qwen3_scale105_materialized_nooverride_seed42_wrongimage_sameanswertype_n3000.json
outputs/v9_confirmation/pope_eval_v9_qwen3_scale105_materialized_nooverride.json
outputs/v9_confirmation/gqa_eval_v9_qwen3_scale105_materialized_nooverride_n500.json
outputs/v9_confirmation/gqa_eval_v9_qwen3_scale105_materialized_nooverride_n1000.json
outputs/v9_confirmation/gqa_eval_v3_robust_currentcache_n1000_seed42_diagnostic.json
outputs/v9_analysis/v9_qwen3_scale105_materialized_nooverride_leakage_audit.json
outputs/v9_analysis/v9_qwen3_scale105_materialized_nooverride_answer_analysis.json
```

Remote artifact directory:

```text
/checkpoints/v9_confirmation/
```

## Operational Notes

The first smoke run caught a current-code Modal packaging issue before the
expensive bundle was launched:

```text
ModuleNotFoundError: No module named 'v1_v2_compare_inference'
```

The evaluator imports were updated to use
`scripts.inference.v1_v2_compare_inference`, after which compile and repo
health checks passed and the smoke completed successfully.
