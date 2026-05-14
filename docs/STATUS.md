# Project Status

Last updated: 2026-05-13

## Current Best Candidate

The current Qwen/Qwen3-8B frontier candidate is the V11 C1-salvage checkpoint:

```text
/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125
```

It uses:

- Architecture family: `anymal_v3`
- Decoder backbone: `Qwen/Qwen3-8B`
- Vision tower: SigLIP2-So400m at 384px
- Connector: V3 Perceiver-style connector with 128 image tokens
- Materialized connector output scale: `1.125`
- Learned 2D patch-position feature path present but attenuated with
  `patch_position_feature_scale: 0.0`
- Checkpoint contents: `model_meta.json` and `projector.pt`
- Resume status: eval/inference checkpoint only, not an optimizer-resume
  checkpoint

The V11 result ledger lives at `experiments/v11_qwen/results.md`. The V9 result
ledger remains at `experiments/v9_qwen/results.md`. The follow-on V12 ceiling
search ledger lives at `experiments/v12_qwen/results.md`.

## Latest V11 Ceiling Note

The V11 Qwen3 ceiling search on 2026-05-13 found a no-training C1 salvage point
that improves trusted GQA n1000 to `44.9` while bringing corrupted-image
controls back inside the V9-style guardrails. The key setting is to keep C1's
common projector weights, materialize connector scale `1.125`, and set
`patch_position_feature_scale` to `0.0`.

The cheap C1 salvage surface appears locally plateaued: V9-C1 interpolation did
not keep controls, fine scale checks around `1.125` did not exceed `44.9`,
prefix-only patch-position-table training did not move GQA, and a 10-step
all-projector continuation regressed GQA to `44.0`.

## Latest Confirmation Note

The materialized no-override V11 confirmation bundle on 2026-05-13 includes:

```text
GQA trusted n1000:       44.900
Clean VQA n3000 mean:   65.922 across seeds 42/43/44
Blank n3000 seed42:     39.078
Shuffled n3000 seed42:  36.767
Wrong-image n3000:      37.178
POPE adversarial n1000: 80.100
```

Leakage audit status: no confirmed leakage. The COCO/VQA/POPE split-aware audit
passes with zero exact/numeric/raw overlaps, and direct GQA image/question ID
inspection found zero overlap. The broad generic audit reports 18 splitless
numeric/raw overlaps against a GQA train source; these are treated as known
false-positive filename-stem collisions and are recorded in the V11 ledger.

## Promotion Gates

The V11 frontier passes the available old replacement gates recorded in the
current ledger:

| Gate | Required | V11 result |
| --- | ---: | ---: |
| Clean VQA | `>= 62.967` | `65.922` n=3000 mean over seeds 42/43/44 |
| Blank image | `<= 39.733` | `39.078` n=3000 seed42 |
| Shuffled image | `<= 37.367` | `36.767` n=3000 seed42 |
| Wrong image, same answer type | `<= 38.900` | `37.178` n=3000 seed42 |
| Perturbation mean | `>= 60.189` | `65.633` over mild blur/crop/translate n=1000 |
| POPE | `>= 77.100` | `80.100` n=1000 |
| GQA | `>= 43.800` | `44.900` trusted n=1000 |
| EOS rate | `>= 0.98` | pass |
| Max-token hit | `<= 0.02` | pass |
| Assistant-prefix rate | `<= 0.01` | pass |
| Strict-clean gap | `<= 1.0` | pass on checked evals |
| Leakage audit | pass | no confirmed leakage; see note above |

## Active Direction

The V12 aggressive Qwen3 ceiling search pushed the major planned high-ceiling
directions: repaired visual cross-attention, larger image-token budgets,
controlled V11 continuations, DeepStack/multi-level features, higher-resolution
eval/training smoke, and vision-side SigLIP adaptation. No branch beat V11 on a
robust matched check. The best apparent n1000 moves either tied V11 exactly on
matched n3000 or rediscovered the same V11 basin.

Current operating stance:

- Treat V11 C1-salvage as the Qwen frontier.
- Treat V9 scale-1.05 as the stable fallback.
- Do not rerun V12 DeepStack, token-budget, spatial-contrastive, or final-block
  SigLIP recipes without a new mechanism or diagnostic.
- Keep generated eval dumps out of git unless they are curated summaries.

Do not rely on older V3/V4 handoff text as the current state. Those documents
remain useful provenance, but they predate the V8/V9 Qwen campaigns.
