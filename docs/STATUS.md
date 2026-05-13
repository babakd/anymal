# Project Status

Last updated: 2026-05-13

## Current Best Candidate

The current viable replacement candidate is the V9 Qwen/Qwen3-8B checkpoint:

```text
/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105
```

It uses:

- Architecture family: `anymal_v3`
- Decoder backbone: `Qwen/Qwen3-8B`
- Vision tower: SigLIP2-So400m at 384px
- Connector: V3 Perceiver-style connector with 128 image tokens
- Materialized connector output scale: `1.05`
- Checkpoint contents: `model_meta.json` and `projector.pt`
- Resume status: eval/inference checkpoint only, not an optimizer-resume
  checkpoint

The full V9 result ledger lives at `experiments/v9_qwen/results.md`. The latest
experiment index lives at `experiments/LATEST.md`, and the confirmation plan
lives at `experiments/v9_qwen/confirm.md`.

## Latest V10 Ceiling Note

The V10 Qwen3 ceiling search on 2026-05-13 did not produce a promotable
successor. The C1 learned-2D-position branch reached GQA `44.5` but regressed
corrupted-image controls, and the final E1 gated visual-cross-attention branch
screened at GQA `43.4`. Treat the V9/Batch-A family as the active Qwen path;
see `experiments/v10_qwen/ceiling_results.md`.

## Latest Confirmation Note

The materialized no-override confirmation bundle on 2026-05-12 reproduced the
canonical gates, including GQA `44.000` on the established n=500 slice. A larger
trusted GQA n=1000 diagnostic landed at `43.100` for Qwen scale-1.05 versus
`43.700` for the V3 robust incumbent on the same slice. Treat promotion as on
hold pending the GQA-slice decision; see
`experiments/v9_qwen/confirmation_results.md`.

## Promotion Gates

The V9 candidate passed the replacement gates recorded in the final ledger:

| Gate | Required | V9 result |
| --- | ---: | ---: |
| Clean VQA | `>= 62.967` | `66.133` n=3000 seed42 |
| Blank image | `<= 39.733` | `38.811` n=3000 seed42 |
| Shuffled image | `<= 37.367` | `36.900` n=3000 seed42 |
| Wrong image, same answer type | `<= 38.900` | `37.967` n=1000 seed42 |
| Perturbation mean | `>= 60.189` | `66.856` |
| POPE | `>= 77.100` | `79.100` |
| GQA | `>= 43.800` | `44.000` |
| EOS rate | `>= 0.98` | pass |
| Max-token hit | `<= 0.02` | pass |
| Assistant-prefix rate | `<= 0.01` | pass |
| Strict-clean gap | `<= 1.0` | `0.000` |
| Leakage audit | pass | pass |

## Active Direction

The next research work should treat V9 Qwen scale-1.05 as the incumbent
candidate, then focus on:

- confirming reproducibility of the materialized checkpoint and eval scripts,
- reducing duplicated VQA/GQA/POPE evaluator code,
- making Modal training/eval entrypoints thinner and easier to smoke-test,
- moving generated artifacts out of git unless they are curated summary
  artifacts.

Do not rely on older V3/V4 handoff text as the current state. Those documents
remain useful provenance, but they predate the V8/V9 Qwen campaigns.
