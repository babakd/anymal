# Latest Experiment Snapshot

Last updated: 2026-05-13

This is the quick index for agents who need the current story without reading
every historical ledger.

## Current Incumbent Candidate

| Field | Value |
| --- | --- |
| Candidate | V9 Qwen scale-1.05 |
| Checkpoint | `/checkpoints/pretrain-output/v9-qwen3-stage1b2350-scale105-final/checkpoint-2350-scale105` |
| Architecture | `anymal_v3` |
| Decoder | `Qwen/Qwen3-8B` |
| Vision tower | SigLIP2-So400m at 384px |
| Connector | V3 Perceiver-style connector, 128 image tokens |
| Connector scale | `1.05`, materialized in checkpoint metadata |
| Resume status | eval/inference checkpoint only |

Primary ledger:

- `experiments/v9_qwen/results.md`
- `experiments/v9_qwen/confirm.md`
- `experiments/v9_qwen/confirmation_results.md`
- `experiments/v10_qwen/plan.md`
- `experiments/v10_qwen/ceiling_results.md`
- `docs/STATUS.md`

## Latest V10 Ceiling Note

The 2026-05-13 V10 Qwen3 ceiling search did not produce a promotable
successor. The best true GQA mover was learned 2D patch features at `44.5`,
but that branch damaged corrupted-image controls. The final gated
visual-cross-attention branch stayed at `43.4` GQA on early screens and was
stopped at checkpoint-50. See `experiments/v10_qwen/ceiling_results.md`.

## Latest Confirmation Note

The 2026-05-12 materialized no-override bundle reproduced the canonical pass
set, but a larger trusted GQA n=1000 diagnostic landed at `43.100` for Qwen
scale-1.05 versus `43.700` for V3 robust on the same slice. Promotion is on hold
pending the GQA-slice decision.

## Final V9 Gate Results

| Gate | Required | Result |
| --- | ---: | ---: |
| Clean VQA | `>= 62.967` | `66.133` n=3000 seed42 |
| Blank image | `<= 39.733` | `38.811` n=3000 seed42 |
| Shuffled image | `<= 37.367` | `36.900` n=3000 seed42 |
| Wrong image, same answer type | `<= 38.900` | `37.967` n=1000 seed42 |
| Perturbation mean | `>= 60.189` | `66.856` |
| POPE | `>= 77.100` | `79.100` |
| GQA | `>= 43.800` | `44.000` |
| Generation hygiene | EOS/max-hit/prefix gates | pass |
| Leakage audit | pass | pass |

## Recent Campaigns

| Campaign | Where to read | Current interpretation |
| --- | --- | --- |
| V10 Qwen ceiling search | `experiments/v10_qwen/` | No promotable successor; V9/Batch-A remains the active Qwen family. |
| V9 Qwen Connection V2 | `experiments/v9_qwen/` | Current promoted candidate. |
| V8 Qwen decoder swap | `experiments/v8_qwen/` | Integration and compute-matched Qwen groundwork. |
| V7/V6 controls | `experiments/v7/`, `docs/V6_*` | LLaMA-side controls and causal falsification context. |
| V4/V5 spatial/recipe work | `docs/history/`, `docs/V5_RESEARCH_PLAN_20260509.md` | Historical evidence; do not treat V4 as current. |

## Next Useful Work

- Reconfirm the materialized V9 checkpoint without eval-time scale overrides.
- Reduce duplicate evaluator loading and artifact-writing code.
- Split `scripts/modal/train.py` into smaller Modal/training/data modules.
- Keep generated predictions and large eval dumps outside git unless they are
  curated summaries.
