# Latest Experiment Snapshot

Last updated: 2026-05-14

This is the quick index for agents who need the current story without reading
every historical ledger.

## Current Incumbent Candidate

| Field | Value |
| --- | --- |
| Candidate | V11 Qwen C1-salvage frontier |
| Checkpoint | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` |
| Architecture | `anymal_v3` |
| Decoder | `Qwen/Qwen3-8B` |
| Vision tower | SigLIP2-So400m at 384px |
| Connector | V3 Perceiver-style connector, 128 image tokens |
| Connector scale | `1.125`, materialized in checkpoint metadata |
| Resume status | eval/inference checkpoint only |

Primary ledger:

- `experiments/v15_qwen/results.md`
- `experiments/v13_qwen/results.md`
- `experiments/v12_qwen/results.md`
- `experiments/v11_qwen/results.md`
- `experiments/v9_qwen/results.md`
- `experiments/v10_qwen/ceiling_results.md`
- `docs/STATUS.md`

## Latest V15 Capability Note

The 2026-05-14/15 V15 campaign moved from connector-only geometry search to
ChartQA/TextVQA capability training with V11 replay KL. The final 128-token
spatial-grid/no-position connector-isolation ablation was negative, closing
pure connector replacement as the main hill.

The connector-only balanced branch is useful but not promotable:

| Metric | V11 | V15 balanced checkpoint-3000 |
| --- | ---: | ---: |
| TextVQA n1000 | exact 28.5 / soft 26.47 | exact 32.3 / soft 29.93 |
| ChartQA n1000 | 7.8 | 9.7 |
| GQA n1000 | 44.9 trusted search | 42.2 |
| Clean VQA | 65.922 n3000 mean | 63.967 n1000 seed42 |
| POPE adversarial | 80.100 n1000 | 80.000 n1000 |

Attribution: the gain came from data/objective plus connector adaptation.
Vision-only, joint vision+connector, and high-resolution frozen-SigLIP controls
did not beat the connector-only balanced run. Keep V11 as incumbent; the next
V15-style run should strengthen GQA retention and add explicit connector-output
drift logging. See `experiments/v15_qwen/results.md`.

## Latest V13 Substrate-Break Note

The 2026-05-14 V13 substrate-break campaign did not find a robust successor to
V11. A no-branch C1 diagnostic underperformed V11 on GQA search/confirm, which
suggests the disabled C1 branch changed the optimization path. Spatial-grid,
AnyRes/pass-through, V11 spatial-tail, repaired visual cross-attention, Qwen
q/v LoRA, and a small SigLIP attention/norm adapter all failed to beat V11 on
matched GQA checks. The best new substrate, repaired visual cross-attention
with V11 teacher KL, tied V11-like n200 slices but landed at `44.6` GQA n1000
and `42.433` n3000 versus V11's `44.9` and `42.6`. A new ChartQA val n200
expanded probe also favored V11, `6.0` exact match versus V13's `5.5`. See
`experiments/v13_qwen/results.md`.

## Latest V12 Ceiling Note

The 2026-05-13 V12 aggressive Qwen3 ceiling search did not beat the V11
frontier on a robust matched check. A1 eval calibration briefly reached `45.0`
trusted GQA n1000, but tied V11 exactly at `43.2` on matched n3000. C1 mining
tied V11 at `44.9` n1000 and also matched V11 exactly on n3000. Larger
token budgets, DeepStack/multi-level features, higher resolution,
spatial-contrastive objectives, and vision-side SigLIP adaptation all regressed
or failed cheap promotion screens. See `experiments/v12_qwen/results.md`.

## Latest V11 Frontier Note

The 2026-05-13 V11 Qwen3 ceiling search found a C1-salvage checkpoint that
improves trusted GQA n1000 to `44.9` while preserving the available corrupted
image controls. This remains the current Qwen frontier.

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
| V15 Qwen capability/vision adaptation | `experiments/v15_qwen/` | TextVQA/ChartQA signal from connector-only data recipe, but GQA retention loss prevents promotion. |
| V13 Qwen substrate break | `experiments/v13_qwen/` | No robust successor; connector-only geometry changes appear exhausted for this stack. |
| V12 Qwen ceiling search | `experiments/v12_qwen/` | No robust successor; V11 remains the frontier after matched n3000 checks. |
| V11 Qwen frontier | `experiments/v11_qwen/` | Current Qwen frontier and active incumbent. |
| V10 Qwen ceiling search | `experiments/v10_qwen/` | No promotable successor; superseded by V11 C1-salvage. |
| V9 Qwen Connection V2 | `experiments/v9_qwen/` | Stable fallback and prior promoted Qwen candidate. |
| V8 Qwen decoder swap | `experiments/v8_qwen/` | Integration and compute-matched Qwen groundwork. |
| V7/V6 controls | `experiments/v7/`, `docs/V6_*` | LLaMA-side controls and causal falsification context. |
| V4/V5 spatial/recipe work | `docs/history/`, `docs/V5_RESEARCH_PLAN_20260509.md` | Historical evidence; do not treat V4 as current. |

## Next Useful Work

- Keep V11 as the current Qwen frontier unless a genuinely new mechanism beats
  it on matched larger GQA checks.
- Reduce duplicate evaluator loading and artifact-writing code.
- Split `scripts/modal/train.py` into smaller Modal/training/data modules.
- Keep generated predictions and large eval dumps outside git unless they are
  curated summaries.
