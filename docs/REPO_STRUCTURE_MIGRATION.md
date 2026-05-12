# Repository Structure Migration

Last updated: 2026-05-12

This file is a breadcrumb for humans and agents that remember the old root-heavy
layout.

## Start Here Now

- Current project state: `docs/STATUS.md`
- Agent operating guide: `AGENTS.md`
- Historical agent handoff: `docs/history/CLAUDE_legacy.md`
- Historical experiment ledgers: `experiments/`

## Historical Documents

| Old path | New path |
| --- | --- |
| `EXPERIMENTS.md` | `experiments/README.md` |
| `v7_experiments.md` | `experiments/v7/README.md` |
| `V8_experiment.md` | `experiments/v8_qwen/V8_experiment.md` |
| `V8_QWEN3_plan.md` | `experiments/v8_qwen/V8_QWEN3_plan.md` |
| `V8_CORE_LLM_SWAP_RESULTS.md` | `experiments/v8_qwen/V8_CORE_LLM_SWAP_RESULTS.md` |
| `v9_qwen_plan.md` | `experiments/v9_qwen/plan.md` |
| `v9_qwen_experiment_results.md` | `experiments/v9_qwen/results.md` |
| old long `CLAUDE.md` | `docs/history/CLAUDE_legacy.md` |

Root files with these old names are now short redirect stubs only.

## Checkpoint Evaluators

The implementation moved, but the old commands still work:

```bash
modal run vqa_checkpoint_eval.py --help
modal run gqa_checkpoint_eval.py --help
modal run pope_checkpoint_eval.py --help
```

Implementation files now live under:

```text
evaluation/checkpoint_eval/
```

## Root Scripts

Root-level Python files are now limited to compatibility wrappers and small
shared modules. Bulky implementations moved here:

| Old path | New path |
| --- | --- |
| `modal_train.py` | wrapper for `scripts/modal/train.py` |
| `modal_v8_llm_swap_smoke.py` | wrapper for `scripts/modal/v8_llm_swap_smoke.py` |
| `arch_sxs_inference.py` | `scripts/inference/arch_sxs_inference.py` |
| `compare_inference.py` | `scripts/inference/compare_inference.py` |
| `three_way_inference.py` | `scripts/inference/three_way_inference.py` |
| `v1_v2_compare_inference.py` | `scripts/inference/v1_v2_compare_inference.py` |
| `v2_compare_inference.py` | `scripts/inference/v2_compare_inference.py` |
| `analyze_v2_compare.py` | `scripts/analysis/analyze_v2_compare.py` |
| `analyze_v2_probe.py` | `scripts/analysis/analyze_v2_probe.py` |
| `v2_quality_diagnostics.py` | `scripts/analysis/v2_quality_diagnostics.py` |
| `modal_viewer.py` | `scripts/viewer/modal_viewer.py` |

## Smoke Checks

Use these after structure changes:

```bash
python3 scripts/repo_health_check.py
python3 -m pytest tests/test_health_monitor.py -q
modal run scripts/modal_repo_smoke.py
```
