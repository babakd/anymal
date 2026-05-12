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

## Smoke Checks

Use these after structure changes:

```bash
python3 scripts/repo_health_check.py
python3 -m pytest tests/test_health_monitor.py -q
modal run scripts/modal_repo_smoke.py
```

