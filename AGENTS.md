# Agent Guide

This repository is an active multimodal-LLM research codebase. It contains
production-like training/evaluation utilities, historical experiment ledgers,
and generated artifacts. Read this file before changing code.

## Start Here

1. Read `docs/STATUS.md` for the current model, checkpoint, gates, and active
   research direction.
2. Read `README.md` for the user-facing overview and basic commands.
3. If an old root-level experiment doc is missing or looks tiny, read
   `docs/REPO_STRUCTURE_MIGRATION.md`; old root docs are now redirect stubs.
4. Use `experiments/` for historical plans, ledgers, and campaign writeups.
5. Treat `docs/history/` as frozen provenance unless a task explicitly asks to
   correct a historical record.

## Current Operating Rules

- Keep runtime behavior stable unless the task asks for a model/training change.
- Do not add new experiment JSON, prediction dumps, W&B logs, checkpoints, or
  local data files to the repo root.
- Write regeneratable eval artifacts under `results/` only when they are small
  and intentionally tracked. Larger artifacts should stay in Modal volumes,
  W&B, external storage, or ignored `outputs/`.
- Preserve checkpoint metadata compatibility. Architecture/backbone guards are
  part of the safety contract, not incidental validation.
- Prefer small, testable refactors over sweeping moves that alter Modal command
  behavior.

## Useful Smoke Tests

Local syntax/import-light checks:

```bash
python3 -m compileall -q models training evaluation data scripts
python3 scripts/repo_health_check.py
```

Local unit tests require a Python environment with the heavy ML dependencies
installed:

```bash
python3 -m pytest tests -q
```

Tiny Modal smoke examples:

```bash
modal run scripts/modal_repo_smoke.py
modal run modal_train.py --use-dummy-data --max-steps 1 --batch-size 1
modal run modal_v8_llm_swap_smoke.py --llm-backbone Qwen/Qwen3-8B --max-new-tokens 2
```

Use GPU smoke tests when a change touches Modal wiring, model construction,
checkpoint metadata, tokenizer/backbone handling, or training/eval entrypoints.

## Layout

```text
configs/       Training configs
data/          Dataset loaders, transforms, collators, chat templates
docs/          Current runbooks and status
docs/history/  Frozen handoffs and historical provenance
evaluation/    Reusable evaluation code
experiments/   Versioned research plans and result ledgers
models/        AnyMAL architectures, encoders, projectors, LLM wrappers
results/       Small tracked eval artifacts only
scripts/       Local tooling and thin operational entrypoints
tests/         Unit tests
training/      Trainers, distributed helpers, health and throughput monitoring
```
