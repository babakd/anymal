#!/usr/bin/env python3
"""Launch or print V19 cheap-screen eval commands for saved checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


RUN_NAME = "v19-qwen3-midtraining-v18sup-v11kl-w2-6000"
CHECKPOINT_ROOT = f"/checkpoints/pretrain-output/{RUN_NAME}"
REMOTE_ROOT = "/checkpoints/v19_qwen/cheap_screens"
LLM_BACKBONE = "Qwen/Qwen3-8B"


def _base(script: str, step: int, task: str, *, detach: bool) -> list[str]:
    command = ["modal", "run"]
    if detach:
        command.append("--detach")
    command.extend(
        [
            script,
            "--candidate-checkpoint",
            f"{CHECKPOINT_ROOT}/checkpoint-{step}",
            "--candidate-label",
            f"v19_step{step}_{task}",
            "--candidate-architecture",
            "v3",
            "--llm-backbone",
            LLM_BACKBONE,
            "--batch-size",
            "8",
            "--seed",
            "42",
            "--eval-schema-version",
            "v19_cheap_screen",
        ]
    )
    return command


def _finish(command: list[str], step: int, task: str) -> list[str]:
    remote = f"{REMOTE_ROOT}/step{step}/{task}.json"
    command.extend(
        [
            "--remote-output-path",
            remote,
            "--output",
            f"/tmp/v19_step{step}_{task}.json",
        ]
    )
    return command


def task_commands(step: int, *, detach: bool) -> list[dict[str, object]]:
    specs: list[tuple[str, list[str]]] = []

    command = _base("gqa_checkpoint_eval.py", step, "gqa_n200", detach=detach)
    command.extend(
        [
            "--gqa-split",
            "testdev_balanced",
            "--max-samples",
            "200",
            "--sample-offset",
            "0",
            "--eval-slice-name",
            "search_n200",
            "--prediction-samples",
            "200",
        ]
    )
    specs.append(("gqa_n200", command))

    command = _base("chartqa_checkpoint_eval.py", step, "chartqa_val_n200", detach=detach)
    command.extend(["--split", "val", "--max-samples", "200", "--prediction-samples", "200"])
    specs.append(("chartqa_val_n200", command))

    command = _base("textvqa_checkpoint_eval.py", step, "textvqa_val_n200", detach=detach)
    command.extend(
        ["--split", "validation", "--max-samples", "200", "--prediction-samples", "200"]
    )
    specs.append(("textvqa_val_n200", command))

    command = _base("vqa_checkpoint_eval.py", step, "vqa_clean_n200", detach=detach)
    command.extend(
        [
            "--max-samples",
            "200",
            "--image-perturbation",
            "none",
            "--prediction-samples",
            "200",
            "--no-include-baselines",
        ]
    )
    specs.append(("vqa_clean_n200", command))

    for split in ("adversarial", "popular"):
        task = f"pope_{split}_n200"
        command = _base("pope_checkpoint_eval.py", step, task, detach=detach)
        command.extend(
            [
                "--pope-split",
                split,
                "--max-samples",
                "200",
                "--prediction-samples",
                "200",
            ]
        )
        specs.append((task, command))

    drift_task = "connector_drift_n64"
    command = ["modal", "run"]
    if detach:
        command.append("--detach")
    command.extend(
        [
            "evaluation/checkpoint_eval/connector_drift_eval.py",
            "--candidate-checkpoint",
            f"{CHECKPOINT_ROOT}/checkpoint-{step}",
            "--candidate-label",
            f"v19_step{step}_{drift_task}",
            "--candidate-architecture",
            "v3",
            "--llm-backbone",
            LLM_BACKBONE,
            "--max-samples",
            "64",
            "--seed",
            "42",
            "--batch-size",
            "4",
        ]
    )
    specs.append((drift_task, command))

    return [
        {
            "step": int(step),
            "task": task,
            "remote_output_path": f"{REMOTE_ROOT}/step{step}/{task}.json",
            "command": _finish(command, step, task),
        }
        for task, command in specs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        default="500,1000,1500,2000,3000,4000,4500,6000",
        help="Comma/space-separated checkpoint steps.",
    )
    parser.add_argument("--tasks", default="", help="Optional comma/space-separated task filter.")
    parser.add_argument("--foreground", action="store_true", help="Run without Modal --detach.")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing JSON.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/v19_qwen/cheap_screen_manifest.json"),
    )
    args = parser.parse_args()

    steps = [int(part) for part in args.steps.replace(",", " ").split() if part.strip()]
    task_filter = {part.strip() for part in args.tasks.replace(",", " ").split() if part.strip()}
    rows = []
    for step in steps:
        for row in task_commands(step, detach=not args.foreground):
            if task_filter and str(row["task"]) not in task_filter:
                continue
            rows.append(row)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if not args.run:
        print(json.dumps(rows, indent=2))
        print(f"Wrote manifest: {args.manifest}")
        return

    for row in rows:
        command = [str(part) for part in row["command"]]
        print(f"Launching step {row['step']} {row['task']}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
