#!/usr/bin/env python3
"""Launch or print V19 full V17-harness confirmation eval commands."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


RUN_NAME = "v19-qwen3-midtraining-v18sup-v11kl-w2-6000"
CHECKPOINT_ROOT = f"/checkpoints/pretrain-output/{RUN_NAME}"
REMOTE_ROOT = "/checkpoints/v19_qwen/v17_fixed_harness"
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
            "v17_fixed_harness",
        ]
    )
    return command


def _finish(command: list[str], step: int, task: str) -> list[str]:
    command.extend(
        [
            "--remote-output-path",
            f"{REMOTE_ROOT}/step{step}/{task}.json",
            "--output",
            f"/tmp/v17_fixed_harness_v19_step{step}_{task}.json",
        ]
    )
    return command


def task_commands(step: int, *, detach: bool) -> list[dict[str, object]]:
    specs: list[tuple[str, list[str]]] = []

    command = _base("gqa_checkpoint_eval.py", step, "gqa_search_n1000", detach=detach)
    command.extend(
        [
            "--gqa-split",
            "testdev_balanced",
            "--max-samples",
            "1000",
            "--sample-offset",
            "0",
            "--eval-slice-name",
            "search_n1000",
            "--prediction-samples",
            "1000",
        ]
    )
    specs.append(("gqa_search_n1000", command))

    command = _base(
        "gqa_checkpoint_eval.py",
        step,
        "gqa_confirm_n3000_offset1000",
        detach=detach,
    )
    command.extend(
        [
            "--gqa-split",
            "testdev_balanced",
            "--max-samples",
            "3000",
            "--sample-offset",
            "1000",
            "--eval-slice-name",
            "confirm_n3000_offset1000",
            "--prediction-samples",
            "3000",
        ]
    )
    specs.append(("gqa_confirm_n3000_offset1000", command))

    command = _base("chartqa_checkpoint_eval.py", step, "chartqa_val_full", detach=detach)
    command.extend(["--split", "val", "--max-samples", "0", "--prediction-samples", "1000000"])
    specs.append(("chartqa_val_full", command))

    command = _base(
        "textvqa_checkpoint_eval.py",
        step,
        "textvqa_validation_full",
        detach=detach,
    )
    command.extend(
        ["--split", "validation", "--max-samples", "0", "--prediction-samples", "1000000"]
    )
    specs.append(("textvqa_validation_full", command))

    for split in ("adversarial", "popular"):
        task = f"pope_{split}_n1000"
        command = _base("pope_checkpoint_eval.py", step, task, detach=detach)
        command.extend(
            [
                "--pope-split",
                split,
                "--max-samples",
                "1000",
                "--prediction-samples",
                "1000",
            ]
        )
        specs.append((task, command))

    vqa_tasks = {
        "vqa_clean_n3000_seed42": "none",
        "vqa_blank_n3000_seed42": "blank_image",
        "vqa_shuffled_n3000_seed42": "shuffled_image",
        "vqa_wrong_image_n3000_seed42": "wrong_image_same_answer_type",
    }
    for task, perturbation in vqa_tasks.items():
        command = _base("vqa_checkpoint_eval.py", step, task, detach=detach)
        command.extend(
            [
                "--max-samples",
                "3000",
                "--image-perturbation",
                perturbation,
                "--prediction-samples",
                "3000",
                "--no-include-baselines",
            ]
        )
        specs.append((task, command))

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
        required=True,
        help="Comma/space-separated candidate checkpoint steps.",
    )
    parser.add_argument("--tasks", default="", help="Optional comma/space-separated task filter.")
    parser.add_argument("--foreground", action="store_true", help="Run without Modal --detach.")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing JSON.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/v19_qwen_midtraining/confirmation_eval_manifest.json"),
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
