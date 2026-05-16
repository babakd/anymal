#!/usr/bin/env python3
"""Launch or print the V17 corrected-harness checkpoint eval suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


V11_CHECKPOINT = (
    "/checkpoints/pretrain-output/"
    "v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125"
)

CANDIDATES = {
    "v11": {
        "label": "v11_frontier",
        "checkpoint": V11_CHECKPOINT,
    },
    "v16_phase2a_step3000": {
        "label": "v16_phase2a_step3000",
        "checkpoint": (
            "/checkpoints/pretrain-output/"
            "v16-qwen3-v11-visionlast2-only-connfrozen-cachekl-lr1e6-3000/"
            "checkpoint-3000"
        ),
    },
    "v16_phase2b_step2000": {
        "label": "v16_phase2b_step2000",
        "checkpoint": (
            "/checkpoints/pretrain-output/"
            "v16-qwen3-v11-visionlast4-only-connfrozen-cachekl-lr1e6-3000/"
            "checkpoint-2000"
        ),
    },
    "v16_phase2b_step3000": {
        "label": "v16_phase2b_step3000",
        "checkpoint": (
            "/checkpoints/pretrain-output/"
            "v16-qwen3-v11-visionlast4-only-connfrozen-cachekl-lr1e6-3000/"
            "checkpoint-3000"
        ),
    },
}


def _base_args(script: str, candidate: dict[str, str], task: str, *, detach: bool) -> list[str]:
    label = f"{candidate['label']}_{task}"
    command = [
        "modal",
        "run",
    ]
    if detach:
        command.append("--detach")
    command.extend(
        [
            script,
            "--candidate-checkpoint",
            candidate["checkpoint"],
            "--candidate-label",
            label,
            "--candidate-architecture",
            "v3",
            "--llm-backbone",
            "Qwen/Qwen3-8B",
            "--batch-size",
            "8",
            "--seed",
            "42",
            "--eval-schema-version",
            "v17_fixed_harness",
        ]
    )
    return command


def _finish_args(
    args: list[str],
    *,
    remote_dir: str,
    task: str,
) -> list[str]:
    args = list(args)
    args.extend(
        [
            "--remote-output-path",
            f"{remote_dir}/{task}.json",
            "--output",
            f"/tmp/v17_fixed_harness_{Path(remote_dir).name}_{task}.json",
        ]
    )
    return args


def _task_commands(candidate_key: str, candidate: dict[str, str], *, detach: bool) -> list[dict[str, object]]:
    remote_dir = f"/checkpoints/v17_fixed_harness/{candidate_key}"
    specs: list[tuple[str, list[str]]] = []

    args = _base_args("gqa_checkpoint_eval.py", candidate, "gqa_search_n1000", detach=detach)
    args.extend(
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
    specs.append(("gqa_search_n1000", args))

    args = _base_args(
        "gqa_checkpoint_eval.py",
        candidate,
        "gqa_confirm_n3000_offset1000",
        detach=detach,
    )
    args.extend(
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
    specs.append(("gqa_confirm_n3000_offset1000", args))

    args = _base_args("chartqa_checkpoint_eval.py", candidate, "chartqa_val_full", detach=detach)
    args.extend(["--split", "val", "--max-samples", "0", "--prediction-samples", "1000000"])
    specs.append(("chartqa_val_full", args))

    args = _base_args(
        "textvqa_checkpoint_eval.py",
        candidate,
        "textvqa_validation_full",
        detach=detach,
    )
    args.extend(
        ["--split", "validation", "--max-samples", "0", "--prediction-samples", "1000000"]
    )
    specs.append(("textvqa_validation_full", args))

    for split in ("adversarial", "popular"):
        task = f"pope_{split}_n1000"
        args = _base_args("pope_checkpoint_eval.py", candidate, task, detach=detach)
        args.extend(
            [
                "--pope-split",
                split,
                "--max-samples",
                "1000",
                "--prediction-samples",
                "1000",
            ]
        )
        specs.append((task, args))

    vqa_tasks = {
        "vqa_clean_n3000_seed42": "none",
        "vqa_blank_n3000_seed42": "blank_image",
        "vqa_shuffled_n3000_seed42": "shuffled_image",
        "vqa_wrong_image_n3000_seed42": "wrong_image_same_answer_type",
    }
    for task, perturbation in vqa_tasks.items():
        args = _base_args("vqa_checkpoint_eval.py", candidate, task, detach=detach)
        args.extend(
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
        specs.append((task, args))

    return [
        {
            "candidate": candidate_key,
            "task": task,
            "remote_output_path": f"{remote_dir}/{task}.json",
            "command": _finish_args(args, remote_dir=remote_dir, task=task),
        }
        for task, args in specs
    ]


def build_commands(candidate_keys: list[str], task_filter: set[str], *, detach: bool) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    for key in candidate_keys:
        if key not in CANDIDATES:
            raise KeyError(f"Unknown candidate {key!r}; choices={sorted(CANDIDATES)}")
        for row in _task_commands(key, CANDIDATES[key], detach=detach):
            if task_filter and str(row["task"]) not in task_filter:
                continue
            commands.append(row)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="v11,v16_phase2a_step3000,v16_phase2b_step2000,v16_phase2b_step3000",
        help="Comma-separated candidate keys.",
    )
    parser.add_argument("--tasks", default="", help="Comma-separated task names to keep.")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run without Modal --detach and wait for each eval.",
    )
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing JSON.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/v17_harness_and_data/eval_manifest.json"),
    )
    args = parser.parse_args()

    candidate_keys = [part.strip() for part in args.candidates.split(",") if part.strip()]
    task_filter = {part.strip() for part in args.tasks.split(",") if part.strip()}
    commands = build_commands(candidate_keys, task_filter, detach=not args.foreground)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2)

    if not args.run:
        print(json.dumps(commands, indent=2))
        print(f"Wrote manifest: {args.manifest}")
        return

    for row in commands:
        command = [str(part) for part in row["command"]]
        print(f"Launching {row['candidate']} {row['task']}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
