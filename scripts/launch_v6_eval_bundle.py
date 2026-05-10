#!/usr/bin/env python3
"""Launch the standard V6 evaluation bundle for one checkpoint on Modal.

The script intentionally launches background eval calls that write their JSON
artifacts into the shared Modal checkpoint volume. Fetch them with:

    modal volume get anymal-checkpoints /v6_remote/<artifact>.json outputs/v6_remote --force
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath


SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)


@dataclass(frozen=True)
class Job:
    script: str
    artifact: str
    extra_args: tuple[str, ...]


VQA_JOBS = (
    ("seed42_clean", 42, "none"),
    ("seed43_clean", 43, "none"),
    ("seed44_clean", 44, "none"),
    ("seed42_resize_up", 42, "resize_up"),
    ("seed42_mildblur", 42, "mild_blur"),
    ("seed42_center_crop90", 42, "center_crop_90"),
    ("seed42_translate5pct", 42, "translate_5pct"),
    ("seed42_blankimage", 42, "blank_image"),
    ("seed42_shuffleimage", 42, "shuffled_image"),
    ("seed42_wrongimage_sameanswertype", 42, "wrong_image_same_answer_type"),
)


def _remote_path(remote_dir: str, artifact: str) -> str:
    return str(PurePosixPath(remote_dir) / artifact)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for suffix, seed, perturbation in VQA_JOBS:
        artifact = f"vqa_eval_{args.artifact_prefix}_{suffix}_leftpad.json"
        extra_args = [
            "--candidate-checkpoint",
            args.checkpoint,
            "--candidate-label",
            args.label,
            "--candidate-architecture",
            args.architecture,
            "--no-include-baselines",
            "--max-samples",
            str(args.vqa_samples),
            "--seed",
            str(seed),
            "--prompt-style",
            "training_chat",
            "--system-prompt",
            SYSTEM_PROMPT,
            "--image-perturbation",
            perturbation,
            "--prediction-samples",
            str(args.vqa_samples),
            "--eval-schema-version",
            args.eval_schema_version,
            "--remote-output-path",
            _remote_path(args.remote_dir, artifact),
            "--output",
            str(PurePosixPath(args.local_output_dir) / artifact),
        ]
        if args.background:
            extra_args.append("--background")
        jobs.append(
            Job(
                script="vqa_checkpoint_eval.py",
                artifact=artifact,
                extra_args=tuple(extra_args),
            )
        )

    if args.second_benchmarks:
        pope_artifact = f"pope_eval_{args.artifact_prefix}_adversarial_seed42_leftpad.json"
        pope_args = [
            "--candidate-checkpoint",
            args.checkpoint,
            "--candidate-label",
            args.label,
            "--candidate-architecture",
            args.architecture,
            "--pope-split",
            "adversarial",
            "--max-samples",
            str(args.pope_samples),
            "--seed",
            "42",
            "--prompt-style",
            "training_chat",
            "--system-prompt",
            SYSTEM_PROMPT,
            "--prediction-samples",
            str(args.pope_samples),
            "--eval-schema-version",
            args.eval_schema_version,
            "--remote-output-path",
            _remote_path(args.remote_dir, pope_artifact),
            "--output",
            str(PurePosixPath(args.local_output_dir) / pope_artifact),
        ]
        if args.background:
            pope_args.append("--background")
        jobs.append(
            Job(
                script="pope_checkpoint_eval.py",
                artifact=pope_artifact,
                extra_args=tuple(pope_args),
            )
        )

        gqa_artifact = f"gqa_eval_{args.artifact_prefix}_testdev500_seed42_leftpad.json"
        gqa_args = [
            "--candidate-checkpoint",
            args.checkpoint,
            "--candidate-label",
            args.label,
            "--candidate-architecture",
            args.architecture,
            "--gqa-split",
            "testdev_balanced",
            "--max-samples",
            str(args.gqa_samples),
            "--seed",
            "42",
            "--prompt-style",
            "training_chat",
            "--system-prompt",
            SYSTEM_PROMPT,
            "--prediction-samples",
            str(args.gqa_samples),
            "--eval-schema-version",
            args.eval_schema_version,
            "--remote-output-path",
            _remote_path(args.remote_dir, gqa_artifact),
            "--output",
            str(PurePosixPath(args.local_output_dir) / gqa_artifact),
        ]
        if args.background:
            gqa_args.append("--background")
        jobs.append(
            Job(
                script="gqa_checkpoint_eval.py",
                artifact=gqa_artifact,
                extra_args=tuple(gqa_args),
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--remote-dir", default="/checkpoints/v6_remote")
    parser.add_argument("--local-output-dir", default="/tmp")
    parser.add_argument("--vqa-samples", type=int, default=1000)
    parser.add_argument("--pope-samples", type=int, default=1000)
    parser.add_argument("--gqa-samples", type=int, default=500)
    parser.add_argument("--eval-schema-version", default="v6")
    parser.add_argument("--second-benchmarks", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based index into the planned job list; useful after a partial launch.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end index into the planned job list.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)[args.start_index:args.end_index]
    for job in jobs:
        cmd = ["modal", "run", job.script, *job.extra_args]
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    print("Artifacts:")
    for job in jobs:
        print(_remote_path(args.remote_dir, job.artifact))


if __name__ == "__main__":
    main()
