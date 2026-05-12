#!/usr/bin/env python3
"""Launch fixed tiny V8 Stage 1 VQA image-use probes on Modal."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath


SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)

PROBE_JOBS = (
    ("clean", "none"),
    ("blank", "blank_image"),
    ("shuffled", "shuffled_image"),
    ("wrongimage_sameanswertype", "wrong_image_same_answer_type"),
)


@dataclass(frozen=True)
class Job:
    artifact: str
    command: list[str]


def _remote_path(remote_dir: str, artifact: str) -> str:
    return str(PurePosixPath(remote_dir) / artifact)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for suffix, perturbation in PROBE_JOBS:
        artifact = f"vqa_probe_{args.artifact_prefix}_seed{args.seed}_n{args.samples}_{suffix}.json"
        cmd = [
            "modal",
            "run",
            "vqa_checkpoint_eval.py",
            "--candidate-checkpoint",
            args.checkpoint,
            "--candidate-label",
            args.label,
            "--candidate-architecture",
            args.architecture,
            "--llm-backbone",
            args.llm_backbone,
            "--no-include-baselines",
            "--max-samples",
            str(args.samples),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--prompt-style",
            "training_chat",
            "--system-prompt",
            SYSTEM_PROMPT,
            "--image-perturbation",
            perturbation,
            "--prediction-samples",
            str(args.samples),
            "--eval-schema-version",
            args.eval_schema_version,
            "--remote-output-path",
            _remote_path(args.remote_dir, artifact),
            "--output",
            str(PurePosixPath(args.local_output_dir) / artifact),
        ]
        if args.background:
            cmd.append("--background")
        jobs.append(Job(artifact=artifact, command=cmd))
    return jobs


def _run_parallel(jobs: list[Job], parallelism: int, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    active: list[tuple[subprocess.Popen, object, str]] = []
    queued = list(jobs)
    while queued or active:
        while queued and len(active) < parallelism:
            job = queued.pop(0)
            log_path = os.path.join(log_dir, f"{job.artifact}.log")
            log_file = open(log_path, "w", encoding="utf-8")
            print(f"Starting {job.artifact}; log: {log_path}")
            proc = subprocess.Popen(
                job.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active.append((proc, log_file, job.artifact))

        still_active: list[tuple[subprocess.Popen, object, str]] = []
        for proc, log_file, artifact in active:
            returncode = proc.poll()
            if returncode is None:
                still_active.append((proc, log_file, artifact))
                continue
            log_file.close()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, artifact)
            print(f"Finished {artifact}")
        active = still_active
        if active:
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="Qwen3 V8 Stage 1")
    parser.add_argument("--architecture", default="anymal_v3")
    parser.add_argument("--llm-backbone", default="Qwen/Qwen3-8B")
    parser.add_argument("--artifact-prefix")
    parser.add_argument("--tag", dest="artifact_prefix")
    parser.add_argument("--remote-dir", default="/checkpoints/v8_qwen3_stage1_probes")
    parser.add_argument("--local-output-dir", default="/tmp")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-schema-version", default="v8_stage1_probe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--log-dir", default="/tmp/v8_stage1_probe_logs")
    args = parser.parse_args()
    if not args.artifact_prefix:
        parser.error("--artifact-prefix is required (or use --tag as an alias)")

    jobs = build_jobs(args)
    for job in jobs:
        print(shlex.join(job.command))

    if args.dry_run:
        return

    if args.parallelism <= 1:
        for job in jobs:
            subprocess.run(job.command, check=True)
    else:
        _run_parallel(jobs, parallelism=args.parallelism, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
