#!/usr/bin/env python3
"""Launch the V8 Qwen decoder-swap evaluation bundle on Modal.

Artifacts are written to the shared Modal checkpoint volume. Fetch them with:

    modal volume get anymal-checkpoints /v8_remote/<artifact>.json outputs/v8_remote --force
"""

from __future__ import annotations

import argparse
import os
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


VQA_1K_JOBS = (
    ("seed42_clean_currentcache", 42, "none", 1000),
    ("seed42_blank_currentcache", 42, "blank_image", 1000),
    ("seed42_shuffled_currentcache", 42, "shuffled_image", 1000),
    ("seed42_wrongimage_sameanswertype_currentcache", 42, "wrong_image_same_answer_type", 1000),
    ("seed42_mildblur_currentcache", 42, "mild_blur", 1000),
    ("seed42_centercrop90_currentcache", 42, "center_crop_90", 1000),
    ("seed42_translate5pct_currentcache", 42, "translate_5pct", 1000),
)

VQA_3K_JOBS = (
    ("seed42_n3000_clean_currentcache", 42, "none", 3000),
    ("seed42_n3000_blank_currentcache", 42, "blank_image", 3000),
    ("seed42_n3000_shuffled_currentcache", 42, "shuffled_image", 3000),
    ("seed42_n3000_wrongimage_sameanswertype_currentcache", 42, "wrong_image_same_answer_type", 3000),
)


def _remote_path(remote_dir: str, artifact: str) -> str:
    return str(PurePosixPath(remote_dir) / artifact)


def _vqa_job(args: argparse.Namespace, suffix: str, seed: int, perturbation: str, samples: int) -> Job:
    artifact = f"vqa_eval_{args.artifact_prefix}_{suffix}.json"
    extra_args = [
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
        str(samples),
        "--seed",
        str(seed),
        "--batch-size",
        str(args.vqa_batch_size),
        "--prompt-style",
        "training_chat",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--image-perturbation",
        perturbation,
        "--prediction-samples",
        str(samples),
        "--eval-schema-version",
        args.eval_schema_version,
        "--remote-output-path",
        _remote_path(args.remote_dir, artifact),
        "--output",
        str(PurePosixPath(args.local_output_dir) / artifact),
    ]
    if args.background:
        extra_args.append("--background")
    return Job(script="vqa_checkpoint_eval.py", artifact=artifact, extra_args=tuple(extra_args))


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = [
        _vqa_job(args, suffix, seed, perturbation, samples)
        for suffix, seed, perturbation, samples in VQA_1K_JOBS
    ]
    if args.include_n3000:
        jobs.extend(
            _vqa_job(args, suffix, seed, perturbation, samples)
            for suffix, seed, perturbation, samples in VQA_3K_JOBS
        )

    pope_artifact = f"pope_eval_{args.artifact_prefix}_currentcache.json"
    pope_args = [
        "--candidate-checkpoint",
        args.checkpoint,
        "--candidate-label",
        args.label,
        "--candidate-architecture",
        args.architecture,
        "--llm-backbone",
        args.llm_backbone,
        "--pope-split",
        "adversarial",
        "--max-samples",
        str(args.pope_samples),
        "--seed",
        "42",
        "--batch-size",
        str(args.second_benchmark_batch_size),
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
    jobs.append(Job(script="pope_checkpoint_eval.py", artifact=pope_artifact, extra_args=tuple(pope_args)))

    gqa_artifact = f"gqa_eval_{args.artifact_prefix}_currentcache.json"
    gqa_args = [
        "--candidate-checkpoint",
        args.checkpoint,
        "--candidate-label",
        args.label,
        "--candidate-architecture",
        args.architecture,
        "--llm-backbone",
        args.llm_backbone,
        "--gqa-split",
        "testdev_balanced",
        "--max-samples",
        str(args.gqa_samples),
        "--seed",
        "42",
        "--batch-size",
        str(args.second_benchmark_batch_size),
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
    jobs.append(Job(script="gqa_checkpoint_eval.py", artifact=gqa_artifact, extra_args=tuple(gqa_args)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="Qwen3 V8")
    parser.add_argument("--architecture", default="anymal_v3")
    parser.add_argument("--llm-backbone", default="Qwen/Qwen3-8B")
    parser.add_argument("--artifact-prefix", default="v8_qwen3_v3_robustcal_ckpt100")
    parser.add_argument("--remote-dir", default="/checkpoints/v8_remote")
    parser.add_argument("--local-output-dir", default="/tmp")
    parser.add_argument("--vqa-batch-size", type=int, default=8)
    parser.add_argument("--pope-samples", type=int, default=1000)
    parser.add_argument("--gqa-samples", type=int, default=500)
    parser.add_argument("--second-benchmark-batch-size", type=int, default=8)
    parser.add_argument("--eval-schema-version", default="v8")
    parser.add_argument("--include-n3000", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--log-dir", default="/tmp/v8_eval_logs")
    args = parser.parse_args()

    jobs = build_jobs(args)[args.start_index : args.end_index]
    commands = [["modal", "run", job.script, *job.extra_args] for job in jobs]
    for cmd in commands:
        print(shlex.join(cmd))

    if not args.dry_run:
        if args.parallelism <= 1:
            for cmd in commands:
                subprocess.run(cmd, check=True)
        else:
            os.makedirs(args.log_dir, exist_ok=True)
            active: list[tuple[subprocess.Popen, object, str]] = []
            queued = list(zip(jobs, commands))
            while queued or active:
                while queued and len(active) < args.parallelism:
                    job, cmd = queued.pop(0)
                    log_path = os.path.join(args.log_dir, f"{job.artifact}.log")
                    log_file = open(log_path, "w", encoding="utf-8")
                    print(f"Starting {job.artifact}; log: {log_path}")
                    proc = subprocess.Popen(
                        cmd,
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
                    import time

                    time.sleep(5)

    print("Artifacts:")
    for job in jobs:
        print(_remote_path(args.remote_dir, job.artifact))


if __name__ == "__main__":
    main()
