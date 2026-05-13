#!/usr/bin/env python3
"""Launch V10 Qwen3 Batch A evaluation grids on Modal.

This is a thin orchestrator around the existing checkpoint evaluators. It adds
connector scale overrides, V10 artifact naming, and the cheap GQA-first screens
from ``v10_qwen3_ceiling.md`` without changing evaluator behavior.
"""

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

VQA_METRICS = {
    "clean": "none",
    "blank": "blank_image",
    "shuffled": "shuffled_image",
    "wrong": "wrong_image_same_answer_type",
    "mildblur": "mild_blur",
    "centercrop90": "center_crop_90",
    "translate5pct": "translate_5pct",
}

MODE_METRICS = {
    "gqa-screen": ("gqa",),
    "screen": ("gqa", "shuffled", "blank", "clean"),
    "full-1k": (
        "gqa",
        "shuffled",
        "blank",
        "clean",
        "wrong",
        "mildblur",
        "centercrop90",
        "translate5pct",
        "pope",
    ),
    "full-n3000": ("clean", "blank", "shuffled", "wrong"),
}


@dataclass(frozen=True)
class Job:
    artifact: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    checkpoint: str


def _scale_tag(scale: float) -> str:
    return f"scale{int(round(float(scale) * 1000)):04d}"


def _remote_path(remote_dir: str, artifact: str) -> str:
    return str(PurePosixPath(remote_dir) / artifact)


def _local_path(local_output_dir: str, artifact: str) -> str:
    return str(PurePosixPath(local_output_dir) / artifact)


def _base_args(
    args: argparse.Namespace,
    candidate: Candidate,
    scale: float,
    label: str,
) -> list[str]:
    base = [
        "--candidate-checkpoint",
        candidate.checkpoint,
        "--candidate-label",
        label,
        "--candidate-architecture",
        args.architecture,
        "--llm-backbone",
        args.llm_backbone,
        "--connector-output-scale-override",
        f"{scale:.6g}",
        "--batch-size",
        str(args.batch_size),
        "--prompt-style",
        "training_chat",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--eval-schema-version",
        args.eval_schema_version,
    ]
    return base


def _vqa_job(
    args: argparse.Namespace,
    prefix: str,
    candidate: Candidate,
    scale: float,
    metric: str,
    samples: int,
    seed: int,
    label: str,
) -> Job:
    artifact = f"vqa_eval_{prefix}_seed{seed}_{metric}_n{samples}.json"
    extra = [
        * _base_args(args, candidate, scale, label),
        "--no-include-baselines",
        "--max-samples",
        str(samples),
        "--seed",
        str(seed),
        "--image-perturbation",
        VQA_METRICS[metric],
        "--prediction-samples",
        str(samples),
        "--remote-output-path",
        _remote_path(args.remote_dir, artifact),
        "--output",
        _local_path(args.local_output_dir, artifact),
    ]
    if args.background:
        extra.append("--background")
    return Job(artifact=artifact, command=("modal", "run", args.vqa_script, *extra))


def _gqa_job(
    args: argparse.Namespace,
    prefix: str,
    candidate: Candidate,
    scale: float,
    samples: int,
    seed: int,
    label: str,
) -> Job:
    artifact = f"gqa_eval_{prefix}_testdev_balanced_seed{seed}_n{samples}.json"
    extra = [
        * _base_args(args, candidate, scale, label),
        "--gqa-split",
        "testdev_balanced",
        "--max-samples",
        str(samples),
        "--seed",
        str(seed),
        "--prediction-samples",
        str(samples),
        "--remote-output-path",
        _remote_path(args.remote_dir, artifact),
        "--output",
        _local_path(args.local_output_dir, artifact),
    ]
    if args.background:
        extra.append("--background")
    return Job(artifact=artifact, command=("modal", "run", args.gqa_script, *extra))


def _pope_job(
    args: argparse.Namespace,
    prefix: str,
    candidate: Candidate,
    scale: float,
    samples: int,
    seed: int,
    label: str,
) -> Job:
    artifact = f"pope_eval_{prefix}_adversarial_seed{seed}_n{samples}.json"
    extra = [
        * _base_args(args, candidate, scale, label),
        "--pope-split",
        "adversarial",
        "--max-samples",
        str(samples),
        "--seed",
        str(seed),
        "--prediction-samples",
        str(samples),
        "--remote-output-path",
        _remote_path(args.remote_dir, artifact),
        "--output",
        _local_path(args.local_output_dir, artifact),
    ]
    if args.background:
        extra.append("--background")
    return Job(artifact=artifact, command=("modal", "run", args.pope_script, *extra))


def _parse_csv_floats(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated int")
    return values


def _safe_name(value: str) -> str:
    safe = []
    for char in value.strip():
        if char.isalnum():
            safe.append(char.lower())
        elif char in {"-", "_", "."}:
            safe.append(char.replace(".", "p"))
        else:
            safe.append("_")
    rendered = "".join(safe).strip("_")
    return rendered or "candidate"


def _parse_checkpoint_specs(raw: str) -> list[Candidate]:
    candidates = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" in piece:
            name, checkpoint = piece.split("=", 1)
        else:
            checkpoint = piece
            name = PurePosixPath(checkpoint.rstrip("/")).name
        name = _safe_name(name)
        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise argparse.ArgumentTypeError(f"empty checkpoint in spec: {piece!r}")
        candidates.append(Candidate(name=name, checkpoint=checkpoint))
    if not candidates:
        raise argparse.ArgumentTypeError("expected at least one checkpoint spec")
    return candidates


def build_jobs(args: argparse.Namespace) -> list[Job]:
    metrics = args.metrics or list(MODE_METRICS[args.mode])
    jobs: list[Job] = []
    samples = args.samples
    if args.mode == "full-n3000" and args.samples == 1000:
        samples = 3000

    for candidate in args.candidates:
        for scale in args.scales:
            prefix = f"{args.artifact_prefix}_{candidate.name}_{_scale_tag(scale)}"
            label = f"{args.label} {candidate.name} {_scale_tag(scale)}"
            for metric in metrics:
                if metric == "gqa":
                    jobs.append(
                        _gqa_job(args, prefix, candidate, scale, args.gqa_samples, args.seed, label)
                    )
                elif metric == "pope":
                    jobs.append(
                        _pope_job(args, prefix, candidate, scale, args.pope_samples, args.seed, label)
                    )
                elif metric in VQA_METRICS:
                    vqa_seeds = args.clean_seeds if metric == "clean" else [args.seed]
                    for vqa_seed in vqa_seeds:
                        jobs.append(
                            _vqa_job(args, prefix, candidate, scale, metric, samples, vqa_seed, label)
                        )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
    return jobs[args.start_index : args.end_index]


def _run_commands(jobs: list[Job], args: argparse.Namespace) -> None:
    os.makedirs(args.local_output_dir, exist_ok=True)
    if args.parallelism <= 1:
        for job in jobs:
            subprocess.run(job.command, check=True)
        return

    os.makedirs(args.log_dir, exist_ok=True)
    active: list[tuple[subprocess.Popen, object, str]] = []
    queued = list(jobs)
    while queued or active:
        while queued and len(active) < args.parallelism:
            job = queued.pop(0)
            log_path = os.path.join(args.log_dir, f"{job.artifact}.log")
            log_file = open(log_path, "w", encoding="utf-8")
            print(f"Starting {job.artifact}; log: {log_path}", flush=True)
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
            print(f"Finished {artifact}", flush=True)
        active = still_active
        if active:
            time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint-specs",
        type=_parse_checkpoint_specs,
        help="Comma-separated name=/checkpoints/path specs for multi-checkpoint grids.",
    )
    parser.add_argument("--label", default="V10 Qwen3")
    parser.add_argument("--architecture", default="anymal_v3")
    parser.add_argument("--llm-backbone", default="Qwen/Qwen3-8B")
    parser.add_argument("--scales", type=_parse_csv_floats, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--remote-dir", default="/checkpoints/v10_qwen_ceiling/batch_a")
    parser.add_argument("--local-output-dir", default="/tmp")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_METRICS),
        default="screen",
        help="Predefined metric bundle.",
    )
    parser.add_argument(
        "--metrics",
        choices=sorted((*VQA_METRICS.keys(), "gqa", "pope")),
        nargs="+",
        help="Override the selected mode with an explicit metric list.",
    )
    parser.add_argument("--samples", type=int, default=1000, help="VQA sample count.")
    parser.add_argument("--gqa-samples", type=int, default=1000)
    parser.add_argument("--pope-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-seeds", type=_parse_csv_ints, default=[42])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-schema-version", default="v10")
    parser.add_argument("--vqa-script", default="vqa_checkpoint_eval.py")
    parser.add_argument("--gqa-script", default="gqa_checkpoint_eval.py")
    parser.add_argument("--pope-script", default="pope_checkpoint_eval.py")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--log-dir", default="/tmp/v10_eval_grid_logs")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_specs:
        args.candidates = args.checkpoint_specs
    elif args.checkpoint:
        args.candidates = [Candidate(name="candidate", checkpoint=args.checkpoint)]
    else:
        parser.error("provide --checkpoint or --checkpoint-specs")

    jobs = build_jobs(args)
    for job in jobs:
        print(shlex.join(job.command))
    if not args.dry_run:
        _run_commands(jobs, args)

    print("Artifacts:")
    for job in jobs:
        print(_remote_path(args.remote_dir, job.artifact))


if __name__ == "__main__":
    main()
