#!/usr/bin/env python3
"""Fetch a V8 evaluation bundle from the Modal checkpoint volume."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

from launch_v8_eval_bundle import VQA_1K_JOBS, VQA_3K_JOBS


def _volume_remote_path(remote_dir: str, artifact: str) -> str:
    remote = str(PurePosixPath(remote_dir) / artifact)
    checkpoint_prefix = "/checkpoints/"
    if remote.startswith(checkpoint_prefix):
        return "/" + remote[len(checkpoint_prefix) :]
    if remote == "/checkpoints":
        return f"/{artifact}"
    return remote


def expected_artifacts(prefix: str, include_n3000: bool) -> list[str]:
    artifacts = [
        f"vqa_eval_{prefix}_{suffix}.json"
        for suffix, _seed, _perturbation, _samples in VQA_1K_JOBS
    ]
    if include_n3000:
        artifacts.extend(
            f"vqa_eval_{prefix}_{suffix}.json"
            for suffix, _seed, _perturbation, _samples in VQA_3K_JOBS
        )
    artifacts.extend(
        [
            f"pope_eval_{prefix}_currentcache.json",
            f"gqa_eval_{prefix}_currentcache.json",
        ]
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--remote-dir", default="/checkpoints/v8_qwen3_final_remote")
    parser.add_argument("--local-dir", default="outputs/v8_qwen3_final_remote")
    parser.add_argument("--volume", default="anymal-checkpoints")
    parser.add_argument("--include-n3000", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for artifact in expected_artifacts(args.artifact_prefix, args.include_n3000):
        remote_path = _volume_remote_path(args.remote_dir, artifact)
        cmd = [
            "modal",
            "volume",
            "get",
            args.volume,
            remote_path,
            str(local_dir),
            "--force",
        ]
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(remote_path)
            if not args.keep_going:
                raise SystemExit(result.returncode)
    if failures:
        print("Missing artifacts:")
        for path in failures:
            print(path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
