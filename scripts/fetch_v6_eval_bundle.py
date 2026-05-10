#!/usr/bin/env python3
"""Fetch a standard V6 evaluation bundle from the Modal checkpoint volume."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

from launch_v6_eval_bundle import VQA_JOBS


def expected_artifacts(prefix: str, second_benchmarks: bool) -> list[str]:
    artifacts = [
        f"vqa_eval_{prefix}_{suffix}_leftpad.json"
        for suffix, _seed, _perturbation in VQA_JOBS
    ]
    if second_benchmarks:
        artifacts.extend(
            [
                f"pope_eval_{prefix}_adversarial_seed42_leftpad.json",
                f"gqa_eval_{prefix}_testdev500_seed42_leftpad.json",
            ]
        )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--remote-dir", default="/v6_remote")
    parser.add_argument("--local-dir", default="outputs/v6_remote")
    parser.add_argument("--volume", default="anymal-checkpoints")
    parser.add_argument("--second-benchmarks", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for artifact in expected_artifacts(args.artifact_prefix, args.second_benchmarks):
        remote_path = str(PurePosixPath(args.remote_dir) / artifact)
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
