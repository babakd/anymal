#!/usr/bin/env python3
"""Print compact metrics for V6 eval artifacts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


METRIC_KEYS = (
    "accuracy",
    "strict_accuracy",
    "accuracy_yes_no",
    "accuracy_number",
    "accuracy_other",
    "pope_accuracy",
    "pope_f1",
    "pope_yes_ratio",
    "gqa_accuracy",
    "gqa_exact_match_rate",
    "eos_rate",
    "hit_max_new_tokens_rate",
    "assistant_role_prefix_rate",
    "predicted_yes_no_rate_on_number",
    "predicted_yes_no_rate_on_other",
)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _artifact_kind(payload: dict, path: str) -> str:
    benchmark = payload.get("benchmark")
    if benchmark:
        return str(benchmark)
    perturbation = payload.get("image_perturbation")
    if perturbation:
        return str(perturbation)
    return Path(path).name


def _iter_paths(patterns: list[str]):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        yield from (matches or [pattern])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args()

    sep = "\t" if args.tsv else "  "
    columns = ["artifact", "label", "kind", "seed", *METRIC_KEYS]
    print(sep.join(columns))
    for path in _iter_paths(args.artifacts):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for run in payload.get("runs", []):
            metrics = run.get("metrics", {})
            row = [
                Path(path).name,
                str(run.get("label", "")),
                _artifact_kind(payload, path),
                str(payload.get("seed", "")),
                *[_fmt(metrics.get(key)) for key in METRIC_KEYS],
            ]
            print(sep.join(row))


if __name__ == "__main__":
    main()
