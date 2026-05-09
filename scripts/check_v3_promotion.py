#!/usr/bin/env python3
"""Check whether V3 candidates clear the V1 held-out VQA baseline.

This intentionally reads saved eval JSON artifacts only. It is a cheap guardrail
for recipe selection before spending another Modal run on a candidate.
"""

import argparse
import glob
import json
from pathlib import Path


METRIC_KEYS = (
    "accuracy",
    "accuracy_number",
    "accuracy_other",
    "accuracy_yes_no",
    "eos_rate",
    "hit_max_new_tokens_rate",
)


def _load_runs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    for run in runs:
        run = dict(run)
        run["source_file"] = str(path)
        yield run


def _expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    seen = set()
    result = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _architecture(run):
    return str(run.get("architecture", "")).lower()


def _score_key(run):
    metrics = run.get("metrics", {})
    return (
        float(metrics.get("accuracy", 0.0)),
        -float(metrics.get("hit_max_new_tokens_rate", 1.0)),
        float(metrics.get("eos_rate", 0.0)),
    )


def _fmt_metric(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Check V3 eval artifacts against the V1 baseline."
    )
    parser.add_argument(
        "--baseline",
        default="vqa_checkpoint_eval_baselines_1000_seed42_training_chat.json",
        help="Eval JSON containing the V1 baseline run.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["vqa_eval_v3*_training_chat.json"],
        help="Candidate eval JSON files or glob patterns.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Required V3 overall-accuracy margin over V1.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Print the report but exit 0 even if V3 does not clear the bar.",
    )
    args = parser.parse_args()

    baseline_runs = list(_load_runs(args.baseline))
    v1_runs = [run for run in baseline_runs if _architecture(run) in {"v1", "anymal_v1"}]
    if not v1_runs:
        raise SystemExit(f"No V1 run found in {args.baseline}")
    v1 = max(v1_runs, key=_score_key)

    candidate_paths = _expand_paths(args.candidates)
    missing = [path for path in candidate_paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing candidate files: {missing}")

    v3_runs = []
    for path in candidate_paths:
        v3_runs.extend(
            run for run in _load_runs(path)
            if _architecture(run) in {"v3", "anymal_v3"}
        )
    if not v3_runs:
        raise SystemExit("No V3 candidate runs found.")

    best = max(v3_runs, key=_score_key)
    v1_metrics = v1.get("metrics", {})
    best_metrics = best.get("metrics", {})
    margin = float(best_metrics.get("accuracy", 0.0)) - float(v1_metrics.get("accuracy", 0.0))
    clears = margin >= float(args.min_margin)

    print("V3 promotion check")
    print(f"  V1 baseline: {v1.get('label', v1.get('key'))}")
    print(f"  Best V3:     {best.get('label', best.get('key'))}")
    print(f"  Source:      {best.get('source_file')}")
    print(f"  Margin:      {margin:.3f} points")
    print("")
    print("Metric                  V1        Best V3   Delta")
    for key in METRIC_KEYS:
        v1_value = v1_metrics.get(key)
        best_value = best_metrics.get(key)
        if v1_value is None or best_value is None:
            delta = "n/a"
        else:
            delta = f"{float(best_value) - float(v1_value):+.3f}"
        print(f"{key:24s}{_fmt_metric(v1_metrics, key):>9s}{_fmt_metric(best_metrics, key):>11s}{delta:>9s}")

    if clears:
        print("\nPASS: V3 clears the V1 overall-accuracy bar.")
        return

    message = (
        f"\nFAIL: V3 margin {margin:.3f} is below required margin "
        f"{float(args.min_margin):.3f}."
    )
    if args.no_fail:
        print(message)
        return
    raise SystemExit(message)


if __name__ == "__main__":
    main()
