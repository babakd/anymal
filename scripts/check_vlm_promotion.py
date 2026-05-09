#!/usr/bin/env python3
"""Check VLM candidate eval artifacts against V1 and incumbent baselines.

This guard is intentionally artifact-only: it reads saved VQA eval JSON files
and applies the promotion rules from the architecture plans. It should be run
before calling any V4 checkpoint stable.
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
    "avg_generated_tokens",
)


def _normalize_arch(name: str) -> str:
    value = str(name or "").strip().lower()
    aliases = {
        "v1": "anymal_v1",
        "v2": "anymal_v2",
        "v3": "anymal_v3",
        "v4": "anymal_v4",
    }
    return aliases.get(value, value)


def _load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_source_file"] = str(path)
    return payload


def _load_runs(path):
    payload = _load_payload(path)
    for run in payload.get("runs", []):
        run = dict(run)
        run["source_file"] = str(path)
        run["_payload"] = payload
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
    return _normalize_arch(run.get("architecture", ""))


def _score_key(run):
    metrics = run.get("metrics", {})
    return (
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("accuracy_yes_no", 0.0)),
        -float(metrics.get("hit_max_new_tokens_rate", 1.0)),
        float(metrics.get("eos_rate", 0.0)),
    )


def _fmt_metric(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _find_best_run(paths, arch, label):
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing {label} files: {missing}")

    runs = []
    for path in paths:
        runs.extend(run for run in _load_runs(path) if _architecture(run) == arch)
    if not runs:
        raise SystemExit(f"No {label} runs found for architecture {arch}.")
    return max(runs, key=_score_key)


def _protocol_value(run, key):
    payload = run.get("_payload", {})
    if key == "image_perturbation" and key not in payload:
        return "none"
    return payload.get(key)


def _check_protocol(run, args, label):
    expected = {
        "max_samples": args.max_samples,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "image_perturbation": args.image_perturbation,
    }
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        found = _protocol_value(run, key)
        if found != expected_value:
            raise SystemExit(
                f"{label} protocol mismatch for {key}: "
                f"found {found!r}, expected {expected_value!r}."
            )


def _print_comparison(title, left_name, left_metrics, right_name, right_metrics):
    print(title)
    print(f"Metric                  {left_name:>12s} {right_name:>12s}      Delta")
    for key in METRIC_KEYS:
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        if left_value is None or right_value is None:
            delta = "n/a"
        else:
            delta = f"{float(right_value) - float(left_value):+.3f}"
        print(
            f"{key:24s}{_fmt_metric(left_metrics, key):>13s}"
            f"{_fmt_metric(right_metrics, key):>13s}{delta:>11s}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check VLM candidate eval artifacts against V1 and incumbent gates."
    )
    parser.add_argument(
        "--v1-baseline",
        default="vqa_checkpoint_eval_baselines_1000_seed42_training_chat_directprompt_postprocessfix.json",
        help="Eval JSON containing the V1 floor.",
    )
    parser.add_argument(
        "--incumbent",
        default="vqa_eval_v3_direct_calibration_ckpt100_training_chat_postprocessfix.json",
        help="Eval JSON containing the current incumbent.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["vqa_eval_v4*_training_chat.json"],
        help="Candidate eval JSON files or glob patterns.",
    )
    parser.add_argument("--candidate-arch", default="anymal_v4")
    parser.add_argument("--incumbent-arch", default="anymal_v3")
    parser.add_argument("--v1-margin", type=float, default=0.0)
    parser.add_argument("--incumbent-margin", type=float, default=0.0)
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.10,
        help="Overall points below incumbent allowed for a yes/no recovery pass.",
    )
    parser.add_argument("--min-yesno-gain", type=float, default=1.0)
    parser.add_argument("--max-other-drop", type=float, default=0.25)
    parser.add_argument("--min-eos-rate", type=float, default=0.98)
    parser.add_argument("--max-hit-max-rate", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-style", default="training_chat")
    parser.add_argument("--image-perturbation", default="none")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    candidate_arch = _normalize_arch(args.candidate_arch)
    incumbent_arch = _normalize_arch(args.incumbent_arch)

    v1 = _find_best_run([args.v1_baseline], "anymal_v1", "V1 baseline")
    incumbent = _find_best_run([args.incumbent], incumbent_arch, "incumbent")
    candidate = _find_best_run(
        _expand_paths(args.candidates),
        candidate_arch,
        "candidate",
    )

    for label, run in (
        ("V1 baseline", v1),
        ("incumbent", incumbent),
        ("candidate", candidate),
    ):
        _check_protocol(run, args, label)

    v1_metrics = v1.get("metrics", {})
    incumbent_metrics = incumbent.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})

    candidate_acc = float(candidate_metrics.get("accuracy", 0.0))
    v1_acc = float(v1_metrics.get("accuracy", 0.0))
    incumbent_acc = float(incumbent_metrics.get("accuracy", 0.0))
    candidate_yesno = float(candidate_metrics.get("accuracy_yes_no", 0.0))
    incumbent_yesno = float(incumbent_metrics.get("accuracy_yes_no", 0.0))
    candidate_other = float(candidate_metrics.get("accuracy_other", 0.0))
    incumbent_other = float(incumbent_metrics.get("accuracy_other", 0.0))
    eos_rate = float(candidate_metrics.get("eos_rate", 0.0))
    hit_max = float(candidate_metrics.get("hit_max_new_tokens_rate", 1.0))

    clears_v1 = candidate_acc >= v1_acc + args.v1_margin
    beats_incumbent = candidate_acc >= incumbent_acc + args.incumbent_margin
    yesno_recovery = (
        candidate_acc >= incumbent_acc - args.match_tolerance
        and candidate_yesno >= incumbent_yesno + args.min_yesno_gain
        and candidate_other >= incumbent_other - args.max_other_drop
    )
    hygiene = eos_rate >= args.min_eos_rate and hit_max <= args.max_hit_max_rate
    promoted = clears_v1 and hygiene and (beats_incumbent or yesno_recovery)

    print("VLM promotion check")
    print(f"  V1 floor:    {v1.get('label', v1.get('key'))}")
    print(f"  Incumbent:   {incumbent.get('label', incumbent.get('key'))}")
    print(f"  Candidate:   {candidate.get('label', candidate.get('key'))}")
    print(f"  Source:      {candidate.get('source_file')}")
    print("")
    _print_comparison("Against V1 floor", "V1", v1_metrics, "candidate", candidate_metrics)
    print("")
    _print_comparison(
        "Against incumbent",
        "incumbent",
        incumbent_metrics,
        "candidate",
        candidate_metrics,
    )
    print("")
    print("Gates")
    print(f"  V1 floor:          {'PASS' if clears_v1 else 'FAIL'}")
    print(f"  Beats incumbent:   {'PASS' if beats_incumbent else 'FAIL'}")
    print(f"  Yes/no recovery:   {'PASS' if yesno_recovery else 'FAIL'}")
    print(f"  Hygiene:           {'PASS' if hygiene else 'FAIL'}")

    if promoted:
        print("\nPASS: candidate clears the promotion gate.")
        return

    message = "\nFAIL: candidate does not clear the promotion gate."
    if args.no_fail:
        print(message)
        return
    raise SystemExit(message)


if __name__ == "__main__":
    main()
