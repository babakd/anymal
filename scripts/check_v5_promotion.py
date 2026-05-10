#!/usr/bin/env python3
"""Check a V5 candidate against the promoted V4 recipe with stricter gates."""

import argparse
import glob
import json
import math
from pathlib import Path


ANSWER_TYPE_KEYS = ("yes_no", "number", "other")


def _normalize_arch(name):
    value = str(name or "").strip().lower()
    return {
        "v1": "anymal_v1",
        "v2": "anymal_v2",
        "v3": "anymal_v3",
        "v4": "anymal_v4",
        "v5": "anymal_v5",
    }.get(value, value)


def _load_runs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for run in payload.get("runs", []):
        row = dict(run)
        row["_payload"] = payload
        row["_source_file"] = str(path)
        yield row


def _expand(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def _best_run(paths, arch, label):
    runs = []
    for path in paths:
        if not Path(path).exists():
            raise SystemExit(f"Missing {label} artifact: {path}")
        runs.extend(run for run in _load_runs(path) if _normalize_arch(run.get("architecture")) == arch)
    if not runs:
        raise SystemExit(f"No {label} run found for architecture {arch}.")
    return max(runs, key=lambda run: float(run.get("metrics", {}).get("accuracy", 0.0)))


def _metric(run, key, default=0.0):
    return float(run.get("metrics", {}).get(key, default))


def _optional_metric(run, key):
    metrics = run.get("metrics", {})
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def _prediction_sample_count(run):
    samples = run.get("prediction_samples")
    return len(samples) if isinstance(samples, list) else 0


def _predicted_kind_rate(run, answer_type_key, predicted_kind):
    key = f"predicted_{predicted_kind}_rate_on_{answer_type_key}"
    value = _optional_metric(run, key)
    if value is not None:
        return value

    metrics = run.get("metrics", {})
    has_answer_type = f"num_samples_{answer_type_key}" in metrics
    has_kind_rates = any(
        metric_key.startswith("predicted_")
        and metric_key.endswith(f"_rate_on_{answer_type_key}")
        for metric_key in metrics
    )
    if has_answer_type and has_kind_rates:
        return 0.0
    return None


def _fmt(value):
    return "MISSING" if value is None or math.isnan(value) else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4-incumbent",
        default="vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_training_chat_seed42_leftpad.json",
    )
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--candidate-arch", default="anymal_v4")
    parser.add_argument("--v4-arch", default="anymal_v4")
    parser.add_argument("--clean-margin", type=float, default=0.0)
    parser.add_argument(
        "--allow-clean-drop",
        type=float,
        default=0.0,
        help="Allowed clean-accuracy drop. Keep at 0 once the incumbent is raw-clean.",
    )
    parser.add_argument("--max-strict-clean-gap", type=float, default=1.0)
    parser.add_argument("--max-assistant-prefix-rate", type=float, default=0.01)
    parser.add_argument("--max-assistant-prefix-rate-per-type", type=float, default=0.02)
    parser.add_argument("--max-non-yesno-yesno-rate", type=float, default=0.05)
    parser.add_argument("--min-eos-rate", type=float, default=0.98)
    parser.add_argument("--max-hit-max-rate", type=float, default=0.02)
    parser.add_argument("--min-prediction-samples", type=int, default=1000)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    incumbent = _best_run([args.v4_incumbent], _normalize_arch(args.v4_arch), "V4 incumbent")
    candidate = _best_run(_expand(args.candidates), _normalize_arch(args.candidate_arch), "candidate")

    v4_clean = _metric(incumbent, "accuracy")
    cand_clean = _optional_metric(candidate, "accuracy")
    cand_strict = _optional_metric(candidate, "strict_accuracy")
    cand_prefix = _optional_metric(candidate, "assistant_role_prefix_rate")
    cand_eos = _optional_metric(candidate, "eos_rate")
    cand_hit_max = _optional_metric(candidate, "hit_max_new_tokens_rate")
    per_type_prefix = {
        key: _optional_metric(candidate, f"assistant_role_prefix_rate_{key}")
        for key in ANSWER_TYPE_KEYS
    }
    non_yesno_rates = {
        "number": _predicted_kind_rate(candidate, "number", "yes_no"),
        "other": _predicted_kind_rate(candidate, "other", "yes_no"),
    }
    present_non_yesno_rates = [
        value for value in non_yesno_rates.values() if value is not None
    ]
    non_yesno_yesno_rate = max(present_non_yesno_rates) if present_non_yesno_rates else None
    prediction_samples = _prediction_sample_count(candidate)

    missing_evidence = []
    for key, value in (
        ("accuracy", cand_clean),
        ("strict_accuracy", cand_strict),
        ("assistant_role_prefix_rate", cand_prefix),
        ("eos_rate", cand_eos),
        ("hit_max_new_tokens_rate", cand_hit_max),
    ):
        if value is None:
            missing_evidence.append(key)
    for key, value in per_type_prefix.items():
        if value is None:
            missing_evidence.append(f"assistant_role_prefix_rate_{key}")
    for key, value in non_yesno_rates.items():
        if value is None:
            missing_evidence.append(f"predicted_yes_no_rate_on_{key}")
    if prediction_samples < args.min_prediction_samples:
        missing_evidence.append(
            f"prediction_samples<{args.min_prediction_samples} ({prediction_samples})"
        )

    clean_gate = (
        cand_clean is not None
        and (
            cand_clean >= v4_clean + args.clean_margin
            or cand_clean >= v4_clean - args.allow_clean_drop
        )
    )
    strict_gate = (
        cand_clean is not None
        and cand_strict is not None
        and abs(cand_clean - cand_strict) <= args.max_strict_clean_gap
    )
    prefix_gate = (
        cand_prefix is not None
        and cand_prefix <= args.max_assistant_prefix_rate
    )
    per_type_prefix_gate = (
        bool(per_type_prefix)
        and all(
            value is not None and value <= args.max_assistant_prefix_rate_per_type
            for value in per_type_prefix.values()
        )
    )
    hygiene_gate = (
        cand_eos is not None
        and cand_hit_max is not None
        and cand_eos >= args.min_eos_rate
        and cand_hit_max <= args.max_hit_max_rate
    )
    collapse_gate = (
        non_yesno_yesno_rate is not None
        and non_yesno_yesno_rate <= args.max_non_yesno_yesno_rate
    )
    samples_gate = prediction_samples >= args.min_prediction_samples
    promoted = (
        not missing_evidence
        and clean_gate
        and strict_gate
        and prefix_gate
        and per_type_prefix_gate
        and hygiene_gate
        and collapse_gate
        and samples_gate
    )

    print("V5 promotion check")
    print(f"  V4 incumbent: {incumbent.get('label')} ({incumbent.get('_source_file')})")
    print(f"  Candidate:    {candidate.get('label')} ({candidate.get('_source_file')})")
    print("")
    print(f"  V4 clean accuracy:          {v4_clean:.3f}")
    print(f"  Candidate clean accuracy:   {_fmt(cand_clean)}")
    print(f"  Candidate strict accuracy:  {_fmt(cand_strict)}")
    print(
        "  Strict-clean gap:           "
        f"{_fmt(abs(cand_clean - cand_strict) if cand_clean is not None and cand_strict is not None else None)}"
    )
    print(f"  Assistant prefix rate:      {_fmt(cand_prefix)}")
    for key in ANSWER_TYPE_KEYS:
        print(f"  Assistant prefix {key}:     {_fmt(per_type_prefix[key])}")
    print(f"  Non-yes/no yes/no rate max: {_fmt(non_yesno_yesno_rate)}")
    print(f"  EOS rate:                   {_fmt(cand_eos)}")
    print(f"  Max-token-hit rate:         {_fmt(cand_hit_max)}")
    print(f"  Prediction samples:         {prediction_samples}")
    if missing_evidence:
        print(f"  Missing evidence:           {', '.join(missing_evidence)}")
    print("")
    print("Gates")
    print(f"  Complete evidence:       {'PASS' if not missing_evidence else 'FAIL'}")
    print(f"  Clean accuracy:          {'PASS' if clean_gate else 'FAIL'}")
    print(f"  Strict/raw parity:       {'PASS' if strict_gate else 'FAIL'}")
    print(f"  Role-prefix clean:       {'PASS' if prefix_gate else 'FAIL'}")
    print(f"  Per-type role-prefix:    {'PASS' if per_type_prefix_gate else 'FAIL'}")
    print(f"  No yes/no collapse:      {'PASS' if collapse_gate else 'FAIL'}")
    print(f"  Hygiene:                 {'PASS' if hygiene_gate else 'FAIL'}")
    print(f"  Full prediction samples: {'PASS' if samples_gate else 'FAIL'}")

    if promoted:
        print("\nPASS: candidate clears the V5 gate.")
        return

    print("\nFAIL: candidate does not clear the V5 gate.")
    if not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
