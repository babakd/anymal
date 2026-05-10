#!/usr/bin/env python3
"""Check V6 causal-campaign artifacts against preregistered gates."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


ANSWER_TYPE_KEYS = ("yes_no", "number", "other")


def _normalize_arch(name: str | None) -> str:
    value = str(name or "").strip().lower()
    return {
        "anymal": "anymal_v1",
        "v1": "anymal_v1",
        "v2": "anymal_v2",
        "v3": "anymal_v3",
        "v4": "anymal_v4",
        "anymal_v1": "anymal_v1",
        "anymal_v2": "anymal_v2",
        "anymal_v3": "anymal_v3",
        "anymal_v4": "anymal_v4",
    }.get(value, value)


def _expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def _load_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected eval artifact object in {path}")
    payload["_source_file"] = path
    return payload


def _iter_runs(payload: dict):
    for run in payload.get("runs", []):
        row = dict(run)
        row["_payload"] = payload
        row["_source_file"] = payload.get("_source_file")
        yield row


def _select_run(path: str, arch: str | None) -> dict:
    payload = _load_payload(path)
    runs = list(_iter_runs(payload))
    if not runs:
        raise SystemExit(f"No runs found in {path}")
    if arch:
        normalized = _normalize_arch(arch)
        runs = [
            run for run in runs
            if _normalize_arch(run.get("candidate_architecture") or run.get("architecture")) == normalized
        ]
    if not runs:
        raise SystemExit(f"No matching run for architecture {arch} in {path}")
    return max(runs, key=lambda run: float(run.get("metrics", {}).get("accuracy", 0.0)))


def _metric(run: dict, key: str) -> float | None:
    value = run.get("metrics", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_count(run: dict) -> int:
    samples = run.get("prediction_samples")
    return len(samples) if isinstance(samples, list) else 0


def _predicted_kind_rate(run: dict, answer_type_key: str, predicted_kind: str) -> float | None:
    key = f"predicted_{predicted_kind}_rate_on_{answer_type_key}"
    value = _metric(run, key)
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


def _fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "MISSING"
    return f"{value:.3f}"


def _schema_ok(run: dict, allow_legacy_schema: bool) -> bool:
    schema = run.get("_payload", {}).get("eval_schema_version")
    return bool(allow_legacy_schema or schema == "v6")


def _hygiene_gates(run: dict, args) -> tuple[bool, list[str]]:
    clean = _metric(run, "accuracy")
    strict = _metric(run, "strict_accuracy")
    prefix = _metric(run, "assistant_role_prefix_rate")
    eos = _metric(run, "eos_rate")
    max_hit = _metric(run, "hit_max_new_tokens_rate")
    samples = _prediction_count(run)
    per_type_prefix = {
        answer_type: _metric(run, f"assistant_role_prefix_rate_{answer_type}")
        for answer_type in ANSWER_TYPE_KEYS
    }
    non_yesno_rates = [
        _predicted_kind_rate(run, "number", "yes_no"),
        _predicted_kind_rate(run, "other", "yes_no"),
    ]

    failures: list[str] = []
    if not _schema_ok(run, args.allow_legacy_schema):
        failures.append("eval_schema_version is not v6")
    if clean is None:
        failures.append("missing accuracy")
    if strict is None:
        failures.append("missing strict_accuracy")
    if clean is not None and strict is not None and abs(clean - strict) > args.max_strict_clean_gap:
        failures.append(
            f"strict-clean gap {abs(clean - strict):.3f} > {args.max_strict_clean_gap:.3f}"
        )
    if prefix is None or prefix > args.max_assistant_prefix_rate:
        failures.append(
            f"assistant prefix rate {_fmt(prefix)} > {args.max_assistant_prefix_rate:.3f}"
        )
    for answer_type, value in per_type_prefix.items():
        if value is None or value > args.max_assistant_prefix_rate_per_type:
            failures.append(
                f"assistant prefix {answer_type} {_fmt(value)} > "
                f"{args.max_assistant_prefix_rate_per_type:.3f}"
            )
    present_non_yesno = [value for value in non_yesno_rates if value is not None]
    if len(present_non_yesno) != len(non_yesno_rates):
        failures.append("missing predicted yes/no rates for non-yes/no answer types")
    elif max(present_non_yesno) > args.max_non_yesno_yesno_rate:
        failures.append(
            f"non-yes/no yes/no rate {max(present_non_yesno):.3f} > "
            f"{args.max_non_yesno_yesno_rate:.3f}"
        )
    if eos is None or eos < args.min_eos_rate:
        failures.append(f"EOS rate {_fmt(eos)} < {args.min_eos_rate:.3f}")
    if max_hit is None or max_hit > args.max_hit_max_rate:
        failures.append(f"max-token-hit rate {_fmt(max_hit)} > {args.max_hit_max_rate:.3f}")
    if samples < args.min_prediction_samples:
        failures.append(f"prediction samples {samples} < {args.min_prediction_samples}")
    return not failures, failures


def _accuracy(run: dict) -> float:
    value = _metric(run, "accuracy")
    if value is None:
        raise SystemExit(f"Missing accuracy in {run.get('_source_file')}")
    return value


def _image_use_gate(clean_run: dict, control_run: dict, min_drop: float) -> tuple[bool, str]:
    clean = _accuracy(clean_run)
    control = _accuracy(control_run)
    drop = clean - control
    return drop >= min_drop, f"{drop:.3f} point drop ({clean:.3f} -> {control:.3f})"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _gte_with_tolerance(left: float, right: float, tolerance: float) -> bool:
    return left + tolerance >= right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, help="Primary clean VQAv2 artifact")
    parser.add_argument("--candidate-arch", default="", help="Run architecture to select")
    parser.add_argument("--blank", help="Blank/gray image-control artifact")
    parser.add_argument("--shuffled", help="Shuffled-image control artifact")
    parser.add_argument("--wrong-image", help="Wrong-image same-answer-type control artifact")
    parser.add_argument("--candidate-perturbations", nargs="*", default=[])
    parser.add_argument("--incumbent-perturbations", nargs="*", default=[])
    parser.add_argument("--allow-legacy-schema", action="store_true")
    parser.add_argument("--max-strict-clean-gap", type=float, default=1.0)
    parser.add_argument("--max-assistant-prefix-rate", type=float, default=0.01)
    parser.add_argument("--max-assistant-prefix-rate-per-type", type=float, default=0.02)
    parser.add_argument("--max-non-yesno-yesno-rate", type=float, default=0.05)
    parser.add_argument("--min-eos-rate", type=float, default=0.98)
    parser.add_argument("--max-hit-max-rate", type=float, default=0.02)
    parser.add_argument("--min-prediction-samples", type=int, default=1000)
    parser.add_argument("--min-image-control-drop", type=float, default=8.0)
    parser.add_argument("--max-single-perturbation-loss", type=float, default=1.0)
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    clean_run = _select_run(args.clean, args.candidate_arch)
    hygiene_ok, hygiene_failures = _hygiene_gates(clean_run, args)

    print("V6 campaign check")
    print(f"  Clean artifact: {args.clean}")
    print(f"  Candidate:      {clean_run.get('label')}")
    print(f"  Clean accuracy: {_fmt(_metric(clean_run, 'accuracy'))}")
    print(f"  Strict accuracy:{_fmt(_metric(clean_run, 'strict_accuracy'))}")
    print(f"  Prefix rate:    {_fmt(_metric(clean_run, 'assistant_role_prefix_rate'))}")
    print(f"  EOS rate:       {_fmt(_metric(clean_run, 'eos_rate'))}")
    print(f"  Max-token hit:  {_fmt(_metric(clean_run, 'hit_max_new_tokens_rate'))}")
    print(f"  Samples:        {_prediction_count(clean_run)}")

    all_ok = hygiene_ok
    print(f"\nGeneration hygiene: {'PASS' if hygiene_ok else 'FAIL'}")
    for failure in hygiene_failures:
        print(f"  - {failure}")

    for label, path in (
        ("blank", args.blank),
        ("shuffled", args.shuffled),
        ("wrong-image", args.wrong_image),
    ):
        if not path:
            continue
        control_run = _select_run(path, args.candidate_arch)
        ok, detail = _image_use_gate(clean_run, control_run, args.min_image_control_drop)
        all_ok = all_ok and ok
        print(f"\nImage-use {label}: {'PASS' if ok else 'FAIL'}")
        print(f"  {detail}; required >= {args.min_image_control_drop:.3f}")

    candidate_paths = _expand(args.candidate_perturbations)
    incumbent_paths = _expand(args.incumbent_perturbations)
    if candidate_paths or incumbent_paths:
        if len(candidate_paths) != len(incumbent_paths):
            raise SystemExit("--candidate-perturbations and --incumbent-perturbations must have equal length")
        candidate_scores = [_accuracy(_select_run(path, args.candidate_arch)) for path in candidate_paths]
        incumbent_scores = [_accuracy(_select_run(path, "")) for path in incumbent_paths]
        mean_ok = _gte_with_tolerance(
            _mean(candidate_scores),
            _mean(incumbent_scores),
            args.float_tolerance,
        )
        single_ok = all(
            _gte_with_tolerance(
                cand,
                inc - args.max_single_perturbation_loss,
                args.float_tolerance,
            )
            for cand, inc in zip(candidate_scores, incumbent_scores)
        )
        all_ok = all_ok and mean_ok and single_ok
        print(f"\nRobustness mean: {'PASS' if mean_ok else 'FAIL'}")
        print(f"  candidate={_mean(candidate_scores):.3f} incumbent={_mean(incumbent_scores):.3f}")
        print(f"Single perturbation losses: {'PASS' if single_ok else 'FAIL'}")
        for cand_path, inc_path, cand, inc in zip(candidate_paths, incumbent_paths, candidate_scores, incumbent_scores):
            print(f"  {Path(cand_path).name}: {cand:.3f}; {Path(inc_path).name}: {inc:.3f}; delta={cand - inc:.3f}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
