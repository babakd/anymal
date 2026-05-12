#!/usr/bin/env python3
"""Summarize a fetched V8 Qwen3 evaluation bundle and check final gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VQA_SUFFIXES = {
    "clean": "seed42_clean_currentcache",
    "blank": "seed42_blank_currentcache",
    "shuffled": "seed42_shuffled_currentcache",
    "wrong": "seed42_wrongimage_sameanswertype_currentcache",
    "mild_blur": "seed42_mildblur_currentcache",
    "center_crop_90": "seed42_centercrop90_currentcache",
    "translate_5pct": "seed42_translate5pct_currentcache",
}

VQA_N3000_SUFFIXES = {
    "clean_n3000": "seed42_n3000_clean_currentcache",
    "blank_n3000": "seed42_n3000_blank_currentcache",
    "shuffled_n3000": "seed42_n3000_shuffled_currentcache",
    "wrong_n3000": "seed42_n3000_wrongimage_sameanswertype_currentcache",
}

GATES = {
    "clean_min": 62.967,
    "blank_max": 39.733,
    "shuffled_max": 37.367,
    "wrong_max": 38.900,
    "blank_gap_min": 23.233,
    "shuffled_gap_min": 25.600,
    "wrong_gap_min": 24.067,
    "perturb_mean_min": 60.108,
    "pope_min": 77.100,
    "gqa_min": 43.800,
    "clean_n3000_min": 63.556,
    "strict_clean_gap_max": 1.0,
    "eos_min": 0.98,
    "max_token_hit_max": 0.02,
    "assistant_prefix_max": 0.01,
    "assistant_prefix_bucket_max": 0.02,
    "yes_no_on_non_yes_no_max": 0.05,
}

GATE_EPSILON = 5e-4


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics(path: Path) -> dict[str, Any]:
    data = _load(path)
    return data["runs"][0]["metrics"]


def _run_payload(path: Path) -> dict[str, Any]:
    data = _load(path)
    return data["runs"][0]


def _artifact_path(root: Path, stem: str) -> Path:
    path = root / stem
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _ok_min(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value >= threshold - GATE_EPSILON


def _ok_max(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value <= threshold + GATE_EPSILON


def _fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.3f}"


def _pass_text(value: bool | None) -> str:
    if value is None:
        return "missing"
    return "pass" if value else "FAIL"


def summarize(root: Path, prefix: str, include_n3000: bool) -> dict[str, Any]:
    vqa: dict[str, dict[str, Any]] = {}
    vqa_prediction_counts: dict[str, int] = {}
    for key, suffix in VQA_SUFFIXES.items():
        artifact = f"vqa_eval_{prefix}_{suffix}.json"
        run = _run_payload(_artifact_path(root, artifact))
        vqa[key] = run["metrics"]
        vqa_prediction_counts[key] = len(run.get("prediction_samples") or [])

    n3000: dict[str, dict[str, Any]] = {}
    if include_n3000:
        for key, suffix in VQA_N3000_SUFFIXES.items():
            artifact = f"vqa_eval_{prefix}_{suffix}.json"
            n3000[key] = _metrics(_artifact_path(root, artifact))

    pope = _metrics(_artifact_path(root, f"pope_eval_{prefix}_currentcache.json"))
    gqa = _metrics(_artifact_path(root, f"gqa_eval_{prefix}_currentcache.json"))

    clean = float(vqa["clean"]["accuracy"])
    blank = float(vqa["blank"]["accuracy"])
    shuffled = float(vqa["shuffled"]["accuracy"])
    wrong = float(vqa["wrong"]["accuracy"])
    mild_blur = float(vqa["mild_blur"]["accuracy"])
    center_crop_90 = float(vqa["center_crop_90"]["accuracy"])
    translate_5pct = float(vqa["translate_5pct"]["accuracy"])
    perturb_mean = (mild_blur + center_crop_90 + translate_5pct) / 3.0
    strict_clean_gap = clean - float(vqa["clean"].get("strict_accuracy", clean))
    pope_acc = float(pope.get("pope_accuracy", pope["accuracy"]))
    gqa_acc = float(gqa.get("gqa_accuracy", gqa["accuracy"]))

    vqa_artifacts = list(vqa.values())
    eos_min = min(float(m.get("eos_rate", 0.0)) for m in vqa_artifacts)
    max_token_hit_max = max(float(m.get("hit_max_new_tokens_rate", 1.0)) for m in vqa_artifacts)
    assistant_prefix_max = max(float(m.get("assistant_role_prefix_rate", 1.0)) for m in vqa_artifacts)
    prefix_bucket_values: list[float] = []
    yes_no_on_non_yes_no_values: list[float] = []
    for m in vqa_artifacts:
        for key, value in m.items():
            if key.startswith("assistant_role_prefix_rate_"):
                prefix_bucket_values.append(float(value))
        for key in ("predicted_yes_no_rate_on_number", "predicted_yes_no_rate_on_other"):
            if key in m:
                yes_no_on_non_yes_no_values.append(float(m[key]))
    assistant_prefix_bucket_max = max(prefix_bucket_values or [0.0])
    yes_no_on_non_yes_no_max = max(yes_no_on_non_yes_no_values or [0.0])

    clean_n3000 = None
    if n3000:
        clean_n3000 = float(n3000["clean_n3000"]["accuracy"])

    min_prediction_samples = min(vqa_prediction_counts.values())
    min_vqa_num_samples = min(int(m["num_samples"]) for m in vqa_artifacts)
    summary = {
        "clean": clean,
        "blank": blank,
        "shuffled": shuffled,
        "wrong": wrong,
        "blank_gap": clean - blank,
        "shuffled_gap": clean - shuffled,
        "wrong_gap": clean - wrong,
        "mild_blur": mild_blur,
        "center_crop_90": center_crop_90,
        "translate_5pct": translate_5pct,
        "perturb_mean": perturb_mean,
        "pope": pope_acc,
        "gqa": gqa_acc,
        "strict_clean_gap": strict_clean_gap,
        "eos_min_across_vqa": eos_min,
        "max_token_hit_max_across_vqa": max_token_hit_max,
        "assistant_prefix_max_across_vqa": assistant_prefix_max,
        "assistant_prefix_bucket_max_across_vqa": assistant_prefix_bucket_max,
        "yes_no_on_non_yes_no_max_across_vqa": yes_no_on_non_yes_no_max,
        "clean_n3000": clean_n3000,
        "min_vqa_prediction_samples": min_prediction_samples,
        "min_vqa_num_samples": min_vqa_num_samples,
    }
    gates = {
        "clean": _ok_min(clean, GATES["clean_min"]),
        "blank": _ok_max(blank, GATES["blank_max"]),
        "shuffled": _ok_max(shuffled, GATES["shuffled_max"]),
        "wrong": _ok_max(wrong, GATES["wrong_max"]),
        "blank_gap": _ok_min(summary["blank_gap"], GATES["blank_gap_min"]),
        "shuffled_gap": _ok_min(summary["shuffled_gap"], GATES["shuffled_gap_min"]),
        "wrong_gap": _ok_min(summary["wrong_gap"], GATES["wrong_gap_min"]),
        "perturb_mean": _ok_min(perturb_mean, GATES["perturb_mean_min"]),
        "pope": _ok_min(pope_acc, GATES["pope_min"]),
        "gqa": _ok_min(gqa_acc, GATES["gqa_min"]),
        "strict_clean_gap": _ok_max(strict_clean_gap, GATES["strict_clean_gap_max"]),
        "eos": _ok_min(eos_min, GATES["eos_min"]),
        "max_token_hit": _ok_max(max_token_hit_max, GATES["max_token_hit_max"]),
        "assistant_prefix": _ok_max(assistant_prefix_max, GATES["assistant_prefix_max"]),
        "assistant_prefix_bucket": _ok_max(
            assistant_prefix_bucket_max, GATES["assistant_prefix_bucket_max"]
        ),
        "yes_no_on_non_yes_no": _ok_max(
            yes_no_on_non_yes_no_max, GATES["yes_no_on_non_yes_no_max"]
        ),
        "vqa_prediction_samples": min_prediction_samples >= min_vqa_num_samples,
    }
    if include_n3000:
        gates["clean_n3000"] = _ok_min(clean_n3000, GATES["clean_n3000_min"])

    return {"prefix": prefix, "summary": summary, "gates": gates, "all_gates_pass": all(gates.values())}


def print_markdown(result: dict[str, Any]) -> None:
    summary = result["summary"]
    gates = result["gates"]
    print(f"# {result['prefix']}")
    print()
    print("| Metric | Value | Gate | Result |")
    print("|---|---:|---:|---|")
    rows = [
        ("clean", summary["clean"], f">= {GATES['clean_min']:.3f}", gates["clean"]),
        ("blank", summary["blank"], f"<= {GATES['blank_max']:.3f}", gates["blank"]),
        ("shuffled", summary["shuffled"], f"<= {GATES['shuffled_max']:.3f}", gates["shuffled"]),
        ("wrong", summary["wrong"], f"<= {GATES['wrong_max']:.3f}", gates["wrong"]),
        ("blank_gap", summary["blank_gap"], f">= {GATES['blank_gap_min']:.3f}", gates["blank_gap"]),
        (
            "shuffled_gap",
            summary["shuffled_gap"],
            f">= {GATES['shuffled_gap_min']:.3f}",
            gates["shuffled_gap"],
        ),
        ("wrong_gap", summary["wrong_gap"], f">= {GATES['wrong_gap_min']:.3f}", gates["wrong_gap"]),
        (
            "perturb_mean",
            summary["perturb_mean"],
            f">= {GATES['perturb_mean_min']:.3f}",
            gates["perturb_mean"],
        ),
        ("POPE", summary["pope"], f">= {GATES['pope_min']:.3f}", gates["pope"]),
        ("GQA", summary["gqa"], f">= {GATES['gqa_min']:.3f}", gates["gqa"]),
        (
            "strict_clean_gap",
            summary["strict_clean_gap"],
            f"<= {GATES['strict_clean_gap_max']:.3f}",
            gates["strict_clean_gap"],
        ),
        ("EOS min", summary["eos_min_across_vqa"], f">= {GATES['eos_min']:.3f}", gates["eos"]),
        (
            "max token hit max",
            summary["max_token_hit_max_across_vqa"],
            f"<= {GATES['max_token_hit_max']:.3f}",
            gates["max_token_hit"],
        ),
        (
            "assistant prefix max",
            summary["assistant_prefix_max_across_vqa"],
            f"<= {GATES['assistant_prefix_max']:.3f}",
            gates["assistant_prefix"],
        ),
        (
            "assistant prefix bucket max",
            summary["assistant_prefix_bucket_max_across_vqa"],
            f"<= {GATES['assistant_prefix_bucket_max']:.3f}",
            gates["assistant_prefix_bucket"],
        ),
        (
            "yes/no on non-yes/no max",
            summary["yes_no_on_non_yes_no_max_across_vqa"],
            f"<= {GATES['yes_no_on_non_yes_no_max']:.3f}",
            gates["yes_no_on_non_yes_no"],
        ),
        (
            "VQA prediction samples min",
            summary["min_vqa_prediction_samples"],
            f">= {summary['min_vqa_num_samples']}",
            gates["vqa_prediction_samples"],
        ),
    ]
    if "clean_n3000" in gates:
        rows.append(
            (
                "clean_n3000",
                summary["clean_n3000"],
                f">= {GATES['clean_n3000_min']:.3f}",
                gates["clean_n3000"],
            )
        )
    for name, value, gate, ok in rows:
        print(f"| {name} | {_fmt(value)} | {gate} | {_pass_text(ok)} |")
    print()
    print(f"all_gates_pass: {str(result['all_gates_pass']).lower()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--dir", default="outputs/v8_qwen3_final_remote")
    parser.add_argument("--include-n3000", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    result = summarize(Path(args.dir), args.artifact_prefix, args.include_n3000)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_markdown(result)


if __name__ == "__main__":
    main()
