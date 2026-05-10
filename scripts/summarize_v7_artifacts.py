#!/usr/bin/env python3
"""Summarize V7 grounding eval artifacts.

The V7 experiment phase judges candidates by image-use controls and diagnostic
benchmarks, not clean VQAv2 alone. This script is intentionally artifact-only:
give it VQA/POPE/GQA JSON files and it prints compact per-artifact metrics plus
grounding gaps whenever a label has clean and control VQA runs.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


VQA_CONTROL_KINDS = (
    "blank_image",
    "shuffled_image",
    "wrong_image_same_answer_type",
)

METRIC_KEYS = (
    "accuracy",
    "strict_accuracy",
    "accuracy_yes_no",
    "accuracy_number",
    "accuracy_other",
    "eos_rate",
    "hit_max_new_tokens_rate",
    "assistant_role_prefix_rate",
    "predicted_yes_no_rate_on_number",
    "predicted_yes_no_rate_on_other",
    "pope_accuracy",
    "pope_f1",
    "pope_yes_ratio",
    "gqa_accuracy",
    "gqa_exact_match_rate",
)


def _expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object artifact: {path}")
    payload["_source_file"] = path
    return payload


def _kind(payload: dict[str, Any], path: str) -> str:
    if payload.get("benchmark"):
        return str(payload["benchmark"])
    if payload.get("image_perturbation"):
        return str(payload["image_perturbation"])
    return Path(path).stem


def _metric(run: dict[str, Any], key: str) -> float | None:
    value = run.get("metrics", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _top_answers(run: dict[str, Any], key: str = "top_answers", limit: int = 8) -> str:
    answers = run.get("metrics", {}).get(key) or []
    rendered = []
    for item in answers[:limit]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rendered.append(f"{item[0]}:{item[1]}")
    return ", ".join(rendered)


def _row(payload: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    path = str(payload["_source_file"])
    metrics = run.get("metrics", {})
    accuracy = _metric(run, "accuracy")
    strict = _metric(run, "strict_accuracy")
    return {
        "artifact": Path(path).name,
        "path": path,
        "label": run.get("label", ""),
        "architecture": run.get("candidate_architecture") or run.get("architecture", ""),
        "checkpoint": run.get("candidate_checkpoint") or run.get("checkpoint", ""),
        "kind": _kind(payload, path),
        "seed": payload.get("seed"),
        "max_samples": payload.get("max_samples"),
        "eval_schema_version": payload.get("eval_schema_version"),
        "strict_clean_gap": (
            abs(accuracy - strict) if accuracy is not None and strict is not None else None
        ),
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS if key in metrics},
        "top_answers": metrics.get("top_answers", [])[:20],
        "top_raw_answers": metrics.get("top_raw_answers", [])[:20],
    }


def _rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load(path)
        for run in payload.get("runs", []):
            rows.append(_row(payload, run))
    return rows


def _grounding_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["label"], row["architecture"], row["seed"])][row["kind"]] = row

    gaps: list[dict[str, Any]] = []
    for (label, architecture, seed), by_kind in sorted(grouped.items()):
        clean = by_kind.get("none")
        if not clean:
            continue
        clean_acc = clean["metrics"].get("accuracy")
        if clean_acc is None:
            continue
        gap_row: dict[str, Any] = {
            "label": label,
            "architecture": architecture,
            "seed": seed,
            "clean_accuracy": clean_acc,
        }
        for kind in VQA_CONTROL_KINDS:
            control = by_kind.get(kind)
            if control and control["metrics"].get("accuracy") is not None:
                control_acc = control["metrics"]["accuracy"]
                gap_row[f"{kind}_accuracy"] = control_acc
                gap_row[f"{kind}_gap"] = clean_acc - control_acc
        if len(gap_row) > 4:
            gaps.append(gap_row)
    return gaps


def _print_text(rows: list[dict[str, Any]]) -> None:
    print("V7 artifact metrics")
    header = [
        "label",
        "kind",
        "seed",
        "n",
        "acc",
        "strict",
        "gap",
        "yes/no",
        "number",
        "other",
        "EOS",
        "maxhit",
        "prefix",
        "POPE",
        "GQA",
    ]
    print("\t".join(header))
    for row in rows:
        metrics = row["metrics"]
        values = [
            row["label"],
            row["kind"],
            row["seed"],
            row["max_samples"],
            metrics.get("accuracy"),
            metrics.get("strict_accuracy"),
            row["strict_clean_gap"],
            metrics.get("accuracy_yes_no"),
            metrics.get("accuracy_number"),
            metrics.get("accuracy_other"),
            metrics.get("eos_rate"),
            metrics.get("hit_max_new_tokens_rate"),
            metrics.get("assistant_role_prefix_rate"),
            metrics.get("pope_accuracy"),
            metrics.get("gqa_accuracy"),
        ]
        print("\t".join(_fmt(value) for value in values))
        tops = _top_answers({"metrics": {"top_answers": row["top_answers"]}})
        if tops:
            print(f"  top cleaned: {tops}")
        raw_tops = _top_answers({"metrics": {"top_answers": row["top_raw_answers"]}}, limit=5)
        if raw_tops:
            print(f"  top raw: {raw_tops}")

    gaps = _grounding_gaps(rows)
    if gaps:
        print("\nGrounding gaps")
        gap_header = [
            "label",
            "seed",
            "clean",
            "blank_gap",
            "shuffle_gap",
            "wrong_gap",
        ]
        print("\t".join(gap_header))
        for row in gaps:
            print(
                "\t".join(
                    _fmt(value)
                    for value in (
                        row.get("label"),
                        row.get("seed"),
                        row.get("clean_accuracy"),
                        row.get("blank_image_gap"),
                        row.get("shuffled_image_gap"),
                        row.get("wrong_image_same_answer_type_gap"),
                    )
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", help="Eval JSON files or glob patterns")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    rows = _rows(_expand(args.artifacts))
    result = {"rows": rows, "grounding_gaps": _grounding_gaps(rows)}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_text(rows)


if __name__ == "__main__":
    main()
