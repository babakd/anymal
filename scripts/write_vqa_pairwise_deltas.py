#!/usr/bin/env python3
"""Write paired VQA comparison files from two eval artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ASSISTANT_ROLE_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r]+\s*", re.IGNORECASE)
ARTICLES = {"a", "an", "the"}


def _process_answer(answer: str, strip_assistant_prefix: bool = True) -> str:
    value = str(answer or "").lower().strip()
    if strip_assistant_prefix:
        value = ASSISTANT_ROLE_PREFIX_RE.sub("", value).strip()
    value = value.split("\n")[0].split(".")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    value = re.sub(r"[^\w\s]", "", value)
    return " ".join(word for word in value.split() if word not in ARTICLES)


def _vqa_score(prediction: str, answers: list[str]) -> float:
    processed = [_process_answer(answer) for answer in answers]
    return min(1.0, sum(1 for answer in processed if answer == prediction) / 3.0)


def _load_run(path: Path, label: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    if not runs:
        raise SystemExit(f"No runs in {path}")
    if label:
        runs = [run for run in runs if run.get("label") == label]
        if not runs:
            raise SystemExit(f"No run label {label!r} in {path}")
    return payload, runs[0]


def _key(row: dict[str, Any]) -> str:
    question_id = row.get("question_id")
    if question_id is not None:
        return f"qid:{question_id}"
    return f"image:{row.get('image_id')}|q:{row.get('question')}"


def _indexed_samples(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = run.get("prediction_samples")
    if not isinstance(samples, list):
        raise SystemExit("Run has no prediction_samples list")
    return {_key(row): row for row in samples}


def _answer(row: dict[str, Any]) -> str:
    answer = str(row.get("answer", "") or "").strip().lower()
    if answer:
        return answer
    return _process_answer(str(row.get("raw_answer", "") or ""))


def _delta_rows(
    left: dict[str, Any],
    right: dict[str, Any],
    left_name: str,
    right_name: str,
    min_score: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left_samples = _indexed_samples(left)
    right_samples = _indexed_samples(right)
    for key in sorted(set(left_samples) & set(right_samples)):
        left_row = left_samples[key]
        right_row = right_samples[key]
        answers = left_row.get("answers") or right_row.get("answers") or []
        if not answers:
            continue
        left_answer = _answer(left_row)
        right_answer = _answer(right_row)
        left_score = _vqa_score(left_answer, answers)
        right_score = _vqa_score(right_answer, answers)
        if left_score <= min_score or right_score > min_score:
            continue
        rows.append(
            {
                "question_id": left_row.get("question_id"),
                "image_id": left_row.get("image_id"),
                "source_image_id": left_row.get("source_image_id"),
                "image_control": left_row.get("image_control"),
                "answer_type": left_row.get("answer_type"),
                "question_type": left_row.get("question_type"),
                "question": left_row.get("question"),
                "answers": answers,
                f"{left_name}_answer": left_answer,
                f"{left_name}_raw_answer": left_row.get("raw_answer"),
                f"{left_name}_score": left_score,
                f"{right_name}_answer": right_answer,
                f"{right_name}_raw_answer": right_row.get("raw_answer"),
                f"{right_name}_score": right_score,
            }
        )
    return rows


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path, help="First VQA artifact")
    parser.add_argument("--right", required=True, type=Path, help="Second VQA artifact")
    parser.add_argument("--left-name", required=True, help="Short key for first artifact")
    parser.add_argument("--right-name", required=True, help="Short key for second artifact")
    parser.add_argument("--left-label", help="Optional run label selector for first artifact")
    parser.add_argument("--right-label", help="Optional run label selector for second artifact")
    parser.add_argument("--left-better-output", required=True, type=Path)
    parser.add_argument("--right-better-output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit per output")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Score threshold for treating the losing side as wrong.",
    )
    args = parser.parse_args()

    left_payload, left_run = _load_run(args.left, args.left_label)
    right_payload, right_run = _load_run(args.right, args.right_label)
    left_better = _delta_rows(
        left_run, right_run, args.left_name, args.right_name, args.min_score
    )
    right_better = _delta_rows(
        right_run, left_run, args.right_name, args.left_name, args.min_score
    )
    if args.limit:
        left_better = left_better[: args.limit]
        right_better = right_better[: args.limit]

    common_meta = {
        "left_artifact": str(args.left),
        "right_artifact": str(args.right),
        "left_label": left_run.get("label"),
        "right_label": right_run.get("label"),
        "left_eval_schema_version": left_payload.get("eval_schema_version"),
        "right_eval_schema_version": right_payload.get("eval_schema_version"),
        "left_image_perturbation": left_payload.get("image_perturbation"),
        "right_image_perturbation": right_payload.get("image_perturbation"),
        "left_seed": left_payload.get("seed"),
        "right_seed": right_payload.get("seed"),
    }
    _write(
        args.left_better_output,
        {
            **common_meta,
            "comparison": f"{args.left_name}_correct_{args.right_name}_wrong",
            "num_rows": len(left_better),
            "rows": left_better,
        },
    )
    _write(
        args.right_better_output,
        {
            **common_meta,
            "comparison": f"{args.right_name}_correct_{args.left_name}_wrong",
            "num_rows": len(right_better),
            "rows": right_better,
        },
    )
    print(f"Wrote {args.left_better_output} ({len(left_better)} rows)")
    print(f"Wrote {args.right_better_output} ({len(right_better)} rows)")


if __name__ == "__main__":
    main()
