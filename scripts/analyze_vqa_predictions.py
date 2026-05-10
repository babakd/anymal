#!/usr/bin/env python3
"""Summarize VQA prediction artifacts for generation-format failures.

This is intentionally artifact-only, so it can be run before spending GPU on a
new checkpoint. It is especially useful for detecting role-prefix leakage such
as ``assistant\n\nyes`` and answer-type mode collapse.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ASSISTANT_ROLE_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r]+\s*", re.IGNORECASE)
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
}
ARTICLES = {"a", "an", "the"}


def _iter_predictions(payload):
    if isinstance(payload, list):
        yield from payload
        return

    if isinstance(payload, dict):
        if isinstance(payload.get("prediction_samples"), list):
            yield from payload["prediction_samples"]
        for run in payload.get("runs", []):
            if isinstance(run.get("prediction_samples"), list):
                yield from run["prediction_samples"]


def _prediction_kind(answer):
    answer = str(answer or "").strip().lower()
    if not answer:
        return "empty"
    if answer in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"\d+([./-]\d+)?", answer) or answer in NUMBER_WORDS:
        return "number"
    return "other"


def _process_answer(answer, strip_assistant_prefix=True):
    answer = str(answer or "").lower().strip()
    if strip_assistant_prefix:
        answer = ASSISTANT_ROLE_PREFIX_RE.sub("", answer).strip()
    answer = answer.split("\n")[0].split(".")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(word for word in answer.split() if word not in ARTICLES)


def _vqa_score(prediction, ground_truths):
    processed = [_process_answer(answer) for answer in ground_truths]
    return min(1.0, sum(1 for answer in processed if answer == prediction) / 3.0)


def _summarize(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    predictions = list(_iter_predictions(payload))
    answer_counts = Counter()
    raw_counts = Counter()
    by_answer_type = defaultdict(Counter)
    kind_by_answer_type = defaultdict(Counter)
    prefix_by_answer_type = defaultdict(lambda: {"prefix": 0, "count": 0})
    clean_score = 0.0
    strict_score = 0.0
    scored_count = 0

    for row in predictions:
        raw_answer = str(row.get("raw_answer", "") or "")
        answer = str(row.get("answer", "") or "").strip().lower()
        if raw_answer:
            answer = _process_answer(raw_answer, strip_assistant_prefix=True)
        strict_answer = (
            str(row.get("strict_answer", "") or "").strip().lower()
            or _process_answer(raw_answer, strip_assistant_prefix=False)
        )
        answer_type = str(row.get("answer_type", "") or "unknown")
        has_prefix = bool(row.get("assistant_role_prefix")) or bool(
            ASSISTANT_ROLE_PREFIX_RE.match(raw_answer)
        )
        ground_truths = row.get("answers") or []
        if ground_truths:
            clean_score += _vqa_score(answer, ground_truths)
            strict_score += _vqa_score(strict_answer, ground_truths)
            scored_count += 1

        answer_counts[answer] += 1
        if raw_answer:
            raw_counts[raw_answer.lower()] += 1
        by_answer_type[answer_type][answer] += 1
        kind_by_answer_type[answer_type][_prediction_kind(answer)] += 1
        prefix_by_answer_type[answer_type]["count"] += 1
        if has_prefix:
            prefix_by_answer_type[answer_type]["prefix"] += 1

    return {
        "path": str(path),
        "num_predictions": len(predictions),
        "clean_accuracy": 100.0 * clean_score / scored_count if scored_count else None,
        "strict_accuracy": 100.0 * strict_score / scored_count if scored_count else None,
        "assistant_role_prefix_rate": (
            sum(bucket["prefix"] for bucket in prefix_by_answer_type.values())
            / len(predictions)
            if predictions
            else 0.0
        ),
        "top_answers": answer_counts.most_common(20),
        "top_raw_answers": raw_counts.most_common(20),
        "answer_types": {
            answer_type: {
                "count": sum(counter.values()),
                "assistant_role_prefix_rate": (
                    prefix_by_answer_type[answer_type]["prefix"]
                    / prefix_by_answer_type[answer_type]["count"]
                    if prefix_by_answer_type[answer_type]["count"]
                    else 0.0
                ),
                "top_answers": counter.most_common(12),
                "predicted_kinds": dict(kind_by_answer_type[answer_type]),
            }
            for answer_type, counter in sorted(by_answer_type.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VQA prediction samples for role-prefix and mode-collapse signals."
    )
    parser.add_argument("artifacts", nargs="+", help="Prediction/eval JSON files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a compact text report.",
    )
    args = parser.parse_args()

    summaries = [_summarize(Path(path)) for path in args.artifacts]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for summary in summaries:
        print(summary["path"])
        print(f"  predictions: {summary['num_predictions']}")
        if summary["clean_accuracy"] is not None:
            print(f"  clean_accuracy: {summary['clean_accuracy']:.3f}")
            print(f"  strict_accuracy: {summary['strict_accuracy']:.3f}")
        print(
            "  assistant_role_prefix_rate: "
            f"{summary['assistant_role_prefix_rate']:.3f}"
        )
        print(f"  top_answers: {summary['top_answers']}")
        if summary["top_raw_answers"]:
            print(f"  top_raw_answers: {summary['top_raw_answers'][:8]}")
        for answer_type, bucket in summary["answer_types"].items():
            print(
                f"  {answer_type}: n={bucket['count']} "
                f"prefix={bucket['assistant_role_prefix_rate']:.3f} "
                f"kinds={bucket['predicted_kinds']} "
                f"top={bucket['top_answers'][:8]}"
            )


if __name__ == "__main__":
    main()
