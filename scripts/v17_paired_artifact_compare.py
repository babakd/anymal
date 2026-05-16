#!/usr/bin/env python3
"""Compute V17 paired-bootstrap deltas from corrected eval artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

PROJECT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from evaluation.checkpoint_eval.paired_bootstrap import paired_bootstrap_mean_ci


ASSISTANT_ROLE_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r]+\s*", re.IGNORECASE)
THINKING_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
THINKING_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)
CHAT_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
ARTICLES = {"a", "an", "the"}
DIGIT_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
CONTRACTIONS = {
    "can't": "cant",
    "couldn't": "couldnt",
    "didn't": "didnt",
    "doesn't": "doesnt",
    "don't": "dont",
    "hadn't": "hadnt",
    "hasn't": "hasnt",
    "haven't": "havent",
    "isn't": "isnt",
    "shouldn't": "shouldnt",
    "wasn't": "wasnt",
    "weren't": "werent",
    "won't": "wont",
    "wouldn't": "wouldnt",
}


def _load_run(path: Path, label: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs") or []
    if label:
        runs = [run for run in runs if run.get("label") == label or run.get("key") == label]
    if not runs:
        raise SystemExit(f"No matching run in {path}")
    return payload, runs[0]


def _rows_by_key(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = run.get("prediction_samples")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            f"Run {run.get('label') or run.get('key') or '<unknown>'} has no prediction_samples. "
            "Re-run the eval with --prediction-samples equal to the slice size."
        )
    indexed = {}
    for row in rows:
        qid = row.get("question_id")
        key = f"qid:{qid}" if qid is not None else f"image:{row.get('image_id')}|q:{row.get('question')}"
        indexed[key] = row
    return indexed


def _clean_answer(answer: Any, strip_assistant_prefix: bool = True) -> str:
    text = str(answer or "").strip()
    if THINKING_CLOSE_RE.search(text):
        text = THINKING_CLOSE_RE.split(text)[-1].strip()
    elif THINKING_TAG_RE.search(text):
        text = ""
    text = CHAT_SPECIAL_TOKEN_RE.sub(" ", text).lower().strip()
    if strip_assistant_prefix:
        text = ASSISTANT_ROLE_PREFIX_RE.sub("", text).strip()
    text = re.split(r"[\n\r]+", text, maxsplit=1)[0]
    text = re.split(r"(?<!\d)\.(?!\d)", text, maxsplit=1)[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    words = [CONTRACTIONS.get(word, word) for word in text.split()]
    words = [DIGIT_WORD_TO_DIGIT.get(word, word) for word in words]
    text = re.sub(r"[^\w\s]", "", " ".join(words))
    words = [DIGIT_WORD_TO_DIGIT.get(word, word) for word in text.split() if word not in ARTICLES]
    return " ".join(words).strip()


def _chartqa_strip(text: Any) -> str:
    value = str(text or "").strip()
    if THINKING_CLOSE_RE.search(value):
        value = THINKING_CLOSE_RE.split(value)[-1].strip()
    elif THINKING_TAG_RE.search(value):
        value = ""
    value = CHAT_SPECIAL_TOKEN_RE.sub(" ", value)
    value = ASSISTANT_ROLE_PREFIX_RE.sub("", value).strip().lower()
    value = value.split("\n", 1)[0].strip()
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return value


def _chartqa_number(text: Any) -> float | None:
    value = _chartqa_strip(text)
    value = re.sub(r"^[\$€£¥]\s*", "", value).replace(",", "").strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    value = value.rstrip(".").strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _chartqa_text(text: Any) -> str:
    value = _chartqa_strip(text)
    value = re.sub(r"[^\w\s.]", " ", value)
    value = re.sub(r"(?<!\d)\.(?!\d)", " ", value)
    return " ".join(word for word in value.split() if word not in ARTICLES)


def _chartqa_relaxed_match(pred: Any, gold: Any) -> bool:
    gold_num = _chartqa_number(gold)
    if gold_num is not None:
        pred_num = _chartqa_number(pred)
        if pred_num is None:
            return False
        return abs(pred_num - gold_num) / max(abs(gold_num), 1e-12) <= 0.05
    return _chartqa_text(pred) == _chartqa_text(gold)


def _answers(row: dict[str, Any]) -> list[str]:
    raw = row.get("answers") or []
    if raw and isinstance(raw[0], dict):
        return [str(item.get("answer", "")) for item in raw]
    return [str(item) for item in raw]


def _vqa_score(row: dict[str, Any]) -> float:
    pred = _clean_answer(row.get("answer") or row.get("raw_answer"))
    answers = [_clean_answer(answer) for answer in _answers(row)]
    if not answers:
        return 0.0
    return min(1.0, sum(1 for answer in answers if pred and pred == answer) / 3.0)


def _exact_score(row: dict[str, Any], key: str, fallback: Callable[[dict[str, Any]], bool]) -> float:
    if isinstance(row.get(key), bool):
        return float(row[key])
    return float(fallback(row))


def _pope_score(row: dict[str, Any]) -> float:
    pred = str(row.get("pope_answer") or row.get("answer") or "").strip().lower()
    if pred not in {"yes", "no"}:
        raw = str(row.get("raw_answer") or "")
        first = raw.split(".", 1)[0].replace(",", " ")
        words = {word.strip().lower() for word in first.split()}
        pred = "no" if {"no", "not"} & words else "yes"
    answers = _answers(row)
    target = str(answers[0] if answers else "").strip().lower()
    return float(pred == target)


def _gqa_score(row: dict[str, Any]) -> float:
    if isinstance(row.get("gqa_exact_match"), bool):
        return float(row["gqa_exact_match"])
    target = str(row.get("gqa_answer") or (_answers(row) or [""])[0])
    return float(_clean_answer(row.get("raw_answer") or row.get("answer")) == _clean_answer(target))


def _chartqa_exact_score(row: dict[str, Any]) -> float:
    pred = _clean_answer(row.get("answer") or row.get("raw_answer"))
    return float(bool(pred) and pred in {_clean_answer(answer) for answer in _answers(row)})


def _chartqa_relaxed_score(row: dict[str, Any]) -> float:
    return float(
        any(
            _chartqa_relaxed_match(str(row.get("raw_answer") or row.get("answer") or ""), answer)
            for answer in _answers(row)
        )
    )


def _textvqa_soft_score(row: dict[str, Any]) -> float:
    if row.get("textvqa_soft_accuracy") is not None:
        return float(row["textvqa_soft_accuracy"])
    return _vqa_score(row)


def _score_functions(benchmark: str) -> dict[str, Callable[[dict[str, Any]], float]]:
    benchmark = benchmark.lower()
    if benchmark == "gqa":
        return {"gqa_accuracy": _gqa_score}
    if benchmark == "chartqa":
        return {
            "chartqa_relaxed_match": lambda row: _exact_score(
                row, "chartqa_relaxed_match", _chartqa_relaxed_score
            ),
            "chartqa_exact_match": lambda row: _exact_score(
                row, "chartqa_exact_match", _chartqa_exact_score
            ),
        }
    if benchmark == "textvqa":
        return {
            "textvqa_exact_match": lambda row: _exact_score(
                row, "textvqa_exact_match", lambda item: _vqa_score(item) > 0.0
            ),
            "textvqa_soft_accuracy": _textvqa_soft_score,
        }
    if benchmark == "pope":
        return {"pope_accuracy": _pope_score}
    if benchmark in {"vqa", "vqav2"}:
        return {"vqa_accuracy": _vqa_score}
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def compare_artifacts(
    *,
    candidate_path: Path,
    baseline_path: Path,
    benchmark: str,
    candidate_label: str | None = None,
    baseline_label: str | None = None,
    seed: int = 12345,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    candidate_payload, candidate_run = _load_run(candidate_path, candidate_label)
    baseline_payload, baseline_run = _load_run(baseline_path, baseline_label)
    candidate_rows = _rows_by_key(candidate_run)
    baseline_rows = _rows_by_key(baseline_run)
    common_keys = sorted(set(candidate_rows) & set(baseline_rows))
    if not common_keys:
        raise SystemExit("No paired prediction rows overlap by question_id/key")
    metrics = {}
    for metric_name, score_fn in _score_functions(benchmark).items():
        candidate_scores = [score_fn(candidate_rows[key]) for key in common_keys]
        baseline_scores = [score_fn(baseline_rows[key]) for key in common_keys]
        ci = paired_bootstrap_mean_ci(
            candidate_scores,
            baseline_scores,
            seed=seed,
            n_resamples=n_resamples,
        )
        metrics[metric_name] = {
            "candidate_accuracy": 100.0 * sum(candidate_scores) / len(candidate_scores),
            "baseline_accuracy": 100.0 * sum(baseline_scores) / len(baseline_scores),
            "observed_delta": 100.0 * ci["observed_delta"],
            "ci95_low": 100.0 * ci["ci_low"],
            "ci95_high": 100.0 * ci["ci_high"],
            "p_value_two_sided": ci["p_value_two_sided"],
            "significant": not (ci["ci_low"] <= 0.0 <= ci["ci_high"]),
        }
    return {
        "benchmark": benchmark,
        "candidate_artifact": str(candidate_path),
        "baseline_artifact": str(baseline_path),
        "candidate_run_label": candidate_run.get("label") or candidate_run.get("key"),
        "baseline_run_label": baseline_run.get("label") or baseline_run.get("key"),
        "candidate_eval_schema_version": candidate_payload.get("eval_schema_version"),
        "baseline_eval_schema_version": baseline_payload.get("eval_schema_version"),
        "candidate_seed": candidate_payload.get("seed"),
        "baseline_seed": baseline_payload.get("seed"),
        "paired_total": len(common_keys),
        "candidate_only_rows": len(set(candidate_rows) - set(baseline_rows)),
        "baseline_only_rows": len(set(baseline_rows) - set(candidate_rows)),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(n_resamples),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--candidate-label")
    parser.add_argument("--baseline-label")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = compare_artifacts(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        benchmark=args.benchmark,
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
        seed=args.seed,
        n_resamples=args.resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
