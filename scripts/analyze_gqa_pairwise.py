#!/usr/bin/env python3
"""Write GQA pairwise deltas and heuristic taxonomy summaries.

The Modal GQA evaluator stores full prediction rows in ``runs[0].prediction_samples``
when launched with ``--prediction-samples``. This script compares those rows with
GQA exact-match semantics instead of VQA consensus scoring, then adds coarse
question-text taxonomy labels for quick experiment steering.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARTICLES = {"a", "an", "the"}
ASSISTANT_ROLE_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r]+\s*", re.IGNORECASE)
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "grey",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
DIRECTION_WORDS = {"left", "right", "front", "back", "behind", "above", "below"}


def _process_answer(answer: Any, strip_assistant_prefix: bool = True) -> str:
    text = str(answer or "").lower().strip()
    if strip_assistant_prefix:
        text = ASSISTANT_ROLE_PREFIX_RE.sub("", text).strip()
    text = text.split("\n")[0].split(".")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = re.sub(r"[^\w\s]", "", text)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words).strip()


def _answer_kind(answer: str) -> str:
    value = _process_answer(answer)
    if value in {"yes", "no"}:
        return "yes_no"
    if value.isdigit():
        return "number"
    if value in COLOR_WORDS:
        return "color"
    if value in DIRECTION_WORDS:
        return "direction"
    if not value:
        return "empty"
    return "other"


def _taxonomy(question: Any, answer: Any = "") -> dict[str, Any]:
    q = f" {str(question or '').lower()} "
    answer_kind = _answer_kind(str(answer or ""))
    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    if answer_kind == "number" or " how many " in q or " number of " in q:
        add("counting")
    if " color " in q or q.strip().startswith("what color") or answer_kind == "color":
        add("color")
    if " left " in q or " right " in q:
        add("left_right")
    spatial_terms = (
        " above ",
        " below ",
        " under ",
        " over ",
        " behind ",
        " in front ",
        " next to ",
        " near ",
        " between ",
        " inside ",
        " outside ",
        " on top ",
        " to the left ",
        " to the right ",
        " where ",
        " side ",
        " position ",
    )
    if any(term in q for term in spatial_terms) or any(label in labels for label in ("left_right",)):
        add("spatial_relation")
    comparison_terms = (
        " same ",
        " different ",
        " more ",
        " less ",
        " fewer ",
        " larger ",
        " smaller ",
        " taller ",
        " shorter ",
        " closer ",
        " farther ",
        " than ",
    )
    if any(term in q for term in comparison_terms):
        add("comparison")
    logical_terms = (
        " and ",
        " or ",
        " not ",
        " both ",
        " either ",
        " all ",
        " any ",
        " with ",
        " without ",
        " that is ",
    )
    if any(term in q for term in logical_terms):
        add("logical_compositional")
    starts_yes_no = bool(
        re.match(
            r"^(is|are|am|was|were|do|does|did|can|could|has|have|had|will|would)\b",
            str(question or "").strip().lower(),
        )
    )
    presence_terms = (
        " there ",
        " visible ",
        " see ",
        " shown ",
        " in the image ",
        " in this image ",
        " in the picture ",
    )
    if answer_kind == "yes_no" or (starts_yes_no and any(term in q for term in presence_terms)):
        add("yes_no_object_presence")
    attribute_terms = (
        " what kind ",
        " what type ",
        " what shape ",
        " what size ",
        " material ",
        " wearing ",
        " doing ",
        " made of ",
        " age ",
    )
    if any(term in q for term in attribute_terms):
        add("attribute")
    identity_starts = (
        "what is ",
        "what are ",
        "which ",
        "who ",
        "where ",
        "name ",
    )
    if any(str(question or "").strip().lower().startswith(prefix) for prefix in identity_starts):
        add("object_identity")
    if not labels:
        add("other")

    primary_priority = (
        "counting",
        "color",
        "left_right",
        "spatial_relation",
        "comparison",
        "yes_no_object_presence",
        "attribute",
        "object_identity",
        "logical_compositional",
        "other",
    )
    primary = next(label for label in primary_priority if label in labels)
    return {"primary": primary, "labels": labels, "answer_kind": answer_kind}


def _parse_artifact_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("artifact specs must be NAME=PATH")
    name, path = raw.split("=", 1)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    if not safe_name:
        raise argparse.ArgumentTypeError(f"empty artifact name in {raw!r}")
    return safe_name, Path(path).expanduser()


def _load_artifact(path: Path, label: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs") or []
    if label:
        runs = [run for run in runs if run.get("label") == label]
    if not runs:
        raise SystemExit(f"No matching runs in {path}")
    return payload, runs[0]


def _target_answer(row: dict[str, Any]) -> str:
    if row.get("gqa_answer") is not None:
        return str(row.get("gqa_answer") or "")
    answers = row.get("answers") or []
    if answers:
        return str(answers[0] or "")
    return ""


def _raw_answer(row: dict[str, Any]) -> str:
    if row.get("raw_answer") is not None:
        return str(row.get("raw_answer") or "")
    return str(row.get("answer") or "")


def _is_correct(row: dict[str, Any]) -> bool:
    if isinstance(row.get("gqa_exact_match"), bool):
        return bool(row["gqa_exact_match"])
    target = _process_answer(_target_answer(row))
    pred = _process_answer(_raw_answer(row))
    return bool(target) and pred == target


def _sample_key(row: dict[str, Any]) -> str:
    qid = row.get("question_id")
    if qid is not None:
        return str(qid)
    return f"image:{row.get('image_id')}|q:{row.get('question')}"


def _indexed_samples(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = run.get("prediction_samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(
            f"Run {run.get('label') or run.get('key') or '<unknown>'} has no prediction_samples"
        )
    indexed = {}
    for row in samples:
        enriched = dict(row)
        target = _target_answer(enriched)
        tax = _taxonomy(enriched.get("question"), target)
        enriched["_gqa_processed_target"] = _process_answer(target)
        enriched["_gqa_processed_prediction"] = _process_answer(_raw_answer(enriched))
        enriched["_gqa_correct"] = _is_correct(enriched)
        enriched["_taxonomy_primary"] = tax["primary"]
        enriched["_taxonomy_labels"] = tax["labels"]
        enriched["_answer_kind"] = tax["answer_kind"]
        indexed[_sample_key(enriched)] = enriched
    return indexed


def _artifact_summary(name: str, payload: dict[str, Any], run: dict[str, Any], samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, Counter] = defaultdict(Counter)
    answer_kind_counts: dict[str, Counter] = defaultdict(Counter)
    top_predictions = Counter()
    for row in samples.values():
        correct = bool(row["_gqa_correct"])
        category_counts[row["_taxonomy_primary"]]["total"] += 1
        category_counts[row["_taxonomy_primary"]]["correct"] += int(correct)
        for label in row["_taxonomy_labels"]:
            category_counts[f"label:{label}"]["total"] += 1
            category_counts[f"label:{label}"]["correct"] += int(correct)
        answer_kind_counts[row["_answer_kind"]]["total"] += 1
        answer_kind_counts[row["_answer_kind"]]["correct"] += int(correct)
        top_predictions[row["_gqa_processed_prediction"] or "<empty>"] += 1

    def render(counter_by_name: dict[str, Counter]) -> dict[str, dict[str, Any]]:
        rendered = {}
        for key, counts in sorted(counter_by_name.items()):
            total = int(counts["total"])
            correct = int(counts["correct"])
            rendered[key] = {
                "correct": correct,
                "total": total,
                "accuracy": (100.0 * correct / total) if total else 0.0,
            }
        return rendered

    total = len(samples)
    correct = sum(1 for row in samples.values() if row["_gqa_correct"])
    return {
        "name": name,
        "artifact": str(payload.get("_artifact_path", "")),
        "label": run.get("label"),
        "checkpoint": run.get("candidate_checkpoint") or run.get("checkpoint"),
        "eval_schema_version": payload.get("eval_schema_version"),
        "reported_gqa_accuracy": (run.get("metrics") or {}).get("gqa_accuracy"),
        "sample_accuracy": (100.0 * correct / total) if total else 0.0,
        "num_samples": total,
        "taxonomy": render(category_counts),
        "answer_kinds": render(answer_kind_counts),
        "top_predictions": top_predictions.most_common(25),
    }


def _delta_rows(
    left_name: str,
    left_samples: dict[str, dict[str, Any]],
    right_name: str,
    right_samples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(left_samples) & set(right_samples)):
        left = left_samples[key]
        right = right_samples[key]
        if not left["_gqa_correct"] or right["_gqa_correct"]:
            continue
        rows.append(
            {
                "question_id": left.get("question_id"),
                "image_id": left.get("image_id"),
                "source_image_id": left.get("source_image_id"),
                "source_index": left.get("source_index"),
                "question": left.get("question"),
                "gqa_answer": _target_answer(left),
                "taxonomy_primary": left["_taxonomy_primary"],
                "taxonomy_labels": left["_taxonomy_labels"],
                "answer_kind": left["_answer_kind"],
                f"{left_name}_answer": left.get("answer"),
                f"{left_name}_raw_answer": left.get("raw_answer"),
                f"{left_name}_processed_answer": left["_gqa_processed_prediction"],
                f"{right_name}_answer": right.get("answer"),
                f"{right_name}_raw_answer": right.get("raw_answer"),
                f"{right_name}_processed_answer": right["_gqa_processed_prediction"],
            }
        )
    return rows


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_primary = Counter(row["taxonomy_primary"] for row in rows)
    by_answer_kind = Counter(row["answer_kind"] for row in rows)
    by_label = Counter(label for row in rows for label in row.get("taxonomy_labels", []))
    return {
        "num_rows": len(rows),
        "by_taxonomy_primary": dict(sorted(by_primary.items())),
        "by_taxonomy_label": dict(sorted(by_label.items())),
        "by_answer_kind": dict(sorted(by_answer_kind.items())),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GQA Pairwise Taxonomy Summary",
        "",
        "Taxonomy labels are heuristic question-text labels for experiment steering, not publication-grade GQA metadata.",
        "",
        "## Artifacts",
        "",
    ]
    for name, item in summary["artifacts"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- GQA accuracy: `{item['sample_accuracy']:.3f}` sample / `{item.get('reported_gqa_accuracy')}` reported",
                f"- Samples: `{item['num_samples']}`",
                f"- Checkpoint: `{item.get('checkpoint')}`",
                "",
                "| Taxonomy | Correct | Total | Accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, bucket in item["taxonomy"].items():
            if label.startswith("label:"):
                continue
            lines.append(
                f"| {label} | {bucket['correct']} | {bucket['total']} | {bucket['accuracy']:.2f} |"
            )
        lines.append("")
    lines.extend(["## Pairwise Deltas", ""])
    lines.append("| Comparison | Rows | Top Primary Buckets | Answer Kinds |")
    lines.append("| --- | ---: | --- | --- |")
    for comparison, item in summary["pairwise"].items():
        top_buckets = ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                item["by_taxonomy_primary"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:6]
        )
        answer_kinds = ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                item["by_answer_kind"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        lines.append(f"| {comparison} | {item['num_rows']} | {top_buckets} | {answer_kinds} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact_spec,
        help="Artifact spec as NAME=PATH. Repeat for V9/C1/V3/etc.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--comparison",
        action="append",
        help="Optional ordered pair NAME:NAME. Repeat; default compares every ordered pair.",
    )
    args = parser.parse_args()

    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in args.artifact:
        payload, run = _load_artifact(path)
        payload["_artifact_path"] = str(path)
        artifacts[name] = {
            "payload": payload,
            "run": run,
            "samples": _indexed_samples(run),
        }
    if len(artifacts) < 2:
        raise SystemExit("Need at least two artifacts")

    if args.comparison:
        comparisons = []
        for raw in args.comparison:
            if ":" not in raw:
                raise SystemExit(f"Comparison must be LEFT:RIGHT, got {raw!r}")
            left, right = raw.split(":", 1)
            if left not in artifacts or right not in artifacts:
                raise SystemExit(f"Unknown artifact in comparison {raw!r}")
            comparisons.append((left, right))
    else:
        names = list(artifacts)
        comparisons = [(left, right) for left in names for right in names if left != right]

    summary = {
        "artifacts": {
            name: _artifact_summary(
                name,
                item["payload"],
                item["run"],
                item["samples"],
            )
            for name, item in artifacts.items()
        },
        "pairwise": {},
    }
    for left, right in comparisons:
        rows = _delta_rows(left, artifacts[left]["samples"], right, artifacts[right]["samples"])
        comparison_name = f"{left}_correct_{right}_wrong"
        payload = {
            "comparison": comparison_name,
            "left_artifact": artifacts[left]["payload"].get("_artifact_path"),
            "right_artifact": artifacts[right]["payload"].get("_artifact_path"),
            "left_label": artifacts[left]["run"].get("label"),
            "right_label": artifacts[right]["run"].get("label"),
            **_pair_summary(rows),
            "rows": rows,
        }
        summary["pairwise"][comparison_name] = {
            key: value for key, value in payload.items() if key != "rows"
        }
        _write_json(args.output_dir / f"{comparison_name}.json", payload)

    _write_json(args.output_dir / "summary.json", summary)
    _write_markdown(args.output_dir / "summary.md", summary)
    print(f"Wrote GQA pairwise taxonomy outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
