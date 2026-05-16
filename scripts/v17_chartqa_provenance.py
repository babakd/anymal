#!/usr/bin/env python3
"""Verify ChartQA mirror provenance for V17.

The primary vis-nlp/ChartQA dataset is not always public through Hugging Face,
so this script records its accessibility and compares the pinned anhdang000
ChartQA-V2 mirror against the accessible HuggingFaceM4/ChartQA mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v17-chartqa-provenance")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "datasets>=2.19.0",
        "huggingface_hub>=0.19.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
    )
    .add_local_dir(Path(__file__).resolve().parents[1], remote_path="/root/anymal", copy=False)
)


def _json_dump(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _answer_text(row: dict[str, Any]) -> str:
    for key in ("label", "answer", "answers"):
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else ""
        if value is not None and str(value).strip():
            return _norm_text(value)
    return ""


def _query_text(row: dict[str, Any]) -> str:
    return _norm_text(row.get("query") or row.get("question") or row.get("Question"))


def _question_family(question: str) -> str:
    question = _norm_text(question)
    if question.startswith("is "):
        return "is"
    if question.startswith("are "):
        return "are"
    if question.startswith("what "):
        return "what"
    if question.startswith("which "):
        return "which"
    if question.startswith("how many"):
        return "how_many"
    if question.startswith("how "):
        return "how"
    return question.split(" ", 1)[0] if question else "empty"


def _hf_revision(dataset_id: str) -> tuple[str | None, str | None, str | None]:
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(dataset_id, files_metadata=False)
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    card = getattr(info, "card_data", None)
    license_name = None
    if hasattr(card, "to_dict"):
        license_name = card.to_dict().get("license")
    elif isinstance(card, dict):
        license_name = card.get("license")
    return str(info.sha), license_name, None


def _dataset_summary(dataset_id: str, revision: str | None, split: str, config: str | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    rows = load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir="/checkpoints/hf_datasets",
        revision=revision,
    )
    qa_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    query_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    for row in rows:
        query = _query_text(row)
        answer = _answer_text(row)
        qa_counter[f"{query}\t{answer}"] += 1
        family_counter[_question_family(query)] += 1
        query_counter[query] += 1
        answer_counter[answer] += 1

    cache_files = []
    for item in getattr(rows, "cache_files", []) or []:
        filename = item.get("filename")
        if filename and os.path.exists(filename):
            cache_files.append(
                {
                    "path": filename,
                    "bytes": os.path.getsize(filename),
                    "sha256": _sha256_file(filename),
                }
            )
    canonical = hashlib.sha256()
    for key, count in sorted(qa_counter.items()):
        canonical.update(f"{key}\t{count}\n".encode("utf-8"))
    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "config": config,
        "split": split,
        "rows": len(rows),
        "cache_files": cache_files,
        "canonical_qa_multiset_sha256": canonical.hexdigest(),
        "question_family_counts": dict(family_counter.most_common()),
        "top_answers": dict(answer_counter.most_common(20)),
        "top_questions": dict(query_counter.most_common(20)),
        "_qa_counter": qa_counter,
    }


def _compare_summaries(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_counter: Counter[str] = left["_qa_counter"]
    right_counter: Counter[str] = right["_qa_counter"]
    overlap = sum((left_counter & right_counter).values())
    left_total = sum(left_counter.values())
    right_total = sum(right_counter.values())
    union = sum((left_counter | right_counter).values())
    return {
        "left_rows": left["rows"],
        "right_rows": right["rows"],
        "row_delta": left["rows"] - right["rows"],
        "qa_multiset_overlap": overlap,
        "left_overlap_rate": overlap / left_total if left_total else 0.0,
        "right_overlap_rate": overlap / right_total if right_total else 0.0,
        "qa_multiset_jaccard": overlap / union if union else 0.0,
        "same_canonical_qa_hash": left["canonical_qa_multiset_sha256"] == right["canonical_qa_multiset_sha256"],
    }


def run_provenance(output_path: str) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/anymal")
    from evaluation.checkpoint_eval.dataset_revisions import pinned_revision

    community = "anhdang000/ChartQA-V2"
    accessible_primary = "HuggingFaceM4/ChartQA"
    inaccessible_primary = "vis-nlp/ChartQA"
    community_revision = pinned_revision(community)
    hfm4_revision, hfm4_license, hfm4_error = _hf_revision(accessible_primary)
    vis_revision, vis_license, vis_error = _hf_revision(inaccessible_primary)

    splits = {}
    comparisons = {}
    for split in ("train", "val"):
        community_summary = _dataset_summary(community, community_revision, split)
        hfm4_summary = _dataset_summary(accessible_primary, hfm4_revision, split)
        splits[f"community_{split}"] = {
            key: value for key, value in community_summary.items() if key != "_qa_counter"
        }
        splits[f"hfm4_{split}"] = {
            key: value for key, value in hfm4_summary.items() if key != "_qa_counter"
        }
        comparisons[split] = _compare_summaries(community_summary, hfm4_summary)

    result = {
        "chosen_eval_source": community,
        "chosen_eval_revision": community_revision,
        "chosen_eval_license": "unknown-community-mirror",
        "accessible_reference_source": accessible_primary,
        "accessible_reference_revision": hfm4_revision,
        "accessible_reference_license": hfm4_license,
        "published_primary_source": inaccessible_primary,
        "published_primary_revision": vis_revision,
        "published_primary_license": vis_license,
        "published_primary_access_error": vis_error,
        "splits": splits,
        "comparisons": comparisons,
        "decision": (
            "Keep anhdang000/ChartQA-V2 pinned for V17 eval reproducibility; "
            "HuggingFaceM4/ChartQA is separately inventoried as GPL-3.0 and should "
            "not be silently substituted into deployed-model training."
        ),
    }
    _json_dump(output_path, result)
    volume.commit()
    return result


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=3 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def run_provenance_remote(output_path: str) -> dict[str, Any]:
    return run_provenance(output_path)


@app.local_entrypoint()
def main(
    output: str = "v17_chartqa_provenance.json",
    remote_output_path: str = "/checkpoints/v17_reports/chartqa_provenance.json",
):
    result = run_provenance_remote.remote(str(remote_output_path))
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="v17_chartqa_provenance.json")
    parser.add_argument("--remote-output-path", default="/checkpoints/v17_reports/chartqa_provenance.json")
    parser.parse_args()
