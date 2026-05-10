#!/usr/bin/env python3
"""Audit VQAv2 eval image IDs against V5 calibration sources on Modal.

The local entrypoint reads eval artifacts from the workspace, then the remote
function reads training source JSONs from the shared Modal checkpoint volume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple

import modal


DEFAULT_V5_SOURCES = [
    "/checkpoints/vqa_data/vqa_train2014_direct_yes_no_balanced_40000.json",
    "/checkpoints/vqa_data/vqa_train2014_direct_number_50000.json",
    "/checkpoints/vqa_data/vqa_train2014_direct_other_80000.json",
    "/checkpoints/llava_data/coco_object_direct_train2017.json",
    "/checkpoints/llava_data/mix665k_direct_answer_filtered.json",
]

IMAGE_RE = re.compile(
    r"COCO_(?P<split>train|val)(?P<year>20\d{2})_(?P<id>\d{12})\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
GENERIC_ID_RE = re.compile(r"(?P<id>\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE)
IMAGE_ID_NUMERIC_RE = re.compile(r"\d+")

app = modal.App("anymal-vqa-leakage-audit")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _iter_predictions(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("prediction_samples"), list):
            yield from payload["prediction_samples"]
        for run in payload.get("runs", []):
            if isinstance(run.get("prediction_samples"), list):
                yield from run["prediction_samples"]


def _eval_image_ids(paths: list[str]) -> Set[int]:
    image_ids: Set[int] = set()
    missing = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in _iter_predictions(payload):
            if row.get("image_id") is None:
                missing += 1
                continue
            image_id = str(row["image_id"]).strip()
            if image_id.isdigit():
                image_ids.add(int(image_id))
                continue
            match = IMAGE_ID_NUMERIC_RE.search(image_id)
            if not match:
                raise SystemExit(f"Could not parse numeric image_id from {image_id!r} in {path}")
            image_ids.add(int(match.group(0)))
    if missing:
        raise SystemExit(
            f"{missing} eval predictions are missing image_id. "
            "Re-run evaluation with the updated evaluator."
        )
    if not image_ids:
        raise SystemExit("No eval image IDs found in artifact prediction_samples.")
    return image_ids


def _extract_image_ref(value: Any) -> Tuple[str, int] | None:
    if not value:
        return None
    text = str(value)
    name = text.rsplit("/", 1)[-1]
    match = IMAGE_RE.search(name)
    if match:
        split = f"{match.group('split').lower()}{match.group('year')}"
        return split, int(match.group("id"))
    match = GENERIC_ID_RE.search(name)
    if match:
        return "unknown", int(match.group("id"))
    return None


def _iter_samples(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict):
        for key in ("samples", "data", "annotations"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def audit_remote(eval_image_ids, train_sources):
    eval_ids = {int(image_id) for image_id in eval_image_ids}
    results = []
    total_exact_val_overlap = 0
    total_numeric_overlap = 0
    for source in train_sources:
        source_path = Path(source)
        if not source_path.exists():
            results.append({"source": source, "missing": True})
            continue
        with open(source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        split_counts: Dict[str, int] = {}
        numeric_ids: Set[int] = set()
        exact_val_ids: Set[int] = set()
        samples = 0
        refs = 0
        for sample in _iter_samples(payload):
            samples += 1
            ref = _extract_image_ref(sample.get("image"))
            if ref is None:
                continue
            split, image_id = ref
            refs += 1
            split_counts[split] = split_counts.get(split, 0) + 1
            numeric_ids.add(image_id)
            if split == "val2014":
                exact_val_ids.add(image_id)

        exact_overlap = sorted(exact_val_ids & eval_ids)
        numeric_overlap = sorted(numeric_ids & eval_ids)
        total_exact_val_overlap += len(exact_overlap)
        total_numeric_overlap += len(numeric_overlap)
        results.append(
            {
                "source": source,
                "missing": False,
                "samples": samples,
                "image_refs": refs,
                "split_counts": split_counts,
                "exact_val2014_overlap": len(exact_overlap),
                "numeric_id_overlap": len(numeric_overlap),
                "overlap_examples": numeric_overlap[:20],
            }
        )

    return {
        "eval_image_ids": len(eval_ids),
        "sources": results,
        "total_exact_val2014_overlap": total_exact_val_overlap,
        "total_numeric_id_overlap": total_numeric_overlap,
        "passes_exact_val2014_gate": total_exact_val_overlap == 0,
        "passes_numeric_id_gate": total_numeric_overlap == 0,
    }


@app.local_entrypoint()
def main(
    eval_artifacts: str,
    train_sources: str = "",
    json_output: bool = False,
) -> None:
    eval_paths = [path for path in re.split(r"[,\s]+", eval_artifacts) if path]
    source_paths = (
        [path for path in re.split(r"[,\s]+", train_sources) if path]
        if train_sources
        else DEFAULT_V5_SOURCES
    )

    eval_ids = sorted(_eval_image_ids(eval_paths))
    result = audit_remote.remote(eval_ids, list(source_paths))
    if json_output:
        print(json.dumps(result, indent=2))
        return

    print("VQA leakage audit")
    print(f"  eval_image_ids: {result['eval_image_ids']}")
    print(f"  exact_val2014_overlap: {result['total_exact_val2014_overlap']}")
    print(f"  numeric_id_overlap: {result['total_numeric_id_overlap']}")
    print(f"  exact_val2014_gate: {'PASS' if result['passes_exact_val2014_gate'] else 'FAIL'}")
    print(f"  numeric_id_gate: {'PASS' if result['passes_numeric_id_gate'] else 'FAIL'}")
    for row in result["sources"]:
        if row.get("missing"):
            print(f"  {row['source']}: MISSING")
            continue
        print(
            f"  {row['source']}: samples={row['samples']} refs={row['image_refs']} "
            f"splits={row['split_counts']} exact_val2014_overlap={row['exact_val2014_overlap']} "
            f"numeric_id_overlap={row['numeric_id_overlap']}"
        )
        if row["overlap_examples"]:
            print(f"    overlap_examples={row['overlap_examples']}")
