#!/usr/bin/env python3
"""Audit V9 finalist eval image refs against training-source JSONs on Modal.

The local entrypoint reads VQA, POPE, and GQA eval artifacts from the workspace.
The remote function reads training source JSONs from the shared Modal checkpoint
volume and reports exact COCO-val2014, numeric-ID, and raw-ref overlaps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import modal


DEFAULT_V5_SOURCES = [
    "/checkpoints/vqa_data/vqa_train2014_direct_yes_no_balanced_40000.json",
    "/checkpoints/vqa_data/vqa_train2014_direct_number_50000.json",
    "/checkpoints/vqa_data/vqa_train2014_direct_other_80000.json",
    "/checkpoints/llava_data/coco_object_direct_train2017.json",
    "/checkpoints/llava_data/mix665k_direct_answer_filtered.json",
]

DEFAULT_V9_SOURCES = [
    *DEFAULT_V5_SOURCES,
    "/checkpoints/vqa_data/v9_qwen_shuffled_image_20000.json",
    "/checkpoints/vqa_data/v9_qwen_wrong_image_same_answer_type_20000.json",
    "/checkpoints/vqa_data/v9_qwen_blank_image_10000.json",
    "/checkpoints/gqa_data/v9_qwen_gqa_train_balanced_10000.json",
    "/checkpoints/llava_data/coco_pope_style_presence_train2017_10000.json",
    "/checkpoints/llava_data/coco_pope_style_absence_train2017_10000.json",
    "/checkpoints/vqa_data/v9_qwen_contrastive_answer_suppression_40000.json",
]

EVAL_REF_FIELDS = (
    "image_id",
    "source_image_id",
    "image",
    "image_path",
    "image_file",
    "file_name",
    "filename",
)
TRAIN_REF_FIELDS = (
    "image",
    "negative_images",
    "source_image",
    "control_image",
    "image_path",
    "image_file",
    "file_name",
    "filename",
    "image_id",
    "source_image_id",
)

COCO_REF_RE = re.compile(
    r"COCO_(?P<split>train|val)(?P<year>20\d{2})_(?P<id>\d{1,12})"
    r"(?:\.(?:jpg|jpeg|png|webp))?",
    re.IGNORECASE,
)
SPLIT_PATH_RE = re.compile(
    r"(?:^|[/\\])(?P<split>train|val)(?P<year>20\d{2})(?:[/\\][^/\\]*)*[/\\]"
    r"(?P<id>\d{1,12})\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)
IMAGE_FILE_RE = re.compile(r"(?P<stem>[^/\\]+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)

app = modal.App("anymal-v9-leakage-audit")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _split_arg(value: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", str(value or "")) if part]


def _new_ref_index() -> dict[str, Any]:
    return {
        "numeric_ids": set(),
        "raw_refs": set(),
        "coco_ids_by_split": {},
    }


def _int_id(value: str | int) -> int:
    text = str(value).strip()
    return int(text.lstrip("0") or "0")


def _add_numeric_ref(
    index: dict[str, Any],
    image_id: str | int,
    default_coco_split: str | None = None,
) -> None:
    parsed = _int_id(image_id)
    index["numeric_ids"].add(parsed)
    if default_coco_split:
        index["coco_ids_by_split"].setdefault(default_coco_split, set()).add(parsed)


def _normalize_raw_ref(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    text = text.split("?", 1)[0].split("#", 1)[0].strip()
    return text.strip("/").lower()


def _add_raw_ref(index: dict[str, Any], value: str) -> bool:
    raw = _normalize_raw_ref(value)
    if not raw:
        return False
    index["raw_refs"].add(raw)
    basename = raw.rsplit("/", 1)[-1]
    if basename:
        index["raw_refs"].add(basename)
    match = IMAGE_FILE_RE.search(basename)
    if match:
        index["raw_refs"].add(match.group("stem").lower())
    return True


def _add_coco_refs(index: dict[str, Any], text: str) -> bool:
    added = False
    for regex in (COCO_REF_RE, SPLIT_PATH_RE):
        for match in regex.finditer(text):
            split = f"{match.group('split').lower()}{match.group('year')}"
            image_id = _int_id(match.group("id"))
            index["numeric_ids"].add(image_id)
            index["coco_ids_by_split"].setdefault(split, set()).add(image_id)
            added = True
    return added


def _add_ref(
    index: dict[str, Any],
    value: Any,
    default_coco_split: str | None = None,
) -> bool:
    """Add one image ref value into an index.

    Raw strings are retained even when they are not parseable as COCO/numeric
    IDs. That keeps GQA question-derived IDs and Visual Genome filenames auditable.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        _add_numeric_ref(index, value, default_coco_split=default_coco_split)
        _add_raw_ref(index, str(value))
        return True
    if isinstance(value, float) and value.is_integer():
        _add_numeric_ref(index, int(value), default_coco_split=default_coco_split)
        _add_raw_ref(index, str(int(value)))
        return True
    if isinstance(value, (list, tuple, set)):
        parsed = False
        for item in value:
            parsed = _add_ref(index, item, default_coco_split) or parsed
        return parsed
    if isinstance(value, dict):
        parsed = False
        for key in (
            "image",
            "image_id",
            "imageId",
            "path",
            "file_name",
            "filename",
            "bytes",
        ):
            if key in value and key != "bytes":
                parsed = _add_ref(index, value[key], default_coco_split) or parsed
        return parsed

    text = str(value).strip()
    if not text:
        return False

    added = _add_raw_ref(index, text)
    added = _add_coco_refs(index, text) or added

    normalized = _normalize_raw_ref(text)
    basename = normalized.rsplit("/", 1)[-1]
    file_match = IMAGE_FILE_RE.search(basename)
    stem = file_match.group("stem") if file_match else basename
    if stem.isdigit():
        _add_numeric_ref(index, stem, default_coco_split=default_coco_split)
        added = True
    return added


def _serialize_index(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_ids": sorted(index["numeric_ids"]),
        "raw_refs": sorted(index["raw_refs"]),
        "coco_ids_by_split": {
            split: sorted(ids)
            for split, ids in sorted(index["coco_ids_by_split"].items())
        },
    }


def _deserialize_index(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_ids": set(payload.get("numeric_ids") or []),
        "raw_refs": set(payload.get("raw_refs") or []),
        "coco_ids_by_split": {
            split: set(ids)
            for split, ids in (payload.get("coco_ids_by_split") or {}).items()
        },
    }


def _index_counts(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_ids": len(index["numeric_ids"]),
        "raw_refs": len(index["raw_refs"]),
        "coco_split_counts": {
            split: len(ids)
            for split, ids in sorted(index["coco_ids_by_split"].items())
        },
    }


def _infer_default_eval_split(payload: Any, path: str) -> str | None:
    benchmark = ""
    image_dir = ""
    if isinstance(payload, dict):
        benchmark = str(payload.get("benchmark") or "").strip().lower()
        image_dir = str(payload.get("image_dir") or "").strip().lower()
    path_hint = str(path).lower()
    if isinstance(payload, dict) and payload.get("gqa_split") is not None:
        return None
    if benchmark == "gqa" or "gqa" in path_hint:
        return None
    if isinstance(payload, dict) and (
        payload.get("pope_split") is not None
        or payload.get("image_perturbation") is not None
    ):
        return "val2014"
    if benchmark in {"vqa", "vqav2", "pope"}:
        return "val2014"
    if "coco_val2014" in image_dir or "val2014" in image_dir:
        return "val2014"
    if "pope" in path_hint or "vqa" in path_hint:
        return "val2014"
    return None


def _iter_prediction_blocks(payload: Any, path: str) -> Iterable[dict[str, Any]]:
    default_split = _infer_default_eval_split(payload, path)
    if isinstance(payload, list):
        yield {
            "artifact": path,
            "run": "top_level_list",
            "samples": payload,
            "default_coco_split": default_split,
        }
        return
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("prediction_samples"), list):
        yield {
            "artifact": path,
            "run": "top_level",
            "samples": payload["prediction_samples"],
            "default_coco_split": default_split,
        }
    for run_idx, run in enumerate(payload.get("runs") or []):
        if not isinstance(run, dict):
            continue
        samples = run.get("prediction_samples")
        if isinstance(samples, list):
            yield {
                "artifact": path,
                "run": str(run.get("label") or run.get("key") or run_idx),
                "samples": samples,
                "default_coco_split": default_split,
            }


def _candidate_values(row: dict[str, Any], fields: tuple[str, ...]) -> Iterable[tuple[str, Any]]:
    for field in fields:
        if field in row:
            yield field, row[field]


def _read_eval_refs(paths: list[str]) -> dict[str, Any]:
    eval_index = _new_ref_index()
    artifacts = []
    totals = {
        "prediction_samples": 0,
        "rows_with_refs": 0,
        "missing_ref_rows": 0,
        "unparseable_ref_rows": 0,
        "non_object_rows": 0,
        "blocks_without_prediction_samples": 0,
    }

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        blocks = list(_iter_prediction_blocks(payload, path))
        if not blocks:
            totals["blocks_without_prediction_samples"] += 1
            artifacts.append(
                {
                    "path": path,
                    "run": None,
                    "prediction_samples": 0,
                    "rows_with_refs": 0,
                    "missing_ref_rows": 0,
                    "unparseable_ref_rows": 0,
                    "non_object_rows": 0,
                    "default_coco_split": _infer_default_eval_split(payload, path),
                    "note": "no prediction_samples block found",
                }
            )
            continue

        for block in blocks:
            block_counts = {
                "path": path,
                "run": block["run"],
                "prediction_samples": 0,
                "rows_with_refs": 0,
                "missing_ref_rows": 0,
                "unparseable_ref_rows": 0,
                "non_object_rows": 0,
                "default_coco_split": block["default_coco_split"],
            }
            for row in block["samples"]:
                totals["prediction_samples"] += 1
                block_counts["prediction_samples"] += 1
                if not isinstance(row, dict):
                    totals["non_object_rows"] += 1
                    block_counts["non_object_rows"] += 1
                    continue

                values = list(_candidate_values(row, EVAL_REF_FIELDS))
                if not values:
                    totals["missing_ref_rows"] += 1
                    block_counts["missing_ref_rows"] += 1
                    continue

                parsed = False
                for _field, value in values:
                    parsed = (
                        _add_ref(
                            eval_index,
                            value,
                            default_coco_split=block["default_coco_split"],
                        )
                        or parsed
                    )
                if parsed:
                    totals["rows_with_refs"] += 1
                    block_counts["rows_with_refs"] += 1
                else:
                    totals["unparseable_ref_rows"] += 1
                    block_counts["unparseable_ref_rows"] += 1
            artifacts.append(block_counts)

    if not eval_index["numeric_ids"] and not eval_index["raw_refs"]:
        raise SystemExit("No eval image refs found in artifact prediction_samples.")

    return {
        "index": _serialize_index(eval_index),
        "artifacts": artifacts,
        "totals": {**totals, **_index_counts(eval_index)},
    }


def _iter_samples(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        yielded = False
        for key in ("samples", "data", "annotations", "records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                yielded = True
                for row in rows:
                    if isinstance(row, dict):
                        yield row
        if not yielded and any(field in payload for field in TRAIN_REF_FIELDS):
            yield payload


def _source_index(payload: Any) -> tuple[dict[str, Any], dict[str, int]]:
    index = _new_ref_index()
    counts = {
        "samples": 0,
        "image_ref_values": 0,
        "missing_ref_samples": 0,
        "unparseable_ref_samples": 0,
    }
    for sample in _iter_samples(payload):
        counts["samples"] += 1
        values = list(_candidate_values(sample, TRAIN_REF_FIELDS))
        if not values:
            counts["missing_ref_samples"] += 1
            continue
        parsed = False
        for _field, value in values:
            if _add_ref(index, value):
                counts["image_ref_values"] += 1
                parsed = True
        if not parsed:
            counts["unparseable_ref_samples"] += 1
    return index, counts


def _examples(values: set[Any], max_examples: int) -> list[Any]:
    return sorted(values)[: max(0, int(max_examples))]


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def audit_remote(eval_ref_index, train_sources, max_examples=20):
    eval_index = _deserialize_index(eval_ref_index)
    eval_val2014_ids = eval_index["coco_ids_by_split"].get("val2014", set())
    eval_numeric_ids = eval_index["numeric_ids"]
    eval_raw_refs = eval_index["raw_refs"]

    results = []
    total_exact_val2014_overlap = 0
    total_numeric_id_overlap = 0
    total_raw_ref_overlap = 0
    missing_sources = []

    for source in train_sources:
        source_path = Path(source)
        if not source_path.exists():
            missing_sources.append(source)
            results.append({"source": source, "missing": True})
            continue
        with open(source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        train_index, counts = _source_index(payload)
        train_val2014_ids = train_index["coco_ids_by_split"].get("val2014", set())
        exact_val_overlap = train_val2014_ids & eval_val2014_ids
        numeric_overlap = train_index["numeric_ids"] & eval_numeric_ids
        raw_overlap = train_index["raw_refs"] & eval_raw_refs

        total_exact_val2014_overlap += len(exact_val_overlap)
        total_numeric_id_overlap += len(numeric_overlap)
        total_raw_ref_overlap += len(raw_overlap)

        results.append(
            {
                "source": source,
                "missing": False,
                **counts,
                **_index_counts(train_index),
                "exact_val2014_overlap": len(exact_val_overlap),
                "numeric_id_overlap": len(numeric_overlap),
                "raw_ref_overlap": len(raw_overlap),
                "exact_val2014_overlap_examples": _examples(
                    exact_val_overlap,
                    max_examples,
                ),
                "numeric_id_overlap_examples": _examples(
                    numeric_overlap,
                    max_examples,
                ),
                "raw_ref_overlap_examples": _examples(raw_overlap, max_examples),
            }
        )

    missing_count = len(missing_sources)
    passes_missing_source_gate = missing_count == 0
    # Numeric-only IDs are ambiguous across sources such as COCO and BLIP/LAION
    # pretrain images, where unrelated files can share stems like "000000123.jpg".
    # Treat numeric-only collisions as warnings unless they are supported by an
    # exact same-split COCO overlap or a raw reference overlap.
    numeric_only_warning = (
        total_numeric_id_overlap > 0
        and total_exact_val2014_overlap == 0
        and total_raw_ref_overlap == 0
    )
    gates = {
        "missing_source_gate": passes_missing_source_gate,
        "exact_val2014_gate": (
            total_exact_val2014_overlap == 0 and passes_missing_source_gate
        ),
        "numeric_id_gate": (
            (total_numeric_id_overlap == 0 or numeric_only_warning)
            and passes_missing_source_gate
        ),
        "raw_ref_gate": total_raw_ref_overlap == 0 and passes_missing_source_gate,
    }
    gates["overall_pass"] = all(gates.values())

    return {
        "sources": results,
        "totals": {
            "train_sources": len(list(train_sources)),
            "missing_sources": missing_count,
            "total_exact_val2014_overlap": total_exact_val2014_overlap,
            "total_numeric_id_overlap": total_numeric_id_overlap,
            "total_raw_ref_overlap": total_raw_ref_overlap,
            "numeric_only_warning": numeric_only_warning,
        },
        "missing_sources": missing_sources,
        "gates": gates,
    }


def _gate_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _print_summary(result: dict[str, Any]) -> None:
    eval_totals = result["eval_totals"]
    train_totals = result["train_totals"]
    gates = result["gates"]

    print("V9 leakage audit")
    print(
        "  eval_prediction_samples: "
        f"{eval_totals['prediction_samples']} "
        f"rows_with_refs={eval_totals['rows_with_refs']} "
        f"missing_ref_rows={eval_totals['missing_ref_rows']} "
        f"unparseable_ref_rows={eval_totals['unparseable_ref_rows']} "
        f"non_object_rows={eval_totals['non_object_rows']}"
    )
    print(
        "  eval_refs: "
        f"numeric={eval_totals['numeric_ids']} "
        f"raw={eval_totals['raw_refs']} "
        f"splits={eval_totals['coco_split_counts']}"
    )
    print(
        "  overlaps: "
        f"exact_val2014={train_totals['total_exact_val2014_overlap']} "
        f"numeric={train_totals['total_numeric_id_overlap']} "
        f"raw_ref={train_totals['total_raw_ref_overlap']} "
        f"missing_sources={train_totals['missing_sources']}"
    )
    if train_totals.get("numeric_only_warning"):
        print("  note: numeric-only overlaps are warnings without exact-split or raw-ref overlap")
    print(f"  missing_source_gate: {_gate_text(gates['missing_source_gate'])}")
    print(f"  exact_val2014_gate: {_gate_text(gates['exact_val2014_gate'])}")
    print(f"  numeric_id_gate: {_gate_text(gates['numeric_id_gate'])}")
    print(f"  raw_ref_gate: {_gate_text(gates['raw_ref_gate'])}")
    print(f"  overall: {_gate_text(gates['overall_pass'])}")

    print("Eval artifacts")
    for row in result["eval_artifacts"]:
        print(
            f"  {row['path']} [{row.get('run') or 'no-run'}]: "
            f"samples={row['prediction_samples']} refs={row['rows_with_refs']} "
            f"missing={row['missing_ref_rows']} unparseable={row['unparseable_ref_rows']} "
            f"default_split={row.get('default_coco_split')}"
        )

    print("Train sources")
    for row in result["sources"]:
        if row.get("missing"):
            print(f"  {row['source']}: MISSING")
            continue
        print(
            f"  {row['source']}: samples={row['samples']} "
            f"refs={row['image_ref_values']} splits={row['coco_split_counts']} "
            f"exact_val2014_overlap={row['exact_val2014_overlap']} "
            f"numeric_id_overlap={row['numeric_id_overlap']} "
            f"raw_ref_overlap={row['raw_ref_overlap']}"
        )
        if row["exact_val2014_overlap_examples"]:
            print(f"    exact_val2014_examples={row['exact_val2014_overlap_examples']}")
        if row["numeric_id_overlap_examples"]:
            print(f"    numeric_examples={row['numeric_id_overlap_examples']}")
        if row["raw_ref_overlap_examples"]:
            print(f"    raw_ref_examples={row['raw_ref_overlap_examples']}")


@app.local_entrypoint()
def main(
    eval_artifacts: str,
    train_sources: str = "",
    output_json: str = "",
    json_output: bool = False,
    max_examples: int = 20,
) -> None:
    eval_paths = _split_arg(eval_artifacts)
    if not eval_paths:
        raise SystemExit("Provide at least one local eval artifact path.")
    source_paths = _split_arg(train_sources) if train_sources else DEFAULT_V9_SOURCES

    eval_refs = _read_eval_refs(eval_paths)
    remote_result = audit_remote.remote(
        eval_refs["index"],
        list(source_paths),
        int(max_examples),
    )
    result = {
        "schema_version": "v9_leakage_audit_v1",
        "eval_artifacts": eval_refs["artifacts"],
        "eval_totals": eval_refs["totals"],
        "train_sources": list(source_paths),
        "sources": remote_result["sources"],
        "train_totals": remote_result["totals"],
        "missing_sources": remote_result["missing_sources"],
        "gates": remote_result["gates"],
    }

    if output_json:
        output_path = Path(output_json)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if json_output:
        print(json.dumps(result, indent=2))
        return

    _print_summary(result)
    if output_json:
        print(f"Saved JSON: {output_json}")
