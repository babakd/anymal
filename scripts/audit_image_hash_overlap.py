#!/usr/bin/env python3
"""Audit train/eval image overlap by hashing referenced image files.

This is intended to run on Modal because most referenced images live on the
shared checkpoint volume. It can also run locally when the same paths exist.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import modal


app = modal.App("anymal-image-hash-overlap-audit")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)

IMAGE_EXT_RE = re.compile(r"\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
COCO_ID_RE = re.compile(r"COCO_(?:train|val)20\d{2}_(\d{1,12})", re.IGNORECASE)
GQA_ID_RE = re.compile(r"^n(\d{1,12})$", re.IGNORECASE)
DIRECT_IMAGE_REF_KEYS = (
    "image",
    "negative_images",
    "source_image",
    "control_image",
    "image_path",
    "image_file",
    "file_name",
    "filename",
    "path",
)
ID_IMAGE_REF_KEYS = ("image_id", "source_image_id")


def _split_arg(value: str | None) -> list[str]:
    return [part for part in re.split(r"[,\s]+", str(value or "")) if part]


def _load_json_or_jsonl(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _parse_image_dirs(values: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in _split_arg(values):
        if "=" not in item:
            mapping["default"] = item
            continue
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _lookup_image_dir(path: str, mapping: dict[str, str], payload: Any = None, kind: str = "") -> str:
    candidates = [
        path,
        os.path.basename(path),
        Path(path).stem,
        kind,
        "default",
    ]
    for candidate in candidates:
        if candidate and candidate in mapping:
            return mapping[candidate]
    if isinstance(payload, dict):
        image_dir = payload.get("image_dir")
        if image_dir:
            return str(image_dir)
        runs = payload.get("runs")
        if isinstance(runs, list) and runs:
            dataset_meta = (runs[0] or {}).get("dataset_meta") or {}
            if dataset_meta.get("image_dir"):
                return str(dataset_meta["image_dir"])
    return ""


def _iter_ref_values(value: Any) -> Iterable[str]:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            yield text
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_ref_values(item)
        return
    if isinstance(value, dict):
        for key in (
            "image",
            "image_id",
            "source_image_id",
            "image_path",
            "image_file",
            "file_name",
            "filename",
            "path",
        ):
            if key in value:
                yield from _iter_ref_values(value[key])


def _candidate_paths(ref: str, image_dir: str) -> list[str]:
    ref = str(ref).strip()
    paths = []
    if not ref:
        return paths
    if os.path.isabs(ref):
        paths.append(ref)
    if image_dir:
        paths.append(os.path.join(image_dir, ref))
        if ref.isdigit():
            paths.append(os.path.join(image_dir, f"VG_{int(ref)}.jpg"))
            paths.append(os.path.join(image_dir, f"COCO_val2014_{int(ref):012d}.jpg"))
            paths.append(os.path.join(image_dir, f"COCO_train2017_{int(ref):012d}.jpg"))
            paths.append(os.path.join(image_dir, f"{ref}.jpg"))
        gqa_match = GQA_ID_RE.fullmatch(ref)
        if gqa_match:
            image_id = int(gqa_match.group(1))
            paths.append(os.path.join(image_dir, f"{ref}.jpg"))
            paths.append(os.path.join(image_dir, f"VG_{image_id}.jpg"))
            paths.append(os.path.join(image_dir, f"{image_id}.jpg"))
            paths.append(os.path.join(image_dir, f"COCO_val2014_{image_id:012d}.jpg"))
            paths.append(os.path.join(image_dir, f"COCO_train2017_{image_id:012d}.jpg"))
    match = COCO_ID_RE.search(ref)
    if match and image_dir:
        image_id = int(match.group(1))
        paths.append(os.path.join(image_dir, f"COCO_val2014_{image_id:012d}.jpg"))
        paths.append(os.path.join(image_dir, f"COCO_train2017_{image_id:012d}.jpg"))
    return list(dict.fromkeys(paths))


def _looks_like_image_ref(ref: str) -> bool:
    text = str(ref).strip()
    return bool(IMAGE_EXT_RE.search(text) or text.isdigit() or GQA_ID_RE.fullmatch(text))


def _hash_first_mib(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read(1024 * 1024))
    return digest.hexdigest()


def _record_path_for_ref(ref: str, image_dir: str) -> str | None:
    for candidate in _candidate_paths(ref, image_dir):
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return candidate
    return None


def _iter_train_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
    elif isinstance(payload, dict):
        for key in ("samples", "data", "rows", "annotations", "questions"):
            rows = payload.get(key)
            if isinstance(rows, list):
                yield from (row for row in rows if isinstance(row, dict))


def _iter_eval_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
        return
    if not isinstance(payload, dict):
        return
    for key in ("predictions", "prediction_samples", "records", "samples"):
        rows = payload.get(key)
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict))
    for run in payload.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for key in ("predictions", "prediction_samples"):
            rows = run.get(key)
            if isinstance(rows, list):
                yield from (row for row in rows if isinstance(row, dict))
        dataset_meta = run.get("dataset_meta") or {}
        image_dir = dataset_meta.get("image_dir") or payload.get("image_dir")
        for image_id in dataset_meta.get("selected_image_ids") or []:
            yield {"image_id": image_id, "_image_dir": image_dir}
        for image_id in dataset_meta.get("source_image_ids") or []:
            yield {"image_id": image_id, "_image_dir": image_dir}


def _hash_refs(
    *,
    source_path: str,
    kind: str,
    image_dirs: dict[str, str],
    workers: int = 1,
) -> dict[str, Any]:
    payload = _load_json_or_jsonl(source_path)
    default_image_dir = _lookup_image_dir(source_path, image_dirs, payload, kind)
    rows = _iter_train_records(payload) if kind == "train" else _iter_eval_records(payload)
    hashes: dict[str, list[str]] = {}
    missing = 0
    missing_examples = []
    checked_refs = 0
    resolved_paths = []
    resolved_ref_cache: dict[tuple[str, str], str | None] = {}
    for row in rows:
        row_image_dir = str(row.get("_image_dir") or default_image_dir or "")
        direct_refs = []
        for key in DIRECT_IMAGE_REF_KEYS:
            if key in row:
                direct_refs.extend(_iter_ref_values(row[key]))
        refs = list(direct_refs)
        has_direct_image_ref = any(_looks_like_image_ref(str(ref)) for ref in direct_refs)
        if not has_direct_image_ref:
            for key in ID_IMAGE_REF_KEYS:
                if key in row:
                    refs.extend(_iter_ref_values(row[key]))
        for ref in refs:
            if not _looks_like_image_ref(str(ref)):
                continue
            checked_refs += 1
            cache_key = (str(ref), row_image_dir)
            if cache_key in resolved_ref_cache:
                path = resolved_ref_cache[cache_key]
            else:
                path = _record_path_for_ref(str(ref), row_image_dir)
                resolved_ref_cache[cache_key] = path
            if not path:
                missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append(
                        {
                            "ref": str(ref),
                            "image_dir": row_image_dir,
                            "candidate_paths": _candidate_paths(str(ref), row_image_dir),
                        }
                    )
                continue
            resolved_paths.append(path)
            if checked_refs % 1000 == 0:
                print(
                    f"{kind} hash audit resolve progress {os.path.basename(source_path)}: "
                    f"checked_refs={checked_refs} resolved_paths={len(resolved_paths)} missing={missing}",
                    flush=True,
                )
    unique_paths = list(dict.fromkeys(resolved_paths))
    path_digest_cache: dict[str, str] = {}
    worker_count = max(1, int(workers or 1))
    if worker_count > 1 and len(unique_paths) > 1:
        hashed_count = 0
        chunk_size = max(1000, worker_count * 64)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for chunk_start in range(0, len(unique_paths), chunk_size):
                chunk = unique_paths[chunk_start : chunk_start + chunk_size]
                futures = {executor.submit(_hash_first_mib, path): path for path in chunk}
                for future in as_completed(futures):
                    path = futures[future]
                    path_digest_cache[path] = future.result()
                    hashed_count += 1
                    if hashed_count % 1000 == 0 or hashed_count == len(unique_paths):
                        print(
                            f"{kind} hash audit hash progress {os.path.basename(source_path)}: "
                            f"hashed_paths={hashed_count}/{len(unique_paths)} missing={missing}",
                            flush=True,
                        )
    else:
        for idx, path in enumerate(unique_paths, start=1):
            path_digest_cache[path] = _hash_first_mib(path)
            if idx % 1000 == 0 or idx == len(unique_paths):
                print(
                    f"{kind} hash audit hash progress {os.path.basename(source_path)}: "
                    f"hashed_paths={idx}/{len(unique_paths)} missing={missing}",
                    flush=True,
                )
    for path in resolved_paths:
        digest = path_digest_cache[path]
        hashes.setdefault(digest, []).append(path)
    print(
        f"{kind} hash audit done {os.path.basename(source_path)}: "
        f"checked_refs={checked_refs} unique_hashes={len(hashes)} missing={missing}",
        flush=True,
    )
    return {
        "source_path": source_path,
        "kind": kind,
        "image_dir": default_image_dir,
        "hashes": hashes,
        "unique_hashes": len(hashes),
        "checked_refs": checked_refs,
        "missing_refs": missing,
        "missing_examples": missing_examples,
    }


def run_audit(
    train_sources: list[str],
    eval_artifacts: list[str],
    image_dirs: dict[str, str],
    *,
    workers: int = 1,
) -> dict[str, Any]:
    train_indexes = []
    for source in train_sources:
        print(f"Starting train hash audit: {source}", flush=True)
        train_indexes.append(
            _hash_refs(source_path=source, kind="train", image_dirs=image_dirs, workers=workers)
        )
    eval_indexes = []
    for artifact in eval_artifacts:
        print(f"Starting eval hash audit: {artifact}", flush=True)
        eval_indexes.append(
            _hash_refs(source_path=artifact, kind="eval", image_dirs=image_dirs, workers=workers)
        )
    pair_reports = []
    for train_index in train_indexes:
        train_hashes = set(train_index["hashes"])
        for eval_index in eval_indexes:
            eval_hashes = set(eval_index["hashes"])
            overlap = sorted(train_hashes & eval_hashes)
            examples = []
            for digest in overlap[:20]:
                examples.append(
                    {
                        "sha256_first_mib": digest,
                        "train_paths": train_index["hashes"][digest][:5],
                        "eval_paths": eval_index["hashes"][digest][:5],
                    }
                )
            pair_reports.append(
                {
                    "train_source": train_index["source_path"],
                    "eval_artifact": eval_index["source_path"],
                    "overlap_count": len(overlap),
                    "overlap_examples": examples,
                }
            )
    return {
        "train_sources": [
            {
                key: value
                for key, value in index.items()
                if key != "hashes"
            }
            for index in train_indexes
        ],
        "eval_artifacts": [
            {
                key: value
                for key, value in index.items()
                if key != "hashes"
            }
            for index in eval_indexes
        ],
        "total_unique_train_hashes": len(
            set().union(*(set(index["hashes"]) for index in train_indexes)) if train_indexes else set()
        ),
        "total_unique_eval_hashes": len(
            set().union(*(set(index["hashes"]) for index in eval_indexes)) if eval_indexes else set()
        ),
        "pairs": pair_reports,
    }


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=6 * 60 * 60)
def audit_image_hash_overlap_remote(
    train_sources: list[str],
    eval_artifacts: list[str],
    image_dirs: dict[str, str],
    workers: int = 1,
    output_path: str | None = None,
) -> dict[str, Any]:
    result = run_audit(train_sources, eval_artifacts, image_dirs, workers=workers)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        volume.commit()
    return result


@app.local_entrypoint()
def main(
    train_sources: str,
    eval_artifacts: str,
    image_dirs: str = "",
    output: str = "image_hash_overlap_audit.json",
    remote_output_path: str = "",
    local: bool = False,
    workers: int = 1,
):
    parsed_train_sources = _split_arg(train_sources)
    parsed_eval_artifacts = _split_arg(eval_artifacts)
    parsed_image_dirs = _parse_image_dirs(image_dirs)
    if local:
        result = run_audit(parsed_train_sources, parsed_eval_artifacts, parsed_image_dirs, workers=workers)
    else:
        result = audit_image_hash_overlap_remote.remote(
            parsed_train_sources,
            parsed_eval_artifacts,
            parsed_image_dirs,
            workers,
            remote_output_path or None,
        )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sources", required=True)
    parser.add_argument("--eval-artifacts", required=True)
    parser.add_argument("--image-dirs", default="")
    parser.add_argument("--output", default="image_hash_overlap_audit.json")
    parser.add_argument("--remote-output-path", default="")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result_payload = run_audit(
        _split_arg(args.train_sources),
        _split_arg(args.eval_artifacts),
        _parse_image_dirs(args.image_dirs),
        workers=args.workers,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(json.dumps(result_payload, indent=2))
