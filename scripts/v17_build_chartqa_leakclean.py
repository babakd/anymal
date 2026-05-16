#!/usr/bin/env python3
"""Build a ChartQA train cache with validation-image hash leakage removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v17-chartqa-leakclean")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "datasets>=2.19.0",
        "huggingface_hub>=0.19.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
    )
    .add_local_dir(Path(__file__).resolve().parents[1], remote_path="/root/anymal", copy=False)
)


def _pinned_revision(dataset_id: str) -> str:
    import sys

    sys.path.insert(0, "/root/anymal")
    from evaluation.checkpoint_eval.dataset_revisions import pinned_revision

    return pinned_revision(dataset_id)


def _safe_hf_filename(prefix: str, split: str, source_index: int) -> str:
    raw = f"{prefix}_{split}_{source_index}"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_")
    return f"{safe}.jpg"


def _first_nonempty_text(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, (list, tuple)):
            nested = _first_nonempty_text(list(value))
            if nested:
                return nested
            continue
        text = " ".join(str(value or "").split())
        if text:
            return text
    return ""


def _has_reserved_chat_markers(text: str) -> bool:
    value = str(text or "")
    return any(marker in value for marker in ("<|im_start|>", "<|im_end|>", "<image>"))


def _hf_image_to_rgb(image_value: Any):
    import requests
    from PIL import Image

    if hasattr(image_value, "convert"):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes"):
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path") and os.path.exists(image_value["path"]):
            return Image.open(image_value["path"]).convert("RGB")
    if isinstance(image_value, str):
        if os.path.exists(image_value):
            return Image.open(image_value).convert("RGB")
        if image_value.startswith(("http://", "https://")):
            response = requests.get(image_value, timeout=60)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
    raise RuntimeError(f"Unsupported HF image payload type: {type(image_value).__name__}")


def _row_image_value(row: dict[str, Any]) -> Any:
    image_value = row.get("image")
    if image_value is None and row.get("images"):
        image_values = row.get("images")
        if isinstance(image_values, (list, tuple)) and image_values:
            image_value = image_values[0]
    return image_value


def _jpeg_bytes_and_hash(image_obj: Any) -> tuple[bytes, str]:
    buffer = BytesIO()
    image_obj.convert("RGB").save(buffer, format="JPEG", quality=95)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload[: 1024 * 1024]).hexdigest()
    return payload, digest


def _hash_file_first_mib(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read(1024 * 1024))
    return digest.hexdigest()


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=6 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def build_chartqa_leakclean_remote(
    max_samples: int = 20000,
    seed: int = 1501,
    dataset_id: str = "anhdang000/ChartQA-V2",
    output_path: str = "/checkpoints/chartqa_data/v17_chartqa_train_leakclean_val_seed1501_n20000.json",
    remote_report_path: str = "/checkpoints/v17_reports/chartqa_leakclean_report.json",
) -> dict[str, Any]:
    from datasets import load_dataset

    image_dir = "/checkpoints/chartqa_images_hf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    revision = _pinned_revision(dataset_id)
    train_rows = load_dataset(
        dataset_id,
        split="train",
        cache_dir="/checkpoints/hf_datasets",
        revision=revision,
    )
    val_rows = load_dataset(
        dataset_id,
        split="val",
        cache_dir="/checkpoints/hf_datasets",
        revision=revision,
    )

    val_hashes: set[str] = set()
    val_hash_errors = 0
    for val_index in range(len(val_rows)):
        try:
            _, digest = _jpeg_bytes_and_hash(_hf_image_to_rgb(_row_image_value(val_rows[int(val_index)])))
        except Exception as exc:
            val_hash_errors += 1
            print(f"Skipping ChartQA val hash row {val_index}: {exc}", flush=True)
            continue
        val_hashes.add(digest)
    cached_val_hashes = 0
    for filename in os.listdir(image_dir):
        if not (filename.startswith("COCO_val2014_") or filename.startswith("chartqa_val_")):
            continue
        cached_path = os.path.join(image_dir, filename)
        if not os.path.isfile(cached_path):
            continue
        try:
            val_hashes.add(_hash_file_first_mib(cached_path))
            cached_val_hashes += 1
        except OSError as exc:
            val_hash_errors += 1
            print(f"Skipping cached ChartQA val hash {cached_path}: {exc}", flush=True)

    indices = list(range(len(train_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    samples = []
    written = 0
    cached = 0
    skipped_reserved_markers = 0
    skipped_val_hash_overlap = 0
    skipped_image_errors = 0
    overlap_examples = []
    for source_index in indices:
        if int(max_samples) > 0 and len(samples) >= int(max_samples):
            break
        row = train_rows[int(source_index)]
        question = _first_nonempty_text([row.get("query"), row.get("question"), row.get("problem")])
        answer = _first_nonempty_text([row.get("label"), row.get("answer"), row.get("answers")])
        if not question or not answer:
            continue
        if _has_reserved_chat_markers(question) or _has_reserved_chat_markers(answer):
            skipped_reserved_markers += 1
            continue
        filename = _safe_hf_filename("chartqa", "train", int(source_index))
        image_path = os.path.join(image_dir, filename)
        jpeg_payload = None
        try:
            if os.path.exists(image_path):
                cached += 1
                image_digest = _hash_file_first_mib(image_path)
            else:
                jpeg_payload, image_digest = _jpeg_bytes_and_hash(_hf_image_to_rgb(_row_image_value(row)))
        except Exception as exc:
            skipped_image_errors += 1
            print(f"Skipping ChartQA train image row {source_index}: {exc}", flush=True)
            continue
        if image_digest in val_hashes:
            skipped_val_hash_overlap += 1
            if len(overlap_examples) < 20:
                overlap_examples.append(
                    {
                        "source_index": int(source_index),
                        "image": filename,
                        "sha256_first_mib": image_digest,
                    }
                )
            continue
        if jpeg_payload is not None:
            with open(image_path, "wb") as f:
                f.write(jpeg_payload)
            written += 1
            if written % 1000 == 0:
                print(
                    f"ChartQA leakclean progress: rows={len(samples)} "
                    f"written={written} cached={cached} skipped_overlap={skipped_val_hash_overlap}",
                    flush=True,
                )
                volume.commit()
        samples.append(
            {
                "id": f"chartqa_train_{int(source_index)}",
                "image": filename,
                "image_sha256_first_mib": image_digest,
                "source_dataset": dataset_id,
                "source_dataset_revision": revision,
                "source_split": "train",
                "source_index": int(source_index),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

    if not samples:
        raise RuntimeError("ChartQA leak-clean builder produced no samples.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f)
    report = {
        "dataset_id": dataset_id,
        "revision": revision,
        "output_path": output_path,
        "image_dir": image_dir,
        "rows_written": len(samples),
        "target_max_samples": int(max_samples),
        "seed": int(seed),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_unique_image_hashes": len(val_hashes),
        "cached_val_image_hashes_checked": cached_val_hashes,
        "val_hash_errors": val_hash_errors,
        "images_written": written,
        "images_cached": cached,
        "skipped_reserved_markers": skipped_reserved_markers,
        "skipped_val_hash_overlap": skipped_val_hash_overlap,
        "skipped_image_errors": skipped_image_errors,
        "overlap_examples": overlap_examples,
        "leakage_filter": "sha256_first_mib_against_pinned_val_split_jpeg_q95_and_cached_eval_images",
    }
    _write_json(output_path.replace(".json", "_manifest.json"), report)
    if remote_report_path:
        _write_json(remote_report_path, report)
    volume.commit()
    return report


@app.local_entrypoint()
def main(
    output: str = "/tmp/v17_chartqa_leakclean_report.json",
    remote_report_path: str = "/checkpoints/v17_reports/chartqa_leakclean_report.json",
    max_samples: int = 20000,
    seed: int = 1501,
):
    output_path = (
        f"/checkpoints/chartqa_data/v17_chartqa_train_leakclean_val_seed{int(seed)}"
        f"_n{int(max_samples)}.json"
    )
    result = build_chartqa_leakclean_remote.remote(
        max_samples=int(max_samples),
        seed=int(seed),
        output_path=output_path,
        remote_report_path=str(remote_report_path or ""),
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
