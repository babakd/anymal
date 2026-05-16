#!/usr/bin/env python3
"""Build and audit V17 training-data hardening artifacts on the Modal volume."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v17-data-artifacts")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "datasets>=2.19.0",
        "huggingface_hub>=0.19.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    )
    .add_local_dir(Path(__file__).resolve().parents[1], remote_path="/root/anymal", copy=False)
)

POPE_URLS = {
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
VG_IMAGE_URLS = (
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
)
GQA_SPATIAL_TERMS = (
    " left ",
    " right ",
    " above ",
    " below ",
    " underneath ",
    " under ",
    " over ",
    " behind ",
    " in front of ",
    " front of ",
    " next to ",
    " beside ",
    " near ",
    " between ",
    " around ",
    " side ",
    " where ",
    " on top of ",
    " at the top ",
    " at the bottom ",
    " to the left ",
    " to the right ",
    " standing on ",
    " sitting on ",
    " lying on ",
    " holding ",
    " wearing ",
    " looking at ",
    " facing ",
    " carrying ",
    " touching ",
    " covering ",
    " parked ",
)


def _pinned_revision(dataset_id: str) -> str:
    import sys

    sys.path.insert(0, "/root/anymal")
    from evaluation.checkpoint_eval.dataset_revisions import pinned_revision

    return pinned_revision(dataset_id)


def _json_dump(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_hf_filename(prefix: str, split: str, source_index: int, image_id: Any = "") -> str:
    raw = f"{prefix}_{split}_{source_index}_{image_id or ''}"
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


def _normalize_training_answer(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split()).rstrip(".,;:!?")


def _hf_image_to_rgb(image_value: Any, fallback_urls: tuple[Any, ...] = ()):
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
    for url in fallback_urls:
        if not url:
            continue
        response = requests.get(str(url), timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    raise RuntimeError(f"Unsupported image payload type: {type(image_value).__name__}")


def _coco_image_id_from_filename(value: Any) -> int | None:
    stem = os.path.basename(str(value or "")).split(".", 1)[0]
    digits = stem.rsplit("_", 1)[-1]
    return int(digits.lstrip("0") or "0") if digits.isdigit() else None


def _pope_eval_coco_ids() -> set[int]:
    import requests

    pope_dir = "/checkpoints/pope_data"
    os.makedirs(pope_dir, exist_ok=True)
    eval_ids: set[int] = set()
    for split, url in POPE_URLS.items():
        path = os.path.join(pope_dir, f"coco_pope_{split}.jsonl")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                image_id = _coco_image_id_from_filename(json.loads(line).get("image"))
                if image_id is not None:
                    eval_ids.add(image_id)
    return eval_ids


def _official_gqa_questions_zip() -> str:
    import requests

    gqa_dir = "/checkpoints/gqa_data"
    os.makedirs(gqa_dir, exist_ok=True)
    path = os.path.join(gqa_dir, "questions1.2.zip")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    tmp_path = f"{path}.tmp"
    with requests.get(GQA_QUESTIONS_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, path)
    volume.commit()
    return path


def _official_gqa_rows(split: str) -> list[tuple[str, dict[str, Any]]]:
    zip_path = _official_gqa_questions_zip()
    target = f"{split}_questions.json"
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [name for name in zf.namelist() if name.endswith(target)]
        if not matches:
            raise RuntimeError(f"Could not find {target} in {zip_path}")
        with zf.open(matches[0]) as f:
            raw = json.load(f)
    rows = [
        (str(question_id), row)
        for question_id, row in raw.items()
        if isinstance(row, dict) and row.get("isBalanced", True)
    ]
    return rows or [(str(question_id), row) for question_id, row in raw.items()]


def _official_gqa_image_filename(image_id: Any) -> str:
    try:
        return f"VG_{int(image_id)}.jpg"
    except Exception:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(image_id))
        return f"VG_{safe}.jpg"


def _valid_image_file(path: str) -> bool:
    from PIL import Image

    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def _ensure_official_gqa_image(image_id: Any, image_dir: str) -> tuple[str, str]:
    import requests
    import threading

    os.makedirs(image_dir, exist_ok=True)
    filename = _official_gqa_image_filename(image_id)
    out_path = os.path.join(image_dir, filename)
    if _valid_image_file(out_path):
        return filename, "cached"
    last_error = ""
    for template in VG_IMAGE_URLS:
        try:
            response = requests.get(template.format(image_id=int(image_id)), timeout=30)
            if response.status_code == 404:
                last_error = "404"
                continue
            response.raise_for_status()
            tmp_path = f"{out_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp_path, "wb") as f:
                f.write(response.content)
            os.replace(tmp_path, out_path)
            if not _valid_image_file(out_path):
                last_error = "downloaded file failed image validation"
                continue
            return filename, "downloaded"
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"Could not fetch GQA image {image_id}: {last_error}")


def _ensure_coco_instance_annotations() -> str:
    import requests

    annotation_dir = "/checkpoints/coco_annotations"
    instances_path = os.path.join(annotation_dir, "annotations", "instances_train2017.json")
    if os.path.exists(instances_path):
        return instances_path
    os.makedirs(annotation_dir, exist_ok=True)
    zip_path = os.path.join(annotation_dir, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        response = requests.get(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            timeout=600,
        )
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(annotation_dir)
    if not os.path.exists(instances_path):
        raise FileNotFoundError(instances_path)
    volume.commit()
    return instances_path


def build_textvqa_majority(max_samples: int = 20000, seed: int = 1502) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "lmms-lab/textvqa"
    revision = _pinned_revision(dataset_id)
    split = "train"
    textvqa_dir = "/checkpoints/textvqa_data"
    image_dir = "/checkpoints/textvqa_images_hf"
    os.makedirs(textvqa_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    output_path = (
        f"{textvqa_dir}/v17_lmms-lab_textvqa_train_majority_seed{seed}"
        f"_n{int(max_samples)}.json"
    )
    rows = load_dataset(dataset_id, split=split, cache_dir="/checkpoints/hf_datasets", revision=revision)
    indices = list(range(len(rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if max_samples:
        indices = indices[: int(max_samples)]

    samples = []
    first_vs_majority_different = 0
    skipped_image = 0
    written = 0
    cached = 0
    for source_index in indices:
        row = rows[int(source_index)]
        question = _first_nonempty_text([row.get("question"), row.get("query")])
        raw_answers = row.get("answers") if isinstance(row.get("answers"), (list, tuple)) else []
        normalized = [
            _normalize_training_answer(answer)
            for answer in raw_answers
            if _normalize_training_answer(answer)
        ]
        if normalized:
            counts = Counter(normalized)
            majority, majority_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            first = normalized[0]
        else:
            majority = _normalize_training_answer(_first_nonempty_text([row.get("answer"), row.get("label")]))
            first = majority
            majority_count = 0
        if not question or not majority:
            continue
        image_id = row.get("image_id", row.get("image_path", source_index))
        filename = _safe_hf_filename("textvqa", split, int(source_index), image_id)
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            cached += 1
        else:
            try:
                _hf_image_to_rgb(
                    row.get("image"),
                    fallback_urls=(
                        row.get("flickr_300k_url"),
                        row.get("flickr_original_url"),
                        row.get("image_url"),
                    ),
                ).save(image_path, format="JPEG", quality=95)
                written += 1
            except Exception:
                skipped_image += 1
                continue
        samples.append(
            {
                "id": f"textvqa_{split}_{source_index}_{image_id}",
                "image": filename,
                "source_dataset": dataset_id,
                "source_dataset_revision": revision,
                "source_split": split,
                "source_index": int(source_index),
                "textvqa_image_id": str(image_id),
                "first_annotator_answer": first,
                "majority_answer": majority,
                "majority_answer_count": int(majority_count),
                "annotator_answer_count": len(normalized),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": majority},
                ],
            }
        )
        first_vs_majority_different += int(first != majority)

    _json_dump(output_path, samples)
    sample_rng = random.Random(1702)
    sample_rows = sample_rng.sample(samples, min(50, len(samples))) if samples else []
    report = {
        "output_path": output_path,
        "image_dir": image_dir,
        "dataset_id": dataset_id,
        "revision": revision,
        "rows": len(samples),
        "source_rows_considered": len(indices),
        "images_written": written,
        "images_cached": cached,
        "skipped_image_rows": skipped_image,
        "first_vs_majority_different": first_vs_majority_different,
        "first_vs_majority_different_rate": (
            first_vs_majority_different / len(samples) if samples else 0.0
        ),
        "sampled_first_majority_pairs": [
            {
                "source_index": row["source_index"],
                "first_annotator_answer": row["first_annotator_answer"],
                "majority_answer": row["majority_answer"],
            }
            for row in sample_rows
        ],
    }
    _json_dump("/checkpoints/v17_reports/textvqa_majority_report.json", report)
    volume.commit()
    return report


def _build_pope_style(kind: str, max_samples: int, image_dir: str, pope_eval_ids: set[int]) -> dict[str, Any]:
    instances_path = _ensure_coco_instance_annotations()
    with open(instances_path, "r", encoding="utf-8") as f:
        instances = json.load(f)
    available_images = {
        filename for filename in os.listdir(image_dir) if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    images = {
        int(image["id"]): image
        for image in instances.get("images", [])
        if image.get("file_name") in available_images and int(image["id"]) not in pope_eval_ids
    }
    dropped_leakage = sum(
        1
        for image in instances.get("images", [])
        if image.get("file_name") in available_images and int(image["id"]) in pope_eval_ids
    )
    categories = {category["id"]: category["name"] for category in instances.get("categories", [])}
    all_categories = sorted(set(categories.values()))
    present_by_image = {image_id: set() for image_id in images}
    for ann in instances.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        image_id = ann.get("image_id")
        if image_id not in present_by_image:
            continue
        category = categories.get(ann.get("category_id"))
        if category:
            present_by_image[int(image_id)].add(category)

    rng = random.Random(7421 if kind == "presence" else 7321)
    image_ids = list(images)
    rng.shuffle(image_ids)
    samples = []
    for image_id in image_ids:
        if kind == "presence":
            pool = sorted(present_by_image.get(image_id, set()))
            if not pool:
                continue
            category = pool[(image_id * 1543) % len(pool)]
            answer = "yes"
            key = "present_category"
            counterfactual_type = "pope_style_object_presence"
        else:
            pool = [name for name in all_categories if name not in present_by_image.get(image_id, set())]
            if not pool:
                continue
            category = pool[(image_id * 1543) % len(pool)]
            answer = "no"
            key = "absent_category"
            counterfactual_type = "pope_style_object_absence"
        article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        samples.append(
            {
                "id": f"coco_{kind}_{image_id}_{category.replace(' ', '_')}",
                "image": images[image_id]["file_name"],
                "counterfactual_type": counterfactual_type,
                key: category,
                "conversations": [
                    {"from": "human", "value": f"<image>\nIs there {article} {category} in the image?"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
        if len(samples) >= int(max_samples):
            break

    output_path = (
        f"/checkpoints/llava_data/coco_pope_style_{kind}_train2017_leakclean_"
        f"{int(max_samples)}.json"
    )
    _json_dump(output_path, samples)
    return {
        "kind": kind,
        "output_path": output_path,
        "rows": len(samples),
        "available_non_eval_images": len(images),
        "dropped_pope_eval_overlap_images": dropped_leakage,
        "pope_eval_image_ids": len(pope_eval_ids),
        "sample_image_ids": [
            _coco_image_id_from_filename(row["image"])
            for row in samples[:20]
        ],
    }


def build_pope_leakclean(max_samples: int = 10000) -> dict[str, Any]:
    image_dir = "/checkpoints/coco_images"
    pope_eval_ids = _pope_eval_coco_ids()
    reports = [
        _build_pope_style("presence", max_samples, image_dir, pope_eval_ids),
        _build_pope_style("absence", max_samples, image_dir, pope_eval_ids),
    ]
    result = {"image_dir": image_dir, "sources": reports}
    _json_dump("/checkpoints/v17_reports/pope_leakclean_report.json", result)
    volume.commit()
    return result


def _gqa_metadata_match(row: dict[str, Any]) -> bool:
    types = row.get("types") or {}
    structural = str(types.get("structural") or "").strip().lower()
    semantic = str(types.get("semantic") or "").strip().lower()
    return semantic in {"rel", "relate"} or structural in {"verify", "logical"}


def _gqa_keyword_match(row: dict[str, Any]) -> bool:
    q = f" {' '.join(str(row.get('question') or '').lower().split())} "
    return any(term in q for term in GQA_SPATIAL_TERMS)


def build_gqa_spatial_compare(max_samples: int = 15000, seed: int = 1503) -> dict[str, Any]:
    dataset_id = "Mineru/GQA"
    revision = _pinned_revision(dataset_id)
    split = "train_balanced"
    rows = _official_gqa_rows(split)
    rng = random.Random(int(seed))
    rng.shuffle(rows)

    metadata_selected = []
    keyword_selected = []
    scanned_rows = 0
    for source_index, (question_id, row) in enumerate(rows):
        scanned_rows += 1
        if _gqa_metadata_match(row):
            metadata_selected.append((source_index, question_id, row))
        if _gqa_keyword_match(row):
            keyword_selected.append((source_index, question_id, row))
        if len(metadata_selected) >= max_samples and len(keyword_selected) >= max_samples:
            break
    metadata_selected = metadata_selected[: int(max_samples)]
    keyword_selected = keyword_selected[: int(max_samples)]

    def to_instruction(selected: list[tuple[int, str, dict[str, Any]]], tag: str) -> tuple[str, list[dict[str, Any]]]:
        image_dir = "/checkpoints/gqa_images_hf"
        os.makedirs(image_dir, exist_ok=True)
        prepared = []

        def prepare(item: tuple[int, str, dict[str, Any]]) -> tuple[int, str, dict[str, Any], str | None, str | None]:
            source_index, question_id, row = item
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                return source_index, question_id, row, None, "missing_question_or_answer"
            image_id = row.get("imageId", row.get("image_id", source_index))
            try:
                filename, _status = _ensure_official_gqa_image(image_id, image_dir)
                return source_index, question_id, row, filename, None
            except Exception as exc:
                return source_index, question_id, row, None, str(exc)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(prepare, item) for item in selected]
            for idx, future in enumerate(as_completed(futures), start=1):
                prepared.append(future.result())
                if idx % 1000 == 0:
                    volume.commit()

        samples = []
        skipped_images = 0
        for source_index, question_id, row, filename, error in sorted(prepared, key=lambda item: item[0]):
            if error or not filename:
                skipped_images += 1
                continue
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            image_id = row.get("imageId", row.get("image_id", source_index))
            samples.append(
                {
                    "id": f"gqa_{tag}_{split}_{question_id}",
                    "image": filename,
                    "source_dataset": dataset_id,
                    "source_dataset_revision": revision,
                    "source_questions_url": GQA_QUESTIONS_URL,
                    "source_split": split,
                    "source_index": int(source_index),
                    "source_question_id": str(question_id),
                    "source_image_id": str(image_id),
                    "gqa_types": row.get("types") or {},
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
        output_path = (
            f"/checkpoints/gqa_data/v17_gqa_{tag}_{split}_seed{seed}_n{int(max_samples)}.json"
        )
        _json_dump(output_path, samples)
        return output_path, samples

    metadata_path, metadata_samples = to_instruction(metadata_selected, "metadata_spatial")
    keyword_path, keyword_samples = to_instruction(keyword_selected, "keyword_spatial")
    sample_rng = random.Random(1717)
    result = {
        "dataset_id": dataset_id,
        "revision": revision,
        "split": split,
        "seed": int(seed),
        "max_samples": int(max_samples),
        "scanned_rows": int(scanned_rows),
        "selection_note": "Official GQA questions1.2 metadata, shuffled deterministically for V17 audit materialization.",
        "metadata_path": metadata_path,
        "keyword_path": keyword_path,
        "metadata_rows": len(metadata_samples),
        "keyword_rows": len(keyword_samples),
        "metadata_sample_rows": [
            {
                "source_index": row["source_index"],
                "question": row["conversations"][0]["value"].replace("<image>\n", ""),
                "answer": row["conversations"][1]["value"],
                "types": row["gqa_types"],
            }
            for row in sample_rng.sample(metadata_samples, min(50, len(metadata_samples)))
        ],
        "keyword_sample_rows": [
            {
                "source_index": row["source_index"],
                "question": row["conversations"][0]["value"].replace("<image>\n", ""),
                "answer": row["conversations"][1]["value"],
                "types": row["gqa_types"],
            }
            for row in sample_rng.sample(keyword_samples, min(50, len(keyword_samples)))
        ],
    }
    _json_dump("/checkpoints/v17_reports/gqa_spatial_metadata_compare.json", result)
    volume.commit()
    return result


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=6 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def build_all_remote(
    textvqa_samples: int = 20000,
    pope_samples: int = 10000,
    gqa_samples: int = 15000,
    remote_output_path: str = "/checkpoints/v17_reports/data_artifacts_report.json",
) -> dict[str, Any]:
    result = {
        "textvqa_majority": build_textvqa_majority(max_samples=textvqa_samples),
        "pope_leakclean": build_pope_leakclean(max_samples=pope_samples),
        "gqa_spatial_compare": build_gqa_spatial_compare(max_samples=gqa_samples),
    }
    if remote_output_path:
        _json_dump(remote_output_path, result)
        volume.commit()
    return result


@app.local_entrypoint()
def main(
    output: str = "v17_data_artifacts_report.json",
    remote_output_path: str = "/checkpoints/v17_reports/data_artifacts_report.json",
    textvqa_samples: int = 20000,
    pope_samples: int = 10000,
    gqa_samples: int = 15000,
):
    result = build_all_remote.remote(
        textvqa_samples=int(textvqa_samples),
        pope_samples=int(pope_samples),
        gqa_samples=int(gqa_samples),
        remote_output_path=str(remote_output_path or ""),
    )
    if remote_output_path:
        result["remote_output_path"] = remote_output_path
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
