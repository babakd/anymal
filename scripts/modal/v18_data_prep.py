"""Acquire and prepare V18 mid-training data artifacts on Modal.

The script follows the Phase 0 policy recorded by ``scripts/modal/v18_phase0.py``:
skip paid, GPL/NC, mixed-license, direct-download, and unverified-license sources
unless the user has explicitly approved the exception for this V18 mixture.
"""

from __future__ import annotations

import json
import os
import random
import hashlib
import re
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v18-data-prep")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)
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
)


V18_ROOT = "/checkpoints/v18_qwen"
V18_DATA = "/checkpoints/v18_data"
IMAGE_EXT_RE = re.compile(r"\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
LCS_HF_REPO = "liuhaotian/LLaVA-Pretrain"
LCS_HF_REVISION = "70f9d1e5e1a697fe35830875cfc7de1dd590d727"
LCS_JSON_FILENAME = "blip_laion_cc_sbu_558k.json"
LCS_IMAGES_ZIP_FILENAME = "images.zip"


LICENSES = {
    "coco": {
        "license": "cc-by-4.0 / COCO terms",
        "license_source": "https://cocodataset.org/#termsofuse",
        "commercial_use_allowed": True,
    },
    "vqav2": {
        "license": "cc-by-4.0 / VQAv2 and COCO terms",
        "license_source": "https://visualqa.org/download.html",
        "commercial_use_allowed": True,
    },
    "gqa": {
        "license": "cc-by-4.0 / GQA terms",
        "license_source": "https://cs.stanford.edu/people/dorarad/gqa/about.html",
        "commercial_use_allowed": True,
    },
    "aokvqa": {
        "license": "apache-2.0",
        "license_source": "https://github.com/allenai/aokvqa",
        "commercial_use_allowed": True,
    },
    "okvqa": {
        "license": "cc-by-4.0 / OK-VQA and COCO terms",
        "license_source": "https://huggingface.co/datasets/HuggingFaceM4/OK-VQA/blob/main/OK-VQA.py",
        "commercial_use_allowed": True,
    },
    "vsr": {
        "license": "cc-by-4.0",
        "license_source": "https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot",
        "commercial_use_allowed": True,
    },
    "ai2d": {
        "license": "apache-2.0",
        "license_source": "https://huggingface.co/datasets/LIME-DATA/ai2d",
        "commercial_use_allowed": True,
    },
    "ocrvqa": {
        "license": "apache-2.0",
        "license_source": "https://huggingface.co/datasets/atc96/OCR-VQA-200K/blob/main/LICENCE.txt",
        "commercial_use_allowed": True,
    },
    "textvqa": {
        "license": "cc-by-4.0 / TextVQA and OpenImages terms",
        "license_source": "https://textvqa.org/dataset/",
        "commercial_use_allowed": True,
    },
    "lcs558k": {
        "license": "mixed; LLaVA-Pretrain caption license noted",
        "license_source": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain",
        "license_note": (
            "User approved inclusion for V18. Source is LLaVA Visual Instruct "
            "Pretrain LCS-558K from blip_laion_cc_sbu_558k.json; HF card "
            "license tag is 'other' and the caption/image provenance is "
            "license-dependent across LLaVA/LAION/CC/SBU components."
        ),
        "commercial_use_allowed": True,
    },
    "visual_genome": {
        "license": "cc-by-4.0 / Visual Genome terms",
        "license_source": "https://homes.cs.washington.edu/~ranjay/visualgenome/api.html",
        "commercial_use_allowed": True,
    },
}


def _json_dump(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_images(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        1
        for entry in os.scandir(path)
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )


def _count_images_recursive(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += sum(1 for name in files if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    return total


def _json_len(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    payload = _load_json(path)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("samples", "data", "annotations", "questions"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return len(rows)
    return None


def _iter_records(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return rows
    for key in ("predictions", "prediction_samples", "records", "samples", "data", "annotations", "questions"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    for run in payload.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for key in ("predictions", "prediction_samples"):
            value = run.get(key)
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        dataset_meta = run.get("dataset_meta") or {}
        image_dir = dataset_meta.get("image_dir") or payload.get("image_dir")
        for image_id in dataset_meta.get("selected_image_ids") or []:
            rows.append({"image_id": image_id, "_image_dir": image_dir})
        for image_id in dataset_meta.get("source_image_ids") or []:
            rows.append({"image_id": image_id, "_image_dir": image_dir})
    return rows


def _row_refs(row: dict[str, Any]) -> list[Any]:
    refs: list[Any] = []
    for key in (
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
    ):
        if key in row:
            value = row[key]
            if isinstance(value, (list, tuple, set)):
                refs.extend(value)
            else:
                refs.append(value)
    return refs


def _candidate_paths(ref: Any, image_dir: str) -> list[str]:
    text = str(ref or "").strip()
    if not text or not image_dir:
        return []
    paths = []
    if os.path.isabs(text):
        paths.append(text)
    basename = os.path.basename(text)
    paths.append(os.path.join(image_dir, text))
    if basename != text:
        paths.append(os.path.join(image_dir, basename))
    stem = os.path.splitext(basename)[0]
    numeric = stem.rsplit("_", 1)[-1]
    if numeric.isdigit():
        image_id = int(numeric)
        paths.extend(
            [
                os.path.join(image_dir, f"{image_id:012d}.jpg"),
                os.path.join(image_dir, f"COCO_val2014_{image_id:012d}.jpg"),
                os.path.join(image_dir, f"COCO_train2014_{image_id:012d}.jpg"),
                os.path.join(image_dir, f"COCO_train2017_{image_id:012d}.jpg"),
                os.path.join(image_dir, f"VG_{image_id}.jpg"),
                os.path.join(image_dir, f"{image_id}.jpg"),
            ]
        )
    if text.lower().startswith("n") and text[1:].isdigit():
        image_id = int(text[1:])
        paths.extend(
            [
                os.path.join(image_dir, f"{text}.jpg"),
                os.path.join(image_dir, f"VG_{image_id}.jpg"),
                os.path.join(image_dir, f"{image_id}.jpg"),
            ]
        )
    return list(dict.fromkeys(paths))


def _resolve_paths(json_path: str, image_dir: str) -> tuple[set[str], list[dict[str, Any]]]:
    payload = _load_json(json_path)
    paths: set[str] = set()
    missing = []
    resolution_cache: dict[tuple[str, str], str | None] = {}
    for row in _iter_records(payload):
        row_dir = str(row.get("_image_dir") or image_dir or "")
        for ref in _row_refs(row):
            text = str(ref or "").strip()
            if not text:
                continue
            if not (IMAGE_EXT_RE.search(text) or text.isdigit() or (text.startswith("n") and text[1:].isdigit())):
                continue
            cache_key = (row_dir, text)
            found = resolution_cache.get(cache_key)
            if cache_key not in resolution_cache:
                found = None
                for candidate in _candidate_paths(text, row_dir):
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        found = candidate
                        break
                resolution_cache[cache_key] = found
            if found:
                paths.add(found)
            elif len(missing) < 20:
                missing.append({"ref": text, "image_dir": row_dir, "json_path": json_path})
    return paths, missing


def _resolve_row_paths(row: dict[str, Any], image_dir: str) -> set[str]:
    paths: set[str] = set()
    row_dir = str(row.get("_image_dir") or image_dir or "")
    for ref in _row_refs(row):
        text = str(ref or "").strip()
        if not text:
            continue
        if not (IMAGE_EXT_RE.search(text) or text.isdigit() or (text.startswith("n") and text[1:].isdigit())):
            continue
        for candidate in _candidate_paths(text, row_dir):
            if os.path.exists(candidate) and os.path.isfile(candidate):
                paths.add(candidate)
                break
    return paths


def _resolve_primary_image_path(row: dict[str, Any], image_dir: str) -> str | None:
    for key in ("image", "image_file", "file_name", "filename", "image_path"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            refs = list(value)
        else:
            refs = [value]
        for ref in refs:
            text = str(ref or "").strip()
            if not text:
                continue
            if not (IMAGE_EXT_RE.search(text) or text.isdigit() or (text.startswith("n") and text[1:].isdigit())):
                continue
            for candidate in _candidate_paths(text, image_dir):
                if os.path.exists(candidate) and os.path.isfile(candidate):
                    return candidate
    return None


def _primary_image_candidate_path(row: dict[str, Any], image_dir: str) -> str | None:
    for key in ("image", "image_file", "file_name", "filename", "image_path"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            refs = list(value)
        else:
            refs = [value]
        for ref in refs:
            text = str(ref or "").strip()
            if not text:
                continue
            if not (IMAGE_EXT_RE.search(text) or text.isdigit() or (text.startswith("n") and text[1:].isdigit())):
                continue
            if os.path.isabs(text):
                return text
            return os.path.join(image_dir, os.path.basename(text))
    return None


def _hash_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read(1024 * 1024))
    return digest.hexdigest()


def _hash_paths(paths: set[str], label: str, max_workers: int = 32) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    sorted_paths = sorted(paths)
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {executor.submit(_hash_path, path): path for path in sorted_paths}
        for idx, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            digest = future.result()
            result.setdefault(digest, []).append(path)
            if idx % 5000 == 0 or idx == len(sorted_paths):
                print(f"hash {label}: {idx}/{len(sorted_paths)} unique paths", flush=True)
    print(f"hash {label}: done {len(paths)} unique paths", flush=True)
    return result


def _sample_rows(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= int(max_samples):
        return list(rows)
    rng = random.Random(int(seed))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[i] for i in indices[: int(max_samples)]]


def _reservoir_add(
    reservoir: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    seen_count: int,
    max_samples: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < int(max_samples):
        reservoir.append(row)
        return
    replacement_index = rng.randrange(int(seen_count))
    if replacement_index < int(max_samples):
        reservoir[replacement_index] = row


def _download_file(url: str, output_path: str, *, expected_min_bytes: int = 1) -> dict[str, Any]:
    import requests

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) >= int(expected_min_bytes):
        return {
            "url": url,
            "path": output_path,
            "bytes": os.path.getsize(output_path),
            "cached": True,
        }
    tmp_path = output_path + ".tmp"
    print(f"Downloading {url} -> {output_path}", flush=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(tmp_path) < int(expected_min_bytes):
        raise RuntimeError(
            f"Downloaded file is too small: {tmp_path} has {os.path.getsize(tmp_path)} bytes"
        )
    os.replace(tmp_path, output_path)
    return {
        "url": url,
        "path": output_path,
        "bytes": os.path.getsize(output_path),
        "cached": False,
    }


def _normal_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normal_answer(value: Any) -> str:
    return _normal_text(value).rstrip(".,;:!?")


def _safe_stem(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return safe.strip("._") or fallback


def _save_image(image_value: Any, output_path: str, fallback_url: str = "") -> bool:
    import requests
    from PIL import Image

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    image_obj = None
    if hasattr(image_value, "convert"):
        image_obj = image_value
    elif isinstance(image_value, dict):
        if image_value.get("bytes"):
            image_obj = Image.open(BytesIO(image_value["bytes"]))
        elif image_value.get("path") and os.path.exists(image_value["path"]):
            image_obj = Image.open(image_value["path"])
        elif image_value.get("src"):
            fallback_url = str(image_value["src"])
    elif isinstance(image_value, str):
        if os.path.exists(image_value):
            image_obj = Image.open(image_value)
        elif image_value.startswith(("http://", "https://")):
            fallback_url = image_value
    if image_obj is None and fallback_url:
        response = requests.get(fallback_url, timeout=60)
        response.raise_for_status()
        image_obj = Image.open(BytesIO(response.content))
    if image_obj is None:
        return False
    image_obj.convert("RGB").save(output_path, format="JPEG", quality=95)
    return True


def _majority_answer(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        stripped = values.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                values = ast.literal_eval(stripped)
            except Exception:
                return _normal_answer(stripped)
        else:
            return _normal_answer(stripped)
    if not isinstance(values, (list, tuple)):
        return _normal_answer(values)
    answers = [_normal_answer(value).lower() for value in values if _normal_answer(value)]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _tag_rows(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    dataset_family: str,
    loss_family: str,
    license_key: str,
    teacher_kl_weight: float,
) -> list[dict[str, Any]]:
    metadata = LICENSES[license_key]
    tagged = []
    for row in rows:
        tagged.append(
            {
                **row,
                "source_name": source_name,
                "dataset_family": dataset_family,
                "loss_family": loss_family,
                "teacher_kl_weight": float(teacher_kl_weight),
                **metadata,
            }
        )
    return tagged


def _write_sampled_source(
    *,
    source_name: str,
    input_path: str,
    output_path: str,
    max_samples: int,
    seed: int,
    dataset_family: str,
    loss_family: str,
    license_key: str,
    teacher_kl_weight: float,
) -> dict[str, Any]:
    rows = _load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON at {input_path}")
    sampled = _sample_rows(rows, int(max_samples), int(seed))
    tagged = _tag_rows(
        sampled,
        source_name=source_name,
        dataset_family=dataset_family,
        loss_family=loss_family,
        license_key=license_key,
        teacher_kl_weight=teacher_kl_weight,
    )
    _json_dump(output_path, tagged)
    return {
        "name": source_name,
        "data_path": output_path,
        "input_path": input_path,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": dataset_family,
        "loss_family": loss_family,
        "teacher_kl_weight": float(teacher_kl_weight),
        **LICENSES[license_key],
    }


def build_coco_captions(max_samples: int = 100000, seed: int = 1802) -> dict[str, Any]:
    annotations_path = "/checkpoints/coco_annotations/annotations/captions_train2017.json"
    image_dir = "/checkpoints/coco_images"
    output_path = f"{V18_DATA}/coco_captions_train2017_seed{seed}_n{max_samples}.json"
    if os.path.exists(output_path) and _json_len(output_path) == int(max_samples):
        return {
            "name": "coco_captions",
            "data_path": output_path,
            "image_dir": image_dir,
            "rows": int(max_samples),
            "cached": True,
            "dataset_family": "broad_alignment",
            "loss_family": "caption",
            "teacher_kl_weight": 1.0,
            **LICENSES["coco"],
        }
    payload = _load_json(annotations_path)
    images = {
        int(image["id"]): image.get("file_name")
        for image in payload.get("images", [])
        if image.get("file_name")
    }
    available = {
        entry.name
        for entry in os.scandir(image_dir)
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    }
    best_caption_by_image: dict[int, str] = {}
    for ann in payload.get("annotations", []):
        image_id = int(ann.get("image_id"))
        file_name = images.get(image_id)
        caption = " ".join(str(ann.get("caption") or "").split())
        if not file_name or file_name not in available or not caption:
            continue
        current = best_caption_by_image.get(image_id, "")
        if len(caption) > len(current):
            best_caption_by_image[image_id] = caption
    rows = [
        {
            "id": f"coco_train2017_caption_{image_id}",
            "image": images[image_id],
            "source_dataset": "COCO Captions train2017",
            "source_split": "train2017",
            "source_image_id": str(image_id),
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt", "value": caption},
            ],
        }
        for image_id, caption in best_caption_by_image.items()
    ]
    sampled = _sample_rows(rows, int(max_samples), int(seed))
    tagged = _tag_rows(
        sampled,
        source_name="coco_captions",
        dataset_family="broad_alignment",
        loss_family="caption",
        license_key="coco",
        teacher_kl_weight=1.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "coco_captions",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "available_caption_images": len(rows),
        "sample_seed": int(seed),
        "dataset_family": "broad_alignment",
        "loss_family": "caption",
        "teacher_kl_weight": 1.0,
        **LICENSES["coco"],
    }


def build_gqa_broad(max_samples: int = 200000, seed: int = 1803) -> dict[str, Any]:
    gqa_zip = "/checkpoints/gqa_data/questions1.2.zip"
    image_dir = "/checkpoints/gqa_images_hf"
    output_path = f"{V18_DATA}/gqa_train_balanced_broad_seed{seed}_n{max_samples}.json"
    if os.path.exists(output_path) and (_json_len(output_path) or 0) >= int(max_samples):
        return {
            "name": "gqa_train_balanced_broad",
            "data_path": output_path,
            "image_dir": image_dir,
            "rows": _json_len(output_path),
            "cached": True,
            "dataset_family": "broad_vqa",
            "loss_family": "direct_answer",
            "teacher_kl_weight": 1.0,
            **LICENSES["gqa"],
        }
    with zipfile.ZipFile(gqa_zip, "r") as zf:
        matches = [name for name in zf.namelist() if name.endswith("train_balanced_questions.json")]
        if not matches:
            raise RuntimeError(f"Could not find train_balanced_questions.json in {gqa_zip}")
        with zf.open(matches[0]) as f:
            raw = json.load(f)
    candidates = []
    for question_id, row in raw.items():
        if not isinstance(row, dict) or not row.get("isBalanced", True):
            continue
        question = _normal_text(row.get("question"))
        answer = _normal_answer(row.get("answer")).lower()
        image_id = row.get("imageId", row.get("image_id"))
        if not question or not answer or image_id is None:
            continue
        try:
            filename = f"VG_{int(image_id)}.jpg"
        except Exception:
            filename = f"VG_{_safe_stem(image_id, str(question_id))}.jpg"
        if not os.path.exists(os.path.join(image_dir, filename)):
            continue
        candidates.append(
            {
                "id": f"gqa_train_balanced_{question_id}",
                "image": filename,
                "source_dataset": "GQA train_balanced",
                "source_split": "train_balanced",
                "source_question_id": str(question_id),
                "source_image_id": str(image_id),
                "gqa_types": row.get("types") or {},
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    sampled = _sample_rows(candidates, int(max_samples), int(seed))
    tagged = _tag_rows(
        sampled,
        source_name="gqa_train_balanced_broad",
        dataset_family="broad_vqa",
        loss_family="direct_answer",
        license_key="gqa",
        teacher_kl_weight=1.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "gqa_train_balanced_broad",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "available_rows_with_cached_images": len(candidates),
        "sample_seed": int(seed),
        "dataset_family": "broad_vqa",
        "loss_family": "direct_answer",
        "teacher_kl_weight": 1.0,
        **LICENSES["gqa"],
    }


def build_aokvqa(max_samples: int = 17000, seed: int = 1804) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "HuggingFaceM4/A-OKVQA"
    image_dir = "/checkpoints/aokvqa_images_hf"
    output_path = f"{V18_DATA}/aokvqa_train_seed{seed}_n{max_samples}.json"
    rows = load_dataset(dataset_id, split="train", cache_dir="/checkpoints/hf_datasets")
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    samples = []
    for out_idx, source_index in enumerate(indices):
        row = rows[int(source_index)]
        question = _normal_text(row.get("question"))
        answer = _majority_answer(row.get("direct_answers"))
        if not answer:
            choices = row.get("choices") or []
            try:
                correct_idx = int(row.get("correct_choice_idx"))
                answer = _normal_answer(choices[correct_idx])
            except Exception:
                answer = ""
        if not question or not answer:
            continue
        qid = row.get("question_id", source_index)
        filename = f"aokvqa_train_{int(source_index):06d}_{_safe_stem(qid, str(source_index))}.jpg"
        if not _save_image(row.get("image"), os.path.join(image_dir, filename)):
            continue
        samples.append(
            {
                "id": f"aokvqa_train_{qid}",
                "image": filename,
                "source_dataset": dataset_id,
                "source_split": "train",
                "source_index": int(source_index),
                "source_question_id": str(qid),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
        if len(samples) >= int(max_samples):
            break
    tagged = _tag_rows(
        samples,
        source_name="aokvqa",
        dataset_family="broad_vqa",
        loss_family="direct_answer",
        license_key="aokvqa",
        teacher_kl_weight=0.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "aokvqa",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": "broad_vqa",
        "loss_family": "direct_answer",
        "teacher_kl_weight": 0.0,
        **LICENSES["aokvqa"],
    }


def build_okvqa(max_samples: int = 9000, seed: int = 1804) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "Multimodal-Fatima/OK-VQA_train"
    image_dir = "/checkpoints/okvqa_images_hf"
    output_path = f"{V18_DATA}/okvqa_train_seed{seed}_n{max_samples}.json"
    rows = load_dataset(dataset_id, split="train", cache_dir="/checkpoints/hf_datasets")
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    samples = []
    for source_index in indices:
        row = rows[int(source_index)]
        question = _normal_text(row.get("question"))
        answer = _majority_answer(row.get("answers"))
        if not question or not answer:
            continue
        qid = row.get("question_id", row.get("id", source_index))
        image_id = row.get("id_image", row.get("image_id", source_index))
        filename = f"okvqa_train_{_safe_stem(image_id, str(source_index))}_{_safe_stem(qid, str(source_index))}.jpg"
        if not _save_image(row.get("image"), os.path.join(image_dir, filename)):
            continue
        samples.append(
            {
                "id": f"okvqa_train_{qid}",
                "image": filename,
                "source_dataset": dataset_id,
                "source_split": "train",
                "source_index": int(source_index),
                "source_question_id": str(qid),
                "source_image_id": str(image_id),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
        if len(samples) >= int(max_samples):
            break
    tagged = _tag_rows(
        samples,
        source_name="okvqa",
        dataset_family="broad_vqa",
        loss_family="direct_answer",
        license_key="okvqa",
        teacher_kl_weight=0.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "okvqa",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": "broad_vqa",
        "loss_family": "direct_answer",
        "teacher_kl_weight": 0.0,
        **LICENSES["okvqa"],
    }


def build_vsr(max_samples: int = 10000, seed: int = 1805) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "cambridgeltl/vsr_zeroshot"
    image_dir = "/checkpoints/vsr_images_hf"
    output_path = f"{V18_DATA}/vsr_train_seed{seed}_n{max_samples}.json"
    rows = list(load_dataset(dataset_id, split="train", cache_dir="/checkpoints/hf_datasets"))
    rows.extend(load_dataset(dataset_id, split="validation", cache_dir="/checkpoints/hf_datasets"))
    sampled = _sample_rows(rows, int(max_samples), int(seed))
    samples = []
    coco_dir = "/checkpoints/coco_images"
    for source_index, row in enumerate(sampled):
        caption = _normal_text(row.get("caption"))
        label = str(row.get("label")).strip()
        answer = "yes" if label in {"1", "true", "True"} else "no"
        image_name = os.path.basename(str(row.get("image") or ""))
        if not caption or not image_name:
            continue
        filename = image_name
        source_path = os.path.join(coco_dir, image_name)
        output_image_path = os.path.join(image_dir, filename)
        if os.path.exists(source_path) and not os.path.exists(output_image_path):
            from PIL import Image

            os.makedirs(image_dir, exist_ok=True)
            Image.open(source_path).convert("RGB").save(output_image_path, format="JPEG", quality=95)
        elif not os.path.exists(output_image_path):
            # Avoid slow sequential COCO URL fetches. V18 uses the cached subset
            # here; missing VSR images are documented in the manifest.
            continue
        samples.append(
            {
                "id": f"vsr_{source_index}_{_safe_stem(image_name, str(source_index))}",
                "image": filename,
                "source_dataset": dataset_id,
                "source_split": "train_plus_validation",
                "source_index": int(source_index),
                "relation": row.get("relation"),
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\nDoes this sentence correctly describe the image? Answer yes or no.\nSentence: {caption}",
                    },
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    tagged = _tag_rows(
        samples,
        source_name="vsr",
        dataset_family="broad_vqa",
        loss_family="spatial_verification",
        license_key="vsr",
        teacher_kl_weight=0.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "vsr",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": "broad_vqa",
        "loss_family": "spatial_verification",
        "teacher_kl_weight": 0.0,
        **LICENSES["vsr"],
    }


def build_ai2d(max_samples: int = 5000, seed: int = 1806) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "LIME-DATA/ai2d"
    image_dir = "/checkpoints/ai2d_images_hf"
    output_path = f"{V18_DATA}/ai2d_train_seed{seed}_n{max_samples}.json"
    rows = load_dataset(dataset_id, split="train", cache_dir="/checkpoints/hf_datasets")
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    samples = []
    for source_index in indices[: int(max_samples)]:
        row = rows[int(source_index)]
        question = _normal_text(row.get("question"))
        options = [str(option) for option in (row.get("options") or [])]
        try:
            answer_index = int(row.get("answer"))
        except Exception:
            answer_index = -1
        if not question or not options or not (0 <= answer_index < len(options)):
            continue
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_text = "\n".join(f"{letters[i]}. {option}" for i, option in enumerate(options))
        answer = letters[answer_index]
        doc_id = row.get("doc_id", source_index)
        filename = f"ai2d_train_{int(source_index):06d}_{_safe_stem(doc_id, str(source_index))}.jpg"
        if not _save_image(row.get("image"), os.path.join(image_dir, filename)):
            continue
        samples.append(
            {
                "id": f"ai2d_train_{source_index}_{doc_id}",
                "image": filename,
                "source_dataset": dataset_id,
                "source_split": "train",
                "source_index": int(source_index),
                "conversations": [
                    {"from": "human", "value": f"<image>\nQuestion: {question}\nOptions:\n{option_text}\nAnswer with the correct option letter."},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    tagged = _tag_rows(
        samples,
        source_name="ai2d",
        dataset_family="capability",
        loss_family="diagram_qa",
        license_key="ai2d",
        teacher_kl_weight=0.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "ai2d",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": "capability",
        "loss_family": "diagram_qa",
        "teacher_kl_weight": 0.0,
        **LICENSES["ai2d"],
    }


def build_ocrvqa(max_samples: int = 50000, seed: int = 1807) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "howard-hou/OCR-VQA"
    image_dir = "/checkpoints/ocrvqa_images_hf"
    output_path = f"{V18_DATA}/ocrvqa_train_seed{seed}_n{max_samples}.json"
    rows = load_dataset(dataset_id, split="train", cache_dir="/checkpoints/hf_datasets")
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    samples = []
    for source_index in indices:
        row = rows[int(source_index)]
        image_id = row.get("image_id", source_index)
        filename = f"ocrvqa_train_{_safe_stem(image_id, str(source_index))}.jpg"
        if not _save_image(row.get("image"), os.path.join(image_dir, filename), fallback_url=str(row.get("image_url") or "")):
            continue
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        for pair_index, (question, answer) in enumerate(zip(questions, answers)):
            question = _normal_text(question)
            answer = _normal_answer(answer)
            if not question or not answer:
                continue
            samples.append(
                {
                    "id": f"ocrvqa_train_{image_id}_{pair_index}",
                    "image": filename,
                    "source_dataset": dataset_id,
                    "source_split": "train",
                    "source_index": int(source_index),
                    "source_image_id": str(image_id),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
            if len(samples) >= int(max_samples):
                break
        if len(samples) >= int(max_samples):
            break
    tagged = _tag_rows(
        samples,
        source_name="ocrvqa",
        dataset_family="capability",
        loss_family="ocr_vqa",
        license_key="ocrvqa",
        teacher_kl_weight=0.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "ocrvqa",
        "data_path": output_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "sample_seed": int(seed),
        "dataset_family": "capability",
        "loss_family": "ocr_vqa",
        "teacher_kl_weight": 0.0,
        **LICENSES["ocrvqa"],
    }


def _lcs_zip_members(zip_path: str) -> dict[str, str]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing LCS-558k image zip: {zip_path}")
    members: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not IMAGE_EXT_RE.search(name):
                continue
            ref = name[len("images/") :] if name.startswith("images/") else name
            members[ref] = name
    return members


def _hf_dataset_file_metadata(repo_id: str, revision: str, filenames: list[str]) -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
        )
    except TypeError:
        info = api.dataset_info(repo_id, revision=revision)
    wanted = set(filenames)
    metadata: dict[str, dict[str, Any]] = {}
    for sibling in getattr(info, "siblings", []) or []:
        filename = getattr(sibling, "rfilename", None)
        if filename not in wanted:
            continue
        item: dict[str, Any] = {"rfilename": filename}
        for key in ("size", "blob_id"):
            value = getattr(sibling, key, None)
            if value is not None:
                item[key] = value
        metadata[str(filename)] = item
    missing = sorted(wanted - set(metadata))
    if missing:
        raise RuntimeError(
            f"{repo_id}@{revision} is missing expected LCS-558k files: {missing}"
        )
    return metadata


def _ensure_lcs558k_source_files() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    root = "/checkpoints/llava_pretrain"
    json_dir = "/checkpoints/llava_data"
    os.makedirs(root, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, LCS_JSON_FILENAME)
    zip_path = os.path.join(root, LCS_IMAGES_ZIP_FILENAME)
    file_metadata = _hf_dataset_file_metadata(
        LCS_HF_REPO,
        LCS_HF_REVISION,
        [LCS_JSON_FILENAME, LCS_IMAGES_ZIP_FILENAME],
    )

    for filename, target_path, target_dir in (
        (LCS_JSON_FILENAME, json_path, json_dir),
        (LCS_IMAGES_ZIP_FILENAME, zip_path, root),
    ):
        expected_size = file_metadata.get(filename, {}).get("size")
        if os.path.exists(target_path):
            actual_size = os.path.getsize(target_path)
            if expected_size is not None and int(actual_size) != int(expected_size):
                raise RuntimeError(
                    f"Cached {filename} size mismatch for {LCS_HF_REPO}@{LCS_HF_REVISION}: "
                    f"{actual_size} bytes at {target_path}, expected {expected_size}"
                )
            continue
        downloaded_path = hf_hub_download(
            repo_id=LCS_HF_REPO,
            repo_type="dataset",
            revision=LCS_HF_REVISION,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
            os.replace(downloaded_path, target_path)
        expected_size = file_metadata.get(filename, {}).get("size")
        if expected_size is not None and os.path.getsize(target_path) != int(expected_size):
            raise RuntimeError(
                f"Downloaded {filename} size mismatch for {LCS_HF_REPO}@{LCS_HF_REVISION}: "
                f"{os.path.getsize(target_path)} bytes at {target_path}, expected {expected_size}"
            )

    return {
        "repo_id": LCS_HF_REPO,
        "revision": LCS_HF_REVISION,
        "json_path": json_path,
        "images_zip_path": zip_path,
        "files": file_metadata,
    }


def _ensure_lcs558k_sample_images(
    rows: list[dict[str, Any]],
    *,
    zip_path: str = "/checkpoints/llava_pretrain/images.zip",
    image_dir: str,
) -> dict[str, Any]:
    os.makedirs(image_dir, exist_ok=True)
    refs = sorted({str(row.get("image") or "").strip() for row in rows if row.get("image")})
    missing = [ref for ref in refs if not os.path.exists(os.path.join(image_dir, ref))]
    if not missing:
        return {
            "image_dir": image_dir,
            "sample_image_count": _count_images_recursive(image_dir),
            "sample_refs": len(refs),
            "missing_after_extract": 0,
            "extracted": False,
            "zip_path": zip_path,
            "storage": "sampled_files_from_images_zip",
        }
    members = _lcs_zip_members(zip_path)
    unavailable = [ref for ref in missing if ref not in members]
    if unavailable:
        raise RuntimeError(
            f"LCS-558k image zip is missing {len(unavailable)} selected refs; "
            f"examples={unavailable[:5]}"
        )
    print(
        f"Extracting {len(missing)} selected LCS-558k images from {zip_path} "
        f"to {image_dir}",
        flush=True,
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, ref in enumerate(missing, start=1):
            target_path = os.path.join(image_dir, ref)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            tmp_path = target_path + ".tmp"
            with zf.open(members[ref], "r") as source, open(tmp_path, "wb") as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    if not chunk:
                        break
                    dest.write(chunk)
            os.replace(tmp_path, target_path)
            if idx % 10000 == 0 or idx == len(missing):
                print(f"LCS selected image extract: {idx}/{len(missing)}", flush=True)
    still_missing = [ref for ref in refs if not os.path.exists(os.path.join(image_dir, ref))]
    if still_missing:
        raise RuntimeError(
            f"LCS-558k selected image extraction incomplete: "
            f"{len(still_missing)} refs still missing; examples={still_missing[:5]}"
        )
    return {
        "image_dir": image_dir,
        "sample_image_count": _count_images_recursive(image_dir),
        "sample_refs": len(refs),
        "missing_after_extract": 0,
        "extracted": True,
        "zip_path": zip_path,
        "storage": "sampled_files_from_images_zip",
    }


def build_lcs558k_caption(max_samples: int = 200000, seed: int = 1801) -> dict[str, Any]:
    source_file_report = _ensure_lcs558k_source_files()
    input_path = source_file_report["json_path"]
    zip_path = source_file_report["images_zip_path"]
    image_dir = f"/checkpoints/llava_pretrain/lcs558k_seed{seed}_n{max_samples}_images"
    output_path = f"{V18_DATA}/lcs558k_seed{seed}_n{max_samples}.json"
    if os.path.exists(output_path) and _json_len(output_path) == int(max_samples):
        rows = _load_json(output_path)
        image_report = _ensure_lcs558k_sample_images(rows, image_dir=image_dir)
        return {
            "name": "lcs558k_caption",
            "data_path": output_path,
            "image_dir": image_dir,
            "rows": int(max_samples),
            "cached": True,
            "input_path": input_path,
            "target_rows": int(max_samples),
            "sample_seed": int(seed),
            "dataset_family": "broad_alignment",
            "loss_family": "caption",
            "teacher_kl_weight": 1.0,
            "image_cache": image_report,
            "provenance_verified_against": "blip_laion_cc_sbu_558k.json",
            "hf_dataset": source_file_report,
            "hf_dataset_revision": LCS_HF_REVISION,
            **LICENSES["lcs558k"],
        }
    rows = _load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON at {input_path}")
    zip_members = _lcs_zip_members(zip_path)
    available = []
    missing_examples = []
    for source_index, row in enumerate(rows):
        image_ref = str(row.get("image") or "").strip()
        conversations = row.get("conversations") or []
        if not image_ref or not isinstance(conversations, list) or len(conversations) < 2:
            continue
        if image_ref not in zip_members:
            if len(missing_examples) < 20:
                missing_examples.append(image_ref)
            continue
        available.append(
            {
                **row,
                "id": f"lcs558k_{row.get('id', source_index)}",
                "image": image_ref,
                "source_dataset": "liuhaotian/LLaVA-Pretrain",
                "source_split": "train",
                "source_index": int(source_index),
                "source_image_id": str(row.get("id", source_index)),
            }
        )
    if len(available) < int(max_samples):
        raise RuntimeError(
            f"LCS-558k has only {len(available)} rows with cached images; "
            f"need {max_samples}. Missing examples: {missing_examples[:5]}"
        )
    sampled = _sample_rows(available, int(max_samples), int(seed))
    image_report = _ensure_lcs558k_sample_images(sampled, image_dir=image_dir)
    tagged = _tag_rows(
        sampled,
        source_name="lcs558k_caption",
        dataset_family="broad_alignment",
        loss_family="caption",
        license_key="lcs558k",
        teacher_kl_weight=1.0,
    )
    _json_dump(output_path, tagged)
    return {
        "name": "lcs558k_caption",
        "data_path": output_path,
        "input_path": input_path,
        "image_dir": image_dir,
        "rows": len(tagged),
        "target_rows": int(max_samples),
        "available_rows_with_cached_images": len(available),
        "source_rows": len(rows),
        "sample_seed": int(seed),
        "dataset_family": "broad_alignment",
        "loss_family": "caption",
        "teacher_kl_weight": 1.0,
        "image_cache": image_report,
        "provenance_verified_against": "blip_laion_cc_sbu_558k.json",
        "hf_dataset": source_file_report,
        "hf_dataset_revision": LCS_HF_REVISION,
        **LICENSES["lcs558k"],
    }


VG_BASE_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
VG_ANNOTATION_FILES = [
    "region_descriptions.json.zip",
    "attributes.json.zip",
    "relationships.json.zip",
    "image_data.json.zip",
]


def _ensure_vg_annotations(root: str = "/checkpoints/visual_genome") -> dict[str, Any]:
    os.makedirs(root, exist_ok=True)
    reports = []
    for filename in VG_ANNOTATION_FILES:
        zip_path = os.path.join(root, filename)
        reports.append(_download_file(f"{VG_BASE_URL}/{filename}", zip_path))
        json_name = filename[:-4]
        json_path = os.path.join(root, json_name)
        if not os.path.exists(json_path):
            print(f"Extracting {zip_path}", flush=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                matches = [name for name in zf.namelist() if os.path.basename(name) == json_name]
                if not matches:
                    raise RuntimeError(f"Could not find {json_name} inside {zip_path}")
                with zf.open(matches[0]) as source, open(json_path + ".tmp", "wb") as dest:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        if not chunk:
                            break
                        dest.write(chunk)
                os.replace(json_path + ".tmp", json_path)
        reports[-1]["json_path"] = json_path
        reports[-1]["json_bytes"] = os.path.getsize(json_path)
    return {"root": root, "files": reports}


def _vg_image_name(image_id: Any, image_dir: str) -> str | None:
    try:
        numeric = int(image_id)
    except Exception:
        return None
    for filename in (f"VG_{numeric}.jpg", f"{numeric}.jpg"):
        if os.path.exists(os.path.join(image_dir, filename)):
            return filename
    return None


def _load_eval_hash_blocklist() -> dict[str, Any]:
    eval_paths: set[str] = set()
    eval_missing: dict[str, list[dict[str, Any]]] = {}
    for artifact in EVAL_ARTIFACTS:
        paths, missing = _resolve_paths(artifact["path"], artifact["image_dir"])
        eval_paths.update(paths)
        eval_missing[artifact["name"]] = missing
    nonempty_missing = {name: values for name, values in eval_missing.items() if values}
    if nonempty_missing:
        raise RuntimeError(
            "Cannot build Visual Genome eval-overlap blocklist because eval image "
            f"refs failed to resolve: {nonempty_missing}"
        )
    eval_hashes = _hash_paths(eval_paths, "eval:all-for-vg-prefilter")
    return {
        "hashes": set(eval_hashes),
        "unique_paths": len(eval_paths),
        "unique_hashes": len(eval_hashes),
        "missing_examples": eval_missing,
    }


def _vg_image_allowed(
    *,
    filename: str,
    image_dir: str,
    blocked_eval_hashes: set[str],
    digest_cache: dict[str, str],
) -> tuple[bool, str | None]:
    path = os.path.join(image_dir, filename)
    digest = digest_cache.get(path)
    if digest is None:
        digest = _hash_path(path)
        digest_cache[path] = digest
    return digest not in blocked_eval_hashes, digest


def _first_name(value: Any) -> str:
    if isinstance(value, list) and value:
        return _normal_text(value[0])
    return _normal_text(value)


def build_visual_genome(max_regions: int = 50000, max_attributes: int = 25000, max_relationships: int = 25000) -> dict[str, Any]:
    root = "/checkpoints/visual_genome"
    image_dir = "/checkpoints/gqa_images_hf"
    annotation_report = _ensure_vg_annotations(root)
    image_data = _load_json(os.path.join(root, "image_data.json"))
    cached_image_filenames: dict[str, str] = {}
    for row in image_data:
        image_id = row.get("image_id", row.get("id"))
        filename = _vg_image_name(image_id, image_dir)
        if filename:
            cached_image_filenames[str(int(image_id))] = filename
    if not cached_image_filenames:
        raise RuntimeError(
            "Visual Genome could not reuse any cached images from /checkpoints/gqa_images_hf"
        )

    eval_blocklist = _load_eval_hash_blocklist()
    digest_cache: dict[str, str] = {}
    allowed_cache: dict[str, tuple[bool, str | None]] = {}
    blocked_cached_images = set()

    def image_for(image_id: Any) -> str | None:
        try:
            key = str(int(image_id))
        except Exception:
            return None
        filename = cached_image_filenames.get(key)
        if not filename:
            return None
        if key not in allowed_cache:
            allowed, digest = _vg_image_allowed(
                filename=filename,
                image_dir=image_dir,
                blocked_eval_hashes=eval_blocklist["hashes"],
                digest_cache=digest_cache,
            )
            allowed_cache[key] = (allowed, digest)
            if not allowed:
                blocked_cached_images.add(key)
        allowed, _digest = allowed_cache[key]
        if not allowed:
            return None
        return filename

    def region_rows() -> tuple[list[dict[str, Any]], int]:
        rng = random.Random(1807)
        samples: list[dict[str, Any]] = []
        seen = 0
        payload = _load_json(os.path.join(root, "region_descriptions.json"))
        rng.shuffle(payload)
        for image_entry_index, image_entry in enumerate(payload, start=1):
            regions = list(image_entry.get("regions") or [])
            rng.shuffle(regions)
            for region in regions:
                if len(samples) >= int(max_regions):
                    break
                phrase = _normal_text(region.get("phrase"))
                image_id = region.get("image_id", image_entry.get("image_id", image_entry.get("id")))
                filename = image_for(image_id)
                if not phrase or not filename:
                    continue
                try:
                    x = int(float(region.get("x", 0)))
                    y = int(float(region.get("y", 0)))
                    w = int(float(region.get("width", region.get("w", 0))))
                    h = int(float(region.get("height", region.get("h", 0))))
                except Exception:
                    x = y = w = h = 0
                seen += 1
                samples.append(
                    {
                        "id": f"vg_region_{region.get('region_id', seen)}",
                        "image": filename,
                        "source_dataset": "Visual Genome",
                        "source_split": "train",
                        "source_index": int(seen),
                        "source_image_id": str(image_id),
                        "source_region_id": str(region.get("region_id", seen)),
                        "vg_subsource": "regions",
                        "region": {"x": x, "y": y, "width": w, "height": h},
                        "conversations": [
                            {
                                "from": "human",
                                "value": (
                                    "<image>\nDescribe the region centered at "
                                    f"({x + w // 2}, {y + h // 2}) with size ({w}, {h})."
                                ),
                            },
                            {"from": "gpt", "value": phrase},
                        ],
                    }
                )
            if len(samples) >= int(max_regions):
                break
            if image_entry_index % 10000 == 0:
                print(
                    f"VG regions scan: images={image_entry_index} candidates={seen} samples={len(samples)}",
                    flush=True,
                )
        print(f"VG regions scan done: candidates={seen} samples={len(samples)}", flush=True)
        return samples, seen

    def attribute_rows() -> tuple[list[dict[str, Any]], int]:
        rng = random.Random(1808)
        samples: list[dict[str, Any]] = []
        seen = 0
        payload = _load_json(os.path.join(root, "attributes.json"))
        rng.shuffle(payload)
        for image_entry_index, image_entry in enumerate(payload, start=1):
            objects = list(image_entry.get("attributes") or [])
            rng.shuffle(objects)
            for obj in objects:
                if len(samples) >= int(max_attributes):
                    break
                attrs = [_normal_text(value) for value in obj.get("attributes") or []]
                attrs = [value for value in attrs if value]
                name = _first_name(obj.get("names") or obj.get("name") or "object")
                image_id = obj.get("image_id", image_entry.get("image_id", image_entry.get("id")))
                filename = image_for(image_id)
                if not attrs or not name or not filename:
                    continue
                seen += 1
                samples.append(
                    {
                        "id": f"vg_attribute_{obj.get('object_id', seen)}",
                        "image": filename,
                        "source_dataset": "Visual Genome",
                        "source_split": "train",
                        "source_index": int(seen),
                        "source_image_id": str(image_id),
                        "source_object_id": str(obj.get("object_id", seen)),
                        "vg_subsource": "attributes",
                        "object_name": name,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\nWhat attributes does the {name} have?",
                            },
                            {"from": "gpt", "value": ", ".join(attrs)},
                        ],
                    }
                )
            if len(samples) >= int(max_attributes):
                break
            if image_entry_index % 10000 == 0:
                print(
                    f"VG attributes scan: images={image_entry_index} candidates={seen} samples={len(samples)}",
                    flush=True,
                )
        print(f"VG attributes scan done: candidates={seen} samples={len(samples)}", flush=True)
        return samples, seen

    def relationship_rows() -> tuple[list[dict[str, Any]], int]:
        rng = random.Random(1809)
        samples: list[dict[str, Any]] = []
        seen = 0
        payload = _load_json(os.path.join(root, "relationships.json"))
        rng.shuffle(payload)
        for image_entry_index, image_entry in enumerate(payload, start=1):
            relationships = list(image_entry.get("relationships") or [])
            rng.shuffle(relationships)
            for rel in relationships:
                if len(samples) >= int(max_relationships):
                    break
                predicate = _normal_answer(rel.get("predicate")).lower()
                subject = rel.get("subject") or {}
                obj = rel.get("object") or {}
                subject_name = _first_name(subject.get("names") or subject.get("name") or "subject")
                object_name = _first_name(obj.get("names") or obj.get("name") or "object")
                image_id = rel.get("image_id", image_entry.get("image_id", image_entry.get("id")))
                filename = image_for(image_id)
                if not predicate or not subject_name or not object_name or not filename:
                    continue
                seen += 1
                samples.append(
                    {
                        "id": f"vg_relationship_{rel.get('relationship_id', seen)}",
                        "image": filename,
                        "source_dataset": "Visual Genome",
                        "source_split": "train",
                        "source_index": int(seen),
                        "source_image_id": str(image_id),
                        "source_relationship_id": str(rel.get("relationship_id", seen)),
                        "vg_subsource": "relationships",
                        "subject_name": subject_name,
                        "object_name": object_name,
                        "conversations": [
                            {
                                "from": "human",
                                "value": (
                                    "<image>\nWhat is the relationship between "
                                    f"{subject_name} and {object_name}?"
                                ),
                            },
                            {"from": "gpt", "value": predicate},
                        ],
                    }
                )
            if len(samples) >= int(max_relationships):
                break
            if image_entry_index % 10000 == 0:
                print(
                    f"VG relationships scan: images={image_entry_index} candidates={seen} samples={len(samples)}",
                    flush=True,
                )
        print(
            f"VG relationships scan done: candidates={seen} samples={len(samples)}",
            flush=True,
        )
        return samples, seen

    regions, region_candidates = region_rows()
    attributes, attribute_candidates = attribute_rows()
    relationships, relationship_candidates = relationship_rows()
    if len(regions) < int(max_regions) or len(attributes) < int(max_attributes) or len(relationships) < int(max_relationships):
        raise RuntimeError(
            "Visual Genome had insufficient cached, eval-clean rows: "
            f"regions={len(regions)}/{max_regions}, "
            f"attributes={len(attributes)}/{max_attributes}, "
            f"relationships={len(relationships)}/{max_relationships}"
        )

    outputs = {
        "regions": f"{V18_DATA}/vg_regions_seed1807_n{max_regions}.json",
        "attributes": f"{V18_DATA}/vg_attributes_seed1808_n{max_attributes}.json",
        "relationships": f"{V18_DATA}/vg_relationships_seed1809_n{max_relationships}.json",
        "combined": f"{V18_DATA}/vg_regions_attrs_rels_seed1807_1808_1809_n{max_regions + max_attributes + max_relationships}.json",
    }
    tagged_regions = _tag_rows(
        regions,
        source_name="vg_regions",
        dataset_family="compositional_grounding",
        loss_family="vg_region_description",
        license_key="visual_genome",
        teacher_kl_weight=0.0,
    )
    tagged_attributes = _tag_rows(
        attributes,
        source_name="vg_attributes",
        dataset_family="compositional_grounding",
        loss_family="vg_attribute",
        license_key="visual_genome",
        teacher_kl_weight=0.0,
    )
    tagged_relationships = _tag_rows(
        relationships,
        source_name="vg_relationships",
        dataset_family="compositional_grounding",
        loss_family="vg_relationship",
        license_key="visual_genome",
        teacher_kl_weight=0.0,
    )
    combined = [*tagged_regions, *tagged_attributes, *tagged_relationships]
    _json_dump(outputs["regions"], tagged_regions)
    _json_dump(outputs["attributes"], tagged_attributes)
    _json_dump(outputs["relationships"], tagged_relationships)
    _json_dump(outputs["combined"], combined)
    manifest_source = {
        "name": "vg_regions_attrs_rels",
        "data_path": outputs["combined"],
        "image_dir": image_dir,
        "rows": len(combined),
        "target_rows": int(max_regions + max_attributes + max_relationships),
        "component_paths": outputs,
        "component_rows": {
            "regions": len(tagged_regions),
            "attributes": len(tagged_attributes),
            "relationships": len(tagged_relationships),
        },
        "component_candidate_rows_after_cache_and_eval_filter": {
            "regions": int(region_candidates),
            "attributes": int(attribute_candidates),
            "relationships": int(relationship_candidates),
        },
        "sample_seeds": {
            "regions": 1807,
            "attributes": 1808,
            "relationships": 1809,
        },
        "dataset_family": "compositional_grounding",
        "loss_family": "visual_genome_grounding",
        "teacher_kl_weight": 0.0,
        "annotation_report": annotation_report,
        "image_cache_reuse": {
            "image_dir": image_dir,
            "image_data_rows": len(image_data),
            "cached_vg_images": len(cached_image_filenames),
            "eval_blocked_cached_images": len(blocked_cached_images),
            "eval_unique_paths": eval_blocklist["unique_paths"],
            "eval_unique_hashes": eval_blocklist["unique_hashes"],
            "eval_missing_examples": eval_blocklist["missing_examples"],
        },
        **LICENSES["visual_genome"],
    }
    return manifest_source


def build_existing_sources() -> list[dict[str, Any]]:
    os.makedirs(V18_DATA, exist_ok=True)
    sources = [
        build_coco_captions(max_samples=100000, seed=1802),
        {
            **_write_sampled_source(
                source_name="vqav2_train_broad",
                input_path="/checkpoints/vqa_data/vqa_train2014_direct_150000.json",
                output_path=f"{V18_DATA}/vqav2_train2014_direct_seed1802_n150000.json",
                max_samples=150000,
                seed=1802,
                dataset_family="broad_vqa",
                loss_family="direct_answer",
                license_key="vqav2",
                teacher_kl_weight=1.0,
            ),
            "image_dir": "/checkpoints/coco_train2014_vqa",
        },
        build_gqa_broad(max_samples=200000, seed=1803),
        {
            **_write_sampled_source(
                source_name="textvqa_majority",
                input_path="/checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json",
                output_path=f"{V18_DATA}/textvqa_train_majority_seed1502_n20000.json",
                max_samples=20000,
                seed=1502,
                dataset_family="capability",
                loss_family="textvqa",
                license_key="textvqa",
                teacher_kl_weight=0.0,
            ),
            "image_dir": "/checkpoints/textvqa_images_hf",
        },
        {
            **_write_sampled_source(
                source_name="gqa_spatial_metadata",
                input_path="/checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json",
                output_path=f"{V18_DATA}/gqa_spatial_metadata_seed1503_n15000.json",
                max_samples=15000,
                seed=1503,
                dataset_family="compositional_grounding",
                loss_family="gqa_spatial_relation",
                license_key="gqa",
                teacher_kl_weight=0.0,
            ),
            "image_dir": "/checkpoints/gqa_images_hf",
        },
    ]
    return sources


def build_hf_sources() -> list[dict[str, Any]]:
    os.makedirs(V18_DATA, exist_ok=True)
    sources = [
        build_aokvqa(max_samples=17000, seed=1804),
        build_vsr(max_samples=10000, seed=1805),
        build_ai2d(max_samples=5000, seed=1806),
        build_ocrvqa(max_samples=50000, seed=1807),
    ]
    return sources


def build_named_hf_source(name: str) -> dict[str, Any]:
    key = str(name).strip().lower()
    if key == "aokvqa":
        return build_aokvqa(max_samples=17000, seed=1804)
    if key == "okvqa":
        return build_okvqa(max_samples=9000, seed=1804)
    if key == "vsr":
        return build_vsr(max_samples=10000, seed=1805)
    if key == "ai2d":
        return build_ai2d(max_samples=5000, seed=1806)
    if key == "ocrvqa":
        return build_ocrvqa(max_samples=50000, seed=1807)
    raise ValueError(f"Unsupported HF source: {name}")


def write_manifest(sources: list[dict[str, Any]], path: str) -> dict[str, Any]:
    image_dirs = {}
    for image_dir in sorted({str(source.get("image_dir") or "") for source in sources if source.get("image_dir")}):
        if "llava_pretrain" in image_dir:
            image_dirs[image_dir] = _count_images_recursive(image_dir)
        else:
            image_dirs[image_dir] = _count_images(image_dir)
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": (
            "V18 supplement: include user-approved LCS-558k license posture and "
            "Visual Genome Stanford direct download; continue excluding GPL/NC "
            "and still-unverified sources."
        ),
        "sources": sources,
        "total_rows": sum(int(source.get("rows") or 0) for source in sources),
        "image_dirs": image_dirs,
    }
    _json_dump(path, manifest)
    return manifest


FINAL_WEIGHTS = {
    "lcs558k_caption": 30.0,
    "coco_captions": 20.0,
    "vqav2_train_broad": 12.0,
    "gqa_train_balanced_broad": 17.0,
    "aokvqa": 3.0,
    "okvqa": 2.0,
    "vsr": 1.0,
    "ocrvqa": 6.0,
    "textvqa_majority": 3.0,
    "ai2d": 1.0,
    "gqa_spatial_metadata": 3.0,
    "vg_regions_attrs_rels": 2.0,
}


def _weights_for_sources(sources: list[dict[str, Any]]) -> dict[str, float]:
    names = {str(source.get("name") or "") for source in sources}
    return {name: float(weight) for name, weight in FINAL_WEIGHTS.items() if name in names}


def _weights_payload(sources: list[dict[str, Any]], created_at_utc: str) -> dict[str, Any]:
    weights = _weights_for_sources(sources)
    family_weights: dict[str, float] = {}
    by_name = {str(source.get("name") or ""): source for source in sources}
    for name, weight in weights.items():
        family = str(by_name.get(name, {}).get("dataset_family") or "unknown")
        family_weights[family] = family_weights.get(family, 0.0) + float(weight)
    return {
        "created_at_utc": created_at_utc,
        "strategy": "weighted",
        "epoch_length": 384000,
        "weighted_index_mode": "hash",
        "weights": weights,
        "family_weights": family_weights,
        "total_weight": sum(weights.values()),
        "notes": (
            "Supplement weights rebalance V18 to broad_alignment=50, "
            "broad_vqa=35, capability=10, compositional_grounding=5 after "
            "adding LCS-558k and Visual Genome."
        ),
    }


def _source_license_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    license_counts: dict[str, int] = {}
    commercial_values = []
    entries = []
    for source in sources:
        license_name = str(source.get("license") or "unknown")
        license_counts[license_name] = license_counts.get(license_name, 0) + 1
        if source.get("commercial_use_allowed") is not None:
            commercial_values.append(bool(source.get("commercial_use_allowed")))
        entries.append(
            {
                "name": source.get("name"),
                "license": license_name,
                "license_source": source.get("license_source"),
                "license_note": source.get("license_note"),
                "commercial_use_allowed": source.get("commercial_use_allowed"),
                "rows": source.get("rows"),
                "dataset_family": source.get("dataset_family"),
                "loss_family": source.get("loss_family"),
            }
        )
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": entries,
        "license_counts": license_counts,
        "aggregate_commercial_use_allowed": all(commercial_values) if commercial_values else None,
        "posture_note": (
            "aggregate_commercial_use_allowed=true with LLaVA-Pretrain caption "
            "license noted when lcs558k_caption is present"
        ),
        "excluded_by_policy": [
            "ChartQA GPL-3.0",
            "ShareGPT4V CC-BY-NC-4.0",
            "LLaVA-Mix-665k NC-mixed",
            "RefCOCO family (unverified HF license tags)",
            "VizWiz/NLVR2 requested mirrors unavailable",
            "DocVQA train unavailable in selected public HF mirror",
        ],
    }


EVAL_ARTIFACTS = [
    {
        "name": "vqa_wrong_image_n3000_seed42",
        "path": "/checkpoints/v17_fixed_harness/v11/vqa_wrong_image_n3000_seed42.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "vqa_shuffled_n3000_seed42",
        "path": "/checkpoints/v17_fixed_harness/v11/vqa_shuffled_n3000_seed42.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "vqa_blank_n3000_seed42",
        "path": "/checkpoints/v17_fixed_harness/v11/vqa_blank_n3000_seed42.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "vqa_clean_n3000_seed42",
        "path": "/checkpoints/v17_fixed_harness/v11/vqa_clean_n3000_seed42.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "pope_popular_n1000",
        "path": "/checkpoints/v17_fixed_harness/v11/pope_popular_n1000.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "pope_adversarial_n1000",
        "path": "/checkpoints/v17_fixed_harness/v11/pope_adversarial_n1000.json",
        "image_dir": "/checkpoints/coco_val2014",
    },
    {
        "name": "textvqa_validation_full",
        "path": "/checkpoints/v17_fixed_harness/v11/textvqa_validation_full.json",
        "image_dir": "/checkpoints/textvqa_images_hf",
    },
    {
        "name": "chartqa_val_full",
        "path": "/checkpoints/v17_fixed_harness/v11/chartqa_val_full.json",
        "image_dir": "/checkpoints/chartqa_images_hf",
    },
    {
        "name": "gqa_confirm_n3000_offset1000",
        "path": "/checkpoints/v17_fixed_harness/v11/gqa_confirm_n3000_offset1000.json",
        "image_dir": "/checkpoints/gqa_images_hf",
    },
    {
        "name": "gqa_search_n1000",
        "path": "/checkpoints/v17_fixed_harness/v11/gqa_search_n1000.json",
        "image_dir": "/checkpoints/gqa_images_hf",
    },
]


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=10 * 60)
def combine_manifests_remote() -> dict[str, Any]:
    manifest_paths = [
        f"{V18_ROOT}/existing_sources_manifest.json",
        f"{V18_ROOT}/hf_source_aokvqa_manifest.json",
        f"{V18_ROOT}/hf_source_okvqa_manifest.json",
        f"{V18_ROOT}/hf_source_vsr_manifest.json",
        f"{V18_ROOT}/hf_source_ai2d_manifest.json",
        f"{V18_ROOT}/hf_source_ocrvqa_manifest.json",
        f"{V18_ROOT}/supplement_lcs558k_manifest.json",
        f"{V18_ROOT}/supplement_visual_genome_manifest.json",
    ]
    sources: list[dict[str, Any]] = []
    for path in manifest_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cannot combine V18 manifest because required source manifest is missing: {path}"
            )
        payload = _load_json(path)
        sources.extend(payload.get("sources") or [])
    by_name = {str(source["name"]): source for source in sources}
    missing = [name for name in FINAL_WEIGHTS if name not in by_name]
    if missing:
        raise RuntimeError(f"Cannot combine V18 manifest; missing sources: {missing}")
    ordered_sources = [by_name[name] for name in FINAL_WEIGHTS]
    for source in ordered_sources:
        source["weight"] = FINAL_WEIGHTS[str(source["name"])]
    manifest = write_manifest(ordered_sources, f"{V18_ROOT}/source_manifest.json")
    weights = _weights_payload(ordered_sources, manifest["created_at_utc"])
    license_summary = _source_license_summary(ordered_sources)
    _json_dump(f"{V18_ROOT}/final_mixture_weights.json", weights)
    _json_dump(f"{V18_ROOT}/mixture_license_summary.json", license_summary)
    volume.commit()
    result = {
        "manifest": manifest,
        "final_mixture_weights": weights,
        "mixture_license_summary": license_summary,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=6 * 60 * 60)
def audit_v18_hashes_remote() -> dict[str, Any]:
    manifest = _load_json(f"{V18_ROOT}/source_manifest.json")
    os.makedirs(f"{V18_ROOT}/audits", exist_ok=True)
    eval_indexes = []
    for artifact in EVAL_ARTIFACTS:
        paths, missing = _resolve_paths(artifact["path"], artifact["image_dir"])
        hashes = _hash_paths(paths, f"eval:{artifact['name']}")
        eval_indexes.append({**artifact, "paths": paths, "hashes": hashes, "missing_examples": missing})

    source_reports = []
    pair_reports = []
    for source in manifest.get("sources") or []:
        paths, missing = _resolve_paths(source["data_path"], source["image_dir"])
        train_hashes = _hash_paths(paths, f"train:{source['name']}")
        source_reports.append(
            {
                "name": source["name"],
                "data_path": source["data_path"],
                "image_dir": source["image_dir"],
                "rows": source.get("rows"),
                "resolved_unique_paths": len(paths),
                "unique_hashes": len(train_hashes),
                "missing_examples": missing,
            }
        )
        train_digest_set = set(train_hashes)
        for eval_index in eval_indexes:
            overlap = sorted(train_digest_set & set(eval_index["hashes"]))
            pair_reports.append(
                {
                    "train_source": source["name"],
                    "eval_artifact": eval_index["name"],
                    "overlap_count": len(overlap),
                    "overlap_examples": [
                        {
                            "sha256_first_mib": digest,
                            "train_paths": train_hashes[digest][:5],
                            "eval_paths": eval_index["hashes"][digest][:5],
                        }
                        for digest in overlap[:20]
                    ],
                }
            )
    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": source_reports,
        "eval_artifacts": [
            {
                "name": item["name"],
                "path": item["path"],
                "image_dir": item["image_dir"],
                "resolved_unique_paths": len(item["paths"]),
                "unique_hashes": len(item["hashes"]),
                "missing_examples": item["missing_examples"],
            }
            for item in eval_indexes
        ],
        "pairs": pair_reports,
        "passed": all(pair["overlap_count"] == 0 for pair in pair_reports),
    }
    _json_dump(f"{V18_ROOT}/audits/all_sources_vs_v17_v11_evals.json", result)
    for source in source_reports:
        source_pairs = [pair for pair in pair_reports if pair["train_source"] == source["name"]]
        _json_dump(
            f"{V18_ROOT}/audits/{source['name']}_vs_v17_v11_evals.json",
            {
                "created_at_utc": result["created_at_utc"],
                "source": source,
                "pairs": source_pairs,
                "passed": all(pair["overlap_count"] == 0 for pair in source_pairs),
            },
        )
    volume.commit()
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def _audit_paths_for_source(source: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    name = str(source.get("name") or "")
    image_dir = str(source.get("image_dir") or "")
    if name == "lcs558k_caption":
        payload = _load_json(str(source["data_path"]))
        paths: set[str] = set()
        missing = []
        refs = sorted({str(row.get("image") or "").strip() for row in _iter_records(payload) if row.get("image")})
        for ref in refs:
            path = ref if os.path.isabs(ref) else os.path.join(image_dir, ref)
            if os.path.exists(path) and os.path.isfile(path):
                paths.add(path)
            elif len(missing) < 20:
                missing.append({"ref": ref, "image_dir": image_dir, "json_path": source["data_path"]})
        expected = int(source.get("rows") or 0)
        if expected and len(refs) != expected and len(missing) < 20:
            missing.append(
                {
                    "message": "LCS row count does not match unique image refs",
                    "expected_rows": expected,
                    "unique_image_refs": len(refs),
                    "image_dir": image_dir,
                }
            )
        return paths, missing
    return _resolve_paths(str(source["data_path"]), image_dir)


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=6 * 60 * 60, memory=32768)
def audit_supplement_hashes_remote() -> dict[str, Any]:
    os.makedirs(f"{V18_ROOT}/audits", exist_ok=True)
    supplement_paths = [
        f"{V18_ROOT}/supplement_lcs558k_manifest.json",
        f"{V18_ROOT}/supplement_visual_genome_manifest.json",
    ]
    sources: list[dict[str, Any]] = []
    for path in supplement_paths:
        payload = _load_json(path)
        sources.extend(source for source in payload.get("sources") or [] if isinstance(source, dict))
    if len(sources) != 2:
        raise RuntimeError(f"Expected 2 supplement sources to audit, found {len(sources)}")

    eval_indexes = []
    for artifact in EVAL_ARTIFACTS:
        paths, missing = _resolve_paths(artifact["path"], artifact["image_dir"])
        hashes = _hash_paths(paths, f"eval:{artifact['name']}")
        eval_indexes.append({**artifact, "paths": paths, "hashes": hashes, "missing_examples": missing})

    source_reports = []
    pair_reports = []
    for source in sources:
        paths, missing = _audit_paths_for_source(source)
        train_hashes = _hash_paths(paths, f"train:{source['name']}")
        source_reports.append(
            {
                "name": source["name"],
                "data_path": source["data_path"],
                "image_dir": source["image_dir"],
                "rows": source.get("rows"),
                "resolved_unique_paths": len(paths),
                "unique_hashes": len(train_hashes),
                "missing_examples": missing,
            }
        )
        train_digest_set = set(train_hashes)
        for eval_index in eval_indexes:
            overlap = sorted(train_digest_set & set(eval_index["hashes"]))
            pair_reports.append(
                {
                    "train_source": source["name"],
                    "eval_artifact": eval_index["name"],
                    "overlap_count": len(overlap),
                    "overlap_examples": [
                        {
                            "sha256_first_mib": digest,
                            "train_paths": train_hashes[digest][:5],
                            "eval_paths": eval_index["hashes"][digest][:5],
                        }
                        for digest in overlap[:20]
                    ],
                }
            )

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": source_reports,
        "eval_artifacts": [
            {
                "name": item["name"],
                "path": item["path"],
                "image_dir": item["image_dir"],
                "resolved_unique_paths": len(item["paths"]),
                "unique_hashes": len(item["hashes"]),
                "missing_examples": item["missing_examples"],
            }
            for item in eval_indexes
        ],
        "pairs": pair_reports,
        "passed": (
            all(pair["overlap_count"] == 0 for pair in pair_reports)
            and all(not source["missing_examples"] for source in source_reports)
            and all(not item["missing_examples"] for item in eval_indexes)
        ),
    }
    _json_dump(f"{V18_ROOT}/audits/supplement_lcs_vg_vs_v17_v11_evals.json", result)
    for source in source_reports:
        source_pairs = [pair for pair in pair_reports if pair["train_source"] == source["name"]]
        _json_dump(
            f"{V18_ROOT}/audits/{source['name']}_vs_v17_v11_evals.json",
            {
                "created_at_utc": result["created_at_utc"],
                "source": source,
                "pairs": source_pairs,
                "passed": (
                    all(pair["overlap_count"] == 0 for pair in source_pairs)
                    and not source["missing_examples"]
                    and all(not item["missing_examples"] for item in eval_indexes)
                ),
            },
        )
    volume.commit()
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def _load_manifest_sources(path: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list):
        raise ValueError(f"Expected manifest with sources list at {path}")
    return [source for source in sources if isinstance(source, dict)]


def _write_manifest_and_license_artifacts(sources: list[dict[str, Any]], manifest_path: str) -> dict[str, Any]:
    manifest = write_manifest(sources, manifest_path)
    if manifest_path == f"{V18_ROOT}/source_manifest.json":
        _json_dump(f"{V18_ROOT}/final_mixture_weights.json", _weights_payload(sources, manifest["created_at_utc"]))
        _json_dump(f"{V18_ROOT}/mixture_license_summary.json", _source_license_summary(sources))
    return manifest


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=2 * 60 * 60)
def filter_gqa_eval_overlaps_remote() -> dict[str, Any]:
    """Rewrite GQA V18 sources without images that hash-match V17 eval images."""

    os.makedirs(f"{V18_ROOT}/audits", exist_ok=True)
    eval_paths: set[str] = set()
    eval_missing: dict[str, list[dict[str, Any]]] = {}
    for artifact in EVAL_ARTIFACTS:
        paths, missing = _resolve_paths(artifact["path"], artifact["image_dir"])
        eval_paths.update(paths)
        eval_missing[artifact["name"]] = missing
    eval_hashes = _hash_paths(eval_paths, "eval:all-for-gqa-cleanup")
    eval_digest_set = set(eval_hashes)

    manifest_path = f"{V18_ROOT}/source_manifest.json"
    sources = _load_manifest_sources(manifest_path)
    replacements: dict[str, dict[str, Any]] = {}
    cleanup_reports = []
    target_names = {"gqa_train_balanced_broad", "gqa_spatial_metadata"}

    for source in sources:
        name = str(source.get("name") or "")
        if name not in target_names:
            continue
        rows = _load_json(str(source["data_path"]))
        if not isinstance(rows, list):
            raise ValueError(f"Expected list JSON for {name}: {source['data_path']}")

        candidate_paths_by_index: list[str | None] = []
        candidate_paths: set[str] = set()
        for row in rows:
            candidate_path = _primary_image_candidate_path(row, str(source["image_dir"]))
            candidate_paths_by_index.append(candidate_path)
            if candidate_path:
                candidate_paths.add(candidate_path)
        source_paths = {
            path for path in candidate_paths if os.path.exists(path) and os.path.isfile(path)
        }
        row_paths_by_index = [
            {path} if path and path in source_paths else set()
            for path in candidate_paths_by_index
        ]
        source_missing = []
        for row, path in zip(rows, candidate_paths_by_index):
            if path and path in source_paths:
                continue
            if len(source_missing) >= 20:
                break
            source_missing.append(
                {
                    "id": row.get("id"),
                    "image": row.get("image"),
                    "source_image_id": row.get("source_image_id"),
                    "candidate_path": path,
                    "image_dir": source["image_dir"],
                }
            )
        train_hashes = _hash_paths(source_paths, f"train:{name}:gqa-cleanup")
        overlap_digests = sorted(set(train_hashes) & eval_digest_set)
        overlap_paths = {
            path
            for digest in overlap_digests
            for path in train_hashes.get(digest, [])
        }
        path_to_digest = {
            path: digest
            for digest, digest_paths in train_hashes.items()
            for path in digest_paths
        }

        kept_rows = []
        removed_rows = []
        for row, row_paths in zip(rows, row_paths_by_index):
            row_overlap_paths = sorted(row_paths & overlap_paths)
            if row_overlap_paths:
                removed_rows.append(
                    {
                        "id": row.get("id"),
                        "source_question_id": row.get("source_question_id"),
                        "source_image_id": row.get("source_image_id"),
                        "image": row.get("image"),
                        "overlap_paths": row_overlap_paths,
                        "overlap_digests": sorted(
                            {path_to_digest[path] for path in row_overlap_paths if path in path_to_digest}
                        ),
                    }
                )
            else:
                kept_rows.append(row)

        output_stem = Path(str(source["data_path"])).stem
        output_path = f"{V18_DATA}/{output_stem}_leakclean_v17evals.json"
        _json_dump(output_path, kept_rows)
        report = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": name,
            "input_path": source["data_path"],
            "output_path": output_path,
            "input_rows": len(rows),
            "output_rows": len(kept_rows),
            "removed_rows": len(removed_rows),
            "source_unique_paths": len(source_paths),
            "source_unique_hashes": len(train_hashes),
            "overlap_unique_hashes": len(overlap_digests),
            "overlap_unique_train_paths": len(overlap_paths),
            "source_missing_examples": source_missing,
            "eval_missing_examples": eval_missing,
            "removed_examples": removed_rows[:200],
        }
        _json_dump(f"{V18_ROOT}/audits/{name}_leakclean_filter_report.json", report)
        cleanup_reports.append(report)
        replacements[name] = {
            **source,
            "data_path": output_path,
            "rows": len(kept_rows),
            "leak_cleaned": True,
            "leak_clean_input_path": source["data_path"],
            "leak_clean_removed_rows": len(removed_rows),
            "leak_clean_overlap_unique_hashes": len(overlap_digests),
            "leak_clean_filter_report": f"{V18_ROOT}/audits/{name}_leakclean_filter_report.json",
        }

    if not replacements:
        raise RuntimeError("No GQA V18 sources found in source_manifest.json")

    def replaced(source: dict[str, Any]) -> dict[str, Any]:
        name = str(source.get("name") or "")
        return replacements.get(name, source)

    updated_manifest = _write_manifest_and_license_artifacts(
        [replaced(source) for source in sources],
        manifest_path,
    )

    existing_manifest_path = f"{V18_ROOT}/existing_sources_manifest.json"
    if os.path.exists(existing_manifest_path):
        existing_sources = _load_manifest_sources(existing_manifest_path)
        _write_manifest_and_license_artifacts(
            [replaced(source) for source in existing_sources],
            existing_manifest_path,
        )

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_unique_paths": len(eval_paths),
        "eval_unique_hashes": len(eval_hashes),
        "cleanup_reports": cleanup_reports,
        "manifest": updated_manifest,
    }
    _json_dump(f"{V18_ROOT}/audits/gqa_leakclean_filter_summary.json", result)
    volume.commit()
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


HF_PROBE_SOURCES = [
    {"name": "gqa", "dataset_id": "Mineru/GQA"},
    {"name": "aokvqa", "dataset_id": "HuggingFaceM4/A-OKVQA"},
    {"name": "okvqa", "dataset_id": "Multimodal-Fatima/OK-VQA_train"},
    {"name": "vizwiz", "dataset_id": "lmms-lab/vizwiz_vqa"},
    {"name": "vsr", "dataset_id": "cambridgeltl/vsr_zeroshot"},
    {"name": "nlvr2", "dataset_id": "lil-lab/nlvr2"},
    {"name": "ocrvqa", "dataset_id": "howard-hou/OCR-VQA"},
    {"name": "textvqa", "dataset_id": "lmms-lab/textvqa"},
    {"name": "ai2d_lmms", "dataset_id": "lmms-lab/ai2d"},
    {"name": "ai2d_lime", "dataset_id": "LIME-DATA/ai2d"},
    {"name": "docvqa", "dataset_id": "lmms-lab/DocVQA"},
]


def _short_value(value: Any) -> Any:
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {"type": "PIL.Image", "size": list(value.size), "mode": value.mode}
    if isinstance(value, dict):
        return {str(k): _short_value(v) for k, v in list(value.items())[:8] if k != "bytes"}
    if isinstance(value, list):
        return [_short_value(v) for v in value[:4]]
    text = str(value)
    return text[:200] + ("..." if len(text) > 200 else "")


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=45 * 60,
)
def probe_hf_sources_remote() -> dict[str, Any]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    reports = []
    for spec in HF_PROBE_SOURCES:
        dataset_id = spec["dataset_id"]
        report: dict[str, Any] = {"name": spec["name"], "dataset_id": dataset_id}
        try:
            info = api.dataset_info(dataset_id)
            card = info.cardData or {}
            if hasattr(card, "to_dict"):
                card = card.to_dict()
            report.update(
                {
                    "sha": getattr(info, "sha", None),
                    "gated": getattr(info, "gated", None),
                    "private": getattr(info, "private", None),
                    "tags": list(getattr(info, "tags", None) or [])[:30],
                    "card_license": card.get("license") if isinstance(card, dict) else None,
                }
            )
        except Exception as exc:
            report["dataset_info_error"] = f"{type(exc).__name__}: {exc}"
        try:
            configs = get_dataset_config_names(dataset_id, token=token)
            report["configs"] = configs[:20]
        except Exception as exc:
            configs = [None]
            report["configs_error"] = f"{type(exc).__name__}: {exc}"
        split_reports = []
        for config in (configs or [None])[:3]:
            try:
                splits = get_dataset_split_names(
                    dataset_id,
                    config_name=config,
                    token=token,
                )
            except Exception as exc:
                split_reports.append(
                    {
                        "config": config,
                        "splits_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            config_report = {"config": config, "splits": splits}
            examples = []
            for split in splits[:3]:
                try:
                    ds = load_dataset(
                        dataset_id,
                        config,
                        split=f"{split}[:1]",
                        token=token,
                        cache_dir="/checkpoints/hf_datasets",
                    )
                    row = ds[0] if len(ds) else {}
                    examples.append(
                        {
                            "split": split,
                            "num_rows_probe": len(ds),
                            "features": list(getattr(ds, "features", {}).keys()),
                            "example": {key: _short_value(row[key]) for key in list(row.keys())[:12]},
                        }
                    )
                except Exception as exc:
                    examples.append(
                        {
                            "split": split,
                            "load_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            config_report["examples"] = examples
            split_reports.append(config_report)
        report["split_reports"] = split_reports
        reports.append(report)
    output = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": reports,
    }
    _json_dump(f"{V18_ROOT}/hf_source_probe.json", output)
    volume.commit()
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return output


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * 60 * 60,
)
def build_existing_remote() -> dict[str, Any]:
    sources = build_existing_sources()
    manifest = write_manifest(sources, f"{V18_ROOT}/existing_sources_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=12 * 60 * 60,
)
def build_hf_remote() -> dict[str, Any]:
    sources = build_hf_sources()
    manifest = write_manifest(sources, f"{V18_ROOT}/hf_sources_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=12 * 60 * 60,
)
def build_hf_source_remote(name: str) -> dict[str, Any]:
    source = build_named_hf_source(name)
    manifest = write_manifest([source], f"{V18_ROOT}/hf_source_{name}_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=24 * 60 * 60,
    memory=32768,
)
def build_lcs558k_remote() -> dict[str, Any]:
    source = build_lcs558k_caption(max_samples=200000, seed=1801)
    manifest = write_manifest([source], f"{V18_ROOT}/supplement_lcs558k_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=12 * 60 * 60, memory=32768)
def build_visual_genome_remote() -> dict[str, Any]:
    source = build_visual_genome()
    manifest = write_manifest([source], f"{V18_ROOT}/supplement_visual_genome_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=30 * 60)
def combine_supplement_remote() -> dict[str, Any]:
    manifest_path = f"{V18_ROOT}/source_manifest.json"
    existing_sources = _load_manifest_sources(manifest_path)
    supplement_paths = [
        f"{V18_ROOT}/supplement_lcs558k_manifest.json",
        f"{V18_ROOT}/supplement_visual_genome_manifest.json",
    ]
    sources = list(existing_sources)
    for path in supplement_paths:
        payload = _load_json(path)
        for source in payload.get("sources") or []:
            if not isinstance(source, dict):
                continue
            name = str(source.get("name") or "")
            if not name:
                raise ValueError(f"Supplement source in {path} is missing name")
            existing_index = next(
                (idx for idx, item in enumerate(sources) if str(item.get("name") or "") == name),
                None,
            )
            if existing_index is None:
                sources.append(source)
            else:
                sources[existing_index] = source
    by_name = {str(source.get("name") or ""): source for source in sources}
    missing = [name for name in FINAL_WEIGHTS if name not in by_name]
    if missing:
        raise RuntimeError(f"Cannot combine V18 supplement; missing sources: {missing}")
    ordered_sources = [by_name[name] for name in FINAL_WEIGHTS]
    for source in ordered_sources:
        source["weight"] = FINAL_WEIGHTS[str(source["name"])]
    manifest = write_manifest(ordered_sources, manifest_path)
    weights = _weights_payload(ordered_sources, manifest["created_at_utc"])
    license_summary = _source_license_summary(ordered_sources)
    _json_dump(f"{V18_ROOT}/final_mixture_weights.json", weights)
    _json_dump(f"{V18_ROOT}/mixture_license_summary.json", license_summary)
    volume.commit()
    result = {
        "manifest": manifest,
        "final_mixture_weights": weights,
        "mixture_license_summary": license_summary,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=12 * 60 * 60,
)
def build_all_remote() -> dict[str, Any]:
    existing = build_existing_sources()
    hf_sources = build_hf_sources()
    sources = [*existing, *hf_sources]
    manifest = write_manifest(sources, f"{V18_ROOT}/source_manifest.json")
    volume.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=10 * 60)
def inventory_remote() -> dict[str, Any]:
    inventory = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": {
            "/checkpoints/coco_images": {"images": _count_images("/checkpoints/coco_images")},
            "/checkpoints/coco_train2014_vqa": {"images": _count_images("/checkpoints/coco_train2014_vqa")},
            "/checkpoints/gqa_images_hf": {"images": _count_images("/checkpoints/gqa_images_hf")},
            "/checkpoints/textvqa_images_hf": {"images": _count_images("/checkpoints/textvqa_images_hf")},
            "/checkpoints/vqa_data/vqa_train2014_direct_150000.json": {
                "rows": _json_len("/checkpoints/vqa_data/vqa_train2014_direct_150000.json")
            },
            "/checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json": {
                "rows": _json_len("/checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json")
            },
            "/checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json": {
                "rows": _json_len("/checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json")
            },
        },
    }
    _json_dump(f"{V18_ROOT}/inventory.json", inventory)
    volume.commit()
    print(json.dumps(inventory, indent=2, ensure_ascii=True))
    return inventory


@app.local_entrypoint()
def main(mode: str = "inventory", source: str = "") -> None:
    if mode == "inventory":
        inventory_remote.remote()
    elif mode == "build-existing":
        build_existing_remote.remote()
    elif mode == "build-hf":
        build_hf_remote.remote()
    elif mode == "build-hf-source":
        if not source:
            raise ValueError("--source is required with --mode build-hf-source")
        build_hf_source_remote.remote(source)
    elif mode == "build-lcs558k":
        build_lcs558k_remote.remote()
    elif mode == "build-visual-genome":
        build_visual_genome_remote.remote()
    elif mode == "build-all":
        build_all_remote.remote()
    elif mode == "combine":
        combine_manifests_remote.remote()
    elif mode == "combine-supplement":
        combine_supplement_remote.remote()
    elif mode == "audit-hashes":
        audit_v18_hashes_remote.remote()
    elif mode == "audit-supplement-hashes":
        audit_supplement_hashes_remote.remote()
    elif mode == "filter-gqa-overlaps":
        filter_gqa_eval_overlaps_remote.remote()
    elif mode == "probe-hf":
        probe_hf_sources_remote.remote()
    else:
        raise ValueError(f"Unsupported mode: {mode}")
