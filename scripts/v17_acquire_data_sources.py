#!/usr/bin/env python3
"""Acquire and inventory V17 candidate training data sources on Modal.

The script intentionally produces conservative, auditable artifacts:

* one inventory JSON per source under /checkpoints/v17_acquired/
* one converted instruction JSON per source when a usable split is available
* a combined manifest with data_path + image_dir for future mixture builders

It does not silently bless restrictive or unknown licenses. Those are recorded in
the inventory and manifest for the V18 data-selection decision.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import modal


app = modal.App("anymal-v17-acquire-data-sources")
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
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    config: str | None
    split: str
    license: str
    license_source: str
    commercial_use_allowed: bool | None
    source_family: str
    preferred: bool = True
    notes: str = ""


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec(
        name="ocr_vqa",
        dataset_id="howard-hou/OCR-VQA",
        config=None,
        split="train",
        license="unknown",
        license_source="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        commercial_use_allowed=None,
        source_family="ocr_vqa",
        notes=(
            "Selected instead of atc96/OCR-VQA-200K because the atc96 mirror "
            "currently exposes license text rather than image/question rows through HF datasets-server."
        ),
    ),
    SourceSpec(
        name="docvqa_public_lmms",
        dataset_id="lmms-lab/DocVQA",
        config="DocVQA",
        split="validation",
        license="apache-2.0",
        license_source="https://huggingface.co/datasets/lmms-lab/DocVQA",
        commercial_use_allowed=True,
        source_family="docvqa",
        preferred=False,
        notes=(
            "Public lmms-lab mirror exposes validation/test only, not train. "
            "Inventory is retained, but this source is not included in training manifests by default."
        ),
    ),
    SourceSpec(
        name="ai2d_train",
        dataset_id="LIME-DATA/ai2d",
        config=None,
        split="train",
        license="apache-2.0",
        license_source="https://huggingface.co/datasets/LIME-DATA/ai2d",
        commercial_use_allowed=True,
        source_family="ai2d",
    ),
    SourceSpec(
        name="chartqa_augmented",
        dataset_id="HuggingFaceM4/ChartQA",
        config=None,
        split="train",
        license="gpl-3.0",
        license_source="https://huggingface.co/datasets/HuggingFaceM4/ChartQA",
        commercial_use_allowed=False,
        source_family="chartqa",
        notes="Restrictive GPL-3.0 source; keep out of deployed-model mixes unless reviewed.",
    ),
    SourceSpec(
        name="sharegpt4v_detailed",
        dataset_id="Lin-Chen/ShareGPT4V",
        config="ShareGPT4V",
        split="train",
        license="cc-by-nc-4.0",
        license_source="https://huggingface.co/datasets/Lin-Chen/ShareGPT4V",
        commercial_use_allowed=False,
        source_family="sharegpt4v",
        notes="Non-commercial license; many rows reference external COCO files by path.",
    ),
    SourceSpec(
        name="visual_genome_regions",
        dataset_id="visual_genome",
        config="region_descriptions",
        split="train",
        license="cc-by-4.0",
        license_source="https://huggingface.co/datasets/visual_genome",
        commercial_use_allowed=True,
        source_family="visual_genome",
    ),
    SourceSpec(
        name="refcoco_train",
        dataset_id="jxu124/refcoco",
        config=None,
        split="train",
        license="unknown",
        license_source="https://huggingface.co/datasets/jxu124/refcoco",
        commercial_use_allowed=None,
        source_family="refcoco",
    ),
    SourceSpec(
        name="refcocoplus_train",
        dataset_id="jxu124/refcocoplus",
        config=None,
        split="train",
        license="unknown",
        license_source="https://huggingface.co/datasets/jxu124/refcocoplus",
        commercial_use_allowed=None,
        source_family="refcoco_plus",
    ),
    SourceSpec(
        name="refcocog_train",
        dataset_id="jxu124/refcocog",
        config=None,
        split="train",
        license="unknown",
        license_source="https://huggingface.co/datasets/jxu124/refcocog",
        commercial_use_allowed=None,
        source_family="refcoco_g",
    ),
]


def _json_dump(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return safe.strip("._") or fallback


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, dict):
            text = _first_text(*value.values())
        elif isinstance(value, (list, tuple)):
            text = _first_text(*value)
        else:
            text = " ".join(str(value or "").split())
        if text:
            return text
    return ""


def _normal_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).rstrip(".,;:!?")


def _choice_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    options = row.get("options")
    if isinstance(answer, int) and isinstance(options, list) and 0 <= answer < len(options):
        return _normal_answer(options[answer])
    if isinstance(answer, list):
        return _normal_answer(_first_text(answer))
    return _normal_answer(_first_text(answer, row.get("answers"), row.get("label")))


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
        if image_value.get("src"):
            fallback_urls = (image_value["src"],) + tuple(fallback_urls)
    if isinstance(image_value, str):
        if os.path.exists(image_value):
            return Image.open(image_value).convert("RGB")
        if image_value.startswith(("http://", "https://")):
            fallback_urls = (image_value,) + tuple(fallback_urls)
    for url in fallback_urls:
        if not url:
            continue
        response = requests.get(str(url), timeout=90)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    raise RuntimeError(f"Unsupported image payload: {type(image_value).__name__}")


def _coco_url_from_ref(ref: Any) -> str | None:
    text = str(ref or "")
    match = re.search(r"(COCO_(train|val)20\d{2}_\d{12}\.jpg)", text)
    if match:
        split = "train2014" if "train2014" in match.group(1) else "val2014"
        if "train2017" in match.group(1):
            split = "train2017"
        return f"http://images.cocodataset.org/{split}/{match.group(1)}"
    match = re.search(r"coco/(train20\d{2}|val20\d{2})/([^/]+\.jpg)", text, re.IGNORECASE)
    if match:
        return f"http://images.cocodataset.org/{match.group(1)}/{match.group(2)}"
    return None


def _copy_or_download_image(row: dict[str, Any], image_dir: str, source_name: str, source_index: int) -> tuple[str, str | None]:
    os.makedirs(image_dir, exist_ok=True)
    raw_ref = (
        row.get("image")
        or row.get("image_path")
        or row.get("file_name")
        or row.get("image_url")
        or row.get("image_id")
        or source_index
    )
    filename = _safe_name(
        os.path.basename(str(raw_ref)) if isinstance(raw_ref, str) else f"{source_name}_{source_index}.jpg",
        f"{source_name}_{source_index}.jpg",
    )
    if not re.search(r"\.(jpg|jpeg|png|webp)$", filename, re.IGNORECASE):
        filename = f"{filename}.jpg"
    path = os.path.join(image_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return filename, None

    fallback_urls = [
        row.get("image_url"),
        row.get("url"),
        row.get("coco_url"),
        row.get("flickr_url"),
        _coco_url_from_ref(raw_ref),
        _coco_url_from_ref(row.get("image_path")),
        _coco_url_from_ref(row.get("file_name")),
    ]
    raw_image_info = row.get("raw_image_info")
    if isinstance(raw_image_info, str):
        try:
            parsed = json.loads(raw_image_info)
            fallback_urls.extend([parsed.get("coco_url"), parsed.get("flickr_url")])
        except Exception:
            pass
    try:
        image = _hf_image_to_rgb(row.get("image"), tuple(url for url in fallback_urls if url))
        image.save(path, format="JPEG", quality=94)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    return filename, None


def _generic_qa_records(row: dict[str, Any]) -> list[tuple[str, str]]:
    questions = row.get("questions")
    answers = row.get("answers")
    if isinstance(questions, list) and isinstance(answers, list):
        pairs = []
        for question, answer in zip(questions, answers):
            q = _first_text(question)
            a = _normal_answer(answer)
            if q and a:
                pairs.append((q, a))
        return pairs
    question = _first_text(row.get("question"), row.get("query"), row.get("prompt"))
    answer = _choice_answer(row)
    return [(question, answer)] if question and answer else []


def _sharegpt4v_records(row: dict[str, Any]) -> list[tuple[str, str]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return []
    pairs: list[tuple[str, str]] = []
    pending_question = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        value = _first_text(turn.get("value") or turn.get("content"))
        if role in {"human", "user"}:
            pending_question = value.replace("<image>", "").strip()
        elif role in {"gpt", "assistant"} and pending_question and value:
            pairs.append((pending_question, value))
            pending_question = ""
    return pairs


def _refcoco_records(row: dict[str, Any]) -> list[tuple[str, str]]:
    captions = row.get("captions")
    if not isinstance(captions, list):
        captions = [_first_text(item.get("sent") if isinstance(item, dict) else item) for item in row.get("sentences") or []]
    captions = [_first_text(caption) for caption in captions if _first_text(caption)]
    if not captions:
        return []
    return [
        (
            "Describe the highlighted region in the image.",
            captions[0],
        )
    ]


def _visual_genome_records(row: dict[str, Any]) -> list[tuple[str, str]]:
    answer = _first_text(row.get("phrase"), row.get("region_description"), row.get("description"))
    if answer:
        return [("Describe the specified region in the image.", answer)]
    return _generic_qa_records(row)


CONVERTERS: dict[str, Callable[[dict[str, Any]], list[tuple[str, str]]]] = {
    "ocr_vqa": _generic_qa_records,
    "docvqa": _generic_qa_records,
    "ai2d": _generic_qa_records,
    "chartqa": _generic_qa_records,
    "sharegpt4v": _sharegpt4v_records,
    "visual_genome": _visual_genome_records,
    "refcoco": _refcoco_records,
    "refcoco_plus": _refcoco_records,
    "refcoco_g": _refcoco_records,
}


def _dataset_info(spec: SourceSpec, revision: str) -> dict[str, Any]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder

    configs: list[str] = []
    splits: dict[str, list[str]] = {}
    split_sizes: dict[str, int | None] = {}
    try:
        configs = list(get_dataset_config_names(spec.dataset_id, revision=revision))
    except Exception:
        if spec.config:
            configs = [spec.config]
    config_names = [spec.config] if spec.config else (configs or [None])
    for config in config_names:
        config_key = config or "default"
        try:
            split_names = list(get_dataset_split_names(spec.dataset_id, config, revision=revision))
        except Exception:
            split_names = [spec.split]
        splits[config_key] = split_names
        try:
            builder = load_dataset_builder(spec.dataset_id, config, revision=revision)
            for split_name, split_info in (builder.info.splits or {}).items():
                split_sizes[f"{config_key}/{split_name}"] = getattr(split_info, "num_examples", None)
        except Exception:
            for split_name in split_names:
                split_sizes.setdefault(f"{config_key}/{split_name}", None)
    return {
        "configs": configs,
        "splits": splits,
        "split_sizes": split_sizes,
    }


def _hf_revision(dataset_id: str) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id, files_metadata=False)
    card = getattr(info, "card_data", None)
    if hasattr(card, "to_dict"):
        card_payload = card.to_dict()
    elif isinstance(card, dict):
        card_payload = dict(card)
    else:
        card_payload = {}
    return str(info.sha), card_payload


def acquire_source(
    spec: SourceSpec,
    *,
    max_samples: int,
    seed: int,
    output_root: str,
) -> dict[str, Any]:
    from datasets import load_dataset

    revision, card_payload = _hf_revision(spec.dataset_id)
    info = _dataset_info(spec, revision)
    image_dir = os.path.join(output_root, "images", spec.name)
    data_path = os.path.join(output_root, "data", f"{spec.name}_{spec.split}_seed{seed}_n{max_samples or 'all'}.json")
    inventory_path = os.path.join(output_root, "inventories", f"data_inventory_{spec.name}.json")

    inventory: dict[str, Any] = {
        **asdict(spec),
        "revision": revision,
        "hf_card_license": card_payload.get("license"),
        "configs": info["configs"],
        "splits": info["splits"],
        "split_sizes": info["split_sizes"],
        "requested_split": spec.split,
        "image_dir": image_dir,
        "data_path": data_path,
        "inventory_path": inventory_path,
        "usable_for_training": bool(spec.preferred and spec.split.lower() == "train"),
        "warnings": [],
    }
    available_splits = set(info["splits"].get(spec.config or "default", []))
    if spec.split not in available_splits and available_splits:
        inventory["warnings"].append(f"requested split {spec.split!r} not advertised in HF metadata")
    if spec.commercial_use_allowed is False:
        inventory["warnings"].append("license is restrictive for commercial/deployed use")
    if spec.commercial_use_allowed is None:
        inventory["warnings"].append("license/commercial-use status is unknown")
    if not inventory["usable_for_training"]:
        inventory["warnings"].append("not a preferred train split; excluded from default future training manifests")

    rows_written = 0
    source_rows_seen = 0
    image_failures = 0
    conversion_failures = 0
    samples: list[dict[str, Any]] = []
    try:
        dataset = load_dataset(
            spec.dataset_id,
            spec.config,
            split=spec.split,
            cache_dir="/checkpoints/hf_datasets",
            revision=revision,
            streaming=max_samples == 0,
        )
    except Exception as exc:
        inventory["warnings"].append(f"load_dataset failed: {type(exc).__name__}: {exc}")
        inventory["rows_written"] = 0
        inventory["source_rows_seen"] = 0
        _json_dump(inventory_path, inventory)
        return inventory

    rng = random.Random(int(seed))
    if max_samples:
        try:
            total = len(dataset)
            indices = list(range(total))
            rng.shuffle(indices)
            iterator = (dataset[int(index)] for index in indices[: int(max_samples)])
        except Exception:
            iterator = iter(dataset.shuffle(seed=int(seed), buffer_size=10000).take(int(max_samples)))
    else:
        iterator = iter(dataset)

    converter = CONVERTERS.get(spec.source_family, _generic_qa_records)
    for source_index, row in enumerate(iterator):
        source_rows_seen += 1
        if not isinstance(row, dict):
            conversion_failures += 1
            continue
        pairs = converter(row)
        if not pairs:
            conversion_failures += 1
            continue
        filename, image_error = _copy_or_download_image(row, image_dir, spec.name, source_index)
        if image_error:
            image_failures += 1
            continue
        for pair_index, (question, answer) in enumerate(pairs):
            if not question or not answer:
                continue
            samples.append(
                {
                    "id": f"{spec.name}_{spec.split}_{source_index}_{pair_index}",
                    "image": filename,
                    "source_dataset": spec.dataset_id,
                    "source_dataset_revision": revision,
                    "source_config": spec.config,
                    "source_split": spec.split,
                    "source_index": source_index,
                    "source_family": spec.source_family,
                    "license": spec.license,
                    "commercial_use_allowed": spec.commercial_use_allowed,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
            rows_written += 1
        if max_samples and source_rows_seen >= int(max_samples):
            break

    _json_dump(data_path, samples)
    inventory.update(
        {
            "rows_written": rows_written,
            "source_rows_seen": source_rows_seen,
            "image_failures": image_failures,
            "conversion_failures": conversion_failures,
            "sample_records": samples[:5],
        }
    )
    _json_dump(inventory_path, inventory)
    return inventory


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=12 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def acquire_all_remote(
    source_names: list[str],
    max_samples: int,
    seed: int,
    output_root: str,
    remote_output_path: str,
) -> dict[str, Any]:
    selected = [spec for spec in SOURCE_SPECS if not source_names or spec.name in source_names]
    inventories = [
        acquire_source(spec, max_samples=max_samples, seed=seed, output_root=output_root)
        for spec in selected
    ]
    manifest = {
        "output_root": output_root,
        "max_samples_per_source": max_samples,
        "seed": seed,
        "sources": inventories,
        "training_ready_sources": [
            {
                "name": item["name"],
                "data_path": item["data_path"],
                "image_dir": item["image_dir"],
                "rows_written": item.get("rows_written", 0),
                "license": item["license"],
                "commercial_use_allowed": item["commercial_use_allowed"],
                "warnings": item.get("warnings", []),
            }
            for item in inventories
            if item.get("usable_for_training") and item.get("rows_written", 0) > 0
        ],
    }
    _json_dump(remote_output_path, manifest)
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    output: str = "v17_acquired_data_manifest.json",
    remote_output_path: str = "/checkpoints/v17_acquired/manifest.json",
    output_root: str = "/checkpoints/v17_acquired",
    sources: str = "",
    max_samples_per_source: int = 2000,
    seed: int = 1701,
):
    source_names = [part for part in re.split(r"[,\s]+", sources or "") if part]
    result = acquire_all_remote.remote(
        source_names=source_names,
        max_samples=int(max_samples_per_source),
        seed=int(seed),
        output_root=str(output_root),
        remote_output_path=str(remote_output_path),
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="v17_acquired_data_manifest.json")
    parser.add_argument("--remote-output-path", default="/checkpoints/v17_acquired/manifest.json")
    parser.add_argument("--output-root", default="/checkpoints/v17_acquired")
    parser.add_argument("--sources", default="")
    parser.add_argument("--max-samples-per-source", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.parse_args()
