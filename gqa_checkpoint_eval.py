"""GQA diagnostic-slice evaluation for AnyMAL checkpoints.

This keeps the V6 second-benchmark path comparable with VQAv2 artifacts:
left-padded decoder generation, answer-only system prompt, exact-match scoring,
and prediction samples with image IDs for leakage audits.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).parent
REMOTE_PROJECT_DIR = "/root/anymal"
if os.path.exists(REMOTE_PROJECT_DIR) and REMOTE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, REMOTE_PROJECT_DIR)

volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.53.0,<5.0.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "datasets>=2.19.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "einops>=0.7.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path=REMOTE_PROJECT_DIR, copy=False)
)

app = modal.App("anymal-gqa-checkpoint-eval")

LLAMA_PATH = "/checkpoints/llama3-8b-instruct"
CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
GQA_DIR = "/checkpoints/gqa_data"
DEFAULT_IMAGE_DIR = "/checkpoints/gqa_images_hf"
HF_GQA_CACHE_DIR = "/checkpoints/hf_datasets"
HF_GQA_DATASET = "Mineru/GQA"
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
VG_IMAGE_URLS = (
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
)


def _ensure_questions_zip() -> str:
    import requests

    os.makedirs(GQA_DIR, exist_ok=True)
    path = os.path.join(GQA_DIR, "questions1.2.zip")
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


def _find_split_member(zf: zipfile.ZipFile, split: str) -> str:
    target = f"{split}_questions.json"
    matches = [name for name in zf.namelist() if name.endswith(target)]
    if not matches:
        raise RuntimeError(
            f"Could not find {target} in GQA questions archive. "
            f"Available JSON examples: {[n for n in zf.namelist() if n.endswith('.json')][:20]}"
        )
    return matches[0]


def _load_gqa_questions(split: str) -> dict:
    zip_path = _ensure_questions_zip()
    with zipfile.ZipFile(zip_path) as zf:
        member = _find_split_member(zf, split)
        with zf.open(member) as f:
            return json.load(f)


def _vqa_layout_name(image_id: int) -> str:
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def _ensure_gqa_images(image_ids: list[int], image_dir: str) -> dict:
    import requests

    os.makedirs(image_dir, exist_ok=True)
    unique_ids = sorted({int(image_id) for image_id in image_ids})
    missing = [
        image_id
        for image_id in unique_ids
        if not os.path.exists(os.path.join(image_dir, _vqa_layout_name(image_id)))
    ]
    if not missing:
        return {"needed": len(unique_ids), "downloaded": 0, "cached": len(unique_ids), "failed": 0}

    def download_one(image_id: int) -> tuple[int, bool, str]:
        out_path = os.path.join(image_dir, _vqa_layout_name(image_id))
        if os.path.exists(out_path):
            return image_id, True, "cached"
        last_error = ""
        for template in VG_IMAGE_URLS:
            url = template.format(image_id=image_id)
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 404:
                    last_error = "404"
                    continue
                response.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(response.content)
                return image_id, True, "downloaded"
            except Exception as exc:  # pragma: no cover - remote network path
                last_error = str(exc)
        return image_id, False, last_error

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_one, image_id) for image_id in missing]
        for future in as_completed(futures):
            _image_id, ok, status = future.result()
            if ok and status == "downloaded":
                downloaded += 1
            elif not ok:
                failed += 1
    volume.commit()
    return {
        "needed": len(unique_ids),
        "downloaded": downloaded,
        "cached": len(unique_ids) - len(missing),
        "failed": failed,
    }


def _answer_type(answer: str) -> str:
    text = str(answer or "").strip().lower()
    if text in {"yes", "no"}:
        return "yes/no"
    if text.isdigit():
        return "number"
    return "other"


def _build_gqa_vqa_files(split: str, max_samples: int, seed: int, image_dir: str) -> tuple[str, str, dict]:
    raw = _load_gqa_questions(split)
    rows = list(raw.items())
    balanced_rows = [
        (question_id, row)
        for question_id, row in rows
        if row.get("isBalanced", True)
    ]
    if balanced_rows:
        rows = balanced_rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples:
        rows = rows[: int(max_samples)]

    image_ids = [int(row["imageId"]) for _question_id, row in rows]
    image_cache = _ensure_gqa_images(image_ids, image_dir)
    if image_cache["failed"]:
        raise RuntimeError(f"Failed to download {image_cache['failed']} GQA images")

    questions = []
    annotations = []
    gqa_meta_by_question = {}
    for question_id, row in rows:
        image_id = int(row["imageId"])
        answer = str(row.get("answer", "")).strip()
        types = row.get("types") or {}
        questions.append(
            {
                "question_id": str(question_id),
                "image_id": image_id,
                "question": str(row["question"]),
            }
        )
        annotations.append(
            {
                "question_id": str(question_id),
                "image_id": image_id,
                "answer_type": _answer_type(answer),
                "question_type": str(types.get("structural") or ""),
                "multiple_choice_answer": answer,
                "answers": [{"answer": answer} for _ in range(10)],
            }
        )
        gqa_meta_by_question[str(question_id)] = {
            "image_id": image_id,
            "answer": answer,
            "full_answer": row.get("fullAnswer"),
            "is_balanced": bool(row.get("isBalanced", True)),
            "types": types,
            "groups": row.get("groups") or row.get("group"),
            "semantic": row.get("semantic"),
            "semantic_str": row.get("semanticStr"),
        }

    base = os.path.join(GQA_DIR, f"{split}_seed{seed}_n{len(rows)}")
    questions_path = f"{base}_questions.json"
    annotations_path = f"{base}_annotations.json"
    meta_path = f"{base}_meta.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f)
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": annotations}, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(gqa_meta_by_question, f)
    volume.commit()

    return questions_path, annotations_path, {
        "gqa_split": split,
        "source_url": GQA_QUESTIONS_URL,
        "questions_zip": os.path.join(GQA_DIR, "questions1.2.zip"),
        "rows": len(rows),
        "original_rows": len(raw),
        "balanced_rows": len(balanced_rows),
        "unique_image_ids": len(set(image_ids)),
        "image_cache": image_cache,
        "meta_path": meta_path,
        "selected_image_ids": image_ids,
    }


def _process_for_exact(answer: str) -> str:
    import re

    text = str(answer or "").lower().strip()
    text = text.split("\n")[0].split(".")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"[^\w\s]", "", text)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words).strip()


def _compute_gqa_metrics(predictions: list[dict], meta_by_question: dict) -> dict:
    total = 0
    correct = 0
    buckets: dict[str, dict[str, int]] = {}
    for row in predictions:
        question_id = str(row.get("question_id"))
        meta = meta_by_question.get(question_id, {})
        target = _process_for_exact(meta.get("answer", ""))
        pred = _process_for_exact(row.get("raw_answer", row.get("answer", "")))
        is_correct = bool(target) and pred == target
        row["gqa_answer"] = meta.get("answer", "")
        row["gqa_exact_match"] = is_correct
        row["gqa_types"] = meta.get("types") or {}
        total += 1
        correct += int(is_correct)
        types = meta.get("types") or {}
        for key in ("structural", "semantic", "detailed"):
            value = str(types.get(key) or "unknown")
            bucket = buckets.setdefault(f"{key}:{value}", {"correct": 0, "total": 0})
            bucket["correct"] += int(is_correct)
            bucket["total"] += 1

    metrics = {
        "gqa_accuracy": 100.0 * correct / total if total else 0.0,
        "gqa_exact_match_rate": correct / total if total else 0.0,
    }
    for name, bucket in sorted(buckets.items()):
        key = name.replace(":", "_").replace("/", "_").replace(" ", "_")
        metrics[f"gqa_accuracy_{key}"] = (
            100.0 * bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        )
        metrics[f"gqa_num_samples_{key}"] = bucket["total"]
    return metrics


def _safe_image_filename(image_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(image_id))
    return f"{safe}.jpg"


def _prepare_hf_gqa_records(split: str, max_samples: int, seed: int, image_dir: str) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(HF_GQA_CACHE_DIR, exist_ok=True)
    dataset = load_dataset(HF_GQA_DATASET, split=split, cache_dir=HF_GQA_CACHE_DIR)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples:
        indices = indices[: int(max_samples)]

    records = []
    downloaded = 0
    cached = 0
    for ordinal, source_idx in enumerate(indices):
        row = dataset[int(source_idx)]
        image_id = str(row["question_id"])
        question_id = f"{split}_{source_idx}_{image_id}"
        answer = str(row["answer"]).strip()
        image_path = os.path.join(image_dir, _safe_image_filename(image_id))
        if os.path.exists(image_path):
            cached += 1
        else:
            image = row["image"].convert("RGB")
            image.save(image_path, format="JPEG")
            downloaded += 1
        records.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": str(row["question"]),
                "answer": answer,
                "answer_type": _answer_type(answer),
                "question_type": "gqa_testdev_balanced",
                "image_path": image_path,
                "source_index": int(source_idx),
                "ordinal": int(ordinal),
            }
        )
    volume.commit()
    meta = {
        "gqa_split": split,
        "source_dataset": HF_GQA_DATASET,
        "source_url": f"https://huggingface.co/datasets/{HF_GQA_DATASET}",
        "source_note": "Hugging Face parquet mirror of GQA testdev_balanced with embedded images.",
        "rows": len(records),
        "original_rows": len(dataset),
        "unique_image_ids": len({row["image_id"] for row in records}),
        "selected_image_ids": [row["image_id"] for row in records],
        "selected_source_indices": [row["source_index"] for row in records],
        "image_cache": {
            "needed": len({row["image_id"] for row in records}),
            "downloaded": downloaded,
            "cached": cached,
            "failed": 0,
        },
    }
    return records, meta


class GQAHFDataset:
    def __init__(
        self,
        records: list[dict],
        transform,
        tokenizer,
        image_placeholder_token_id,
        num_image_tokens: int,
        prompt_style: str,
        system_prompt: str | None,
        chat_template_family: str | None = None,
    ):
        self.records = list(records)
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_placeholder_token_id = image_placeholder_token_id
        self.num_image_tokens = int(num_image_tokens or 0)
        self.prompt_style = str(prompt_style)
        self.system_prompt = system_prompt
        self.chat_template_family = chat_template_family

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        from evaluation.vqa_eval import _build_chat_prompt_ids

        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        if self.prompt_style != "training_chat":
            raise ValueError("GQA V6 eval currently requires prompt_style=training_chat")
        encoded = _build_chat_prompt_ids(
            tokenizer=self.tokenizer,
            question=record["question"],
            image_placeholder_token_id=self.image_placeholder_token_id,
            num_image_tokens=self.num_image_tokens,
            max_length=768,
            system_prompt=self.system_prompt,
            chat_template_family=self.chat_template_family,
        )
        return {
            "image": image_tensor,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "image_id": record["image_id"],
            "source_image_id": record["image_id"],
            "image_control": "none",
            "question_id": record["question_id"],
            "question": record["question"],
            "answers": [record["answer"]] * 10,
            "answer_type": record["answer_type"],
            "question_type": record["question_type"],
        }


def _build_gqa_dataloader(
    model,
    architecture: str,
    records: list[dict],
    batch_size: int,
    prompt_style: str,
    system_prompt: str | None,
):
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from data.data_utils import get_image_transform, get_vision_transform

    if architecture in {"v2", "v3", "v4"}:
        transform = get_vision_transform(
            vision_encoder_type="siglip2",
            vision_model_name="google/siglip2-so400m-patch14-384",
            image_size=384,
            is_train=False,
            use_augmentation=False,
        )
    else:
        transform = get_image_transform(image_size=224, is_train=False, use_augmentation=False)

    dataset = GQAHFDataset(
        records=records,
        transform=transform,
        tokenizer=model.tokenizer,
        image_placeholder_token_id=getattr(model, "image_placeholder_token_id", None),
        num_image_tokens=getattr(model, "num_image_tokens", 0),
        prompt_style=prompt_style,
        system_prompt=system_prompt,
        chat_template_family=getattr(getattr(model, "llm", None), "chat_template_family", None),
    )
    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

    def collate(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        images = __import__("torch").stack([item["image"] for item in batch])
        max_len = max(item["input_ids"].shape[-1] for item in batch)
        input_ids = []
        attention_masks = []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(F.pad(item["input_ids"], (pad_len, 0), value=pad_token_id))
            attention_masks.append(F.pad(item["attention_mask"], (pad_len, 0), value=0))
        torch = __import__("torch")
        return {
            "image": images,
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "image_id": [item["image_id"] for item in batch],
            "source_image_id": [item["source_image_id"] for item in batch],
            "image_control": [item["image_control"] for item in batch],
            "question_id": [item["question_id"] for item in batch],
            "question": [item["question"] for item in batch],
            "answers": [item["answers"] for item in batch],
            "answer_type": [item["answer_type"] for item in batch],
            "question_type": [item["question_type"] for item in batch],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )
    dataset_meta = {
        "questions_file": None,
        "annotations_file": None,
        "image_dir": DEFAULT_IMAGE_DIR,
        "prompt_style": str(prompt_style),
        "max_samples": int(len(records)),
        "original_num_questions": int(len(records)),
        "available_questions": int(len(records)),
        "selected_image_ids": [row["image_id"] for row in records],
        "selected_unique_image_ids": int(len({row["image_id"] for row in records})),
        "source_image_ids": [row["image_id"] for row in records],
        "source_unique_image_ids": int(len({row["image_id"] for row in records})),
        "image_id_namespace": "gqa_hf_image_id",
    }
    image_transform_meta = {
        "name": "none",
        "base_transform": "siglip2_384" if architecture in {"v2", "v3", "v4"} else "clip_224",
        "control_uses_wrong_images": False,
        "control_seed": None,
    }
    return dataloader, dataset_meta, image_transform_meta


def _load_model(run: dict, device, llm_backbone=None):
    import torch

    sys.path.insert(0, "/root/anymal")
    from model_metadata import read_model_metadata
    from models.anymal_v3 import AnyMALv3
    from models.anymal_v4 import AnyMALv4
    from v1_v2_compare_inference import _load_v1_model, _load_v2_model
    from vqa_checkpoint_eval import _ensure_eval_llm_path, _resolve_eval_llm_path

    meta = read_model_metadata(run["checkpoint"]) or {}
    llm_path = _ensure_eval_llm_path(
        _resolve_eval_llm_path(meta, llm_backbone),
        model_meta=meta,
        llm_backbone=llm_backbone,
    )
    if run["architecture"] == "v1":
        model = _load_v1_model(run["checkpoint"], llm_path, device)
    elif run["architecture"] == "v2":
        model = _load_v2_model(run["checkpoint"], llm_path, device)
    elif run["architecture"] == "v3":
        model = AnyMALv3.from_pretrained(
            run["checkpoint"],
            llm_model_name=llm_path,
            vision_encoder_type="siglip2",
            vision_model_name="google/siglip2-so400m-patch14-384",
            connector_type=meta.get("connector_type", "perceiver_resampler"),
            num_image_tokens=int(meta.get("num_image_tokens", 128)),
            connector_layers=int(meta.get("connector_layers", 6)),
            connector_heads=int(meta.get("connector_heads", 16)),
            connector_ff_mult=int(meta.get("connector_ff_mult", 4)),
            connector_output_scale=float(meta.get("connector_output_scale", 1.0)),
            connector_output_gate_init=(
                float(meta["connector_output_gate_init"])
                if meta.get("connector_output_gate_init") is not None
                else None
            ),
            project_directly_to_llm_dim=bool(meta.get("project_directly_to_llm_dim", True)),
            use_qlora=True,
            use_lora=False,
            lora_r=64,
            lora_alpha=16,
            gradient_checkpointing=False,
            use_flash_attention=False,
            llm_device_map="auto",
            llm_torch_dtype=torch.bfloat16,
        )
        model.eval()
        model.image_encoder.to(device)
        model.projector.to(device)
    elif run["architecture"] == "v4":
        connector_type = meta.get("connector_type", "spatial_perceiver_resampler")
        deepstack_layers = meta.get("deepstack_hidden_state_indices") or meta.get("vision_feature_layers")
        deepstack_kwargs = {}
        if connector_type == "deepstack_spatial_perceiver_resampler":
            deepstack_kwargs = {
                "deepstack_num_feature_levels": int(
                    meta.get("deepstack_num_feature_levels") or len(deepstack_layers or []) or 3
                ),
                "deepstack_hidden_state_indices": deepstack_layers,
            }
        model = AnyMALv4.from_pretrained(
            run["checkpoint"],
            llm_model_name=llm_path,
            vision_encoder_type="siglip2",
            vision_model_name="google/siglip2-so400m-patch14-384",
            connector_type=connector_type,
            num_global_image_tokens=int(meta.get("num_global_image_tokens", 64)),
            num_local_image_tokens=int(meta.get("num_local_image_tokens", 64)),
            num_image_tokens=int(meta.get("num_image_tokens", 128)),
            connector_layers=int(meta.get("connector_layers", 6)),
            connector_heads=int(meta.get("connector_heads", 16)),
            connector_ff_mult=int(meta.get("connector_ff_mult", 4)),
            connector_hidden_dim=(
                int(meta["connector_hidden_dim"]) if meta.get("connector_hidden_dim") is not None else None
            ),
            connector_output_scale=float(meta.get("connector_output_scale", 1.0)),
            connector_output_gate_init=(
                float(meta["connector_output_gate_init"])
                if meta.get("connector_output_gate_init") is not None
                else None
            ),
            use_2d_position_features=bool(meta.get("use_2d_position_features", True)),
            project_directly_to_llm_dim=bool(meta.get("project_directly_to_llm_dim", True)),
            use_qlora=True,
            use_lora=False,
            lora_r=64,
            lora_alpha=16,
            gradient_checkpointing=False,
            use_flash_attention=False,
            llm_device_map="auto",
            llm_torch_dtype=torch.bfloat16,
            **deepstack_kwargs,
        )
        model.eval()
        model.image_encoder.to(device)
        model.projector.to(device)
    else:
        raise ValueError(f"Unknown architecture: {run['architecture']}")
    return model, meta


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_gqa(
    candidate_checkpoint=None,
    candidate_label=None,
    candidate_architecture="v4",
    gqa_split="testdev_balanced",
    max_samples=1000,
    seed=42,
    batch_size=8,
    image_dir=DEFAULT_IMAGE_DIR,
    prompt_style="training_chat",
    system_prompt=None,
    prediction_samples=0,
    remote_output_path=None,
    train_sources=None,
    eval_schema_version="v6",
    llm_backbone=CURRENT_LLAMA3_BACKBONE,
):
    import torch

    sys.path.insert(0, "/root/anymal")
    from evaluation.vqa_eval import VQAEvaluator
    from vqa_checkpoint_eval import _default_runs, _parse_train_sources

    records, gqa_meta = _prepare_hf_gqa_records(
        split=str(gqa_split),
        max_samples=int(max_samples),
        seed=int(seed),
        image_dir=image_dir,
    )
    meta_by_question = {
        record["question_id"]: {
            "answer": record["answer"],
            "types": {},
            "image_id": record["image_id"],
            "source_index": record["source_index"],
        }
        for record in records
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed_train_sources = _parse_train_sources(train_sources)

    results = []
    for run in _default_runs(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        include_baselines=False,
    ):
        print(f"Evaluating GQA {gqa_split}: {run['label']} from {run['checkpoint']}")
        model, run_model_meta = _load_model(run, device, llm_backbone=llm_backbone)
        dataloader, dataset_meta, image_transform_meta = _build_gqa_dataloader(
            model=model,
            architecture=run["architecture"],
            batch_size=int(batch_size),
            prompt_style=str(prompt_style),
            records=records,
            system_prompt=system_prompt,
        )
        evaluator = VQAEvaluator(model, device=device, max_new_tokens=16)
        prediction_output = f"/tmp/{run['key']}_gqa_predictions.json"
        hygiene_metrics = evaluator.evaluate(dataloader, output_file=prediction_output)
        with open(prediction_output, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        gqa_metrics = _compute_gqa_metrics(predictions, meta_by_question)
        metrics = {**hygiene_metrics, **gqa_metrics}
        connector_meta_keys = {
            "connector_type",
            "num_image_tokens",
            "num_global_image_tokens",
            "num_local_image_tokens",
            "connector_layers",
            "connector_heads",
            "connector_ff_mult",
            "connector_hidden_dim",
            "connector_output_scale",
            "connector_output_gate_init",
            "use_2d_position_features",
            "project_directly_to_llm_dim",
            "deepstack_num_feature_levels",
            "deepstack_hidden_state_indices",
            "vision_feature_layers",
        }
        result_entry = {
            **run,
            "candidate_checkpoint": run["checkpoint"],
            "candidate_architecture": run["architecture"],
            "model_meta": run_model_meta,
            "connector_meta": {
                key: value for key, value in run_model_meta.items() if key in connector_meta_keys
            },
            "dataset_meta": {**dataset_meta, **gqa_meta},
            "train_source_meta": {
                "train_sources": parsed_train_sources,
                "leakage_audit_required": True,
                "gqa_visual_genome_id_audit_required": True,
            },
            "image_transform_meta": image_transform_meta,
            "metrics": metrics,
        }
        if int(prediction_samples or 0) > 0:
            result_entry["prediction_samples"] = predictions[: int(prediction_samples)]
        results.append(result_entry)

        del evaluator, dataloader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "eval_schema_version": str(eval_schema_version),
        "benchmark": "gqa",
        "llm_backbone": str(llm_backbone),
        "padding_side": "left",
        "generation_mode": "decoder_leftpad_greedy",
        "gqa_split": str(gqa_split),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "image_dir": image_dir,
        "prompt_style": str(prompt_style),
        "system_prompt": system_prompt,
        "runs": results,
    }
    if remote_output_path:
        os.makedirs(os.path.dirname(str(remote_output_path)), exist_ok=True)
        with open(str(remote_output_path), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        print(f"Saved remote GQA result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v4",
    gqa_split: str = "testdev_balanced",
    max_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = DEFAULT_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = None,
    prediction_samples: int = 0,
    output: str = "gqa_checkpoint_eval.json",
    remote_output_path: str = None,
    train_sources: str = "",
    eval_schema_version: str = "v6",
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    background: bool = False,
):
    call = evaluate_gqa.spawn(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        gqa_split=gqa_split,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
        prediction_samples=prediction_samples,
        remote_output_path=remote_output_path,
        train_sources=train_sources,
        eval_schema_version=eval_schema_version,
        llm_backbone=llm_backbone,
    )
    if background:
        print(f"Spawned background GQA eval: {call}")
        print(f"Remote output path: {remote_output_path or '(none)'}")
        return
    result = call.get()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
