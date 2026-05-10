"""POPE object-hallucination evaluation for AnyMAL checkpoints.

The evaluator reuses the VQAv2 prompt/generation machinery so POPE runs are
comparable with V6 VQA artifacts: left-padded decoder generation, answer-only
system prompt, and full prediction samples.
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
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
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "einops>=0.7.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path=REMOTE_PROJECT_DIR, copy=False)
)


app = modal.App("anymal-pope-checkpoint-eval")

LLAMA_PATH = "/checkpoints/llama3-8b-instruct"
CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_IMAGE_DIR = "/checkpoints/coco_val2014"
POPE_DIR = "/checkpoints/pope_data"
POPE_URLS = {
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}
COCO_VAL_URL = "http://images.cocodataset.org/val2014/{filename}"
IMAGE_RE = re.compile(r"COCO_val2014_(?P<id>\d{12})\.jpg$", re.IGNORECASE)


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _image_id_from_name(name: str) -> int:
    match = IMAGE_RE.search(str(name))
    if not match:
        raise ValueError(f"Unsupported POPE image name: {name}")
    return int(match.group("id"))


def _ensure_pope_file(split: str) -> str:
    import requests

    os.makedirs(POPE_DIR, exist_ok=True)
    path = os.path.join(POPE_DIR, f"coco_pope_{split}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = POPE_URLS[split]
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    volume.commit()
    return path


def _ensure_pope_images(rows: list[dict], image_dir: str) -> dict:
    import requests

    os.makedirs(image_dir, exist_ok=True)
    filenames = sorted({str(row["image"]) for row in rows})
    missing = [name for name in filenames if not os.path.exists(os.path.join(image_dir, name))]
    if not missing:
        return {"needed": len(filenames), "downloaded": 0, "cached": len(filenames), "failed": 0}

    def download_one(filename: str) -> tuple[str, bool, str]:
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            return filename, True, "cached"
        try:
            response = requests.get(COCO_VAL_URL.format(filename=filename), timeout=30)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            return filename, True, "downloaded"
        except Exception as exc:  # pragma: no cover - remote network path
            return filename, False, str(exc)

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_one, filename) for filename in missing]
        for future in as_completed(futures):
            _filename, ok, status = future.result()
            if ok and status == "downloaded":
                downloaded += 1
            elif not ok:
                failed += 1
    volume.commit()
    return {
        "needed": len(filenames),
        "downloaded": downloaded,
        "cached": len(filenames) - len(missing),
        "failed": failed,
    }


def _build_pope_vqa_files(split: str, max_samples: int, seed: int, image_dir: str) -> tuple[str, str, dict]:
    source_path = _ensure_pope_file(split)
    rows = _read_jsonl(source_path)
    if max_samples and len(rows) > max_samples:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        rows = [rows[idx] for idx in indices[:max_samples]]
    image_cache = _ensure_pope_images(rows, image_dir)
    if image_cache["failed"]:
        raise RuntimeError(f"Failed to download {image_cache['failed']} POPE images")

    questions = []
    annotations = []
    for idx, row in enumerate(rows):
        image_id = _image_id_from_name(row["image"])
        question_id = int(row.get("question_id", idx + 1))
        label = str(row["label"]).strip().lower()
        questions.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": str(row["text"]),
            }
        )
        annotations.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "answer_type": "yes/no",
                "question_type": f"pope_{split}",
                "multiple_choice_answer": label,
                "answers": [{"answer": label} for _ in range(10)],
            }
        )

    base = os.path.join(POPE_DIR, f"coco_pope_{split}_seed{seed}_n{len(rows)}")
    questions_path = f"{base}_questions.json"
    annotations_path = f"{base}_annotations.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f)
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": annotations}, f)
    volume.commit()

    return questions_path, annotations_path, {
        "pope_split": split,
        "source_url": POPE_URLS[split],
        "source_path": source_path,
        "rows": len(rows),
        "unique_image_ids": len({row["image_id"] for row in questions}),
        "image_cache": image_cache,
    }


def _official_pope_answer(raw_answer: str) -> str:
    text = str(raw_answer or "")
    if "." in text:
        text = text.split(".", 1)[0]
    text = text.replace(",", " ")
    words = {word.strip().lower() for word in text.split()}
    return "no" if {"no", "not"} & words else "yes"


def _compute_pope_metrics(predictions: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    yes_count = 0
    exact_yes_no = 0
    for row in predictions:
        label = str(row.get("answers", [""])[0] if row.get("answers") else "").lower()
        pred = _official_pope_answer(row.get("raw_answer", row.get("answer", "")))
        row["pope_answer"] = pred
        if str(row.get("answer", "")).lower() in {"yes", "no"}:
            exact_yes_no += 1
        if pred == "yes":
            yes_count += 1
        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
        elif pred == "no" and label == "no":
            tn += 1
        elif pred == "no" and label == "yes":
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "pope_accuracy": 100.0 * (tp + tn) / total if total else 0.0,
        "pope_precision": precision,
        "pope_recall": recall,
        "pope_f1": f1,
        "pope_yes_ratio": yes_count / total if total else 0.0,
        "pope_exact_yes_no_rate": exact_yes_no / total if total else 0.0,
        "pope_tp": tp,
        "pope_fp": fp,
        "pope_tn": tn,
        "pope_fn": fn,
    }


def _load_model(run: dict, device, llm_backbone=None):
    import sys

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
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_pope(
    candidate_checkpoint=None,
    candidate_label=None,
    candidate_architecture="v4",
    pope_split="adversarial",
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
    import sys

    import torch

    sys.path.insert(0, "/root/anymal")
    from evaluation.vqa_eval import VQAEvaluator
    from vqa_checkpoint_eval import _build_vqa_dataloader, _default_runs, _parse_train_sources

    pope_split = str(pope_split).lower()
    if pope_split not in POPE_URLS:
        raise ValueError(f"Unsupported POPE split '{pope_split}'. Expected one of {sorted(POPE_URLS)}")

    questions, annotations, pope_meta = _build_pope_vqa_files(
        split=pope_split,
        max_samples=int(max_samples),
        seed=int(seed),
        image_dir=image_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed_train_sources = _parse_train_sources(train_sources)

    results = []
    for run in _default_runs(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        include_baselines=False,
    ):
        print(f"Evaluating POPE {pope_split}: {run['label']} from {run['checkpoint']}")
        model, run_model_meta = _load_model(run, device, llm_backbone=llm_backbone)
        dataloader, dataset_meta, image_transform_meta = _build_vqa_dataloader(
            model=model,
            architecture=run["architecture"],
            questions=questions,
            annotations=annotations,
            image_dir=image_dir,
            max_samples=int(max_samples),
            seed=int(seed),
            batch_size=int(batch_size),
            prompt_style=str(prompt_style),
            image_perturbation="none",
            system_prompt=system_prompt,
        )
        evaluator = VQAEvaluator(model, device=device, max_new_tokens=16)
        prediction_output = f"/tmp/{run['key']}_pope_predictions.json"
        hygiene_metrics = evaluator.evaluate(dataloader, output_file=prediction_output)
        with open(prediction_output, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        pope_metrics = _compute_pope_metrics(predictions)
        metrics = {**hygiene_metrics, **pope_metrics}
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
            "dataset_meta": {**dataset_meta, **pope_meta},
            "train_source_meta": {
                "train_sources": parsed_train_sources,
                "leakage_audit_required": True,
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
        "benchmark": "pope",
        "llm_backbone": str(llm_backbone),
        "padding_side": "left",
        "generation_mode": "decoder_leftpad_greedy",
        "pope_split": pope_split,
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
        print(f"Saved remote POPE result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v4",
    pope_split: str = "adversarial",
    max_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = DEFAULT_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = None,
    prediction_samples: int = 0,
    output: str = "pope_checkpoint_eval.json",
    remote_output_path: str = None,
    train_sources: str = "",
    eval_schema_version: str = "v6",
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    background: bool = False,
):
    call = evaluate_pope.spawn(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        pope_split=pope_split,
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
        print(f"Spawned background POPE eval: {call}")
        print(f"Remote output path: {remote_output_path or '(none)'}")
        return
    result = call.get()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
