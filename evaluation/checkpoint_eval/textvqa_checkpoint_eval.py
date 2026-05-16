"""TextVQA probe for AnyMAL checkpoints.

Builds a deterministic TextVQA slice in the repository's VQA-style layout and
reuses the existing VQA generation evaluator. TextVQA is scored with both exact
match and the common soft VQA-style score over the answer set.
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import modal

REMOTE_PROJECT_DIR = "/root/anymal"


def _resolve_project_dir() -> Path:
    path = Path(__file__).resolve()
    if len(path.parents) >= 3:
        return path.parents[2]
    remote_project = Path(REMOTE_PROJECT_DIR)
    if remote_project.exists():
        return remote_project
    cwd = Path.cwd()
    if (cwd / "models").exists() and (cwd / "evaluation").exists():
        return cwd
    return path.parent


PROJECT_DIR = _resolve_project_dir()
if os.path.exists(REMOTE_PROJECT_DIR) and REMOTE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, REMOTE_PROJECT_DIR)

from evaluation.checkpoint_eval.dataset_revisions import (
    pinned_revision as _pinned_revision,
    slice_fingerprint as _slice_fingerprint,
)
from evaluation.checkpoint_eval.paired_bootstrap import (
    binary_ci_metrics,
    mean_ci_metrics,
    paired_bootstrap_ci,
)


def _ignore_modal_mount(path: Path) -> bool:
    return ".git" in path.parts


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
    .add_local_dir(
        PROJECT_DIR,
        remote_path=REMOTE_PROJECT_DIR,
        copy=False,
        ignore=_ignore_modal_mount,
    )
)

app = modal.App("anymal-textvqa-checkpoint-eval")

CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
TEXTVQA_DIR = "/checkpoints/textvqa_data"
TEXTVQA_IMAGE_DIR = "/checkpoints/textvqa_images_hf"
HF_CACHE_DIR = "/checkpoints/hf_datasets"
HF_TEXTVQA_DATASET = "lmms-lab/textvqa"
TEXTVQA_SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)


def _vqa_layout_name(image_id: int) -> str:
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def _load_hf_dataset(dataset_name: str, split: str):
    from datasets import load_dataset

    revision = _pinned_revision(dataset_name, required=str(dataset_name) == HF_TEXTVQA_DATASET)
    try:
        return load_dataset(
            dataset_name,
            split=split,
            cache_dir=HF_CACHE_DIR,
            revision=revision,
        )
    except RuntimeError as exc:
        if str(dataset_name) != "facebook/textvqa" or "Dataset scripts" not in str(exc):
            raise
        fallback_revision = _pinned_revision(HF_TEXTVQA_DATASET)
        return load_dataset(
            HF_TEXTVQA_DATASET,
            split=split,
            cache_dir=HF_CACHE_DIR,
            revision=fallback_revision,
        )


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


def _labels_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("answers")
    if raw is None:
        raw = row.get("answer")
    if raw is None:
        raw = row.get("label")
    if isinstance(raw, str):
        labels = [raw]
    elif isinstance(raw, (list, tuple)):
        labels = [str(item) for item in raw if str(item).strip()]
    else:
        labels = [str(raw)] if raw is not None else []
    labels = [label.strip() for label in labels if label.strip()]
    return labels or [""]


def _question_from_row(row: dict[str, Any]) -> str:
    question = _first_nonempty_text([row.get("question"), row.get("query")])
    if not question:
        raise KeyError(f"TextVQA row has no question field; keys={sorted(row)}")
    return question


def _image_to_rgb(row: dict[str, Any]):
    from io import BytesIO

    import requests
    from PIL import Image

    image_value = row.get("image")
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
    for key in ("flickr_300k_url", "flickr_original_url", "image_url"):
        url = row.get(key)
        if url:
            response = requests.get(str(url), timeout=60)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
    raise TypeError(f"Unsupported TextVQA image value: {type(image_value)!r}")


def _answer_type(answer: str) -> str:
    text = str(answer or "").strip().lower()
    if text in {"yes", "no"}:
        return "yes/no"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text):
        return "number"
    return "other"


def _build_textvqa_vqa_files(
    *,
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int,
    image_dir: str,
) -> tuple[str, str, dict[str, Any]]:
    os.makedirs(TEXTVQA_DIR, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    resolved_split = "validation" if str(split).lower() == "val" else str(split)
    resolved_dataset_name = (
        HF_TEXTVQA_DATASET if str(dataset_name) == "facebook/textvqa" else str(dataset_name)
    )
    revision = _pinned_revision(
        resolved_dataset_name,
        required=resolved_dataset_name == HF_TEXTVQA_DATASET,
    )
    fingerprint = _slice_fingerprint(
        dataset_id=resolved_dataset_name,
        revision=revision or "",
        split=resolved_split,
        seed=int(seed),
        offset=0,
        max_samples=int(max_samples),
    )
    slice_dir = "/checkpoints/v17_slices"
    os.makedirs(slice_dir, exist_ok=True)
    slice_artifact = os.path.join(slice_dir, f"textvqa_{resolved_split}_{fingerprint}.json")
    if os.path.exists(slice_artifact):
        with open(slice_artifact, "r", encoding="utf-8") as f:
            payload = json.load(f)
        questions_path = payload.get("questions_path")
        annotations_path = payload.get("annotations_path")
        if questions_path and annotations_path and os.path.exists(questions_path) and os.path.exists(annotations_path):
            return (
                str(questions_path),
                str(annotations_path),
                {
                    "source_dataset": resolved_dataset_name,
                    "source_dataset_revision": revision,
                    "source_split": resolved_split,
                    "slice_fingerprint": fingerprint,
                    "slice_artifact": slice_artifact,
                    "split_definition_version": payload.get(
                        "split_definition_version",
                        "hf_textvqa_seeded_rows_v1",
                    ),
                    "selection_seed": int(seed),
                    "source_indices": payload.get("source_indices") or [],
                    "rows": int(payload.get("rows") or 0),
                    "skipped_image_rows": int(payload.get("skipped_image_rows") or 0),
                    "loaded_from_materialized_slice": True,
                },
            )
    dataset = _load_hf_dataset(dataset_name, resolved_split)
    indices = list(range(len(dataset)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if int(max_samples) > 0:
        indices = indices[: int(max_samples)]

    questions = []
    annotations = []
    source_indices = []
    skipped_image = 0
    for ordinal, source_index in enumerate(indices):
        row = dataset[int(source_index)]
        labels = _labels_from_row(row)
        answer = labels[0]
        image_id = 940_000_000 + int(ordinal)
        question_id = 940_000_000 + int(source_index)
        image_path = os.path.join(image_dir, _vqa_layout_name(image_id))
        if not os.path.exists(image_path):
            try:
                _image_to_rgb(row).save(image_path, format="JPEG", quality=95)
            except Exception as exc:
                skipped_image += 1
                if skipped_image <= 5:
                    print(
                        f"Skipping TextVQA row {source_index}; image load failed: {exc}",
                        flush=True,
                    )
                continue
        repeated_answers = (labels * 10)[:10]
        questions.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": _question_from_row(row),
            }
        )
        annotations.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "answer_type": _answer_type(answer),
                "question_type": "textvqa",
                "multiple_choice_answer": answer,
                "answers": [{"answer": item} for item in repeated_answers],
            }
        )
        source_indices.append(int(source_index))

    base = os.path.join(
        TEXTVQA_DIR,
        f"{dataset_name.replace('/', '_')}_{resolved_split}_seed{seed}_n{len(questions)}",
    )
    questions_path = f"{base}_questions.json"
    annotations_path = f"{base}_annotations.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f)
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": annotations}, f)
    with open(slice_artifact, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": resolved_dataset_name,
                "revision": revision,
                "split": resolved_split,
                "seed": int(seed),
                "offset": 0,
                "max_samples": int(max_samples),
                "fingerprint": fingerprint,
                "split_definition_version": "hf_textvqa_seeded_rows_v1",
                "questions_path": questions_path,
                "annotations_path": annotations_path,
                "rows": len(questions),
                "source_indices": source_indices,
                "skipped_image_rows": int(skipped_image),
            },
            f,
            indent=2,
        )
    volume.commit()
    return (
        questions_path,
        annotations_path,
        {
            "source_dataset": resolved_dataset_name,
            "source_dataset_revision": revision,
            "source_split": resolved_split,
            "slice_fingerprint": fingerprint,
            "slice_artifact": slice_artifact,
            "split_definition_version": "hf_textvqa_seeded_rows_v1",
            "selection_seed": int(seed),
            "source_indices": source_indices,
            "rows": len(questions),
            "skipped_image_rows": int(skipped_image),
            "loaded_from_materialized_slice": False,
        },
    )


def _norm_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.split("\n")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"[^\w\s]", "", text)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words).strip()


def _compute_textvqa_metrics(
    predictions: list[dict[str, Any]],
    ci_confidence: float = 0.95,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 12345,
) -> dict[str, Any]:
    total = len(predictions)
    exact = 0
    soft = 0.0
    exact_values: list[int] = []
    soft_values: list[float] = []
    by_type: dict[str, dict[str, float]] = {}
    for row in predictions:
        pred = _norm_answer(row.get("answer", ""))
        answers = [_norm_answer(answer) for answer in row.get("answers", [])]
        matches = sum(1 for answer in answers if pred and pred == answer)
        exact_ok = int(matches > 0)
        exact += exact_ok
        soft_score = min(1.0, matches / 3.0)
        soft += soft_score
        exact_values.append(exact_ok)
        soft_values.append(float(soft_score))
        row["textvqa_exact_match"] = bool(exact_ok)
        row["textvqa_soft_accuracy"] = float(soft_score)
        answer_type = str(row.get("answer_type") or "unknown").replace("/", "_")
        bucket = by_type.setdefault(answer_type, {"exact": 0.0, "soft": 0.0, "total": 0.0})
        bucket["exact"] += float(matches > 0)
        bucket["soft"] += soft_score
        bucket["total"] += 1.0
    metrics: dict[str, Any] = {
        "textvqa_exact_match": 100.0 * exact / total if total else 0.0,
        "textvqa_soft_accuracy": 100.0 * soft / total if total else 0.0,
        "textvqa_correct": exact,
        "textvqa_total": total,
    }
    metrics.update(
        binary_ci_metrics(
            "textvqa_exact_match",
            exact_values,
            seed=bootstrap_seed,
            n_resamples=bootstrap_resamples,
            confidence=ci_confidence,
        )
    )
    metrics.update(
        mean_ci_metrics(
            "textvqa_soft",
            soft_values,
            seed=bootstrap_seed,
            n_resamples=bootstrap_resamples,
            confidence=ci_confidence,
            include_binomial_when_binary=False,
        )
    )
    for key, bucket in sorted(by_type.items()):
        denom = bucket["total"] or 1.0
        metrics[f"textvqa_exact_match_{key}"] = 100.0 * bucket["exact"] / denom
        metrics[f"textvqa_soft_accuracy_{key}"] = 100.0 * bucket["soft"] / denom
        metrics[f"textvqa_num_samples_{key}"] = int(bucket["total"])
    return metrics


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_textvqa(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v3",
    dataset_name: str = HF_TEXTVQA_DATASET,
    split: str = "validation",
    max_samples: int = 200,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = TEXTVQA_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = TEXTVQA_SYSTEM_PROMPT,
    prediction_samples: int = 0,
    max_new_tokens: int = 32,
    remote_output_path: str | None = None,
    eval_schema_version: str = "v1",
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    connector_output_scale_override: float | None = None,
    vision_image_size_override: int | None = None,
):
    import torch

    sys.path.insert(0, REMOTE_PROJECT_DIR)
    from evaluation.checkpoint_eval.gqa_checkpoint_eval import _load_model
    from evaluation.checkpoint_eval.vqa_checkpoint_eval import _build_vqa_dataloader, _default_runs
    from evaluation.vqa_eval import VQAEvaluator

    questions, annotations, textvqa_meta = _build_textvqa_vqa_files(
        dataset_name=str(dataset_name),
        split=str(split),
        max_samples=int(max_samples),
        seed=int(seed),
        image_dir=str(image_dir),
    )
    effective_samples = int(textvqa_meta.get("rows") or max_samples)
    if effective_samples <= 0:
        raise RuntimeError("TextVQA slice produced no available image/question rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for run in _default_runs(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        include_baselines=False,
    ):
        print(f"Evaluating TextVQA {split}: {run['label']} from {run['checkpoint']}", flush=True)
        model, run_model_meta = _load_model(
            run,
            device,
            llm_backbone=llm_backbone,
            connector_output_scale_override=connector_output_scale_override,
        )
        vision_image_size = (
            int(vision_image_size_override)
            if vision_image_size_override is not None
            else int(run_model_meta.get("vision_image_size") or getattr(model, "vision_image_size", 384))
        )
        dataloader, dataset_meta, image_transform_meta = _build_vqa_dataloader(
            model=model,
            architecture=run["architecture"],
            questions=questions,
            annotations=annotations,
            image_dir=image_dir,
            max_samples=effective_samples,
            seed=int(seed),
            batch_size=int(batch_size),
            prompt_style=str(prompt_style),
            image_perturbation="none",
            system_prompt=system_prompt,
            image_size=vision_image_size,
        )
        evaluator = VQAEvaluator(model, device=device, max_new_tokens=int(max_new_tokens))
        prediction_output = f"/tmp/{run['key']}_textvqa_predictions.json"
        hygiene_metrics = evaluator.evaluate(dataloader, output_file=prediction_output)
        with open(prediction_output, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        metrics = {**hygiene_metrics, **_compute_textvqa_metrics(predictions)}
        result_entry = {
            **run,
            "candidate_checkpoint": run["checkpoint"],
            "candidate_architecture": run["architecture"],
            "model_meta": run_model_meta,
            "dataset_meta": {**dataset_meta, **textvqa_meta},
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
        "benchmark": "textvqa",
        "dataset_name": str(dataset_name),
        "split": str(split),
        "max_samples": int(max_samples),
        "effective_samples": int(effective_samples),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "image_dir": str(image_dir),
        "prompt_style": str(prompt_style),
        "system_prompt": system_prompt,
        "max_new_tokens": int(max_new_tokens),
        "runs": results,
    }
    if remote_output_path:
        os.makedirs(os.path.dirname(str(remote_output_path)), exist_ok=True)
        with open(str(remote_output_path), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        print(f"Saved remote TextVQA result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v3",
    dataset_name: str = HF_TEXTVQA_DATASET,
    split: str = "validation",
    max_samples: int = 200,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = TEXTVQA_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = TEXTVQA_SYSTEM_PROMPT,
    prediction_samples: int = 0,
    max_new_tokens: int = 32,
    output: str = "textvqa_checkpoint_eval.json",
    remote_output_path: str = None,
    eval_schema_version: str = "v1",
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    connector_output_scale_override: float = None,
    vision_image_size_override: int = None,
    background: bool = False,
):
    call = evaluate_textvqa.spawn(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
        prediction_samples=prediction_samples,
        max_new_tokens=max_new_tokens,
        remote_output_path=remote_output_path,
        eval_schema_version=eval_schema_version,
        llm_backbone=llm_backbone,
        connector_output_scale_override=connector_output_scale_override,
        vision_image_size_override=vision_image_size_override,
    )
    if background:
        print(f"Spawned background TextVQA eval: {call}")
        print(f"Remote output path: {remote_output_path or '(none)'}")
        return
    result = call.get()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
