"""ChartQA exact-match probe for AnyMAL checkpoints.

This is a lightweight expanded-benchmark harness for V13-style model selection.
It converts a deterministic Hugging Face ChartQA slice into the repository's
VQA-style question/annotation/image layout, then reuses the VQA generation path.
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
from evaluation.checkpoint_eval.paired_bootstrap import binary_ci_metrics, paired_bootstrap_ci


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

app = modal.App("anymal-chartqa-checkpoint-eval")

CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
CHARTQA_DIR = "/checkpoints/chartqa_data"
CHARTQA_IMAGE_DIR = "/checkpoints/chartqa_images_hf"
HF_CACHE_DIR = "/checkpoints/hf_datasets"
HF_CHARTQA_DATASET = "anhdang000/ChartQA-V2"
CHARTQA_SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)


def _vqa_layout_name(image_id: int) -> str:
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def _answer_type(answer: str) -> str:
    text = str(answer or "").strip().lower()
    if text in {"yes", "no"}:
        return "yes/no"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text):
        return "number"
    return "other"


def _labels_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("label")
    if raw is None:
        raw = row.get("answer")
    if raw is None:
        raw = row.get("answers")
    if isinstance(raw, str):
        labels = [raw]
    elif isinstance(raw, (list, tuple)):
        labels = [str(item) for item in raw if str(item).strip()]
    else:
        labels = [str(raw)]
    labels = [label.strip() for label in labels if label.strip()]
    return labels or [""]


def _query_from_row(row: dict[str, Any]) -> str:
    for key in ("query", "question", "problem"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"ChartQA row has no query/question field; keys={sorted(row)}")


def _image_from_row(row: dict[str, Any]):
    value = row.get("image")
    if value is None and row.get("images"):
        images = row.get("images")
        if isinstance(images, (list, tuple)) and images:
            value = images[0]
    if value is None:
        raise KeyError(f"ChartQA row has no image field; keys={sorted(row)}")
    return value


def _build_chartqa_vqa_files(
    *,
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int,
    image_dir: str,
) -> tuple[str, str, dict[str, Any]]:
    from datasets import load_dataset

    os.makedirs(CHARTQA_DIR, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    revision = _pinned_revision(dataset_name)
    fingerprint = _slice_fingerprint(
        dataset_id=dataset_name,
        revision=revision,
        split=split,
        seed=int(seed),
        offset=0,
        max_samples=int(max_samples),
    )
    slice_dir = "/checkpoints/v17_slices"
    os.makedirs(slice_dir, exist_ok=True)
    slice_artifact = os.path.join(slice_dir, f"chartqa_{split}_{fingerprint}.json")
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
                    "source_dataset": dataset_name,
                    "source_dataset_revision": revision,
                    "source_split": split,
                    "slice_fingerprint": fingerprint,
                    "slice_artifact": slice_artifact,
                    "split_definition_version": payload.get(
                        "split_definition_version",
                        "hf_chartqa_seeded_rows_v1",
                    ),
                    "selection_seed": int(seed),
                    "source_indices": payload.get("source_indices") or [],
                    "rows": int(payload.get("rows") or 0),
                    "loaded_from_materialized_slice": True,
                },
            )
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=HF_CACHE_DIR,
        revision=revision,
    )
    indices = list(range(len(dataset)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if int(max_samples) > 0:
        indices = indices[: int(max_samples)]

    questions = []
    annotations = []
    source_indices = []
    for ordinal, source_index in enumerate(indices):
        row = dataset[int(source_index)]
        labels = _labels_from_row(row)
        answer = labels[0]
        image_id = 930_000_000 + int(ordinal)
        question_id = 930_000_000 + int(source_index)
        image_path = os.path.join(image_dir, _vqa_layout_name(image_id))
        if not os.path.exists(image_path):
            image_obj = _image_from_row(row)
            if hasattr(image_obj, "convert"):
                image_obj.convert("RGB").save(image_path, format="JPEG", quality=95)
            elif isinstance(image_obj, str) and os.path.exists(image_obj):
                from PIL import Image

                Image.open(image_obj).convert("RGB").save(image_path, format="JPEG", quality=95)
            else:
                raise TypeError(f"Unsupported ChartQA image value: {type(image_obj)!r}")
        repeated_answers = (labels * 10)[:10]
        questions.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": _query_from_row(row),
            }
        )
        annotations.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "answer_type": _answer_type(answer),
                "question_type": f"chartqa_{row.get('human_or_machine', 'unknown')}",
                "multiple_choice_answer": answer,
                "answers": [{"answer": item} for item in repeated_answers],
            }
        )
        source_indices.append(int(source_index))

    base = os.path.join(
        CHARTQA_DIR,
        f"{dataset_name.replace('/', '_')}_{split}_seed{seed}_n{len(indices)}",
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
                "dataset_id": dataset_name,
                "revision": revision,
                "split": split,
                "seed": int(seed),
                "offset": 0,
                "max_samples": int(max_samples),
                "fingerprint": fingerprint,
                "split_definition_version": "hf_chartqa_seeded_rows_v1",
                "questions_path": questions_path,
                "annotations_path": annotations_path,
                "rows": len(indices),
                "source_indices": source_indices,
            },
            f,
            indent=2,
        )
    volume.commit()
    return (
        questions_path,
        annotations_path,
        {
            "source_dataset": dataset_name,
            "source_dataset_revision": revision,
            "source_split": split,
            "slice_fingerprint": fingerprint,
            "slice_artifact": slice_artifact,
            "split_definition_version": "hf_chartqa_seeded_rows_v1",
            "selection_seed": int(seed),
            "source_indices": source_indices,
            "rows": len(indices),
            "loaded_from_materialized_slice": False,
        },
    )


def _norm_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.split("\n")[0].split(".")[0]
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"[^\w\s]", "", text)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words).strip()


_CHARTQA_ASSISTANT_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r]+\s*", re.IGNORECASE)
_CHARTQA_THINKING_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_CHARTQA_THINKING_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)
_CHARTQA_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")


def _strip_chartqa_answer(text: str) -> str:
    text = str(text or "").strip()
    if _CHARTQA_THINKING_CLOSE_RE.search(text):
        text = _CHARTQA_THINKING_CLOSE_RE.split(text)[-1].strip()
    elif _CHARTQA_THINKING_TAG_RE.search(text):
        text = ""
    text = _CHARTQA_SPECIAL_TOKEN_RE.sub(" ", text)
    text = _CHARTQA_ASSISTANT_PREFIX_RE.sub("", text).strip().lower()
    text = text.split("\n", 1)[0].strip()
    for prefix in ("the answer is", "answer:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def _parse_chartqa_number(text: str) -> float | None:
    text = _strip_chartqa_answer(text)
    text = text.strip()
    text = re.sub(r"^[\$€£¥]\s*", "", text)
    text = text.replace(",", "").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    text = text.rstrip(".").strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _chartqa_text_norm(text: str) -> str:
    text = _strip_chartqa_answer(text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words).strip()


def chartqa_relaxed_match(pred: str, gold: str) -> bool:
    """Return ChartQA relaxed accuracy, with numerals in 5% relative tolerance.

    Spelled-out numerals such as "two" are intentionally not mapped to digits;
    that is a known limitation retained for compatibility with the V17 audit.
    """
    gold_num = _parse_chartqa_number(gold)
    if gold_num is not None:
        pred_num = _parse_chartqa_number(pred)
        if pred_num is None:
            return False
        epsilon = 1e-12
        relative_error = abs(pred_num - gold_num) / max(abs(gold_num), epsilon)
        return relative_error <= 0.05
    return _chartqa_text_norm(pred) == _chartqa_text_norm(gold)


def _compute_chartqa_metrics(
    predictions: list[dict[str, Any]],
    ci_confidence: float = 0.95,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 12345,
) -> dict[str, Any]:
    total = len(predictions)
    exact_correct = 0
    relaxed_correct = 0
    exact_values: list[int] = []
    relaxed_values: list[int] = []
    by_type: dict[str, dict[str, int]] = {}
    for row in predictions:
        pred = _norm_answer(row.get("answer", ""))
        answers = [_norm_answer(answer) for answer in row.get("answers", [])]
        exact_ok = bool(pred and pred in answers)
        relaxed_ok = any(
            chartqa_relaxed_match(row.get("raw_answer", row.get("answer", "")), answer)
            for answer in row.get("answers", [])
        )
        row["chartqa_exact_match"] = exact_ok
        row["chartqa_relaxed_match"] = relaxed_ok
        exact_correct += int(exact_ok)
        relaxed_correct += int(relaxed_ok)
        exact_values.append(int(exact_ok))
        relaxed_values.append(int(relaxed_ok))
        answer_type = str(row.get("answer_type") or "unknown").replace("/", "_")
        bucket = by_type.setdefault(answer_type, {"correct": 0, "total": 0})
        bucket["correct"] += int(relaxed_ok)
        bucket["total"] += 1
    metrics: dict[str, Any] = {
        "chartqa_relaxed_match": 100.0 * relaxed_correct / total if total else 0.0,
        "chartqa_exact_match": 100.0 * exact_correct / total if total else 0.0,
        "chartqa_correct": relaxed_correct,
        "chartqa_exact_correct": exact_correct,
        "chartqa_total": total,
    }
    metrics.update(
        binary_ci_metrics(
            "chartqa_relaxed_match",
            relaxed_values,
            seed=bootstrap_seed,
            n_resamples=bootstrap_resamples,
            confidence=ci_confidence,
        )
    )
    metrics.update(
        binary_ci_metrics(
            "chartqa_exact_match",
            exact_values,
            seed=bootstrap_seed,
            n_resamples=bootstrap_resamples,
            confidence=ci_confidence,
        )
    )
    for key, bucket in sorted(by_type.items()):
        metrics[f"chartqa_relaxed_match_{key}"] = (
            100.0 * bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        )
        metrics[f"chartqa_num_samples_{key}"] = bucket["total"]
    return metrics


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_chartqa(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v3",
    dataset_name: str = HF_CHARTQA_DATASET,
    split: str = "train",
    max_samples: int = 200,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = CHARTQA_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = CHARTQA_SYSTEM_PROMPT,
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

    questions, annotations, chartqa_meta = _build_chartqa_vqa_files(
        dataset_name=str(dataset_name),
        split=str(split),
        max_samples=int(max_samples),
        seed=int(seed),
        image_dir=str(image_dir),
    )
    effective_samples = int(chartqa_meta.get("rows") or max_samples)
    if effective_samples <= 0:
        raise RuntimeError("ChartQA slice produced no available image/question rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for run in _default_runs(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        include_baselines=False,
    ):
        print(f"Evaluating ChartQA {split}: {run['label']} from {run['checkpoint']}", flush=True)
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
        prediction_output = f"/tmp/{run['key']}_chartqa_predictions.json"
        hygiene_metrics = evaluator.evaluate(dataloader, output_file=prediction_output)
        with open(prediction_output, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        metrics = {**hygiene_metrics, **_compute_chartqa_metrics(predictions)}
        result_entry = {
            **run,
            "candidate_checkpoint": run["checkpoint"],
            "candidate_architecture": run["architecture"],
            "model_meta": run_model_meta,
            "dataset_meta": {**dataset_meta, **chartqa_meta},
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
        "benchmark": "chartqa",
        "dataset_name": str(dataset_name),
        "split": str(split),
        "max_samples": int(max_samples),
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
        print(f"Saved remote ChartQA result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    candidate_checkpoint: str,
    candidate_label: str,
    candidate_architecture: str = "v3",
    dataset_name: str = HF_CHARTQA_DATASET,
    split: str = "train",
    max_samples: int = 200,
    seed: int = 42,
    batch_size: int = 8,
    image_dir: str = CHARTQA_IMAGE_DIR,
    prompt_style: str = "training_chat",
    system_prompt: str = CHARTQA_SYSTEM_PROMPT,
    prediction_samples: int = 0,
    max_new_tokens: int = 32,
    output: str = "chartqa_checkpoint_eval.json",
    remote_output_path: str = None,
    eval_schema_version: str = "v1",
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    connector_output_scale_override: float = None,
    vision_image_size_override: int = None,
    background: bool = False,
):
    call = evaluate_chartqa.spawn(
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
        print(f"Spawned background ChartQA eval: {call}")
        print(f"Remote output path: {remote_output_path or '(none)'}")
        return
    result = call.get()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
