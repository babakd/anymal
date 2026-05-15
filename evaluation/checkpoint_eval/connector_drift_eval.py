"""V16 fixed-probe connector drift evaluation.

Builds a deterministic VQAv2-val probe cache with V11 connector outputs, then
compares candidate checkpoints against that cache. The tensor payload is stored
as a `.pt` sidecar because the 64x128x4096 connector cache is too large for a
reasonable JSON artifact; the JSON manifest records the stable row identities
and points at the tensor payload.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import tempfile
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

app = modal.App("anymal-connector-drift-eval")

QWEN3_8B_BACKBONE = "Qwen/Qwen3-8B"
DEFAULT_TEACHER_CHECKPOINT = (
    "/checkpoints/pretrain-output/"
    "v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125"
)
DEFAULT_PROBE_MANIFEST = "/checkpoints/v16_qwen/drift_probe_set_v1.json"
DEFAULT_PROBE_TENSOR = "/checkpoints/v16_qwen/drift_probe_set_v1.pt"
VQA_SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_architecture(value: str) -> str:
    arch = str(value or "").strip().lower()
    aliases = {
        "anymal_v1": "v1",
        "anymal_v2": "v2",
        "anymal_v3": "v3",
        "anymal_v4": "v4",
    }
    return aliases.get(arch, arch)


def _load_run_model(
    *,
    checkpoint: str,
    label: str,
    architecture: str,
    device,
    llm_backbone: str,
):
    from evaluation.checkpoint_eval.gqa_checkpoint_eval import _load_model

    arch = _canonical_architecture(architecture)
    run = {
        "key": label.lower().replace(" ", "_").replace("/", "_"),
        "label": label,
        "architecture": arch,
        "checkpoint": checkpoint,
    }
    return _load_model(run, device, llm_backbone=llm_backbone)


def _build_probe_dataloader(
    *,
    model,
    architecture: str,
    max_samples: int,
    seed: int,
    batch_size: int,
    image_dir: str,
):
    from evaluation.checkpoint_eval.vqa_checkpoint_eval import (
        _build_vqa_dataloader,
        _require_vqa_files,
    )

    questions, annotations, resolved_image_dir = _require_vqa_files(
        min_images=int(max_samples),
        image_dir=image_dir,
        ensure_num_images=max(int(max_samples), 3000),
        image_sample_seed=int(seed),
    )
    return _build_vqa_dataloader(
        model=model,
        architecture=_canonical_architecture(architecture),
        questions=questions,
        annotations=annotations,
        image_dir=resolved_image_dir,
        max_samples=int(max_samples),
        seed=int(seed),
        batch_size=int(batch_size),
        prompt_style="training_chat",
        image_perturbation="none",
        system_prompt=VQA_SYSTEM_PROMPT,
        image_size=int(getattr(model, "vision_image_size", 384)),
    )


def _extract_connector_tokens(model, batch: dict[str, Any], device):
    import torch

    images = batch["image"].to(device)
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    input_ids = input_ids.to(device) if input_ids is not None else None
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    question_summary = None
    target_counts = None
    if (
        input_ids is not None
        and hasattr(model, "_uses_question_summary")
        and model._uses_question_summary()
    ):
        question_summary = model._build_question_summary(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
    if input_ids is not None and hasattr(model, "_extract_placeholder_counts"):
        target_counts = model._extract_placeholder_counts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strict=True,
        )
    with torch.no_grad():
        output = model.encode_images(
            images=images,
            target_num_tokens=target_counts,
            question_summary=question_summary,
        )
    return output[0] if isinstance(output, tuple) else output


def _rows_from_batch(batch: dict[str, Any], start_index: int) -> list[dict[str, Any]]:
    rows = []
    batch_size = len(batch["question"])
    for local_index in range(batch_size):
        answers = batch.get("answers", [[]])[local_index] or []
        rows.append(
            {
                "probe_index": int(start_index + local_index),
                "image_id": int(_scalar(batch["image_id"][local_index])),
                "source_image_id": int(_scalar(batch.get("source_image_id", batch["image_id"])[local_index])),
                "question_id": str(_scalar(batch["question_id"][local_index])),
                "question": str(batch["question"][local_index]),
                "answers": [str(answer) for answer in answers],
                "answer_type": str(batch.get("answer_type", [""])[local_index] or ""),
                "question_type": str(batch.get("question_type", [""])[local_index] or ""),
            }
        )
    return rows


def _first_answer(answers: list[Any]) -> str:
    for answer in answers or []:
        text = " ".join(str(answer or "").split())
        if text:
            return text
    return ""


def _make_answer_labeled_batch(model, batch: dict[str, Any], device, start_index: int):
    import torch
    import torch.nn.functional as F

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_rows = []
    mask_rows = []
    label_rows = []
    images = []
    kept_probe_indices = []
    batch_size = len(batch["question"])
    for local_index in range(batch_size):
        answer = _first_answer(batch.get("answers", [[]])[local_index])
        if not answer:
            continue
        prompt_ids = batch["input_ids"][local_index]
        prompt_mask = batch["attention_mask"][local_index].bool()
        prompt_ids = prompt_ids[prompt_mask].detach().cpu()
        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        if not answer_ids:
            continue
        answer_ids = torch.tensor(answer_ids, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        full_mask = torch.ones_like(full_ids)
        labels = torch.cat(
            [
                torch.full_like(prompt_ids, -100),
                answer_ids.clone(),
            ],
            dim=0,
        )
        input_rows.append(full_ids)
        mask_rows.append(full_mask)
        label_rows.append(labels)
        images.append(batch["image"][local_index])
        kept_probe_indices.append(int(start_index + local_index))
    if not input_rows:
        return None

    max_len = max(row.shape[0] for row in input_rows)
    input_ids = torch.stack(
        [F.pad(row, (max_len - row.shape[0], 0), value=pad_id) for row in input_rows]
    ).to(device)
    attention_mask = torch.stack(
        [F.pad(row, (max_len - row.shape[0], 0), value=0) for row in mask_rows]
    ).to(device)
    labels = torch.stack(
        [F.pad(row, (max_len - row.shape[0], 0), value=-100) for row in label_rows]
    ).to(device)
    image_tensor = torch.stack(images).to(device)
    return image_tensor, input_ids, attention_mask, labels, kept_probe_indices


def _collect_answer_logits(model, dataloader, device):
    import torch

    logit_chunks = []
    token_chunks = []
    probe_index_chunks = []
    offset = 0
    for batch in dataloader:
        if batch is None:
            continue
        batch_size = len(batch["question"])
        labeled = _make_answer_labeled_batch(model, batch, device, offset)
        offset += batch_size
        if labeled is None:
            continue
        images, input_ids, attention_mask, labels, probe_indices = labeled
        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                return_dict=True,
            )
            logits = outputs.logits
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            answer_mask = shifted_labels != -100
            selected_logits = shifted_logits[answer_mask].detach().to("cpu", dtype=torch.float16)
            selected_tokens = shifted_labels[answer_mask].detach().to("cpu", dtype=torch.long)
            per_position_indices = torch.tensor(
                probe_indices,
                dtype=torch.long,
                device=shifted_labels.device,
            ).unsqueeze(1).expand_as(shifted_labels)
            selected_probe_indices = per_position_indices[answer_mask].detach().cpu()
        if selected_logits.numel():
            logit_chunks.append(selected_logits)
            token_chunks.append(selected_tokens)
            probe_index_chunks.append(selected_probe_indices)
    if not logit_chunks:
        return {
            "answer_logits": None,
            "answer_token_ids": None,
            "answer_probe_indices": None,
        }
    return {
        "answer_logits": torch.cat(logit_chunks, dim=0),
        "answer_token_ids": torch.cat(token_chunks, dim=0),
        "answer_probe_indices": torch.cat(probe_index_chunks, dim=0),
    }


def _generate_probe_predictions(model, dataloader, device, max_new_tokens: int):
    from evaluation.vqa_eval import VQAEvaluator

    evaluator = VQAEvaluator(model, device=device, max_new_tokens=int(max_new_tokens))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name
    try:
        metrics = evaluator.evaluate(dataloader, output_file=output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
    return metrics, predictions


def _tensor_summary(values) -> dict[str, float]:
    import torch

    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _compute_drift_metrics(candidate_tokens, teacher_tokens) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    candidate = candidate_tokens.detach().float().cpu()
    teacher = teacher_tokens.detach().float().cpu()
    if tuple(candidate.shape) != tuple(teacher.shape):
        raise ValueError(
            "Candidate connector output shape does not match probe cache: "
            f"candidate={tuple(candidate.shape)} teacher={tuple(teacher.shape)}"
        )
    cand_flat = candidate.reshape(candidate.shape[0], -1)
    teacher_flat = teacher.reshape(teacher.shape[0], -1)
    cand_rms = cand_flat.pow(2).mean(dim=1).sqrt()
    teacher_rms = teacher_flat.pow(2).mean(dim=1).sqrt()
    mse = (cand_flat - teacher_flat).pow(2).mean(dim=1)
    cosine = F.cosine_similarity(cand_flat, teacher_flat, dim=1)
    return {
        "connector_output_rms": _tensor_summary(cand_rms),
        "teacher_connector_output_rms": _tensor_summary(teacher_rms),
        "connector_output_mse_to_v11": _tensor_summary(mse),
        "connector_output_cosine_to_v11": _tensor_summary(cosine),
    }


def _compute_answer_kl(candidate_logits, teacher_logits):
    import torch
    import torch.nn.functional as F

    if candidate_logits is None or teacher_logits is None:
        return {"answer_token_kl_to_v11": None, "answer_token_kl_tokens": 0}
    if tuple(candidate_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "Candidate answer logits shape does not match probe cache: "
            f"candidate={tuple(candidate_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits.float(), dim=-1)
    per_token = torch.exp(teacher_log_probs) * (teacher_log_probs - candidate_log_probs)
    per_token = per_token.sum(dim=-1)
    return {
        "answer_token_kl_to_v11": float(per_token.mean().item()) if per_token.numel() else None,
        "answer_token_kl_tokens": int(per_token.numel()),
        "answer_token_kl_to_v11_std": (
            float(per_token.std(unbiased=False).item()) if per_token.numel() > 1 else 0.0
        ),
    }


def _answer_agreement(candidate_predictions: list[dict[str, Any]], teacher_predictions: list[dict[str, Any]]):
    total = min(len(candidate_predictions), len(teacher_predictions))
    if total <= 0:
        return {"student_v11_exact_answer_agreement": None, "agreement_total": 0}
    exact = 0
    strict = 0
    mismatches = []
    for idx in range(total):
        candidate = candidate_predictions[idx]
        teacher = teacher_predictions[idx]
        cand_answer = str(candidate.get("answer", ""))
        teacher_answer = str(teacher.get("answer", ""))
        cand_strict = str(candidate.get("strict_answer", ""))
        teacher_strict = str(teacher.get("strict_answer", ""))
        exact += int(cand_answer == teacher_answer)
        strict += int(cand_strict == teacher_strict)
        if cand_answer != teacher_answer and len(mismatches) < 10:
            mismatches.append(
                {
                    "probe_index": idx,
                    "question_id": str(candidate.get("question_id", "")),
                    "question": str(candidate.get("question", "")),
                    "candidate_answer": cand_answer,
                    "v11_answer": teacher_answer,
                }
            )
    return {
        "student_v11_exact_answer_agreement": exact / total,
        "student_v11_strict_answer_agreement": strict / total,
        "agreement_total": total,
        "agreement_mismatch_samples": mismatches,
    }


def _build_probe_cache(
    *,
    teacher_checkpoint: str,
    teacher_architecture: str,
    max_samples: int,
    seed: int,
    batch_size: int,
    image_dir: str,
    probe_manifest_path: str,
    probe_tensor_path: str,
    llm_backbone: str,
    force: bool,
    max_new_tokens: int,
):
    import torch

    if (
        not force
        and os.path.exists(probe_manifest_path)
        and os.path.exists(probe_tensor_path)
    ):
        with open(probe_manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(probe_manifest_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_meta = _load_run_model(
        checkpoint=teacher_checkpoint,
        label="v11_drift_teacher",
        architecture=teacher_architecture,
        device=device,
        llm_backbone=llm_backbone,
    )

    dataloader, dataset_meta, image_transform_meta = _build_probe_dataloader(
        model=model,
        architecture=teacher_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
    )
    token_chunks = []
    rows = []
    offset = 0
    for batch in dataloader:
        if batch is None:
            continue
        rows.extend(_rows_from_batch(batch, offset))
        with torch.no_grad():
            tokens = _extract_connector_tokens(model, batch, device)
        token_chunks.append(tokens.detach().cpu().to(dtype=torch.float16))
        offset += len(batch["question"])
    teacher_tokens = torch.cat(token_chunks, dim=0)

    dataloader, _, _ = _build_probe_dataloader(
        model=model,
        architecture=teacher_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
    )
    teacher_hygiene, teacher_predictions = _generate_probe_predictions(
        model,
        dataloader,
        device,
        max_new_tokens=max_new_tokens,
    )

    dataloader, _, _ = _build_probe_dataloader(
        model=model,
        architecture=teacher_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=max(1, min(batch_size, 2)),
        image_dir=image_dir,
    )
    teacher_answer_payload = _collect_answer_logits(model, dataloader, device)

    payload = {
        "schema_version": "v16_connector_drift_probe_v1",
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_architecture": teacher_architecture,
        "teacher_model_meta": model_meta,
        "max_samples": int(max_samples),
        "seed": int(seed),
        "rows": rows,
        "dataset_meta": dataset_meta,
        "image_transform_meta": image_transform_meta,
        "teacher_connector_outputs": teacher_tokens,
        "teacher_predictions": teacher_predictions,
        "teacher_hygiene_metrics": teacher_hygiene,
        **teacher_answer_payload,
    }
    torch.save(payload, probe_tensor_path)
    manifest = {
        "schema_version": "v16_connector_drift_probe_manifest_v1",
        "tensor_path": probe_tensor_path,
        "tensor_sha256": _sha256(probe_tensor_path),
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_architecture": teacher_architecture,
        "max_samples": int(max_samples),
        "seed": int(seed),
        "rows": rows,
        "connector_output_shape": list(teacher_tokens.shape),
        "connector_output_dtype": str(teacher_tokens.dtype),
        "answer_token_kl_cache_tokens": (
            0
            if teacher_answer_payload["answer_token_ids"] is None
            else int(teacher_answer_payload["answer_token_ids"].numel())
        ),
        "teacher_hygiene_metrics": teacher_hygiene,
        "dataset_meta": dataset_meta,
        "image_transform_meta": image_transform_meta,
    }
    with open(probe_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_connector_drift(
    candidate_checkpoint: str = "",
    candidate_label: str = "",
    candidate_architecture: str = "v3",
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT,
    teacher_architecture: str = "v3",
    max_samples: int = 64,
    seed: int = 42,
    batch_size: int = 4,
    image_dir: str = "/checkpoints/coco_val2014",
    probe_manifest_path: str = DEFAULT_PROBE_MANIFEST,
    probe_tensor_path: str = DEFAULT_PROBE_TENSOR,
    remote_output_path: str | None = None,
    llm_backbone: str = QWEN3_8B_BACKBONE,
    force_rebuild_probe: bool = False,
    max_new_tokens: int = 16,
    build_only: bool = False,
):
    import torch

    manifest = _build_probe_cache(
        teacher_checkpoint=teacher_checkpoint,
        teacher_architecture=teacher_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
        probe_manifest_path=probe_manifest_path,
        probe_tensor_path=probe_tensor_path,
        llm_backbone=llm_backbone,
        force=force_rebuild_probe,
        max_new_tokens=max_new_tokens,
    )
    if build_only:
        return {"probe_manifest": manifest}
    if not candidate_checkpoint:
        raise ValueError("candidate_checkpoint is required unless build_only=True")

    probe = torch.load(probe_tensor_path, map_location="cpu", weights_only=False)
    teacher_tokens = probe["teacher_connector_outputs"]
    teacher_predictions = probe.get("teacher_predictions") or []
    teacher_logits = probe.get("answer_logits")
    teacher_token_ids = probe.get("answer_token_ids")
    teacher_probe_indices = probe.get("answer_probe_indices")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_meta = _load_run_model(
        checkpoint=candidate_checkpoint,
        label=candidate_label or "candidate",
        architecture=candidate_architecture,
        device=device,
        llm_backbone=llm_backbone,
    )
    dataloader, dataset_meta, image_transform_meta = _build_probe_dataloader(
        model=model,
        architecture=candidate_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
    )
    candidate_chunks = []
    for batch in dataloader:
        if batch is None:
            continue
        with torch.no_grad():
            candidate_chunks.append(
                _extract_connector_tokens(model, batch, device)
                .detach()
                .cpu()
                .to(dtype=torch.float16)
            )
    candidate_tokens = torch.cat(candidate_chunks, dim=0)
    drift_metrics = _compute_drift_metrics(candidate_tokens, teacher_tokens)

    dataloader, _, _ = _build_probe_dataloader(
        model=model,
        architecture=candidate_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
    )
    candidate_hygiene, candidate_predictions = _generate_probe_predictions(
        model,
        dataloader,
        device,
        max_new_tokens=max_new_tokens,
    )
    agreement_metrics = _answer_agreement(candidate_predictions, teacher_predictions)

    dataloader, _, _ = _build_probe_dataloader(
        model=model,
        architecture=candidate_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=max(1, min(batch_size, 2)),
        image_dir=image_dir,
    )
    candidate_answer_payload = _collect_answer_logits(model, dataloader, device)
    candidate_logits = candidate_answer_payload["answer_logits"]
    candidate_token_ids = candidate_answer_payload["answer_token_ids"]
    candidate_probe_indices = candidate_answer_payload["answer_probe_indices"]
    if teacher_token_ids is not None and candidate_token_ids is not None:
        if not torch.equal(teacher_token_ids, candidate_token_ids):
            raise ValueError("Candidate answer-token labels do not match the probe cache")
        if not torch.equal(teacher_probe_indices, candidate_probe_indices):
            raise ValueError("Candidate answer-token probe indices do not match the probe cache")
    kl_metrics = _compute_answer_kl(candidate_logits, teacher_logits)

    result = {
        "schema_version": "v16_connector_drift_eval_v1",
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_label": candidate_label,
        "candidate_architecture": candidate_architecture,
        "candidate_model_meta": model_meta,
        "probe_manifest_path": probe_manifest_path,
        "probe_tensor_path": probe_tensor_path,
        "probe_manifest": manifest,
        "dataset_meta": dataset_meta,
        "image_transform_meta": image_transform_meta,
        "metrics": {
            **drift_metrics,
            **agreement_metrics,
            **kl_metrics,
            "candidate_probe_hygiene": candidate_hygiene,
            "teacher_probe_hygiene": probe.get("teacher_hygiene_metrics", {}),
        },
        "prediction_samples": candidate_predictions[:10],
    }
    if remote_output_path:
        os.makedirs(os.path.dirname(str(remote_output_path)), exist_ok=True)
        with open(str(remote_output_path), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        print(f"Saved remote connector drift result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    candidate_checkpoint: str = "",
    candidate_label: str = "",
    candidate_architecture: str = "v3",
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT,
    teacher_architecture: str = "v3",
    max_samples: int = 64,
    seed: int = 42,
    batch_size: int = 4,
    image_dir: str = "/checkpoints/coco_val2014",
    probe_manifest_path: str = DEFAULT_PROBE_MANIFEST,
    probe_tensor_path: str = DEFAULT_PROBE_TENSOR,
    output: str = "connector_drift_eval.json",
    remote_output_path: str = None,
    llm_backbone: str = QWEN3_8B_BACKBONE,
    force_rebuild_probe: bool = False,
    max_new_tokens: int = 16,
    build_only: bool = False,
    background: bool = False,
):
    call = evaluate_connector_drift.spawn(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        teacher_checkpoint=teacher_checkpoint,
        teacher_architecture=teacher_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        image_dir=image_dir,
        probe_manifest_path=probe_manifest_path,
        probe_tensor_path=probe_tensor_path,
        remote_output_path=remote_output_path,
        llm_backbone=llm_backbone,
        force_rebuild_probe=force_rebuild_probe,
        max_new_tokens=max_new_tokens,
        build_only=build_only,
    )
    if background:
        print(f"Spawned background connector drift eval: {call}")
        print(f"Remote output path: {remote_output_path or '(none)'}")
        return
    result = call.get()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
