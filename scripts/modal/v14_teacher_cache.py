"""Build V14 cached V11 teacher answer-token distributions on Modal."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import modal


REMOTE_PROJECT_DIR = "/root/anymal"


def _resolve_project_dir() -> Path:
    current = Path(__file__).resolve()
    if len(current.parents) >= 3:
        return current.parents[2]
    cwd = Path.cwd()
    if (cwd / "models").exists() and (cwd / "training").exists():
        return cwd
    return current.parent


PROJECT_DIR = _resolve_project_dir()


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

app = modal.App("anymal-v14-teacher-cache")


def _selected_answer_rows(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    rows = []
    for row_logits, row_labels in zip(shift_logits, shift_labels):
        mask = row_labels != -100
        rows.append((row_logits[mask], row_labels[mask], mask.nonzero(as_tuple=True)[0]))
    return rows


def _teacher_kl_weights(batch: dict[str, Any], batch_size: int) -> list[float]:
    raw_weights = batch.get("teacher_kl_weight")
    if raw_weights is None:
        return [1.0] * int(batch_size)
    if hasattr(raw_weights, "detach"):
        values = raw_weights.detach().cpu().view(-1).tolist()
    else:
        values = list(raw_weights)
    if len(values) != int(batch_size):
        raise ValueError(
            f"teacher_kl_weight length mismatch: got {len(values)}, expected {batch_size}"
        )
    return [
        1.0 if value is None or value == "" else float(value)
        for value in values
    ]


def _sample_metadata_without_loading_image(dataset_obj: Any, idx: int) -> tuple[str, float, str]:
    """Return (sample_id, teacher_kl_weight, source) without calling __getitem__."""
    if hasattr(dataset_obj, "dataset") and hasattr(dataset_obj, "indices"):
        return _sample_metadata_without_loading_image(
            dataset_obj.dataset,
            int(dataset_obj.indices[int(idx)]),
        )

    if hasattr(dataset_obj, "strategy") and hasattr(dataset_obj, "datasets"):
        if dataset_obj.strategy == "balanced":
            source_idx = idx % len(dataset_obj.datasets)
            local_idx = (idx // len(dataset_obj.datasets)) % len(dataset_obj.datasets[source_idx])
        elif dataset_obj.strategy == "weighted":
            source_idx = dataset_obj._weighted_cycle[idx % len(dataset_obj._weighted_cycle)]
            stride = idx // len(dataset_obj._weighted_cycle)
            if dataset_obj._weighted_index_mode == "hash":
                local_idx = (
                    stride * 1_000_003
                    + (idx + 1) * 97_531
                    + (source_idx + 1) * 31_337
                ) % len(dataset_obj.datasets[source_idx])
            else:
                local_idx = stride % len(dataset_obj.datasets[source_idx])
        else:
            for source_idx, end in enumerate(dataset_obj._cumulative_lengths):
                start = 0 if source_idx == 0 else dataset_obj._cumulative_lengths[source_idx - 1]
                if idx < end:
                    local_idx = idx - start
                    break
            else:
                raise IndexError(idx)
        source_name = str(dataset_obj.source_names[source_idx])
        sample = dataset_obj.datasets[source_idx].samples[local_idx]
        base_id = sample.get("id", local_idx)
        return (
            f"{source_name}:{int(local_idx)}:{base_id}",
            float(sample.get("teacher_kl_weight") or 0.0),
            source_name,
        )

    sample = dataset_obj.samples[idx]
    return (
        str(sample.get("id", idx)),
        float(sample.get("teacher_kl_weight") or 0.0),
        str(sample.get("mixture_source") or ""),
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def build_v14_teacher_cache(
    *,
    dataset: str,
    output_path: str,
    teacher_checkpoint: str,
    llm_backbone: str = "Qwen/Qwen3-8B",
    teacher_image_tokens: int = 128,
    max_entries: int = 0,
    batch_size: int = 4,
    top_k: int = 128,
    split: str = "train",
    checkpoint_every_batches: int = 1000,
    resume: bool = True,
    seed_cache_path: str = "",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    sys.path.insert(0, REMOTE_PROJECT_DIR)
    from data import ImageTextCollator
    from data.dataset_splitter import deterministic_train_val_split
    from evaluation.checkpoint_eval.vqa_checkpoint_eval import (
        _ensure_eval_llm_path,
        _resolve_eval_llm_path,
    )
    from model_metadata import read_model_metadata
    from models.anymal_v3 import AnyMALv3
    from scripts.modal.train import load_finetune_dataset

    if int(top_k) <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if int(teacher_image_tokens) <= 0:
        raise ValueError(
            f"teacher_image_tokens must be > 0, got {teacher_image_tokens}"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    teacher_meta = read_model_metadata(teacher_checkpoint) or {}
    llm_path = _ensure_eval_llm_path(
        _resolve_eval_llm_path(teacher_meta, llm_backbone),
        model_meta=teacher_meta,
        llm_backbone=llm_backbone,
    )
    print(
        "Loading V11 teacher for cache: "
        f"checkpoint={teacher_checkpoint}, llm={llm_path}, top_k={top_k}",
        flush=True,
    )
    model = AnyMALv3.from_pretrained(
        teacher_checkpoint,
        llm_model_name=llm_path,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        vision_image_size=int(teacher_meta.get("vision_image_size") or 384),
        connector_type=teacher_meta.get("connector_type", "perceiver_resampler"),
        num_image_tokens=int(teacher_meta.get("num_image_tokens", teacher_image_tokens)),
        connector_layers=int(teacher_meta.get("connector_layers", 6)),
        connector_heads=int(teacher_meta.get("connector_heads", 16)),
        connector_ff_mult=int(teacher_meta.get("connector_ff_mult", 4)),
        connector_output_scale=float(teacher_meta.get("connector_output_scale", 1.0)),
        connector_output_gate_init=teacher_meta.get("connector_output_gate_init"),
        project_directly_to_llm_dim=bool(
            teacher_meta.get("project_directly_to_llm_dim", True)
        ),
        use_qlora=True,
        use_lora=False,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
        llm_device_map="auto",
        llm_torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.image_encoder.to(device)
    model.projector.to(device)
    if hasattr(model, "visual_cross_attention_adapters"):
        model.visual_cross_attention_adapters.to(device)
    model.eval()

    max_length = int(teacher_image_tokens) + 384
    dataset_obj = load_finetune_dataset(
        model.tokenizer,
        dataset=dataset,
        num_image_tokens=int(teacher_image_tokens),
        image_token_policy="fixed",
        min_image_tokens=int(teacher_image_tokens),
        max_image_tokens=int(teacher_image_tokens),
        image_size=int(teacher_meta.get("vision_image_size") or 384),
        max_length=max_length,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        image_view_mode="single",
    )
    split_key = str(split or "train").lower()
    if split_key in {"all", "full", "dataset"}:
        selected_dataset = dataset_obj
    else:
        train_dataset, val_dataset = deterministic_train_val_split(
            dataset_obj,
            val_fraction=0.05,
        )
        selected_dataset = val_dataset if split_key == "val" else train_dataset
    partial_path = output_path + ".partial.pt"
    entries = {}
    total_answer_tokens = 0
    total_remainder = 0.0
    processed_batches = 0
    skipped_kl_disabled = 0
    resumed_from_partial = False
    seeded_from_cache = ""
    if bool(resume) and os.path.exists(partial_path):
        partial = torch.load(partial_path, map_location="cpu")
        if partial.get("schema") != "v14_teacher_kl_topk_v1":
            raise RuntimeError(f"Unexpected teacher-cache partial schema in {partial_path}")
        if partial.get("dataset") != dataset or partial.get("split") != split:
            raise RuntimeError(
                "Teacher-cache partial does not match requested dataset/split: "
                f"{partial.get('dataset')}/{partial.get('split')} vs {dataset}/{split}"
            )
        entries = dict(partial.get("entries") or {})
        total_answer_tokens = int(partial.get("answer_tokens") or 0)
        total_remainder = float(partial.get("total_remainder_sum") or 0.0)
        processed_batches = int(partial.get("processed_batches") or 0)
        skipped_kl_disabled = int(partial.get("skipped_kl_disabled") or 0)
        resumed_from_partial = True
        print(
            "Resuming teacher cache partial: "
            f"path={partial_path}, processed_batches={processed_batches}, "
            f"entries={len(entries)}",
            flush=True,
        )
    elif seed_cache_path:
        seed = torch.load(seed_cache_path, map_location="cpu")
        if seed.get("schema") != "v14_teacher_kl_topk_v1":
            raise RuntimeError(f"Unexpected teacher-cache seed schema in {seed_cache_path}")
        seed_top_k = int(seed.get("top_k") or 0)
        if seed_top_k != int(top_k):
            raise RuntimeError(
                f"Teacher-cache seed top_k mismatch: seed={seed_top_k}, requested={top_k}"
            )
        seed_checkpoint = str(seed.get("teacher_checkpoint") or "")
        if seed_checkpoint and os.path.normpath(seed_checkpoint) != os.path.normpath(
            teacher_checkpoint
        ):
            raise RuntimeError(
                "Teacher-cache seed checkpoint mismatch: "
                f"seed={seed_checkpoint}, requested={teacher_checkpoint}"
            )
        entries = dict(seed.get("entries") or {})
        total_answer_tokens = int(seed.get("answer_tokens") or 0)
        total_remainder = float(seed.get("total_remainder_sum") or 0.0)
        skipped_kl_disabled = int(seed.get("skipped_kl_disabled") or 0)
        seeded_from_cache = str(seed_cache_path)
        print(
            "Seeding teacher cache build from existing cache: "
            f"path={seed_cache_path}, entries={len(entries)}",
            flush=True,
        )

    resume_start = int(processed_batches) * int(batch_size)
    if resume_start > 0:
        total_len = len(selected_dataset)
        if resume_start >= total_len:
            selected_dataset = Subset(selected_dataset, [])
        else:
            selected_dataset = Subset(
                selected_dataset,
                list(range(resume_start, total_len)),
            )

    collator = ImageTextCollator(tokenizer=model.tokenizer, max_length=max_length)
    loader = DataLoader(
        selected_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    def _save_cache(path: str, *, partial: bool) -> None:
        payload = {
            "schema": "v14_teacher_kl_topk_v1",
            "dataset": dataset,
            "split": split,
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_checkpoint_metadata": teacher_meta,
            "teacher_image_tokens": int(teacher_image_tokens),
            "top_k": int(top_k),
            "entries_count": len(entries),
            "answer_tokens": int(total_answer_tokens),
            "skipped_kl_disabled": int(skipped_kl_disabled),
            "processed_batches": int(processed_batches),
            "total_remainder_sum": float(total_remainder),
            "mean_remainder_prob_per_entry": (
                total_remainder / max(len(entries), 1)
            ),
            "prompt_template_metadata": {
                "chat_template_family": getattr(model, "chat_template_family", None),
                "image_placeholder_token": getattr(model, "image_placeholder_token", None),
                "image_placeholder_count": int(teacher_image_tokens),
            },
            "partial": bool(partial),
            "seeded_from_cache": seeded_from_cache,
            "entries": entries,
        }
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        volume.commit()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching V11 teacher"):
            processed_batches += 1
            if int(max_entries or 0) > 0 and len(entries) >= int(max_entries):
                break
            raw_sample_ids = [str(x) for x in batch["sample_id"]]
            raw_sample_weights = _teacher_kl_weights(batch, len(raw_sample_ids))
            active_indices = [
                idx for idx, value in enumerate(raw_sample_weights) if float(value) > 0.0
            ]
            skipped_kl_disabled += len(raw_sample_ids) - len(active_indices)
            active_indices = [
                idx for idx in active_indices if raw_sample_ids[idx] not in entries
            ]
            if not active_indices:
                if (
                    int(checkpoint_every_batches or 0) > 0
                    and processed_batches % int(checkpoint_every_batches) == 0
                ):
                    _save_cache(partial_path, partial=True)
                continue
            sample_ids = [raw_sample_ids[idx] for idx in active_indices]
            sample_weights = [raw_sample_weights[idx] for idx in active_indices]
            images = batch["images"][active_indices].to(device, non_blocking=True)
            input_ids = batch["input_ids"][active_indices].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"][active_indices].to(device, non_blocking=True)
            labels = batch["labels"][active_indices].to(device, non_blocking=True)
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )
            answer_rows = _selected_answer_rows(outputs.logits, labels)
            for row_idx, (answer_logits, answer_labels, positions) in enumerate(answer_rows):
                if int(max_entries or 0) > 0 and len(entries) >= int(max_entries):
                    break
                raw_idx = active_indices[row_idx]
                if answer_logits.numel() == 0:
                    continue
                probs = torch.softmax(answer_logits.float(), dim=-1)
                top_probs, top_ids = torch.topk(probs, k=int(top_k), dim=-1)
                remainder = (1.0 - top_probs.sum(dim=-1)).clamp_min(0.0)
                greedy_ids = top_ids[:, 0].detach().cpu()
                key = sample_ids[row_idx]
                entries[key] = {
                    "labels": answer_labels.detach().cpu().to(torch.long),
                    "answer_token_positions": positions.detach().cpu().to(torch.long),
                    "topk_ids": top_ids.detach().cpu().to(torch.int32),
                    "topk_probs": top_probs.detach().cpu().to(torch.float16),
                    "remainder_probs": remainder.detach().cpu().to(torch.float32),
                    "teacher_greedy_token_ids": greedy_ids.to(torch.int32),
                    "teacher_greedy_answer": model.tokenizer.decode(
                        greedy_ids.tolist(),
                        skip_special_tokens=True,
                    ).strip(),
                    "sample_id": key,
                    "image_ref": batch.get("image_ref", [""] * len(raw_sample_ids))[raw_idx],
                    "question": batch.get("question_text", [""] * len(raw_sample_ids))[raw_idx],
                    "answer": batch.get("answer_text", [""] * len(raw_sample_ids))[raw_idx],
                    "mixture_source": batch.get(
                        "mixture_source",
                        [""] * len(raw_sample_ids),
                    )[raw_idx],
                    "teacher_kl_weight": float(sample_weights[row_idx]),
                }
                total_answer_tokens += int(answer_labels.numel())
                total_remainder += float(remainder.detach().mean().item())
            if (
                int(checkpoint_every_batches or 0) > 0
                and processed_batches % int(checkpoint_every_batches) == 0
            ):
                _save_cache(partial_path, partial=True)

    _save_cache(output_path, partial=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)
        volume.commit()
    result = {
        "output_path": output_path,
        "entries": len(entries),
        "answer_tokens": int(total_answer_tokens),
        "top_k": int(top_k),
        "skipped_kl_disabled": int(skipped_kl_disabled),
        "processed_batches": int(processed_batches),
        "resumed_from_partial": bool(resumed_from_partial),
        "seeded_from_cache": seeded_from_cache,
    }
    print(result, flush=True)
    return result


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def audit_teacher_cache_coverage(
    *,
    dataset: str,
    cache_path: str,
    teacher_checkpoint: str,
    llm_backbone: str = "Qwen/Qwen3-8B",
    teacher_image_tokens: int = 128,
    split: str = "train",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import Subset
    from transformers import AutoTokenizer

    sys.path.insert(0, REMOTE_PROJECT_DIR)
    from data.dataset_splitter import deterministic_train_val_split
    from evaluation.checkpoint_eval.vqa_checkpoint_eval import (
        _ensure_eval_llm_path,
        _resolve_eval_llm_path,
    )
    from model_metadata import read_model_metadata
    from scripts.modal.train import load_finetune_dataset

    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("schema") != "v14_teacher_kl_topk_v1":
        raise RuntimeError(f"Unexpected teacher-cache schema in {cache_path}")
    cache_entries = payload.get("entries") or {}

    teacher_meta = read_model_metadata(teacher_checkpoint) or {}
    llm_path = _ensure_eval_llm_path(
        _resolve_eval_llm_path(teacher_meta, llm_backbone),
        model_meta=teacher_meta,
        llm_backbone=llm_backbone,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    max_length = int(teacher_image_tokens) + 384
    dataset_obj = load_finetune_dataset(
        tokenizer,
        dataset=dataset,
        num_image_tokens=int(teacher_image_tokens),
        image_token_policy="fixed",
        min_image_tokens=int(teacher_image_tokens),
        max_image_tokens=int(teacher_image_tokens),
        image_size=int(teacher_meta.get("vision_image_size") or 384),
        max_length=max_length,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        image_view_mode="single",
    )
    split_key = str(split or "train").lower()
    if split_key in {"all", "full", "dataset"}:
        selected_dataset = dataset_obj
    else:
        train_dataset, val_dataset = deterministic_train_val_split(
            dataset_obj,
            val_fraction=0.05,
        )
        selected_dataset = val_dataset if split_key == "val" else train_dataset

    if isinstance(selected_dataset, Subset):
        base_dataset = selected_dataset.dataset
        indices = list(selected_dataset.indices)
    else:
        base_dataset = selected_dataset
        indices = list(range(len(selected_dataset)))

    active_ids = set()
    active_by_source: dict[str, int] = {}
    missing = []
    for idx in indices:
        sample_id, weight, source = _sample_metadata_without_loading_image(
            base_dataset,
            int(idx),
        )
        if float(weight) <= 0.0:
            continue
        active_ids.add(sample_id)
        active_by_source[source] = active_by_source.get(source, 0) + 1
        if sample_id not in cache_entries:
            missing.append(sample_id)

    result = {
        "dataset": dataset,
        "split": split,
        "cache_path": cache_path,
        "cache_entries": len(cache_entries),
        "selected_rows": len(indices),
        "active_rows": sum(active_by_source.values()),
        "unique_active_ids": len(active_ids),
        "missing_active_ids": len(missing),
        "missing_preview": missing[:20],
        "active_by_source": active_by_source,
        "cache_extra_entries": max(0, len(set(cache_entries) - active_ids)),
    }
    print(result, flush=True)
    return result


@app.local_entrypoint()
def main(
    dataset: str = "v14_qwen_imitation_replay_stage1b",
    output_path: str = "/checkpoints/v14_qwen/v14_v11_teacher_topk128_train.pt",
    teacher_checkpoint: str = "/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125",
    llm_backbone: str = "Qwen/Qwen3-8B",
    teacher_image_tokens: int = 128,
    max_entries: int = 0,
    batch_size: int = 4,
    top_k: int = 128,
    split: str = "train",
    checkpoint_every_batches: int = 1000,
    resume: bool = True,
    seed_cache_path: str = "",
    audit_only: bool = False,
    cache_path: str = "",
):
    if audit_only:
        print(
            audit_teacher_cache_coverage.remote(
                dataset=dataset,
                cache_path=cache_path or output_path,
                teacher_checkpoint=teacher_checkpoint,
                llm_backbone=llm_backbone,
                teacher_image_tokens=teacher_image_tokens,
                split=split,
            )
        )
        return
    print(
        build_v14_teacher_cache.remote(
            dataset=dataset,
            output_path=output_path,
            teacher_checkpoint=teacher_checkpoint,
            llm_backbone=llm_backbone,
            teacher_image_tokens=teacher_image_tokens,
            max_entries=max_entries,
            batch_size=batch_size,
            top_k=top_k,
            split=split,
            checkpoint_every_batches=checkpoint_every_batches,
            resume=resume,
            seed_cache_path=seed_cache_path,
        )
    )
