#!/usr/bin/env python3
"""Run a one-batch V12 visual cross-attention gradient proof on Modal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


app = modal.App("anymal-v12-vxattn-proof")
PROJECT_DIR = Path(__file__).resolve().parents[1]
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
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "datasets>=2.15.0",
        "requests>=2.31.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)
QWEN3_8B_BACKBONE = "Qwen/Qwen3-8B"


def _parse_layers(raw: str) -> list[int]:
    values = []
    for piece in str(raw or "").replace(" ", ",").split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return sorted(dict.fromkeys(values))


def _first_answer_index(labels) -> int:
    supervised = (labels[0] != -100).nonzero(as_tuple=False)
    if supervised.numel() == 0:
        return int(labels.shape[1])
    return int(supervised[0].item())


@app.function(
    image=image,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
)
def run_gradient_proof(
    base_checkpoint: str,
    output_path: str = "/checkpoints/v12_qwen/vxattn_gradient_proof.json",
    llm_backbone: str = QWEN3_8B_BACKBONE,
    layers: str = "12,18,24,30",
    adapter_dim: int = 512,
    gate_init: float = 1e-3,
    dataset: str = "v10_qwen_gqa_contrastive_stage1b",
    batch_size: int = 1,
    proof_lr: float = 1e-3,
    generation_probe_steps: int = 3,
) -> dict:
    import sys

    sys.path.insert(0, "/root/anymal")

    import torch

    from data import ImageTextCollator, build_dataloader
    from data.dataset_splitter import deterministic_train_val_split
    from model_metadata import read_model_metadata, validate_checkpoint_architecture
    from models import create_model_from_config
    from scripts.modal.train import _ensure_llm_backbone_cached, load_finetune_dataset

    base_checkpoint = os.path.normpath(base_checkpoint)
    if not os.path.isdir(base_checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {base_checkpoint}")
    validate_checkpoint_architecture(base_checkpoint, expected_architecture="anymal_v3")
    meta = read_model_metadata(base_checkpoint) or {}
    num_image_tokens = int(meta.get("num_image_tokens", 128))
    layer_list = _parse_layers(layers)
    if not layer_list:
        raise ValueError("At least one visual cross-attention layer is required")

    llm_path = _ensure_llm_backbone_cached(llm_backbone)
    model_cfg = {
        "model": {
            "architecture": "anymal_v3",
            "llm_model_name": llm_path,
            "llm_backbone": llm_backbone,
            "cache_dir": "/checkpoints/hf_cache",
            "use_qlora": False,
            "use_lora": False,
            "gradient_checkpointing": False,
            "use_flash_attention": False,
            "vision_encoder_type": meta.get("vision_encoder_type", "siglip2"),
            "vision_model_name": "google/siglip2-so400m-patch14-384",
            "connector_type": meta.get("connector_type", "perceiver_resampler"),
            "num_image_tokens": num_image_tokens,
            "connector_layers": int(meta.get("connector_layers", 6)),
            "connector_heads": int(meta.get("connector_heads", 16)),
            "connector_ff_mult": int(meta.get("connector_ff_mult", 4)),
            "connector_output_scale": float(meta.get("connector_output_scale", 1.0)),
            "connector_output_gate_init": meta.get("connector_output_gate_init"),
            "connector_trainable_scale_mode": meta.get(
                "connector_trainable_scale_mode",
                "none",
            ),
            "use_2d_patch_position_features": bool(
                meta.get("use_2d_patch_position_features", False)
            ),
            "patch_position_feature_type": meta.get("patch_position_feature_type"),
            "patch_position_grid_size": int(meta.get("patch_position_grid_size", 32)),
            "patch_position_mlp_hidden_dim": int(
                meta.get("patch_position_mlp_hidden_dim", 128)
            ),
            "patch_position_feature_scale": float(
                meta.get("patch_position_feature_scale", 1.0)
            ),
            "query_conditioned_visual_scale_mode": meta.get(
                "query_conditioned_visual_scale_mode",
                "none",
            ),
            "query_conditioned_visual_scale_min": float(
                meta.get("query_conditioned_visual_scale_min") or 0.95
            ),
            "query_conditioned_visual_scale_max": float(
                meta.get("query_conditioned_visual_scale_max") or 1.15
            ),
            "query_conditioned_visual_scale_init": meta.get(
                "query_conditioned_visual_scale_init"
            ),
            "query_conditioned_patch_selector_mode": meta.get(
                "query_conditioned_patch_selector_mode",
                "none",
            ),
            "query_conditioned_patch_selector_hidden_dim": int(
                meta.get("query_conditioned_patch_selector_hidden_dim", 256)
            ),
            "query_conditioned_patch_selector_max_residual": float(
                meta.get("query_conditioned_patch_selector_max_residual") or 0.25
            ),
            "query_conditioned_patch_selector_normalize_mean": bool(
                meta.get("query_conditioned_patch_selector_normalize_mean", True)
            ),
            "visual_cross_attention_mode": "gated",
            "visual_cross_attention_layers": layer_list,
            "visual_cross_attention_num_heads": 16,
            "visual_cross_attention_adapter_dim": int(adapter_dim),
            "visual_cross_attention_gate_init": float(gate_init),
            "visual_cross_attention_dropout": 0.0,
            "visual_cross_attention_freeze_connector": True,
            "project_directly_to_llm_dim": True,
        }
    }

    model = create_model_from_config(model_cfg, llm_device_map=None)
    projector_state = torch.load(
        os.path.join(base_checkpoint, "projector.pt"),
        map_location="cpu",
    )
    model.projector.load_state_dict(projector_state)
    model.load_visual_cross_attention_adapters(base_checkpoint, allow_missing=True)
    model.set_training_stage(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    ft_dataset = load_finetune_dataset(
        model.tokenizer,
        dataset=dataset,
        num_image_tokens=num_image_tokens,
        image_token_policy="fixed",
        min_image_tokens=num_image_tokens,
        max_image_tokens=num_image_tokens,
        image_size=384,
        max_length=512,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
    )
    train_dataset, _ = deterministic_train_val_split(ft_dataset, val_fraction=0.05)
    collator = ImageTextCollator(tokenizer=model.tokenizer, max_length=256)
    loader = build_dataloader(
        dataset=train_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        distributed=False,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    images = batch.get("images", batch.get("image"))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    answer_start = _first_answer_index(labels)
    prompt_input_ids = input_ids[:, :answer_start]
    prompt_attention_mask = attention_mask[:, :answer_start]

    def _forward_logits():
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        model.train()
        return outputs.logits.detach().float().cpu()

    def _generate_texts():
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                images=images,
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=4,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        texts = model.tokenizer.batch_decode(generated.detach().cpu(), skip_special_tokens=False)
        model.train()
        return texts

    logits_before = _forward_logits()
    generated_before = _generate_texts()

    trainable_named = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    adapter_named = [
        (name, param)
        for name, param in trainable_named
        if name.startswith("visual_cross_attention_adapters.")
    ]
    if not adapter_named:
        raise RuntimeError("No visual cross-attention adapter parameters are trainable")
    non_adapter_trainable = [
        name
        for name, _param in trainable_named
        if not name.startswith("visual_cross_attention_adapters.")
    ]

    before_params = {
        name: param.detach().float().cpu().clone()
        for name, param in adapter_named
    }
    optimizer = torch.optim.AdamW(
        [param for _name, param in adapter_named],
        lr=float(proof_lr),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=device.type == "cuda",
    ):
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
    loss.backward()

    grad_stats = {}
    gate_grad_norms = {}
    nonzero_grad_tensors = 0
    for name, param in adapter_named:
        if param.grad is None:
            grad_norm = 0.0
        else:
            grad_norm = float(param.grad.detach().float().norm().item())
        grad_stats[name] = grad_norm
        if grad_norm > 0:
            nonzero_grad_tensors += 1
        if name.endswith(".gate"):
            gate_grad_norms[name] = grad_norm

    optimizer.step()

    delta_stats = {}
    nonzero_delta_tensors = 0
    for name, param in adapter_named:
        delta = (param.detach().float().cpu() - before_params[name]).abs().max().item()
        delta_stats[name] = float(delta)
        if delta > 0:
            nonzero_delta_tensors += 1

    logits_after = _forward_logits()
    logits_diff = (logits_after - logits_before).abs()
    generated_after = _generate_texts()
    generated_changed = generated_after != generated_before

    extra_probe_steps = 0
    while not generated_changed and extra_probe_steps < int(generation_probe_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            probe_loss = outputs.loss
        probe_loss.backward()
        optimizer.step()
        extra_probe_steps += 1
        generated_after = _generate_texts()
        generated_changed = generated_after != generated_before

    result = {
        "base_checkpoint": base_checkpoint,
        "dataset": dataset,
        "num_image_tokens": num_image_tokens,
        "layers": layer_list,
        "adapter_dim": int(adapter_dim),
        "gate_init": float(gate_init),
        "proof_lr": float(proof_lr),
        "loss": float(loss.detach().float().item()),
        "trainable_tensors": len(trainable_named),
        "adapter_trainable_tensors": len(adapter_named),
        "non_adapter_trainable": non_adapter_trainable,
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "nonzero_delta_tensors": nonzero_delta_tensors,
        "gate_grad_norms": gate_grad_norms,
        "max_adapter_grad_norm": max(grad_stats.values()) if grad_stats else 0.0,
        "max_adapter_delta": max(delta_stats.values()) if delta_stats else 0.0,
        "logits_max_abs_diff_after_one_step": float(logits_diff.max().item()),
        "logits_mean_abs_diff_after_one_step": float(logits_diff.mean().item()),
        "generated_before": generated_before,
        "generated_after": generated_after,
        "generated_changed": bool(generated_changed),
        "extra_generation_probe_steps": extra_probe_steps,
    }
    result["passed"] = bool(
        not non_adapter_trainable
        and nonzero_grad_tensors > 0
        and any(value > 0 for value in gate_grad_norms.values())
        and nonzero_delta_tensors > 0
        and result["logits_max_abs_diff_after_one_step"] > 0
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    base_checkpoint: str,
    output_path: str = "/checkpoints/v12_qwen/vxattn_gradient_proof.json",
    llm_backbone: str = QWEN3_8B_BACKBONE,
    layers: str = "12,18,24,30",
    adapter_dim: int = 512,
    gate_init: float = 1e-3,
    dataset: str = "v10_qwen_gqa_contrastive_stage1b",
    batch_size: int = 1,
    proof_lr: float = 1e-3,
    generation_probe_steps: int = 3,
) -> None:
    result = run_gradient_proof.remote(
        base_checkpoint=base_checkpoint,
        output_path=output_path,
        llm_backbone=llm_backbone,
        layers=layers,
        adapter_dim=adapter_dim,
        gate_init=gate_init,
        dataset=dataset,
        batch_size=batch_size,
        proof_lr=proof_lr,
        generation_probe_steps=generation_probe_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
