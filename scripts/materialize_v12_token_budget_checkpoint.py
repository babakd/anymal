#!/usr/bin/env python3
"""Materialize V12 V3 token-budget checkpoints on the Modal volume.

The V12 token-budget branches need a real checkpoint whose projector and
metadata agree on the larger visual-prefix length. This script expands the V3
Perceiver latent-query tensors from a source checkpoint, copies all compatible
files, and writes a checkpoint that can be used as ``--pretrain-checkpoint``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import PurePosixPath

import modal


app = modal.App("anymal-v12-token-budget")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _display_path(path: str) -> str:
    return str(PurePosixPath(path))


def _expand_first_dim_tensor(
    tensor,
    target_rows: int,
    *,
    method: str,
    noise_std: float,
    seed: int,
    add_noise: bool,
):
    import torch

    if tensor.ndim < 1:
        return tensor
    source_rows = int(tensor.shape[0])
    if source_rows == target_rows:
        return tensor
    if source_rows > target_rows:
        raise ValueError(
            f"Cannot shrink tensor first dimension from {source_rows} to {target_rows}"
        )

    extra_rows = target_rows - source_rows
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    method = str(method or "copy_noise").strip().lower().replace("-", "_")
    if method in {"copy", "copy_noise", "small_noise_copy", "small_noise_copies"}:
        indices = torch.arange(extra_rows) % source_rows
        extra = tensor.detach().cpu()[indices].clone()
    elif method in {"mean", "mean_noise"}:
        mean = tensor.detach().cpu().float().mean(dim=0, keepdim=True)
        extra = mean.expand(extra_rows, *tensor.shape[1:]).clone()
    else:
        raise ValueError(
            "init_method must be one of copy_noise, copy, mean_noise, or mean; "
            f"got {method!r}"
        )

    if add_noise and noise_std > 0:
        noise = torch.randn(extra.shape, generator=generator, dtype=torch.float32)
        extra = extra.float() + noise * float(noise_std)

    extra = extra.to(dtype=tensor.dtype)
    return torch.cat([tensor.detach().cpu(), extra], dim=0)


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    timeout=20 * 60,
)
def materialize_token_budget_checkpoint(
    source_checkpoint: str,
    dest_checkpoint: str,
    num_image_tokens: int,
    init_method: str = "copy_noise",
    noise_std: float = 0.002,
    seed: int = 1203,
    overwrite: bool = False,
):
    import torch

    source_checkpoint = os.path.normpath(source_checkpoint)
    dest_checkpoint = os.path.normpath(dest_checkpoint)
    num_image_tokens = int(num_image_tokens)
    if num_image_tokens <= 0:
        raise ValueError(f"num_image_tokens must be > 0, got {num_image_tokens}")
    if not os.path.isdir(source_checkpoint):
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")
    if os.path.exists(dest_checkpoint):
        if not overwrite:
            raise FileExistsError(
                f"Destination exists; pass overwrite=True to replace: {dest_checkpoint}"
            )
        shutil.rmtree(dest_checkpoint)

    meta_path = os.path.join(source_checkpoint, "model_meta.json")
    projector_path = os.path.join(source_checkpoint, "projector.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model_meta.json: {meta_path}")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector.pt: {projector_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    state = torch.load(projector_path, map_location="cpu")
    if "latents" not in state:
        raise RuntimeError("V3 projector state has no 'latents' tensor")
    source_tokens = int(state["latents"].shape[0])
    if num_image_tokens < source_tokens:
        raise ValueError(
            f"Target token count {num_image_tokens} is smaller than source "
            f"latent count {source_tokens}"
        )

    expanded = dict(state)
    expanded["latents"] = _expand_first_dim_tensor(
        state["latents"],
        num_image_tokens,
        method=init_method,
        noise_std=float(noise_std),
        seed=int(seed),
        add_noise=str(init_method).strip().lower() not in {"copy", "mean"},
    )

    scale_tensor = state.get("trainable_output_log_scale")
    if scale_tensor is not None and getattr(scale_tensor, "ndim", 0) >= 1:
        expanded["trainable_output_log_scale"] = _expand_first_dim_tensor(
            scale_tensor,
            num_image_tokens,
            method="copy",
            noise_std=0.0,
            seed=int(seed),
            add_noise=False,
        )

    os.makedirs(dest_checkpoint, exist_ok=True)
    for name in os.listdir(source_checkpoint):
        if name in {"projector.pt", "model_meta.json"}:
            continue
        src = os.path.join(source_checkpoint, name)
        dst = os.path.join(dest_checkpoint, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    torch.save(expanded, os.path.join(dest_checkpoint, "projector.pt"))

    meta.update(
        {
            "num_image_tokens": num_image_tokens,
            "max_image_tokens": num_image_tokens,
            "min_image_tokens": num_image_tokens,
            "image_tokens": num_image_tokens,
            "image_placeholder_count": num_image_tokens,
            "v12_token_budget_source_checkpoint": source_checkpoint,
            "v12_token_budget_source_num_image_tokens": source_tokens,
            "v12_token_budget_target_num_image_tokens": num_image_tokens,
            "v12_token_budget_extra_latents": num_image_tokens - source_tokens,
            "v12_token_budget_init_method": init_method,
            "v12_token_budget_noise_std": float(noise_std),
            "v12_token_budget_seed": int(seed),
        }
    )
    with open(os.path.join(dest_checkpoint, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    volume.commit()
    return {
        "source_checkpoint": _display_path(source_checkpoint),
        "dest_checkpoint": _display_path(dest_checkpoint),
        "source_num_image_tokens": source_tokens,
        "target_num_image_tokens": num_image_tokens,
        "expanded_keys": [
            key
            for key in ("latents", "trainable_output_log_scale")
            if key in expanded and key in state and expanded[key].shape != state[key].shape
        ],
    }


@app.local_entrypoint()
def main(
    source_checkpoint: str,
    dest_checkpoint: str,
    num_image_tokens: int,
    init_method: str = "copy_noise",
    noise_std: float = 0.002,
    seed: int = 1203,
    overwrite: bool = False,
) -> None:
    result = materialize_token_budget_checkpoint.remote(
        source_checkpoint=source_checkpoint,
        dest_checkpoint=dest_checkpoint,
        num_image_tokens=num_image_tokens,
        init_method=init_method,
        noise_std=noise_std,
        seed=seed,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
