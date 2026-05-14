#!/usr/bin/env python3
"""Inspect a tensor inside Modal checkpoint projector files."""

from __future__ import annotations

import json
import os
from pathlib import PurePosixPath
from typing import Any, Optional

import modal


app = modal.App("anymal-inspect-checkpoint-tensor")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _volume_path(path: str) -> str:
    normalized = "/" + str(path).lstrip("/")
    if normalized.startswith("/checkpoints/"):
        normalized = normalized[len("/checkpoints") :]
    return str(PurePosixPath("/checkpoints") / normalized.lstrip("/"))


def _load_projector_tensor(checkpoint: str, tensor_key: str):
    import torch

    projector_path = os.path.join(_volume_path(checkpoint), "projector.pt")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector.pt: {projector_path}")
    try:
        state = torch.load(
            projector_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(projector_path, map_location="cpu")
    return state.get(tensor_key)


def _tensor_stats(tensor) -> dict[str, Any]:
    import torch

    if tensor is None:
        return {"present": False}
    t = tensor.detach().cpu().float()
    stats: dict[str, Any] = {
        "present": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "abs_mean": float(t.abs().mean().item()),
        "max_abs": float(t.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(t).item()),
        "nonzero": int((t != 0).sum().item()),
    }
    exp_t = torch.exp(t)
    stats["exp"] = {
        "mean": float(exp_t.mean().item()),
        "std": float(exp_t.std(unbiased=False).item()) if exp_t.numel() > 1 else 0.0,
        "min": float(exp_t.min().item()),
        "max": float(exp_t.max().item()),
    }
    return stats


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def inspect_tensors(
    checkpoints: list[str],
    tensor_key: str = "trainable_output_log_scale",
    baseline_checkpoint: Optional[str] = None,
) -> list[dict[str, Any]]:
    baseline = (
        _load_projector_tensor(baseline_checkpoint, tensor_key)
        if baseline_checkpoint
        else None
    )
    baseline_float = baseline.detach().cpu().float() if baseline is not None else None

    results = []
    for checkpoint in checkpoints:
        tensor = _load_projector_tensor(checkpoint, tensor_key)
        item: dict[str, Any] = {
            "checkpoint": checkpoint,
            "tensor_key": tensor_key,
            "stats": _tensor_stats(tensor),
        }
        if tensor is not None and baseline_float is not None:
            current = tensor.detach().cpu().float()
            if list(current.shape) == list(baseline_float.shape):
                delta = current - baseline_float
                item["delta_from_baseline"] = _tensor_stats(delta)
            else:
                item["delta_from_baseline"] = {
                    "present": False,
                    "reason": "shape_mismatch",
                    "baseline_shape": list(baseline_float.shape),
                    "current_shape": list(current.shape),
                }
        results.append(item)
    return results


@app.local_entrypoint()
def main(
    checkpoints: str,
    tensor_key: str = "trainable_output_log_scale",
    baseline_checkpoint: Optional[str] = None,
) -> None:
    checkpoint_list = [item.strip() for item in checkpoints.split(",") if item.strip()]
    results = inspect_tensors.remote(
        checkpoints=checkpoint_list,
        tensor_key=tensor_key,
        baseline_checkpoint=baseline_checkpoint,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
