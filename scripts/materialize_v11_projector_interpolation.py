#!/usr/bin/env python3
"""Materialize V11 projector interpolation checkpoints on Modal.

V11 needs to interpolate the V9 Qwen projector with the C1 learned-2D-position
projector. Those state dicts are intentionally not identical: C1 adds a learned
``patch_position_embedding``. For tensors that exist only in the C1 checkpoint,
this script treats the V9 side as zero, so ``alpha`` controls both the common
projector movement and the strength of the added 2D feature tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import modal


app = modal.App("anymal-v11-materialize-projector-interpolation")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


CORE_META_KEYS = (
    "architecture",
    "llm_backbone",
    "vision_encoder_type",
    "vision_tower",
    "connector_type",
    "num_image_tokens",
    "image_tokens",
    "connector_layers",
    "connector_heads",
    "connector_ff_mult",
    "project_directly_to_llm_dim",
)


def _require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    for filename in ("projector.pt", "model_meta.json"):
        if not (path / filename).exists():
            raise FileNotFoundError(f"Checkpoint missing {filename}: {path}")


def _load_meta(path: Path) -> dict[str, Any]:
    with open(path / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"Expected model_meta.json object: {path}")
    return meta


def _compatible_core_meta(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> None:
    mismatches = []
    for key in CORE_META_KEYS:
        if meta_a.get(key) != meta_b.get(key):
            mismatches.append((key, meta_a.get(key), meta_b.get(key)))
    if mismatches:
        rendered = ", ".join(f"{key}: {left!r} != {right!r}" for key, left, right in mismatches)
        raise ValueError(f"Checkpoint metadata is not V11-interpolation compatible: {rendered}")


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def materialize_interpolation_remote(
    source_checkpoint_a: str,
    source_checkpoint_b: str,
    output_checkpoint: str,
    alpha: float,
    connector_output_scale: Optional[float] = None,
    patch_position_feature_scale: Optional[float] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    import torch

    src_a = Path(source_checkpoint_a)
    src_b = Path(source_checkpoint_b)
    dst = Path(output_checkpoint)
    _require_checkpoint(src_a)
    _require_checkpoint(src_b)
    alpha_value = float(alpha)
    if not 0.0 <= alpha_value <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    meta_a = _load_meta(src_a)
    meta_b = _load_meta(src_b)
    _compatible_core_meta(meta_a, meta_b)

    state_a = torch.load(src_a / "projector.pt", map_location="cpu")
    state_b = torch.load(src_b / "projector.pt", map_location="cpu")
    keys_a = set(state_a)
    keys_b = set(state_b)
    source_only_keys = sorted(keys_a - keys_b)
    target_only_keys = sorted(keys_b - keys_a)
    if source_only_keys:
        raise ValueError(
            "Source checkpoint has tensors not present in target/C1 architecture: "
            f"{source_only_keys}"
        )

    interpolated = {}
    zero_initialized_target_only = []
    for key in sorted(keys_b):
        value_b = state_b[key]
        if key in state_a:
            value_a = state_a[key]
            if value_a.shape != value_b.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: {tuple(value_a.shape)} != {tuple(value_b.shape)}"
                )
            if value_a.dtype.is_floating_point:
                interpolated[key] = value_a.lerp(value_b.to(dtype=value_a.dtype), alpha_value)
            else:
                if not torch.equal(value_a, value_b):
                    raise ValueError(f"Non-floating projector tensor differs for {key}")
                interpolated[key] = value_a
        else:
            if not value_b.dtype.is_floating_point:
                raise ValueError(f"Target-only non-floating tensor is unsupported: {key}")
            interpolated[key] = value_b * alpha_value
            zero_initialized_target_only.append(key)

    if dst.exists():
        for child in dst.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                raise RuntimeError(f"Unexpected directory in output checkpoint: {child}")
    else:
        dst.mkdir(parents=True, exist_ok=True)
    torch.save(interpolated, dst / "projector.pt")

    output_meta = dict(meta_b)
    output_scale = (
        float(connector_output_scale)
        if connector_output_scale is not None
        else float(output_meta.get("connector_output_scale", 1.0))
    )
    output_meta.update(
        {
            "connector_output_scale": output_scale,
            "patch_position_feature_scale": (
                float(patch_position_feature_scale)
                if patch_position_feature_scale is not None
                else float(output_meta.get("patch_position_feature_scale", 1.0))
            ),
            "derived_from_checkpoint": str(src_b),
            "interpolation_source_checkpoint_a": str(src_a),
            "interpolation_source_checkpoint_b": str(src_b),
            "interpolation_alpha": alpha_value,
            "interpolation_formula": "(1 - alpha) * projector_A + alpha * projector_B; target-only tensors use alpha * tensor_B",
            "interpolation_target_only_zero_baseline_keys": zero_initialized_target_only,
            "v11_candidate_type": "v9_c1_projector_interpolation_with_2d_zero_baseline",
        }
    )
    if extra_meta:
        output_meta.update(extra_meta)
    with open(dst / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2, sort_keys=True)

    volume.commit()
    return {
        "output_checkpoint": str(dst),
        "alpha": alpha_value,
        "connector_output_scale": output_scale,
        "patch_position_feature_scale": output_meta["patch_position_feature_scale"],
        "source_only_keys": source_only_keys,
        "target_only_keys": target_only_keys,
        "zero_initialized_target_only_keys": zero_initialized_target_only,
        "files": sorted(path.name for path in dst.iterdir()),
    }


@app.local_entrypoint()
def main(
    source_checkpoint_a: str,
    source_checkpoint_b: str,
    output_checkpoint: str,
    alpha: float,
    connector_output_scale: Optional[float] = None,
    patch_position_feature_scale: Optional[float] = None,
) -> None:
    extra_meta = {
        "v11_note": "projector interpolation for V9-to-C1 spatial-feature salvage",
    }
    result = materialize_interpolation_remote.remote(
        source_checkpoint_a,
        source_checkpoint_b,
        output_checkpoint,
        float(alpha),
        connector_output_scale,
        patch_position_feature_scale,
        extra_meta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
