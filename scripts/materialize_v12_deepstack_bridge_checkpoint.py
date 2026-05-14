#!/usr/bin/env python3
"""Materialize V12 V3-to-V4 DeepStack bridge checkpoints on Modal.

The first bridge target is deliberately conservative: configure V4 DeepStack so
it can emulate the V11 V3 connector as closely as possible, then use that as a
generation/eval canary before enabling more visual feature levels.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import PurePosixPath
from typing import Optional

import modal


app = modal.App("anymal-v12-deepstack-bridge")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _display_path(path: str) -> str:
    return str(PurePosixPath(path))


def _parse_int_list(value: Optional[str], *, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    parts = [part.strip() for part in str(value).split(",")]
    result = [int(part) for part in parts if part]
    if not result:
        raise ValueError(f"Expected at least one integer in {value!r}")
    return result


def _copy_checkpoint_payload(source_checkpoint: str, dest_checkpoint: str) -> None:
    for name in os.listdir(source_checkpoint):
        if name in {"projector.pt", "model_meta.json", "trainer_state.pt"}:
            continue
        src = os.path.join(source_checkpoint, name)
        dst = os.path.join(dest_checkpoint, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def materialize_deepstack_bridge_checkpoint(
    source_checkpoint: str,
    dest_checkpoint: str,
    deepstack_hidden_state_indices: Optional[str] = None,
    num_feature_levels: Optional[int] = None,
    num_global_image_tokens: Optional[int] = None,
    num_local_image_tokens: int = 0,
    use_2d_position_features: bool = False,
    zero_type_embeddings: bool = True,
    zero_level_embeddings: bool = True,
    connector_output_scale: Optional[float] = None,
    overwrite: bool = False,
):
    import torch

    source_checkpoint = os.path.normpath(source_checkpoint)
    dest_checkpoint = os.path.normpath(dest_checkpoint)
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

    with open(meta_path, "r", encoding="utf-8") as f:
        source_meta = json.load(f)
    source_state = torch.load(projector_path, map_location="cpu")
    patch_position_scale = float(source_meta.get("patch_position_feature_scale") or 0.0)
    if patch_position_scale != 0.0:
        raise ValueError(
            "The conservative V3-to-V4 bridge drops V3 patch-position tensors, "
            "so it only supports source checkpoints with "
            f"patch_position_feature_scale=0.0; got {patch_position_scale}."
        )

    required = [
        "latents",
        "input_proj.weight",
        "input_proj.bias",
        "norm.weight",
        "norm.bias",
    ]
    missing_required = [key for key in required if key not in source_state]
    if missing_required:
        raise KeyError(f"Source V3 projector is missing required keys: {missing_required}")

    source_latents = source_state["latents"]
    source_tokens, connector_dim = source_latents.shape
    if num_global_image_tokens is None:
        num_global_image_tokens = int(source_tokens)
    num_global_image_tokens = int(num_global_image_tokens)
    num_local_image_tokens = int(num_local_image_tokens)
    if num_local_image_tokens < 0:
        raise ValueError("num_local_image_tokens must be >= 0")
    total_tokens = num_global_image_tokens + num_local_image_tokens
    if total_tokens != source_tokens:
        raise ValueError(
            "This conservative bridge preserves the V3 latent count exactly; "
            "num_global_image_tokens + num_local_image_tokens must equal "
            f"{source_tokens}, got {num_global_image_tokens} + "
            f"{num_local_image_tokens} = {total_tokens}."
        )

    default_layers = [-1]
    hidden_indices = _parse_int_list(
        deepstack_hidden_state_indices,
        default=default_layers,
    )
    if num_feature_levels is None:
        num_feature_levels = len(hidden_indices)
    num_feature_levels = int(num_feature_levels)
    if num_feature_levels <= 0:
        raise ValueError("num_feature_levels must be > 0")
    if len(hidden_indices) != num_feature_levels:
        raise ValueError(
            "deepstack_hidden_state_indices length must match num_feature_levels: "
            f"{hidden_indices} vs {num_feature_levels}"
        )

    bridge_state = {}
    for key, value in source_state.items():
        if key == "latents":
            continue
        if key.startswith("patch_position_"):
            continue
        if key.startswith("query_"):
            continue
        if key.startswith("v3_spatial_residual_branch."):
            continue
        if key.startswith("trainable_output_log_scale"):
            continue
        bridge_state[key] = value.detach().cpu().clone()

    source_latents_cpu = source_latents.detach().cpu()
    bridge_state["global_latents"] = source_latents_cpu[:num_global_image_tokens].clone()
    bridge_state["local_latents"] = source_latents_cpu[
        num_global_image_tokens:total_tokens
    ].clone()
    if zero_type_embeddings:
        bridge_state["type_embeddings"] = torch.zeros(
            2,
            connector_dim,
            dtype=source_latents.dtype,
        )
    else:
        bridge_state["type_embeddings"] = torch.randn(
            2,
            connector_dim,
            dtype=source_latents.dtype,
        ) * 0.02
    if zero_level_embeddings:
        bridge_state["level_embeddings"] = torch.zeros(
            num_feature_levels,
            connector_dim,
            dtype=source_latents.dtype,
        )
    else:
        bridge_state["level_embeddings"] = torch.randn(
            num_feature_levels,
            connector_dim,
            dtype=source_latents.dtype,
        ) * 0.02

    if use_2d_position_features:
        raise ValueError(
            "use_2d_position_features=True needs a position_mlp init policy; "
            "keep it false for the V11-emulation bridge."
        )

    resolved_connector_output_scale = (
        float(connector_output_scale)
        if connector_output_scale is not None
        else float(source_meta.get("connector_output_scale", 1.0))
    )
    bridge_meta = dict(source_meta)
    bridge_meta.update(
        {
            "architecture": "anymal_v4",
            "vision_image_size": int(source_meta.get("vision_image_size") or 384),
            "connector_type": "deepstack_spatial_perceiver_resampler",
            "num_global_image_tokens": num_global_image_tokens,
            "num_local_image_tokens": num_local_image_tokens,
            "num_image_tokens": total_tokens,
            "max_image_tokens": total_tokens,
            "min_image_tokens": total_tokens,
            "image_tokens": total_tokens,
            "image_placeholder_count": total_tokens,
            "connector_layers": int(source_meta.get("connector_layers", 6)),
            "connector_heads": int(source_meta.get("connector_heads", 16)),
            "connector_ff_mult": int(source_meta.get("connector_ff_mult", 4)),
            "connector_hidden_dim": int(connector_dim),
            "connector_output_scale": resolved_connector_output_scale,
            "connector_output_gate_init": source_meta.get("connector_output_gate_init"),
            "use_2d_position_features": bool(use_2d_position_features),
            "use_2d_patch_position_features": False,
            "project_directly_to_llm_dim": True,
            "vision_feature_strategy": "deepstack_lite",
            "vision_feature_layers": list(hidden_indices),
            "num_vision_feature_levels": num_feature_levels,
            "deepstack_num_feature_levels": num_feature_levels,
            "deepstack_hidden_state_indices": list(hidden_indices),
            "derived_from_checkpoint": source_checkpoint,
            "v12_deepstack_bridge": True,
            "v12_deepstack_bridge_source_architecture": source_meta.get("architecture"),
            "v12_deepstack_bridge_source_num_image_tokens": int(source_tokens),
            "v12_deepstack_bridge_latent_split": {
                "global": int(num_global_image_tokens),
                "local": int(num_local_image_tokens),
            },
            "v12_deepstack_bridge_zero_type_embeddings": bool(zero_type_embeddings),
            "v12_deepstack_bridge_zero_level_embeddings": bool(zero_level_embeddings),
            "v12_deepstack_bridge_source_connector_output_scale": float(
                source_meta.get("connector_output_scale", 1.0)
            ),
        }
    )
    if connector_output_scale is not None:
        bridge_meta["v12_deepstack_bridge_connector_output_scale_override"] = (
            resolved_connector_output_scale
        )

    os.makedirs(dest_checkpoint, exist_ok=True)
    _copy_checkpoint_payload(source_checkpoint, dest_checkpoint)
    torch.save(bridge_state, os.path.join(dest_checkpoint, "projector.pt"))
    with open(os.path.join(dest_checkpoint, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(bridge_meta, f, indent=2, sort_keys=True)
        f.write("\n")

    volume.commit()
    return {
        "source_checkpoint": _display_path(source_checkpoint),
        "dest_checkpoint": _display_path(dest_checkpoint),
        "source_tokens": int(source_tokens),
        "total_tokens": int(total_tokens),
        "connector_dim": int(connector_dim),
        "hidden_state_indices": hidden_indices,
        "num_feature_levels": num_feature_levels,
        "state_keys": sorted(bridge_state),
        "files": sorted(os.listdir(dest_checkpoint)),
    }


@app.local_entrypoint()
def main(
    source_checkpoint: str,
    dest_checkpoint: str,
    deepstack_hidden_state_indices: Optional[str] = None,
    num_feature_levels: Optional[int] = None,
    num_global_image_tokens: Optional[int] = None,
    num_local_image_tokens: int = 0,
    use_2d_position_features: bool = False,
    zero_type_embeddings: bool = True,
    zero_level_embeddings: bool = True,
    connector_output_scale: Optional[float] = None,
    overwrite: bool = False,
) -> None:
    result = materialize_deepstack_bridge_checkpoint.remote(
        source_checkpoint=source_checkpoint,
        dest_checkpoint=dest_checkpoint,
        deepstack_hidden_state_indices=deepstack_hidden_state_indices,
        num_feature_levels=num_feature_levels,
        num_global_image_tokens=num_global_image_tokens,
        num_local_image_tokens=num_local_image_tokens,
        use_2d_position_features=use_2d_position_features,
        zero_type_embeddings=zero_type_embeddings,
        zero_level_embeddings=zero_level_embeddings,
        connector_output_scale=connector_output_scale,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
