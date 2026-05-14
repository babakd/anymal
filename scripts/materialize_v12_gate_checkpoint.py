#!/usr/bin/env python3
"""Materialize V12 eval-time gate overrides as Modal checkpoints."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import PurePosixPath
from typing import Optional

import modal


app = modal.App("anymal-v12-gate-materializer")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _display_path(path: str) -> str:
    return str(PurePosixPath(path))


def _logit(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"sigmoid gate value must be in (0, 1), got {value}")
    return math.log(value / (1.0 - value))


def _prepare_dest(source_checkpoint: str, dest_checkpoint: str, overwrite: bool) -> None:
    if not os.path.isdir(source_checkpoint):
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")
    if os.path.exists(dest_checkpoint):
        if not overwrite:
            raise FileExistsError(
                f"Destination exists; pass overwrite=True to replace: {dest_checkpoint}"
            )
        shutil.rmtree(dest_checkpoint)
    os.makedirs(dest_checkpoint, exist_ok=True)


def _normalize_patch_position_feature_type(value: Optional[str]) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "none", "off", "false", "0"}:
        return "none"
    if text in {"learned", "learned_table", "table", "2d", "2d_table"}:
        return "learned_table"
    if text in {"coord", "coords", "coordinate", "coordinates", "coord_mlp", "coordinate_mlp"}:
        return "coord_mlp"
    raise ValueError(
        "patch_position_feature_type must be one of: none, learned_table, coord_mlp; "
        f"got {value!r}"
    )


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def inspect_gate_checkpoint(source_checkpoint: str):
    import torch

    source_checkpoint = os.path.normpath(source_checkpoint)
    if not os.path.isdir(source_checkpoint):
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")

    result = {"source_checkpoint": _display_path(source_checkpoint)}
    vx_path = os.path.join(source_checkpoint, "visual_cross_attention_adapters.pt")
    if os.path.exists(vx_path):
        vx_state = torch.load(vx_path, map_location="cpu")
        result["visual_cross_attention_gates"] = {
            key: float(value.detach().float().item())
            for key, value in vx_state.items()
            if key.endswith(".gate")
        }
    projector_path = os.path.join(source_checkpoint, "projector.pt")
    if os.path.exists(projector_path):
        projector_state = torch.load(projector_path, map_location="cpu")
        key = "v3_spatial_residual_branch.gate_logit"
        if key in projector_state:
            gate_logit = projector_state[key].detach().float()
            result["spatial_residual_gate"] = float(torch.sigmoid(gate_logit).item())
            result["spatial_residual_gate_logit"] = float(gate_logit.item())
        key = "output_gate_logit"
        if key in projector_state:
            gate_logit = projector_state[key].detach().float()
            result["connector_output_gate"] = float(torch.sigmoid(gate_logit).item())
            result["connector_output_gate_logit"] = float(gate_logit.item())
    return result


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def materialize_gate_checkpoint(
    source_checkpoint: str,
    dest_checkpoint: str,
    visual_cross_attention_gate: Optional[float] = None,
    visual_cross_attention_gate_multiplier: Optional[float] = None,
    spatial_residual_gate: Optional[float] = None,
    spatial_residual_gate_multiplier: Optional[float] = None,
    connector_output_scale: Optional[float] = None,
    patch_position_feature_scale: Optional[float] = None,
    patch_position_feature_type: Optional[str] = None,
    patch_position_mlp_hidden_dim: Optional[int] = None,
    overwrite: bool = False,
):
    import torch

    source_checkpoint = os.path.normpath(source_checkpoint)
    dest_checkpoint = os.path.normpath(dest_checkpoint)
    _prepare_dest(source_checkpoint, dest_checkpoint, bool(overwrite))

    meta_path = os.path.join(source_checkpoint, "model_meta.json")
    projector_path = os.path.join(source_checkpoint, "projector.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model_meta.json: {meta_path}")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector.pt: {projector_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    changed = {}
    projector_state = torch.load(projector_path, map_location="cpu")
    if spatial_residual_gate is not None and spatial_residual_gate_multiplier is not None:
        raise ValueError("Set either spatial_residual_gate or spatial_residual_gate_multiplier, not both")
    if spatial_residual_gate is not None:
        key = "v3_spatial_residual_branch.gate_logit"
        if key not in projector_state:
            raise KeyError(f"Projector state has no spatial residual gate key {key!r}")
        value = float(spatial_residual_gate)
        projector_state[key] = torch.tensor(
            _logit(value),
            dtype=projector_state[key].dtype,
        )
        meta["spatial_residual_gate_init"] = value
        meta["v12_materialized_spatial_residual_gate"] = value
        changed["spatial_residual_gate"] = value
    elif spatial_residual_gate_multiplier is not None:
        key = "v3_spatial_residual_branch.gate_logit"
        if key not in projector_state:
            raise KeyError(f"Projector state has no spatial residual gate key {key!r}")
        multiplier = float(spatial_residual_gate_multiplier)
        if multiplier < 0:
            raise ValueError("spatial_residual_gate_multiplier must be non-negative")
        source_gate = float(torch.sigmoid(projector_state[key].detach().float()).item())
        value = min(max(source_gate * multiplier, 1e-12), 1.0 - 1e-12)
        projector_state[key] = torch.tensor(
            _logit(value),
            dtype=projector_state[key].dtype,
        )
        meta["v12_materialized_spatial_residual_gate_multiplier"] = multiplier
        meta["v12_materialized_spatial_residual_gate_source"] = source_gate
        meta["v12_materialized_spatial_residual_gate"] = value
        changed["spatial_residual_gate_multiplier"] = multiplier
        changed["spatial_residual_gate_source"] = source_gate
        changed["spatial_residual_gate"] = value

    if patch_position_feature_type is not None:
        value = _normalize_patch_position_feature_type(patch_position_feature_type)
        removed_keys = []
        if value in {"none", "coord_mlp"}:
            for key in list(projector_state):
                if key == "patch_position_embedding":
                    removed_keys.append(key)
                    projector_state.pop(key)
        if value in {"none", "learned_table"}:
            for key in list(projector_state):
                if key.startswith("patch_position_mlp."):
                    removed_keys.append(key)
                    projector_state.pop(key)
        meta["patch_position_feature_type"] = value
        meta["use_2d_patch_position_features"] = value != "none"
        meta["v12_materialized_patch_position_feature_type"] = value
        changed["patch_position_feature_type"] = value
        if removed_keys:
            changed["removed_projector_keys"] = sorted(removed_keys)
    if patch_position_mlp_hidden_dim is not None:
        value = int(patch_position_mlp_hidden_dim)
        if value <= 0:
            raise ValueError("patch_position_mlp_hidden_dim must be positive")
        meta["patch_position_mlp_hidden_dim"] = value
        meta["v12_materialized_patch_position_mlp_hidden_dim"] = value
        changed["patch_position_mlp_hidden_dim"] = value
    torch.save(projector_state, os.path.join(dest_checkpoint, "projector.pt"))

    vx_path = os.path.join(source_checkpoint, "visual_cross_attention_adapters.pt")
    if os.path.exists(vx_path):
        vx_state = torch.load(vx_path, map_location="cpu")
        if visual_cross_attention_gate is not None and visual_cross_attention_gate_multiplier is not None:
            raise ValueError(
                "Set either visual_cross_attention_gate or "
                "visual_cross_attention_gate_multiplier, not both"
            )
        if visual_cross_attention_gate is not None:
            value = float(visual_cross_attention_gate)
            gate_keys = [key for key in vx_state if key.endswith(".gate")]
            if not gate_keys:
                raise KeyError("Visual cross-attention state has no *.gate keys")
            for key in gate_keys:
                vx_state[key] = torch.tensor(value, dtype=vx_state[key].dtype)
            meta["visual_cross_attention_gate_init"] = value
            meta["v12_materialized_visual_cross_attention_gate"] = value
            changed["visual_cross_attention_gate"] = value
            changed["visual_cross_attention_gate_keys"] = sorted(gate_keys)
        elif visual_cross_attention_gate_multiplier is not None:
            multiplier = float(visual_cross_attention_gate_multiplier)
            gate_keys = [key for key in vx_state if key.endswith(".gate")]
            if not gate_keys:
                raise KeyError("Visual cross-attention state has no *.gate keys")
            if multiplier < 0:
                raise ValueError("visual_cross_attention_gate_multiplier must be non-negative")
            source_gates = {}
            materialized_gates = {}
            for key in gate_keys:
                source_value = float(vx_state[key].detach().float().item())
                value = source_value * multiplier
                source_gates[key] = source_value
                materialized_gates[key] = value
                vx_state[key] = torch.tensor(value, dtype=vx_state[key].dtype)
            meta["v12_materialized_visual_cross_attention_gate_multiplier"] = multiplier
            meta["v12_materialized_visual_cross_attention_source_gates"] = source_gates
            meta["v12_materialized_visual_cross_attention_gates"] = materialized_gates
            changed["visual_cross_attention_gate_multiplier"] = multiplier
            changed["visual_cross_attention_source_gates"] = source_gates
            changed["visual_cross_attention_gates"] = materialized_gates
        torch.save(vx_state, os.path.join(dest_checkpoint, "visual_cross_attention_adapters.pt"))
    elif visual_cross_attention_gate is not None:
        raise FileNotFoundError(f"Missing visual cross-attention adapter state: {vx_path}")

    for name in os.listdir(source_checkpoint):
        if name in {
            "projector.pt",
            "model_meta.json",
            "visual_cross_attention_adapters.pt",
            "trainer_state.pt",
        }:
            continue
        src = os.path.join(source_checkpoint, name)
        dst = os.path.join(dest_checkpoint, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    if connector_output_scale is not None:
        value = float(connector_output_scale)
        meta["connector_output_scale"] = value
        meta["v12_materialized_connector_output_scale"] = value
        changed["connector_output_scale"] = value
    if patch_position_feature_scale is not None:
        value = float(patch_position_feature_scale)
        if value < 0.0:
            raise ValueError("patch_position_feature_scale must be non-negative")
        meta["patch_position_feature_scale"] = value
        meta["v12_materialized_patch_position_feature_scale"] = value
        changed["patch_position_feature_scale"] = value

    meta["derived_from_checkpoint"] = source_checkpoint
    meta["v12_materialized_gate_checkpoint"] = True
    with open(os.path.join(dest_checkpoint, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    volume.commit()
    return {
        "source_checkpoint": _display_path(source_checkpoint),
        "dest_checkpoint": _display_path(dest_checkpoint),
        "changed": changed,
        "files": sorted(os.listdir(dest_checkpoint)),
    }


@app.local_entrypoint()
def main(
    source_checkpoint: str,
    dest_checkpoint: str,
    visual_cross_attention_gate: Optional[float] = None,
    visual_cross_attention_gate_multiplier: Optional[float] = None,
    spatial_residual_gate: Optional[float] = None,
    spatial_residual_gate_multiplier: Optional[float] = None,
    connector_output_scale: Optional[float] = None,
    patch_position_feature_scale: Optional[float] = None,
    patch_position_feature_type: Optional[str] = None,
    patch_position_mlp_hidden_dim: Optional[int] = None,
    overwrite: bool = False,
    inspect_only: bool = False,
) -> None:
    if inspect_only:
        result = inspect_gate_checkpoint.remote(source_checkpoint=source_checkpoint)
    else:
        result = materialize_gate_checkpoint.remote(
            source_checkpoint=source_checkpoint,
            dest_checkpoint=dest_checkpoint,
            visual_cross_attention_gate=visual_cross_attention_gate,
            visual_cross_attention_gate_multiplier=visual_cross_attention_gate_multiplier,
            spatial_residual_gate=spatial_residual_gate,
            spatial_residual_gate_multiplier=spatial_residual_gate_multiplier,
            connector_output_scale=connector_output_scale,
            patch_position_feature_scale=patch_position_feature_scale,
            patch_position_feature_type=patch_position_feature_type,
            patch_position_mlp_hidden_dim=patch_position_mlp_hidden_dim,
            overwrite=overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
