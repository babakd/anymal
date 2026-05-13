#!/usr/bin/env python3
"""Materialize an interpolated V10 connector checkpoint on Modal.

Only ``projector.pt`` is interpolated. Decoder weights are not touched, and the
output is an eval/inference checkpoint with ``projector.pt`` plus
``model_meta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v10-materialize-projector-interpolation")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


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


def _compatible_meta(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> None:
    keys = (
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
    mismatches = []
    for key in keys:
        if meta_a.get(key) != meta_b.get(key):
            mismatches.append((key, meta_a.get(key), meta_b.get(key)))
    if mismatches:
        rendered = ", ".join(f"{key}: {left!r} != {right!r}" for key, left, right in mismatches)
        raise ValueError(f"Checkpoint metadata is not interpolation-compatible: {rendered}")


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=20 * 60)
def materialize_interpolation_remote(
    source_checkpoint_a: str,
    source_checkpoint_b: str,
    output_checkpoint: str,
    alpha: float,
    connector_output_scale: float | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    src_a = Path(source_checkpoint_a)
    src_b = Path(source_checkpoint_b)
    dst = Path(output_checkpoint)
    _require_checkpoint(src_a)
    _require_checkpoint(src_b)
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    meta_a = _load_meta(src_a)
    meta_b = _load_meta(src_b)
    _compatible_meta(meta_a, meta_b)

    state_a = torch.load(src_a / "projector.pt", map_location="cpu")
    state_b = torch.load(src_b / "projector.pt", map_location="cpu")
    if set(state_a.keys()) != set(state_b.keys()):
        missing_a = sorted(set(state_b) - set(state_a))
        missing_b = sorted(set(state_a) - set(state_b))
        raise ValueError(f"Projector keys differ; missing_a={missing_a}, missing_b={missing_b}")

    alpha_value = float(alpha)
    interp = {}
    for key in sorted(state_a.keys()):
        value_a = state_a[key]
        value_b = state_b[key]
        if value_a.shape != value_b.shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(value_a.shape)} != {tuple(value_b.shape)}")
        if value_a.dtype.is_floating_point:
            interp[key] = value_a.lerp(value_b.to(dtype=value_a.dtype), alpha_value)
        else:
            if not torch.equal(value_a, value_b):
                raise ValueError(f"Non-floating projector tensor differs for {key}")
            interp[key] = value_a

    if dst.exists():
        for child in dst.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                raise RuntimeError(f"Unexpected directory in output checkpoint: {child}")
    else:
        dst.mkdir(parents=True, exist_ok=True)
    torch.save(interp, dst / "projector.pt")

    output_meta = dict(meta_b)
    output_scale = (
        float(connector_output_scale)
        if connector_output_scale is not None
        else float(output_meta.get("connector_output_scale", 1.0))
    )
    output_meta.update(
        {
            "connector_output_scale": output_scale,
            "derived_from_checkpoint": str(src_b),
            "interpolation_source_checkpoint_a": str(src_a),
            "interpolation_source_checkpoint_b": str(src_b),
            "interpolation_alpha": alpha_value,
            "interpolation_formula": "(1 - alpha) * projector_A + alpha * projector_B",
            "v10_candidate_type": "projector_interpolation",
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
        "files": sorted(path.name for path in dst.iterdir()),
    }


@app.local_entrypoint()
def main(
    source_checkpoint_a: str,
    source_checkpoint_b: str,
    output_checkpoint: str,
    alpha: float,
    connector_output_scale: float | None = None,
) -> None:
    extra_meta = {
        "v10_note": "projector-only interpolation candidate; decoder weights unchanged",
    }
    result = materialize_interpolation_remote.remote(
        source_checkpoint_a,
        source_checkpoint_b,
        output_checkpoint,
        float(alpha),
        connector_output_scale,
        extra_meta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
