#!/usr/bin/env python3
"""Materialize an eval-time connector scale override as a Modal checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v9-materialize-scale-checkpoint")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=10 * 60)
def materialize_remote(
    source_checkpoint: str,
    output_checkpoint: str,
    connector_output_scale: float,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src = Path(source_checkpoint)
    dst = Path(output_checkpoint)
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {src}")
    if not (src / "projector.pt").exists():
        raise FileNotFoundError(f"Source checkpoint missing projector.pt: {src}")
    if not (src / "model_meta.json").exists():
        raise FileNotFoundError(f"Source checkpoint missing model_meta.json: {src}")

    if dst.exists():
        for child in dst.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                raise RuntimeError(f"Unexpected directory in output checkpoint: {child}")
    else:
        dst.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Only inference/eval artifacts are needed. Avoid copying trainer_state.pt;
    # optimizer state can be very large and is irrelevant to a materialized scale.
    shutil.copy2(src / "projector.pt", dst / "projector.pt")

    source_meta_path = src / "model_meta.json"
    meta_path = dst / "model_meta.json"
    with open(source_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["connector_output_scale"] = float(connector_output_scale)
    meta["derived_from_checkpoint"] = str(src)
    meta["materialized_from_eval_connector_output_scale_override"] = float(
        connector_output_scale
    )
    if extra_meta:
        meta.update(extra_meta)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    volume.commit()
    return {
        "output_checkpoint": str(dst),
        "connector_output_scale": meta["connector_output_scale"],
        "files": sorted(path.name for path in dst.iterdir()),
    }


@app.local_entrypoint()
def main(
    source_checkpoint: str,
    output_checkpoint: str,
    connector_output_scale: float = 1.05,
) -> None:
    extra_meta = {
        "v9_finalist_config": "stage1b2350_materialized_connector_scale_1.05",
        "training_dataset_mixture": (
            "v8 Stage1A v3_caption_alignment; v8 Stage1B v3_grounding; no Stage2"
        ),
        "train_sources": [
            "/checkpoints/llava_data/blip_laion_cc_sbu_558k.json",
            "/checkpoints/vqa_data/vqa_train2014_direct_150000.json",
            "/checkpoints/llava_data/coco_object_direct_train2017.json",
            "/checkpoints/llava_data/mix665k_direct_answer_filtered.json",
        ],
    }
    result = materialize_remote.remote(
        source_checkpoint,
        output_checkpoint,
        float(connector_output_scale),
        extra_meta,
    )
    print(json.dumps(result, indent=2))
