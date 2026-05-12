#!/usr/bin/env python3
"""Copy a directory tree inside the AnyMAL Modal checkpoint volume."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import modal


app = modal.App("anymal-copy-volume-tree")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


def _volume_path(path: str) -> str:
    normalized = "/" + str(path).lstrip("/")
    checkpoint_prefix = "/checkpoints/"
    if normalized.startswith(checkpoint_prefix):
        normalized = normalized[len("/checkpoints") :]
    return str(Path("/checkpoints") / normalized.lstrip("/"))


@app.function(image=image, volumes={"/checkpoints": volume}, timeout=60 * 60)
def copy_tree_remote(src: str, dst: str, overwrite: bool = False) -> dict:
    src_path = _volume_path(src)
    dst_path = _volume_path(dst)
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"Source directory not found: {src_path}")
    if os.path.exists(dst_path):
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path}")
        shutil.rmtree(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copytree(src_path, dst_path)
    volume.commit()
    copied_files = sum(len(files) for _root, _dirs, files in os.walk(dst_path))
    return {"src": src_path, "dst": dst_path, "copied_files": copied_files}


@app.local_entrypoint()
def main(src: str, dst: str, overwrite: bool = False) -> None:
    result = copy_tree_remote.remote(src, dst, overwrite)
    print(result)
