"""Modal CPU smoke for V13 Track 1 spatial-grid projector."""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("anymal-v13-spatial-grid-smoke")
REMOTE_PROJECT_DIR = "/root/anymal"


def _resolve_project_dir() -> Path:
    current = Path(__file__).resolve()
    candidates = []
    if len(current.parents) > 2:
        candidates.append(current.parents[2])
    candidates.extend([current.parent, Path.cwd(), Path(REMOTE_PROJECT_DIR)])
    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "training").exists():
            return candidate
    return current.parent


PROJECT_DIR = _resolve_project_dir()


def _ignore_modal_mount(path: Path) -> bool:
    return ".git" in path.parts


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.53.0,<5.0.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    .add_local_dir(
        PROJECT_DIR,
        remote_path=REMOTE_PROJECT_DIR,
        copy=False,
        ignore=_ignore_modal_mount,
    )
)


@app.function(image=image, timeout=10 * 60)
def run_spatial_grid_smoke() -> dict:
    import sys

    sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch

    from models.projectors import SpatialGridProjector

    reports = []
    for grid_tokens in (256, 512):
        projector = SpatialGridProjector(
            input_dim=8,
            output_dim=16,
            num_grid_tokens=grid_tokens,
            hidden_dim=12,
            output_scale=1.125,
            output_gate_init=None,
            trainable_scale_mode="none",
            use_2d_patch_position_features=True,
            patch_position_feature_type="coord_mlp",
            patch_position_grid_size=32,
            patch_position_mlp_hidden_dim=10,
            patch_position_feature_scale=1.0,
        )
        height, width = projector.grid_height, projector.grid_width
        if height * width != grid_tokens:
            raise AssertionError(f"{grid_tokens} did not map to an exact grid")

        vision_features = torch.randn(2, 27 * 27, 8, requires_grad=True)
        output = projector(vision_features)
        if tuple(output.shape) != (2, grid_tokens, 16):
            raise AssertionError(
                f"spatial-grid output shape mismatch: {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise AssertionError("spatial-grid output contains non-finite values")
        rms = output.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()
        if not 0.5 < rms < 2.0:
            raise AssertionError(f"unexpected per-token RMS after norm: {rms}")
        loss = output.float().pow(2).mean()
        loss.backward()
        if vision_features.grad is None or not torch.isfinite(vision_features.grad).all():
            raise AssertionError("vision features did not receive finite gradients")

        reports.append(
            {
                "grid_tokens": grid_tokens,
                "grid_height": height,
                "grid_width": width,
                "output_shape": list(output.shape),
                "output_rms_mean": rms,
                "pooling_mode": projector.pooling_mode,
            }
        )
    return {"reports": reports}


@app.local_entrypoint()
def main():
    result = run_spatial_grid_smoke.remote()
    for report in result["reports"]:
        print(report)
