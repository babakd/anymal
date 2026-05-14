"""Modal CPU smoke for V13 Track 3 AnyRes-lite plumbing."""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("anymal-v13-anyres-smoke")
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
def run_anyres_smoke() -> dict:
    import sys

    sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch
    from PIL import Image

    from data.data_utils import get_vision_transform
    from models.projectors import AnyResMLPProjector
    from training.pretrain import PretrainTrainer

    transform = get_vision_transform(
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        image_size=384,
        is_train=True,
        use_augmentation=True,
        image_view_mode="anyres_global_two_crops",
    )
    image = Image.new("RGB", (640, 480), color=(64, 96, 128))
    pack = transform(image)
    if tuple(pack.shape) != (3, 3, 384, 384):
        raise AssertionError(f"AnyRes transform shape mismatch: {tuple(pack.shape)}")
    batch_images = torch.stack([pack, pack], dim=0)
    if tuple(batch_images.shape) != (2, 3, 3, 384, 384):
        raise AssertionError(f"AnyRes batch shape mismatch: {tuple(batch_images.shape)}")
    negative_images = torch.stack([batch_images, batch_images], dim=1)
    if tuple(negative_images.shape) != (2, 2, 3, 3, 384, 384):
        raise AssertionError(
            f"AnyRes negative image shape mismatch: {tuple(negative_images.shape)}"
        )

    projector = AnyResMLPProjector(
        input_dim=8,
        output_dim=16,
        tokens_per_view=(128, 64, 64),
        hidden_dim=12,
        output_scale=1.125,
        output_gate_init=0.25,
        trainable_scale_mode="none",
        use_2d_patch_position_features=True,
        patch_position_feature_type="coord_mlp",
        patch_position_grid_size=32,
        patch_position_mlp_hidden_dim=10,
        patch_position_feature_scale=1.0,
        use_view_embeddings=True,
    )
    vision_features = torch.randn(2, 3, 27 * 27, 8, requires_grad=True)
    output = projector(vision_features)
    if tuple(output.shape) != (2, 256, 16):
        raise AssertionError(f"AnyRes projector shape mismatch: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("AnyRes projector output contains non-finite values")
    loss = output.float().pow(2).mean()
    loss.backward()
    if vision_features.grad is None or not torch.isfinite(vision_features.grad).all():
        raise AssertionError("AnyRes projector input did not receive finite gradients")

    class _FakeImageEncoder(torch.nn.Module):
        def forward(self, images):
            if tuple(images.shape) != (2, 3, 384, 384):
                raise AssertionError(
                    f"teacher sidecar saw wrong image shape: {tuple(images.shape)}"
                )
            return torch.ones(images.shape[0], 27 * 27, 8)

    class _FakeProjector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 16)

        def forward(self, features):
            pooled = features[:, :128, :]
            return self.proj(pooled)

    class _DummyModel:
        image_encoder = _FakeImageEncoder()

    pretrain_trainer = object.__new__(PretrainTrainer)
    pretrain_trainer.unwrapped_model = _DummyModel()
    pretrain_trainer._teacher_kl_sidecar_projector = _FakeProjector()
    pretrain_trainer._teacher_kl_image_tokens = 128
    teacher_input = torch.randn(2, 3, 3, 384, 384)
    teacher_global = pretrain_trainer._encode_sidecar_teacher_image_tokens(teacher_input)
    if tuple(teacher_global.shape) != (2, 128, 16):
        raise AssertionError(
            f"teacher sidecar output shape mismatch: {tuple(teacher_global.shape)}"
        )

    if negative_images.ndim not in {5, 6}:
        raise AssertionError("Finetune negative-image contract should accept 6D AnyRes")

    return {
        "transform_shape": list(pack.shape),
        "batch_shape": list(batch_images.shape),
        "negative_shape": list(negative_images.shape),
        "projector_shape": list(output.shape),
        "projector_grid_shapes": [list(shape) for shape in projector.grid_shapes],
        "teacher_global_shape": list(teacher_global.shape),
    }


@app.local_entrypoint()
def main():
    result = run_anyres_smoke.remote()
    print(result)
