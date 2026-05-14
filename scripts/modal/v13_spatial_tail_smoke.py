"""Modal CPU smoke for V13 V3 spatial-tail connector plumbing."""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("anymal-v13-spatial-tail-smoke")
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
def run_spatial_tail_smoke() -> dict:
    import sys

    sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch

    from models.anymal_v3 import V3SpatialTailBranch
    from training.pretrain import PretrainTrainer

    reports = []
    for tail_tokens in (128, 256, 512, 640):
        height, width = V3SpatialTailBranch._target_grid(tail_tokens)
        if height * width != tail_tokens:
            raise AssertionError(
                f"tail token grid does not tile exactly: "
                f"{tail_tokens} -> {height}x{width}"
            )

        branch = V3SpatialTailBranch(
            input_dim=8,
            output_dim=16,
            num_tail_tokens=tail_tokens,
            hidden_dim=12,
            output_scale=1.0,
            gate_init=1e-4,
            use_2d_position_features=True,
        )
        siglip_patch_features = torch.randn(2, 27 * 27, 8)
        output = branch(siglip_patch_features)
        expected_shape = (2, tail_tokens, 16)
        if tuple(output.shape) != expected_shape:
            raise AssertionError(
                f"tail output shape mismatch: got {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise AssertionError(f"tail output has non-finite values for {tail_tokens}")

        reports.append(
            {
                "tail_tokens": tail_tokens,
                "grid_height": height,
                "grid_width": width,
                "output_shape": list(output.shape),
                "gate_value": branch.gate_value(),
            }
        )

    class _DummyTokenizer:
        pad_token_id = 0

    class _DummyModel:
        image_placeholder_token_id = 99
        tokenizer = _DummyTokenizer()

    trainer = object.__new__(PretrainTrainer)
    trainer.unwrapped_model = _DummyModel()
    trainer._teacher_kl_weight = 0.5
    trainer._teacher_kl_temperature = 1.0
    trainer._teacher_kl_direction = "teacher_to_student"

    active_ids = torch.tensor(
        [10] + [99] * 256 + [20, 21, 22],
        dtype=torch.long,
    )
    input_ids = torch.cat(
        [active_ids, torch.zeros(8, dtype=torch.long)],
        dim=0,
    ).unsqueeze(0)
    attention_mask = torch.cat(
        [
            torch.ones(active_ids.shape[0], dtype=torch.long),
            torch.zeros(8, dtype=torch.long),
        ],
        dim=0,
    ).unsqueeze(0)
    labels = torch.full_like(input_ids, -100)
    labels[0, 257:260] = torch.tensor([20, 21, 22], dtype=torch.long)
    teacher_batch = trainer._make_teacher_prefix_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        teacher_image_tokens=128,
    )
    teacher_placeholder_count = int((teacher_batch["input_ids"] == 99).sum().item())
    if teacher_placeholder_count != 128:
        raise AssertionError(
            f"teacher batch has {teacher_placeholder_count} placeholders, expected 128"
        )

    vocab_size = 128
    student_logits = torch.randn(
        1,
        input_ids.shape[1],
        vocab_size,
        requires_grad=True,
    )
    teacher_logits = torch.randn(
        1,
        teacher_batch["input_ids"].shape[1],
        vocab_size,
        requires_grad=True,
    )
    student_answer_logits, student_answer_labels = trainer._select_answer_token_logits(
        student_logits,
        labels,
    )
    teacher_answer_logits, teacher_answer_labels = trainer._select_answer_token_logits(
        teacher_logits,
        teacher_batch["labels"],
    )
    if student_answer_labels.tolist() != [20, 21, 22]:
        raise AssertionError(
            f"unexpected student labels: {student_answer_labels.tolist()}"
        )
    if not torch.equal(student_answer_labels, teacher_answer_labels):
        raise AssertionError("teacher/student answer-token labels did not align")
    kl_loss, kl_metrics = trainer._teacher_kl_loss(
        student_answer_logits=student_answer_logits,
        student_answer_labels=student_answer_labels,
        teacher_answer_logits=teacher_answer_logits,
        teacher_answer_labels=teacher_answer_labels,
    )
    if not torch.isfinite(kl_loss):
        raise AssertionError("teacher KL loss is not finite")
    kl_loss.backward()
    if student_logits.grad is None or not torch.isfinite(student_logits.grad).all():
        raise AssertionError("student logits did not receive finite KL gradients")
    if teacher_logits.grad is not None:
        raise AssertionError("teacher logits should be detached from KL gradients")

    return {
        "reports": reports,
        "teacher_kl": {
            "teacher_placeholder_count": teacher_placeholder_count,
            "answer_labels": student_answer_labels.tolist(),
            **kl_metrics,
        },
    }


@app.local_entrypoint()
def main():
    result = run_spatial_tail_smoke.remote()
    for report in result["reports"]:
        print(report)
    print(result["teacher_kl"])
