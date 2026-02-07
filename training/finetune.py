"""
Instruction Fine-tuning Trainer for AnyMAL (Stage 2)

Fine-tunes the model on instruction-following data with LoRA adapters.

Educational Notes:
-----------------
Stage 2 Goal:
Teach the model to follow instructions and have helpful conversations
about images.

What's Trained:
- Perceiver Resampler: Continue training from Stage 1
- LoRA Adapters: Small trainable matrices added to LLM attention

What's Frozen:
- Vision Encoder (CLIP): Still frozen
- LLM Base Weights: Only LoRA adapters are trainable

Training Signal:
- Input: [image_tokens, system_prompt, user_instruction]
- Output: [assistant_response]
- Loss: Cross-entropy on assistant response only

Paper Hyperparameters:
- Batch size: 128
- Learning rate: 1e-5 (lower than Stage 1)
- Steps: 3K (much fewer than Stage 1)
- LoRA rank: 64
- LoRA alpha: 16

Why lower LR and fewer steps?
- Stage 1 did the heavy lifting (alignment)
- Stage 2 is fine-tuning an already capable model
- Too much training risks overfitting to instruction format
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .trainer import Trainer, TrainerConfig
from .distributed import print_rank_0, is_main_process


@dataclass
class FinetuneConfig(TrainerConfig):
    """Configuration for instruction fine-tuning."""
    # Stage 2 specific settings
    learning_rate: float = 1e-5  # Lower than Stage 1
    max_steps: int = 3000  # Much fewer steps
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2

    # LoRA settings (should match model initialization)
    lora_r: int = 64
    lora_alpha: int = 16

    # Continue from Stage 1 checkpoint
    pretrain_checkpoint: Optional[str] = None


class FinetuneTrainer(Trainer):
    """
    Trainer for Stage 2 instruction fine-tuning.

    Trains both the Perceiver Resampler and LoRA adapters on
    instruction-following data.

    Args:
        model: AnyMAL model (potentially with Stage 1 weights)
        config: FinetuneConfig
        train_dataloader: DataLoader for instruction data
        eval_dataloader: Optional validation DataLoader

    Example:
        >>> trainer = FinetuneTrainer(
        ...     model=model,
        ...     config=FinetuneConfig(
        ...         output_dir="./outputs/finetune",
        ...         pretrain_checkpoint="./outputs/pretrain/checkpoint-100000",
        ...     ),
        ...     train_dataloader=train_dataloader,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model,
        config: FinetuneConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        # Load Stage 1 checkpoint if provided
        if config.pretrain_checkpoint:
            self._load_pretrain_checkpoint(model, config.pretrain_checkpoint)

        # Configure model for Stage 2
        model.set_training_stage(2)

        # Verify trainable params
        if is_main_process():
            self._verify_trainable_params(model)

        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

    def _load_pretrain_checkpoint(self, model, checkpoint_path: str):
        """Load pretrained projector weights from Stage 1."""
        projector_path = os.path.join(checkpoint_path, "projector.pt")

        if os.path.exists(projector_path):
            print_rank_0(f"Loading projector from {projector_path}")
            state_dict = torch.load(projector_path, map_location="cpu")
            model.projector.load_state_dict(state_dict)
        else:
            print_rank_0(f"WARNING: No projector weights found at {projector_path}")

    def _verify_trainable_params(self, model):
        """Verify that projector + LoRA are trainable."""
        trainable_groups = {
            "projector": 0,
            "lora": 0,
            "other": 0,
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "projector" in name:
                trainable_groups["projector"] += param.numel()
            elif "lora" in name.lower():
                trainable_groups["lora"] += param.numel()
            else:
                trainable_groups["other"] += param.numel()

        print_rank_0("Stage 2 trainable parameters:")
        for group, count in trainable_groups.items():
            print_rank_0(f"  {group}: {count:,}")

        total = sum(trainable_groups.values())
        print_rank_0(f"  Total: {total:,}")

        if trainable_groups["other"] > 0:
            print_rank_0(
                "WARNING: Unexpected parameters are trainable. "
                "Check model configuration."
            )

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step for instruction fine-tuning.

        The batch should contain:
        - images: [B, 3, H, W]
        - input_ids: [B, seq_len] (conversation tokens)
        - attention_mask: [B, seq_len]
        - labels: [B, seq_len] (with non-response parts masked as -100)
        """
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract components
        images = batch.get("images", batch.get("image"))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Detailed logging for the first few micro-steps
        if self.global_step == 0 and not hasattr(self, '_first_step_logged'):
            self._first_step_logged = True
            supervised = (labels != -100).sum().item()
            total = labels.numel()
            print_rank_0(f"\n--- First micro-batch diagnostics ---")
            print_rank_0(f"  images: {list(images.shape)}, range=[{images.min():.2f}, {images.max():.2f}]")
            print_rank_0(f"  input_ids: {list(input_ids.shape)}")
            print_rank_0(f"  attention_mask: {list(attention_mask.shape)}, sum={attention_mask.sum().item()}")
            print_rank_0(f"  labels: {list(labels.shape)}, supervised={supervised}/{total} ({100*supervised/max(total,1):.1f}%)")
            # Check for image placeholder tokens
            placeholder_id = getattr(self.unwrapped_model, 'image_placeholder_token_id', None)
            if placeholder_id is not None:
                n_ph = (input_ids == placeholder_id).sum().item()
                print_rank_0(f"  image placeholder tokens in input_ids: {n_ph}")
            print_rank_0(f"---\n")

        # Forward pass with mixed precision
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(
            device_type=device_type,
            dtype=self.amp_dtype if torch.cuda.is_available() else torch.float32,
            enabled=self.config.use_amp and torch.cuda.is_available(),
        ):
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # NaN/Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                print_rank_0(f"WARNING: loss is {loss.item()} at step {self.global_step}!")

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Return unscaled loss for logging
        return loss.item() * self.config.gradient_accumulation_steps


def run_finetuning(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    output_dir: str = "./outputs/finetune",
    pretrain_checkpoint: Optional[str] = None,
    max_steps: int = 3000,
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    use_wandb: bool = False,
    wandb_project: str = "anymal-finetune",
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to run instruction fine-tuning.

    Args:
        model: AnyMAL model
        train_dataloader: DataLoader for instruction data
        eval_dataloader: Optional validation DataLoader
        output_dir: Output directory for checkpoints
        pretrain_checkpoint: Path to Stage 1 checkpoint
        max_steps: Maximum training steps
        learning_rate: Learning rate
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        **kwargs: Additional config arguments

    Returns:
        Training metrics dictionary
    """
    config = FinetuneConfig(
        output_dir=output_dir,
        pretrain_checkpoint=pretrain_checkpoint,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        **kwargs,
    )

    trainer = FinetuneTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    return trainer.train()
