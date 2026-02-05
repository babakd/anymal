"""
Alignment Pretraining Trainer for AnyMAL (Stage 1)

Trains the Perceiver Resampler to project image features into
the LLM's text embedding space.

Educational Notes:
-----------------
Stage 1 Goal:
Teach the projector to convert CLIP image features into tokens that
the LLM can "understand" as if they were text tokens.

What's Trained:
- Perceiver Resampler: Cross-attention projector (main trainable component)

What's Frozen:
- Vision Encoder (CLIP): Already has good visual representations
- LLM (LLaMA): Already has good language understanding
- No LoRA adapters in Stage 1

Training Signal:
- Input: [image_tokens, "A photo of"]
- Output: [caption_text]
- Loss: Cross-entropy on caption tokens only

The idea is that if the LLM can predict the correct caption given
the image tokens, then the image tokens must contain the right
information in the right format.

Paper Hyperparameters:
- Batch size: 2048
- Learning rate: 2e-4
- Steps: 100K
- Optimizer: AdamW
- LR Schedule: Cosine decay
- Warmup: 1000 steps
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .trainer import Trainer, TrainerConfig
from .distributed import print_rank_0, is_main_process


@dataclass
class PretrainConfig(TrainerConfig):
    """Configuration for alignment pretraining."""
    # Stage 1 specific settings
    learning_rate: float = 2e-4
    max_steps: int = 100000
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4  # To achieve large batch size

    # Projector-only training
    train_projector_only: bool = True

    # Caption prompt for training
    caption_prompt: str = "A photo of"


class PretrainTrainer(Trainer):
    """
    Trainer for Stage 1 alignment pretraining.

    Focuses on training only the Perceiver Resampler while keeping
    both the vision encoder and LLM frozen.

    Args:
        model: AnyMAL model
        config: PretrainConfig
        train_dataloader: DataLoader for LAION data
        eval_dataloader: Optional validation DataLoader

    Example:
        >>> trainer = PretrainTrainer(
        ...     model=model,
        ...     config=PretrainConfig(
        ...         output_dir="./outputs/pretrain",
        ...         max_steps=100000,
        ...     ),
        ...     train_dataloader=train_dataloader,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model,
        config: PretrainConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        # Configure model for Stage 1
        model.set_training_stage(1)

        # Verify only projector is trainable
        if is_main_process():
            self._verify_trainable_params(model)

        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

    def _verify_trainable_params(self, model):
        """Verify that only the projector is trainable."""
        trainable_modules = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                module_name = name.split(".")[0]
                if module_name not in trainable_modules:
                    trainable_modules.append(module_name)

        print_rank_0(f"Stage 1 trainable modules: {trainable_modules}")

        # Count parameters
        proj_params = sum(
            p.numel() for p in model.projector.parameters() if p.requires_grad
        )
        total_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print_rank_0(f"Projector trainable params: {proj_params:,}")
        print_rank_0(f"Total trainable params: {total_trainable:,}")

        if proj_params != total_trainable:
            print_rank_0(
                "WARNING: Non-projector parameters are trainable. "
                "This may not be intended for Stage 1."
            )

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step for alignment pretraining.

        The batch should contain:
        - images: [B, 3, H, W]
        - input_ids: [B, seq_len] (caption tokens)
        - attention_mask: [B, seq_len]
        - labels: [B, seq_len] (with prompt masked as -100)
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

        # Forward pass with mixed precision
        # Use generic autocast that works on both CPU and CUDA
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

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Return unscaled loss for logging
        return loss.item() * self.config.gradient_accumulation_steps


def run_pretraining(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    output_dir: str = "./outputs/pretrain",
    max_steps: int = 100000,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 64,
    gradient_accumulation_steps: int = 4,
    use_wandb: bool = False,
    wandb_project: str = "anymal-pretrain",
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to run alignment pretraining.

    Args:
        model: AnyMAL model
        train_dataloader: DataLoader for LAION data
        eval_dataloader: Optional validation DataLoader
        output_dir: Output directory for checkpoints
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
    config = PretrainConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        **kwargs,
    )

    trainer = PretrainTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    return trainer.train()
