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

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
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
    loss_scale: float = 1.0
    loss_normalization: str = "mean"
    loss_normalization_target_tokens: float = 8.0
    loss_normalization_min_multiplier: float = 0.05
    loss_normalization_max_multiplier: float = 4.0
    connector_warmup_steps: int = 0
    connector_warmup_trainable_prefixes: Tuple[str, ...] = (
        "projector.output_proj",
        "projector.output_gate_logit",
    )

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

        self._connector_warmup_active = False
        self._loss_scale = float(config.loss_scale)
        if self._loss_scale <= 0:
            raise ValueError(f"loss_scale must be > 0, got {config.loss_scale}")
        if self._loss_scale != 1.0:
            print_rank_0(f"Stage 1 loss scale: multiplying backward loss by {self._loss_scale:g}")
        self._loss_normalization = str(config.loss_normalization or "mean").strip().lower()
        self._loss_normalization_target_tokens = float(config.loss_normalization_target_tokens)
        self._loss_normalization_min_multiplier = float(config.loss_normalization_min_multiplier)
        self._loss_normalization_max_multiplier = float(config.loss_normalization_max_multiplier)
        if self._loss_normalization in {"none", "raw"}:
            self._loss_normalization = "mean"
        if self._loss_normalization not in {"mean", "supervised_token_target"}:
            raise ValueError(
                "loss_normalization must be 'mean' or 'supervised_token_target', "
                f"got {config.loss_normalization!r}"
            )
        if self._loss_normalization_target_tokens <= 0:
            raise ValueError(
                "loss_normalization_target_tokens must be > 0, got "
                f"{config.loss_normalization_target_tokens}"
            )
        if self._loss_normalization_min_multiplier <= 0:
            raise ValueError(
                "loss_normalization_min_multiplier must be > 0, got "
                f"{config.loss_normalization_min_multiplier}"
            )
        if self._loss_normalization_max_multiplier < self._loss_normalization_min_multiplier:
            raise ValueError(
                "loss_normalization_max_multiplier must be >= "
                "loss_normalization_min_multiplier"
            )
        if self._loss_normalization == "supervised_token_target":
            print_rank_0(
                "Stage 1 loss normalization: raw token-mean loss is converted to "
                "a supervised-token-target objective with target "
                f"{self._loss_normalization_target_tokens:g} tokens and multiplier clamp "
                f"[{self._loss_normalization_min_multiplier:g}, "
                f"{self._loss_normalization_max_multiplier:g}]"
            )
        self._connector_warmup_steps = int(config.connector_warmup_steps or 0)
        prefixes = config.connector_warmup_trainable_prefixes or ()
        if isinstance(prefixes, str):
            prefixes = tuple(p.strip() for p in prefixes.split(",") if p.strip())
        self._connector_warmup_trainable_prefixes = tuple(prefixes)
        if self._connector_warmup_steps > 0:
            allowed_params = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad and self._is_connector_warmup_trainable_param_name(name)
            ]
            masked_params = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad
                and self._is_connector_param_name(name)
                and not self._is_connector_warmup_trainable_param_name(name)
            ]
            if not allowed_params:
                raise ValueError(
                    "connector_warmup_steps was requested, but no trainable connector "
                    "parameters matched connector_warmup_trainable_prefixes="
                    f"{self._connector_warmup_trainable_prefixes}"
                )
            self._connector_warmup_active = True
            print_rank_0(
                "Stage 1 connector warmup: only "
                f"{self._connector_warmup_trainable_prefixes} gradients update for "
                f"first {self._connector_warmup_steps} optimizer steps "
                f"({len(allowed_params)} active tensors, {len(masked_params)} masked tensors)"
            )

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

        # Count expected Stage 1 adapter parameters.
        adapter_modules = [model.projector]
        if hasattr(model, "token_compressor"):
            adapter_modules.append(model.token_compressor)
        adapter_params = sum(
            p.numel()
            for module in adapter_modules
            for p in module.parameters()
            if p.requires_grad
        )
        total_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print_rank_0(f"Stage 1 adapter trainable params: {adapter_params:,}")
        print_rank_0(f"Total trainable params: {total_trainable:,}")

        if adapter_params != total_trainable:
            print_rank_0(
                "WARNING: Unexpected non-adapter parameters are trainable. "
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
        if (
            self._connector_warmup_active
            and self.global_step >= self._connector_warmup_steps
        ):
            print_rank_0(f"\n[Step {self.global_step}] Stage 1 connector warmup complete")
            self._connector_warmup_active = False

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
            raw_loss = outputs.loss
            objective_loss, objective_metrics = self._normalize_stage1_loss(raw_loss, labels)
            backward_loss = objective_loss * self._loss_scale
            self._last_batch_metrics = objective_metrics
            self._last_batch_metrics["train/raw_loss"] = raw_loss.detach().item()
            self._last_batch_metrics["train/objective_loss"] = objective_loss.detach().item()
            self._last_batch_metrics["train/backward_loss"] = backward_loss.detach().item()

            loss = backward_loss

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self._connector_warmup_active:
            self._zero_masked_connector_warmup_grads()

        # Return the unscaled objective loss for logging and health monitoring.
        # Raw HF token-mean loss is logged separately as train/raw_loss.
        return objective_loss.item()

    def _normalize_stage1_loss(
        self,
        raw_loss: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        supervised_tokens = (labels != -100).sum().to(device=raw_loss.device, dtype=raw_loss.dtype)
        supervised_tokens_float = supervised_tokens.detach().item()
        multiplier = raw_loss.new_tensor(1.0)

        if self._loss_normalization == "supervised_token_target":
            if supervised_tokens_float > 0:
                multiplier = supervised_tokens / self._loss_normalization_target_tokens
                multiplier = multiplier.clamp(
                    min=self._loss_normalization_min_multiplier,
                    max=self._loss_normalization_max_multiplier,
                )
            else:
                multiplier = raw_loss.new_tensor(self._loss_normalization_min_multiplier)

        objective_loss = raw_loss * multiplier
        return objective_loss, {
            "train/supervised_tokens": float(supervised_tokens_float),
            "train/loss_normalization_multiplier": float(multiplier.detach().item()),
        }

    @staticmethod
    def _is_connector_param_name(name: str) -> bool:
        return name.startswith("projector.") or name.startswith("token_compressor.")

    def _is_connector_warmup_trainable_param_name(self, name: str) -> bool:
        return any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in self._connector_warmup_trainable_prefixes
        )

    def _zero_masked_connector_warmup_grads(self) -> None:
        for name, param in self.unwrapped_model.named_parameters():
            if (
                self._is_connector_param_name(name)
                and not self._is_connector_warmup_trainable_param_name(name)
                and param.grad is not None
            ):
                param.grad = None


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
