"""
Base Trainer for AnyMAL

Provides the core training loop infrastructure shared between
pretraining and fine-tuning.

Educational Notes:
-----------------
Training Loop Components:
1. Forward pass: Model processes batch, computes loss
2. Backward pass: Compute gradients via autograd
3. Gradient accumulation: Sum gradients over multiple steps
4. Optimizer step: Update weights
5. LR scheduler step: Adjust learning rate
6. Logging: Track metrics

Memory Optimization Strategies:
1. Gradient checkpointing: Recompute activations during backward
   - Reduces memory ~50%, increases compute ~30%
   - Essential for training large models

2. Mixed precision (bf16/fp16):
   - Forward/backward in 16-bit
   - Weight updates in 32-bit (maintained by optimizer)
   - 2x memory reduction, 2x faster compute

3. Gradient accumulation:
   - Process micro-batches, accumulate gradients
   - Update weights every N steps
   - Simulates larger batch without more memory

4. ZeRO optimization (via FSDP/DeepSpeed):
   - Stage 1: Shard optimizer states
   - Stage 2: Shard gradients
   - Stage 3: Shard parameters
   - Each stage reduces memory further
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from .distributed import (
    is_main_process,
    get_rank,
    get_world_size,
    reduce_tensor,
    synchronize,
    print_rank_0,
)


@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    # Training
    num_epochs: int = 1
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./outputs"

    # Logging
    logging_steps: int = 10
    log_dir: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "anymal"
    wandb_run_name: Optional[str] = None

    # Evaluation
    eval_steps: Optional[int] = None
    eval_strategy: str = "steps"  # "steps" or "epoch"

    # Other
    seed: int = 42
    dataloader_num_workers: int = 4


class Trainer:
    """
    Base trainer class for AnyMAL.

    Handles the training loop, checkpointing, logging, and evaluation.
    Subclassed by PretrainTrainer and FinetuneTrainer for stage-specific logic.

    Args:
        model: The model to train
        config: Trainer configuration
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        optimizer: Optional optimizer (created if not provided)
        scheduler: Optional LR scheduler (created if not provided)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Set random seed for reproducibility
        self._set_seed(config.seed)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device (skip if already placed by device_map)
        # Check if model has any meta tensors (from device_map="auto")
        has_meta_params = any(p.device.type == "meta" for p in model.parameters())
        if has_meta_params:
            # Model was loaded with device_map and has meta tensors - cannot move
            print("Warning: Model has meta tensors from device_map, skipping .to()")
            self.model = model
        else:
            # Check if model was already placed by device_map
            model_devices = set(p.device for p in model.parameters())
            if len(model_devices) == 1 and list(model_devices)[0] != torch.device("cpu"):
                # Model is already on a specific device, don't move
                self.model = model
                self.device = list(model_devices)[0]
            else:
                self.model = model.to(self.device)

        # Wrap model for distributed training
        if get_world_size() > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # Get underlying model (for saving)
        self.unwrapped_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Set up optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Set up scheduler
        self.scheduler = scheduler or self._create_scheduler()

        # Set up mixed precision
        self.scaler = None
        if config.use_amp:
            if config.amp_dtype == "float16":
                self.scaler = GradScaler()
            self.amp_dtype = (
                torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
            )
        else:
            self.amp_dtype = torch.float32

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Set up logging
        self.logger = None
        self.wandb_url = None
        if config.use_wandb and is_main_process():
            self._setup_wandb()

        # Create output directory
        if is_main_process():
            os.makedirs(config.output_dir, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup support."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Calculate total steps
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        warmup_steps = self.config.warmup_steps

        # Create main scheduler
        if self.config.lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - warmup_steps),
            )
        elif self.config.lr_scheduler_type == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(1, total_steps - warmup_steps),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

        # If no warmup, return main scheduler directly
        if warmup_steps == 0:
            return main_scheduler

        # Create warmup scheduler and chain with main scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            self.logger = wandb
            self.wandb_url = wandb.run.get_url() if wandb.run else None
            if self.wandb_url:
                print_rank_0("")
                print_rank_0("=" * 60)
                print_rank_0("W&B RUN URL: " + self.wandb_url)
                print_rank_0("=" * 60)
                print_rank_0("")
        except ImportError:
            print("wandb not installed, skipping")
            self.wandb_url = None

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Returns:
            Dict of training metrics
        """
        print_rank_0(f"Starting training for {self.config.num_epochs} epochs")
        print_rank_0(f"  Total steps: {self._get_total_steps()}")
        print_rank_0(f"  Batch size per GPU: {self.train_dataloader.batch_size}")
        print_rank_0(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print_rank_0(f"  Effective batch size: {self._get_effective_batch_size()}")

        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss

            # Evaluate at end of epoch if configured
            if self.config.eval_strategy == "epoch" and self.eval_dataloader:
                eval_loss = self.evaluate()
                print_rank_0(f"Epoch {epoch+1} - Eval loss: {eval_loss:.4f}")

            # Check if we've hit max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        elapsed = time.time() - start_time
        print_rank_0(f"Training completed in {elapsed/3600:.2f} hours")

        # Print W&B URL prominently at end of training
        if hasattr(self, 'wandb_url') and self.wandb_url:
            print_rank_0("")
            print_rank_0("=" * 60)
            print_rank_0("VIEW TRAINING RESULTS:")
            print_rank_0(self.wandb_url)
            print_rank_0("=" * 60)
            print_rank_0("")

        return {"train_loss": total_loss / (epoch + 1)}

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_steps = 0

        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        # Progress bar for main process
        progress = None
        if is_main_process():
            progress = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.epoch + 1}",
                disable=not is_main_process(),
            )
        else:
            progress = self.train_dataloader

        self.optimizer.zero_grad()
        micro_steps = 0

        for batch in progress:
            # Some collators may return None (e.g. if every sample in a batch is invalid).
            if batch is None:
                continue

            loss = self._train_step(batch)
            micro_steps += 1

            # Accumulate gradients over valid micro-batches
            if micro_steps % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step (warmup is handled by SequentialLR)
                self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({"train/loss": loss})

                # Evaluation
                if (
                    self.config.eval_steps
                    and self.global_step % self.config.eval_steps == 0
                    and self.eval_dataloader
                ):
                    eval_loss = self.evaluate()
                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

            epoch_loss += loss
            num_steps += 1

            # Update progress bar
            if is_main_process() and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{self._get_lr():.2e}",
                    step=self.global_step,
                )

        # If the epoch ends mid-accumulation, drop leftover gradients to avoid carrying them across epochs.
        if micro_steps % self.config.gradient_accumulation_steps != 0:
            self.optimizer.zero_grad()

        return epoch_loss / max(num_steps, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Subclasses can override for custom behavior.
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass with mixed precision
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Return unscaled loss for logging
        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_steps = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not is_main_process()):
            if batch is None:
                continue
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with autocast(
                device_type=device_type,
                dtype=self.amp_dtype if torch.cuda.is_available() else torch.float32,
                enabled=self.config.use_amp and torch.cuda.is_available(),
            ):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs

            total_loss += loss.item()
            num_steps += 1

        # Average across all processes
        avg_loss = total_loss / max(num_steps, 1)
        avg_loss = reduce_tensor(torch.tensor(avg_loss, device=self.device)).item()

        self._log_metrics({"eval/loss": avg_loss})

        return avg_loss

    def _save_checkpoint(self):
        """Save a checkpoint."""
        if not is_main_process():
            return

        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.unwrapped_model.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": vars(self.config),
        }, os.path.join(checkpoint_dir, "trainer_state.pt"))

        print_rank_0(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        checkpoints = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith("checkpoint-"):
                step = int(name.split("-")[1])
                checkpoints.append((step, os.path.join(self.config.output_dir, name)))

        checkpoints.sort(key=lambda x: x[0])

        # Remove oldest checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            _, path = checkpoints.pop(0)
            import shutil
            shutil.rmtree(path)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and/or console."""
        if not is_main_process():
            return

        metrics["train/lr"] = self._get_lr()
        metrics["train/step"] = self.global_step
        metrics["train/epoch"] = self.epoch

        if self.logger is not None:
            self.logger.log(metrics)

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _get_total_steps(self) -> int:
        """Get total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _get_effective_batch_size(self) -> int:
        """Get effective batch size."""
        return (
            self.train_dataloader.batch_size
            * self.config.gradient_accumulation_steps
            * get_world_size()
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
