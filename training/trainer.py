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
from model_metadata import validate_checkpoint_architecture

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
    lora_learning_rate: Optional[float] = None  # Separate LR for LoRA params (defaults to learning_rate)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1  # Cosine scheduler floors at this fraction of peak LR

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./outputs"
    save_llm_checkpoint: bool = True
    save_llm_base_weights: bool = False

    # Logging
    logging_steps: int = 10
    log_dir: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "anymal"
    wandb_run_name: Optional[str] = None

    # Evaluation
    eval_steps: Optional[int] = None
    eval_strategy: str = "steps"  # "steps" or "epoch"
    max_eval_batches: Optional[int] = None  # Clip eval to N batches (None = full)

    # Other
    seed: int = 42
    dataloader_num_workers: int = 4

    # Health monitoring
    enable_health_monitoring: bool = True
    track_per_layer_grad_norms: bool = False

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None


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

        # Initialize health monitor
        self.health_monitor = None
        if config.enable_health_monitoring and is_main_process():
            from .health_monitor import TrainingHealthMonitor, HealthMonitorConfig
            hm_config = HealthMonitorConfig(max_grad_norm=config.max_grad_norm)
            self.health_monitor = TrainingHealthMonitor(hm_config, wandb_logger=self.logger)

        # Initialize throughput tracker
        self.throughput_tracker = None
        if is_main_process():
            from .throughput_tracker import ThroughputTracker
            self.throughput_tracker = ThroughputTracker()

        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self._load_checkpoint(config.resume_from_checkpoint)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay and optional per-component learning rates."""
        lora_lr = self.config.lora_learning_rate or self.config.learning_rate

        # Separate parameters by component and weight decay
        projector_decay = []
        projector_no_decay = []
        lora_decay = []
        lora_no_decay = []
        other_decay = []
        other_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = "bias" in name or "norm" in name or "layernorm" in name

            if "lora" in name.lower():
                (lora_no_decay if is_no_decay else lora_decay).append(param)
            elif "projector" in name:
                (projector_no_decay if is_no_decay else projector_decay).append(param)
            else:
                (other_no_decay if is_no_decay else other_decay).append(param)

        optimizer_groups = []
        if projector_decay:
            optimizer_groups.append({"params": projector_decay, "weight_decay": self.config.weight_decay, "lr": self.config.learning_rate, "label": "projector"})
        if projector_no_decay:
            optimizer_groups.append({"params": projector_no_decay, "weight_decay": 0.0, "lr": self.config.learning_rate, "label": "projector"})
        if lora_decay:
            optimizer_groups.append({"params": lora_decay, "weight_decay": self.config.weight_decay, "lr": lora_lr, "label": "lora"})
        if lora_no_decay:
            optimizer_groups.append({"params": lora_no_decay, "weight_decay": 0.0, "lr": lora_lr, "label": "lora"})
        if other_decay:
            optimizer_groups.append({"params": other_decay, "weight_decay": self.config.weight_decay, "lr": self.config.learning_rate, "label": "other"})
        if other_no_decay:
            optimizer_groups.append({"params": other_no_decay, "weight_decay": 0.0, "lr": self.config.learning_rate, "label": "other"})

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup support."""
        import math
        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

        # Calculate total steps
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        warmup_steps = self.config.warmup_steps

        # Create main scheduler
        if self.config.lr_scheduler_type == "cosine":
            # Cosine with floor: decays to min_lr_ratio * peak_lr (not zero)
            # LambdaLR multiplies each param group's initial_lr by the lambda value,
            # so this respects per-group LRs (projector vs LoRA).
            min_ratio = self.config.min_lr_ratio
            T = max(1, total_steps - warmup_steps)
            def cosine_with_floor(step):
                progress = min(step / T, 1.0)
                return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            main_scheduler = LambdaLR(self.optimizer, cosine_with_floor)
        elif self.config.lr_scheduler_type == "linear":
            min_ratio = self.config.min_lr_ratio
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_ratio,
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
        # When max_steps is set, loop enough epochs to reach it.
        # Otherwise, use the configured num_epochs.
        num_epochs = self.config.num_epochs
        if self.config.max_steps is not None:
            num_epochs = max(num_epochs, 10_000)

        # Determine starting epoch (for resume)
        start_epoch = self.epoch if self.config.resume_from_checkpoint else 0

        print_rank_0(f"Starting training for {self.config.num_epochs} epochs (max_steps={self.config.max_steps})")
        print_rank_0(f"  Total steps: {self._get_total_steps()}")
        print_rank_0(f"  Batch size per GPU: {self.train_dataloader.batch_size}")
        print_rank_0(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print_rank_0(f"  Effective batch size: {self._get_effective_batch_size()}")
        if start_epoch > 0:
            print_rank_0(f"  Resuming from epoch {start_epoch}, step {self.global_step}")

        if is_main_process():
            self._log_training_config()

        self.model.train()
        total_loss = 0.0
        num_completed_epochs = 0
        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss
            num_completed_epochs += 1

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

        return {"train_loss": total_loss / max(num_completed_epochs, 1)}

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

        # When resuming mid-epoch, skip batches that were already processed
        skip_batches = 0
        if self.config.resume_from_checkpoint and self.global_step > 0:
            skip_batches = self.global_step * self.config.gradient_accumulation_steps
            if skip_batches > 0:
                print_rank_0(f"  Skipping {skip_batches} micro-batches to resume position...")
            # Only skip once (clear after first epoch of resumed training)
            self.config.resume_from_checkpoint = None

        for batch_idx, batch in enumerate(progress):
            # Skip already-processed batches when resuming
            if batch_idx < skip_batches:
                continue

            # Some collators may return None (e.g. if every sample in a batch is invalid).
            if batch is None:
                continue

            loss = self._train_step(batch)
            micro_steps += 1

            # Accumulate gradients over valid micro-batches
            if micro_steps % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm_before_clip = None
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    # Capture pre-clip grad norm for health monitoring
                    if self.health_monitor is not None:
                        grad_norm_before_clip = self._compute_total_grad_norm()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                else:
                    grad_norm = None

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

                # Feed health monitor
                if self.health_monitor is not None:
                    gn_val = grad_norm.item() if grad_norm is not None and hasattr(grad_norm, 'item') else (float(grad_norm) if grad_norm is not None else 0.0)
                    gn_before = grad_norm_before_clip if grad_norm_before_clip is not None else None
                    self.health_monitor.on_step(self.global_step, loss, gn_val, gn_before)

                # Feed throughput tracker
                if self.throughput_tracker is not None:
                    # Determine batch size and seq_len from current batch
                    batch_sz = batch.get("input_ids").shape[0] if "input_ids" in batch and hasattr(batch.get("input_ids"), "shape") else 1
                    seq_ln = batch.get("input_ids").shape[1] if "input_ids" in batch and hasattr(batch.get("input_ids"), "shape") and batch.get("input_ids").dim() > 1 else 1
                    self.throughput_tracker.step(batch_sz, seq_ln)

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    metrics = {"train/loss": loss}
                    if grad_norm is not None:
                        gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                        metrics["train/grad_norm"] = gn
                    # Per-component gradient norms (first 10 steps + every 50th step)
                    if self.global_step <= 10 or self.global_step % 50 == 0:
                        comp_norms = self._compute_component_grad_norms()
                        metrics.update(comp_norms)
                    # Health monitor summary
                    if self.health_monitor is not None:
                        health_summary = self.health_monitor.get_summary()
                        for k, v in health_summary.items():
                            if v is not None:
                                metrics[f"health/{k}"] = v
                    # Throughput metrics
                    if self.throughput_tracker is not None:
                        throughput = self.throughput_tracker.get_metrics()
                        for k, v in throughput.items():
                            if v > 0:
                                metrics[f"perf/{k}"] = v
                    # Per-layer gradient norms (if enabled, first 10 steps + every 50th)
                    if self.config.track_per_layer_grad_norms:
                        if self.global_step <= 10 or self.global_step % 50 == 0:
                            layer_norms = self._compute_per_layer_grad_norms()
                            metrics.update(layer_norms)
                    self._log_metrics(metrics)

                # Evaluation
                if (
                    self.config.eval_steps
                    and self.global_step % self.config.eval_steps == 0
                    and self.eval_dataloader
                ):
                    eval_loss = self.evaluate()
                    if self.health_monitor is not None and self.health_monitor.loss_ema is not None:
                        self.health_monitor.on_eval(self.global_step, self.health_monitor.loss_ema, eval_loss)
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

        max_batches = self.config.max_eval_batches
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not is_main_process(),
                          total=max_batches):
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

            if max_batches and num_steps >= max_batches:
                break

        # Average across all processes
        avg_loss = total_loss / max(num_steps, 1)
        avg_loss = reduce_tensor(torch.tensor(avg_loss, device=self.device)).item()

        print_rank_0(f"Eval: {num_steps} valid batches, avg_loss={avg_loss:.4f}")
        if num_steps == 0:
            print_rank_0("WARNING: Eval had 0 valid batches - eval loss is meaningless")

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

        # Save model (prefer lightweight LLM checkpoint payload unless explicitly requested)
        try:
            self.unwrapped_model.save_pretrained(
                checkpoint_dir,
                save_llm=self.config.save_llm_checkpoint,
                save_llm_base=self.config.save_llm_base_weights,
            )
        except TypeError:
            # Backward compatibility for model classes without the new save signature.
            self.unwrapped_model.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler state
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": vars(self.config),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "health_monitor_state": self.health_monitor.get_state() if self.health_monitor else None,
        }
        if self.scaler is not None:
            state_dict["scaler"] = self.scaler.state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, "trainer_state.pt"))

        print_rank_0(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_dir: str):
        """Resume training from a saved checkpoint."""
        import os as _os

        trainer_state_path = _os.path.join(checkpoint_dir, "trainer_state.pt")
        if not _os.path.exists(trainer_state_path):
            print_rank_0(f"WARNING: No trainer_state.pt in {checkpoint_dir}, cannot resume")
            return

        print_rank_0(f"Resuming from checkpoint: {checkpoint_dir}")

        expected_arch = getattr(self.unwrapped_model, "architecture", "anymal_v1")
        validate_checkpoint_architecture(
            checkpoint_dir=checkpoint_dir,
            expected_architecture=expected_arch,
        )

        # Load model weights (projector + optionally LoRA)
        projector_path = _os.path.join(checkpoint_dir, "projector.pt")
        if _os.path.exists(projector_path):
            self.unwrapped_model.projector.load_state_dict(
                torch.load(projector_path, map_location=self.device, weights_only=True)
            )
            print_rank_0("  Loaded projector weights")

        compressor_path = _os.path.join(checkpoint_dir, "token_compressor.pt")
        if _os.path.exists(compressor_path) and hasattr(self.unwrapped_model, "token_compressor"):
            self.unwrapped_model.token_compressor.load_state_dict(
                torch.load(compressor_path, map_location=self.device, weights_only=True)
            )
            print_rank_0("  Loaded token compressor weights")

        llm_path = _os.path.join(checkpoint_dir, "llm")
        if _os.path.exists(llm_path) and hasattr(self.unwrapped_model.llm, "model"):
            try:
                from peft import set_peft_model_state_dict
                import safetensors.torch
                adapter_path = _os.path.join(llm_path, "adapter_model.safetensors")
                if _os.path.exists(adapter_path):
                    lora_state = safetensors.torch.load_file(adapter_path, device=str(self.device))
                    set_peft_model_state_dict(self.unwrapped_model.llm.model, lora_state)
                    print_rank_0("  Loaded LoRA weights")
            except Exception as e:
                print_rank_0(f"  WARNING: Could not load LoRA weights: {e}")

        # Load trainer state (use CPU to avoid moving RNG states to GPU)
        state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]

        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        # Restore RNG states for reproducibility
        rng = state.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            # RNG states must be CPU ByteTensors
            torch_rng = rng["torch"]
            if isinstance(torch_rng, torch.Tensor):
                torch.random.set_rng_state(torch_rng.cpu().byte())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                cuda_states = rng["cuda"]
                torch.cuda.set_rng_state_all([s.cpu().byte() if isinstance(s, torch.Tensor) else s for s in cuda_states])

        # Restore health monitor state
        if self.health_monitor is not None and state.get("health_monitor_state"):
            self.health_monitor.load_state(state["health_monitor_state"])

        print_rank_0(f"  Resumed at global_step={self.global_step}, epoch={self.epoch}")

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

    def _compute_component_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms per model component for diagnostics."""
        norms = {"projector": 0.0, "lora": 0.0, "other": 0.0}
        counts = {"projector": 0, "lora": 0, "other": 0}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            pnorm = param.grad.data.norm(2).item() ** 2
            if "projector" in name:
                norms["projector"] += pnorm
                counts["projector"] += 1
            elif "lora" in name.lower():
                norms["lora"] += pnorm
                counts["lora"] += 1
            else:
                norms["other"] += pnorm
                counts["other"] += 1

        result = {}
        for comp in norms:
            if counts[comp] > 0:
                result[f"train/grad_norm_{comp}"] = norms[comp] ** 0.5
        return result

    def _compute_total_grad_norm(self) -> float:
        """Compute L2 norm of all parameter gradients (before clipping)."""
        total_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm_sq += param.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5

    def _compute_per_layer_grad_norms(self) -> Dict[str, float]:
        """Compute per-layer gradient norms for perceiver and LoRA layers."""
        layer_norms = {}  # group_name -> sum of squared norms

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            pnorm_sq = param.grad.data.norm(2).item() ** 2

            # Group by perceiver layer index
            if "projector" in name and "layers." in name:
                # Extract layer index from names like "projector.layers.0...."
                parts = name.split("layers.")
                if len(parts) > 1:
                    layer_idx = parts[1].split(".")[0]
                    key = f"grad_norm/perceiver_layer_{layer_idx}"
                    layer_norms[key] = layer_norms.get(key, 0.0) + pnorm_sq
            # Group by LoRA layer index
            elif "lora" in name.lower():
                # Extract layer index from names like "...layers.0...lora_A..."
                parts = name.split("layers.")
                if len(parts) > 1:
                    layer_idx = parts[1].split(".")[0]
                    key = f"grad_norm/lora_layer_{layer_idx}"
                    layer_norms[key] = layer_norms.get(key, 0.0) + pnorm_sq

        # Convert squared sums to L2 norms
        return {k: v ** 0.5 for k, v in layer_norms.items()}

    def _log_training_config(self):
        """Log training configuration at start of training."""
        print_rank_0("\n" + "=" * 60)
        print_rank_0("TRAINING CONFIGURATION")
        print_rank_0("=" * 60)
        print_rank_0(f"  Learning rate (projector): {self.config.learning_rate}")
        print_rank_0(f"  LoRA learning rate: {self.config.lora_learning_rate or self.config.learning_rate}")
        print_rank_0(f"  Min LR ratio: {self.config.min_lr_ratio}")
        print_rank_0(f"  LR scheduler: {self.config.lr_scheduler_type}")
        print_rank_0(f"  Warmup steps: {self.config.warmup_steps}")
        print_rank_0(f"  Max grad norm: {self.config.max_grad_norm}")
        print_rank_0(f"  Weight decay: {self.config.weight_decay}")
        print_rank_0(f"  AMP dtype: {self.config.amp_dtype}")
        print_rank_0(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print_rank_0(f"  Eval steps: {self.config.eval_steps}")
        print_rank_0(f"  Health monitoring: {self.config.enable_health_monitoring}")
        print_rank_0(f"  Per-layer grad norms: {self.config.track_per_layer_grad_norms}")
        print_rank_0(f"  Optimizer: {type(self.optimizer).__name__}")
        print_rank_0(f"  Num param groups: {len(self.optimizer.param_groups)}")
        # Log per-group LRs
        for i, group in enumerate(self.optimizer.param_groups):
            print_rank_0(f"  Param group {i}: lr={group['lr']:.2e}, wd={group['weight_decay']}, params={len(group['params'])}")
        print_rank_0("=" * 60 + "\n")

        # Log to W&B config if available
        if self.logger is not None:
            try:
                self.logger.config.update({
                    "lr_scheduler_type": self.config.lr_scheduler_type,
                    "warmup_steps": self.config.warmup_steps,
                    "max_grad_norm": self.config.max_grad_norm,
                    "weight_decay": self.config.weight_decay,
                    "health_monitoring_enabled": self.config.enable_health_monitoring,
                    "track_per_layer_grad_norms": self.config.track_per_layer_grad_norms,
                    "lora_learning_rate": self.config.lora_learning_rate,
                    "min_lr_ratio": self.config.min_lr_ratio,
                }, allow_val_change=True)
            except Exception:
                pass

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and/or console."""
        if not is_main_process():
            return

        metrics["train/lr"] = self._get_lr()
        metrics["train/step"] = self.global_step
        metrics["train/epoch"] = self.epoch

        # Log per-group LRs (projector vs LoRA)
        # Use the LR values directly — groups are ordered: projector_decay, projector_no_decay,
        # lora_decay, lora_no_decay, other_decay, other_no_decay (from _create_optimizer)
        seen_lrs = {}
        for group in self.optimizer.param_groups:
            lr_val = group["lr"]
            label = group.get("label", "")
            if label and label not in seen_lrs:
                seen_lrs[label] = lr_val
        for label, lr_val in seen_lrs.items():
            metrics[f"train/lr_{label}"] = lr_val

        # Always print key metrics to console for monitoring
        loss_str = f"loss={metrics.get('train/loss', 0):.4f}"
        lr_str = f"lr={metrics.get('train/lr', 0):.2e}"
        gn_str = f"grad_norm={metrics.get('train/grad_norm', 0):.4f}" if 'train/grad_norm' in metrics else ""
        parts = [f"[step {self.global_step}]", loss_str, lr_str]
        if gn_str:
            parts.append(gn_str)
        # Component grad norms
        for comp in ["projector", "lora", "other"]:
            key = f"train/grad_norm_{comp}"
            if key in metrics:
                parts.append(f"gn_{comp}={metrics[key]:.4f}")
        print_rank_0("  ".join(parts))

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
