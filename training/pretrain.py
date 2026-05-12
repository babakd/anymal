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
    connector_rms_regularizer_alpha: float = 0.0
    connector_rms_regularizer_eps: float = 1e-6
    connector_rms_regularizer_target: str = "batch_text"
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
        self._connector_rms_regularizer_alpha = float(
            config.connector_rms_regularizer_alpha or 0.0
        )
        if self._connector_rms_regularizer_alpha < 0:
            raise ValueError(
                "connector_rms_regularizer_alpha must be >= 0, "
                f"got {config.connector_rms_regularizer_alpha}"
            )
        self._connector_rms_regularizer_eps = float(
            config.connector_rms_regularizer_eps or 1e-6
        )
        if self._connector_rms_regularizer_eps <= 0:
            raise ValueError(
                "connector_rms_regularizer_eps must be > 0, "
                f"got {config.connector_rms_regularizer_eps}"
            )
        self._connector_rms_regularizer_target = str(
            config.connector_rms_regularizer_target or "batch_text"
        ).strip().lower()
        if self._connector_rms_regularizer_target not in {
            "batch_text",
            "prompt_text",
            "supervised_text",
        }:
            raise ValueError(
                "connector_rms_regularizer_target must be one of "
                "batch_text, prompt_text, supervised_text; got "
                f"{config.connector_rms_regularizer_target!r}"
            )
        if self._connector_rms_regularizer_alpha > 0:
            print_rank_0(
                "Stage 1 connector RMS regularizer enabled: "
                f"alpha={self._connector_rms_regularizer_alpha:g}, "
                f"target={self._connector_rms_regularizer_target}"
            )
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
        contract_metrics = self._validate_stage1_batch_contract(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        embedding_metrics = self._compute_stage1_embedding_metrics(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

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
            rms_loss, rms_metrics = self._connector_rms_regularizer_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                reference_loss=objective_loss,
            )
            if rms_loss is not None:
                objective_loss = objective_loss + rms_loss
                objective_metrics.update(rms_metrics)
            backward_loss = objective_loss * self._loss_scale
            batch_metrics = {}
            batch_metrics.update(contract_metrics)
            batch_metrics.update(embedding_metrics)
            batch_metrics.update(self._get_connector_diagnostics())
            batch_metrics.update(objective_metrics)
            self._add_connector_embedding_ratios(batch_metrics)
            self._last_batch_metrics = batch_metrics
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

    def _connector_rms_regularizer_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        reference_loss: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if self._connector_rms_regularizer_alpha <= 0:
            return None, {}

        connector_rms = getattr(
            self.unwrapped_model,
            "_last_connector_output_rms_tensor",
            None,
        )
        if connector_rms is None:
            return None, {
                "train/connector_rms_regularizer_missing": 1.0,
            }

        target_rms = self._stage1_embedding_rms_target_tensor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if target_rms is None:
            return None, {
                "train/connector_rms_regularizer_missing_target": 1.0,
            }

        eps = reference_loss.new_tensor(self._connector_rms_regularizer_eps)
        connector = connector_rms.to(device=reference_loss.device, dtype=torch.float32)
        target = target_rms.to(device=reference_loss.device, dtype=torch.float32)
        log_ratio = torch.log((connector + eps) / (target + eps))
        rms_loss = reference_loss.new_tensor(
            self._connector_rms_regularizer_alpha
        ) * log_ratio.pow(2)
        return rms_loss.to(dtype=reference_loss.dtype), {
            "train/connector_rms_regularizer_loss": float(rms_loss.detach().item()),
            "train/connector_rms_regularizer_alpha": self._connector_rms_regularizer_alpha,
            "train/connector_rms_regularizer_target": float(target.detach().item()),
            "train/connector_rms_regularizer_log_ratio": float(log_ratio.detach().item()),
        }

    def _validate_stage1_batch_contract(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
    ) -> Dict[str, float]:
        attention_bool = attention_mask.bool()
        supervised_mask = labels != -100
        supervised_tokens = int(supervised_mask.sum().item())
        active_tokens = int(attention_bool.sum().item())
        if supervised_tokens <= 0:
            raise ValueError("Stage 1 batch has zero supervised tokens.")
        if (supervised_mask & ~attention_bool).any():
            raise ValueError("Stage 1 batch has supervised labels on masked attention tokens.")

        placeholder_id = getattr(self.unwrapped_model, "image_placeholder_token_id", None)
        expected_count = getattr(self.unwrapped_model, "num_image_tokens", None)
        if expected_count is None:
            expected_count = getattr(self.unwrapped_model, "max_image_tokens", None)
        base_metrics = {
            "train/active_tokens": float(active_tokens),
            "train/supervised_token_rate": supervised_tokens / max(active_tokens, 1),
            "train/supervised_token_ratio": supervised_tokens / max(active_tokens, 1),
        }
        if placeholder_id is None or expected_count is None:
            base_metrics["train/placeholder_contract_checked"] = 0.0
            return base_metrics

        placeholder_counts = []
        placeholder_label_violations = 0
        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx]
            active = attention_bool[batch_idx]
            placeholder_mask = (ids == placeholder_id) & active
            indices = placeholder_mask.nonzero(as_tuple=True)[0]
            count = int(indices.numel())
            placeholder_counts.append(count)
            if count != int(expected_count):
                raise ValueError(
                    "Stage 1 batch placeholder count mismatch: "
                    f"sample={batch_idx}, expected={expected_count}, got={count}."
                )
            if count:
                if (indices[-1] - indices[0] + 1) != count:
                    raise ValueError(
                        "Stage 1 batch placeholders are not contiguous: "
                        f"sample={batch_idx}, count={count}."
                    )
                placeholder_labels = labels[batch_idx][placeholder_mask]
                placeholder_label_violations += int((placeholder_labels != -100).sum().item())

        if placeholder_label_violations:
            raise ValueError(
                "Stage 1 batch has supervised labels on image placeholder tokens: "
                f"{placeholder_label_violations} violations."
            )

        total_placeholders = int(sum(placeholder_counts))
        base_metrics.update(
            {
                "train/image_placeholder_tokens": float(total_placeholders),
                "train/image_placeholder_count_per_sample": float(
                    total_placeholders / max(len(placeholder_counts), 1)
                ),
                "train/placeholder_contract_checked": 1.0,
                "train/placeholder_contract_valid": 1.0,
            }
        )
        return base_metrics

    @torch.no_grad()
    def _compute_stage1_embedding_metrics(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
    ) -> Dict[str, float]:
        embedding_layer = self.unwrapped_model.llm.get_input_embeddings()
        token_embeds = embedding_layer(input_ids)
        attention_bool = attention_mask.bool()
        placeholder_id = getattr(self.unwrapped_model, "image_placeholder_token_id", None)
        non_placeholder = input_ids != placeholder_id if placeholder_id is not None else torch.ones_like(attention_bool)
        active_text = attention_bool & non_placeholder
        prompt_text = active_text & (labels == -100)
        supervised_text = active_text & (labels != -100)

        metrics = {}
        for name, mask in (
            ("train/qwen_batch_token_embedding_rms", active_text),
            ("train/qwen_prompt_embedding_rms", prompt_text),
            ("train/qwen_supervised_embedding_rms", supervised_text),
        ):
            if mask.any():
                selected = token_embeds[mask]
                metrics[name] = float(selected.detach().float().pow(2).mean().sqrt().item())
        return metrics

    @torch.no_grad()
    def _stage1_embedding_rms_target_tensor(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        embedding_layer = self.unwrapped_model.llm.get_input_embeddings()
        token_embeds = embedding_layer(input_ids)
        attention_bool = attention_mask.bool()
        placeholder_id = getattr(self.unwrapped_model, "image_placeholder_token_id", None)
        non_placeholder = (
            input_ids != placeholder_id
            if placeholder_id is not None
            else torch.ones_like(attention_bool)
        )
        active_text = attention_bool & non_placeholder
        if self._connector_rms_regularizer_target == "prompt_text":
            mask = active_text & (labels == -100)
        elif self._connector_rms_regularizer_target == "supervised_text":
            mask = active_text & (labels != -100)
        else:
            mask = active_text
        if not mask.any():
            return None
        return token_embeds[mask].detach().float().pow(2).mean().sqrt()

    def _get_connector_diagnostics(self) -> Dict[str, float]:
        diagnostics = getattr(self.unwrapped_model, "_last_connector_diagnostics", None)
        if not diagnostics:
            return {}
        return {
            key: float(value)
            for key, value in diagnostics.items()
            if isinstance(value, (int, float))
        }

    @staticmethod
    def _add_connector_embedding_ratios(metrics: Dict[str, float]) -> None:
        connector_rms = metrics.get("train/connector_output_rms")
        if connector_rms is None:
            return
        for denom_key, ratio_key in (
            ("train/qwen_batch_token_embedding_rms", "train/connector_to_qwen_token_rms_ratio"),
            ("train/qwen_prompt_embedding_rms", "train/connector_to_qwen_prompt_rms_ratio"),
            ("train/qwen_supervised_embedding_rms", "train/connector_to_qwen_supervised_rms_ratio"),
        ):
            denom = metrics.get(denom_key)
            if denom and denom > 0:
                metrics[ratio_key] = connector_rms / denom

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
