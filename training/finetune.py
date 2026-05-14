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
from model_metadata import read_model_metadata, validate_checkpoint_architecture


@dataclass
class FinetuneConfig(TrainerConfig):
    """Configuration for instruction fine-tuning."""
    # Stage 2 specific settings
    learning_rate: float = 2e-5  # Projector LR (LoRA uses lora_learning_rate)
    max_steps: int = 3000  # Much fewer steps
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2

    # LoRA settings (should match model initialization)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[tuple] = None

    # Continue from Stage 1 checkpoint
    pretrain_checkpoint: Optional[str] = None

    # Continue from a full Stage 2 checkpoint (projector, compressor, LoRA).
    finetune_checkpoint: Optional[str] = None

    # Adapter warmup: zero multimodal adapter grads for N steps to let LoRA warm up first
    projector_warmup_steps: int = 200

    # Whether Stage 2 should continue training the multimodal adapter
    # (projector / connector / token compressor) alongside LoRA.
    train_adapter: bool = True

    # Multiplies the backward loss only. The unscaled model loss remains the
    # train/loss value used for learning curves and health checks.
    loss_scale: float = 1.0

    # Optional V9 answer-suppression objective. When enabled, batches must
    # include negative_images shaped [B, K, 3, H, W] or [B, K, V, 3, H, W].
    contrastive_answer_suppression: bool = False
    contrastive_lambda: float = 0.1
    contrastive_margin: float = 0.5


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
        # Load Stage 1 checkpoint if provided. Full Stage 2 checkpoint loading is
        # handled by the Modal runner before trainer construction because it can
        # change the PEFT wrapper around the LLM.
        if config.pretrain_checkpoint and not config.finetune_checkpoint:
            self._load_pretrain_checkpoint(model, config.pretrain_checkpoint)

        # Configure model for Stage 2
        model.set_training_stage(2)

        self._train_adapter = bool(config.train_adapter)
        if not self._train_adapter:
            self._set_adapter_requires_grad(model, False)
            print_rank_0("Stage 2 adapter training disabled: training LoRA only")

        # Adapter warmup keeps adapter params in the optimizer from step 0, then
        # zeroes their gradients during warmup. This avoids rebuilding optimizer
        # and scheduler state mid-run.
        self._adapter_warmup_active = False
        self._projector_warmup_steps = config.projector_warmup_steps
        if self._train_adapter and config.projector_warmup_steps > 0:
            self._adapter_warmup_active = True
            print_rank_0(
                f"Multimodal adapter gradients zeroed for first "
                f"{config.projector_warmup_steps} steps (LoRA warmup)"
            )
        elif not self._train_adapter and config.projector_warmup_steps > 0:
            print_rank_0(
                "Ignoring projector_warmup_steps because adapter training is disabled"
            )

        self._loss_scale = float(config.loss_scale)
        if self._loss_scale <= 0:
            raise ValueError(f"loss_scale must be > 0, got {config.loss_scale}")
        if self._loss_scale != 1.0:
            print_rank_0(f"Stage 2 loss scale: multiplying backward loss by {self._loss_scale:g}")

        self._contrastive_answer_suppression = bool(config.contrastive_answer_suppression)
        self._contrastive_lambda = float(config.contrastive_lambda)
        self._contrastive_margin = float(config.contrastive_margin)
        if self._contrastive_answer_suppression:
            print_rank_0(
                "Contrastive answer suppression enabled: "
                f"lambda={self._contrastive_lambda:g}, margin={self._contrastive_margin:g}"
            )

        # Verify trainable params
        if is_main_process():
            self._verify_trainable_params(model)

        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

    @staticmethod
    def _answer_logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~mask, 0)
        token_logp = torch.log_softmax(shift_logits, dim=-1)
        answer_logp = token_logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (answer_logp * mask).sum(dim=1)

    def _load_pretrain_checkpoint(self, model, checkpoint_path: str):
        """Load pretrained projector weights from Stage 1."""
        expected_arch = getattr(model, "architecture", "anymal_v1")
        validate_checkpoint_architecture(
            checkpoint_dir=checkpoint_path,
            expected_architecture=expected_arch,
        )
        if expected_arch == "anymal_v2":
            meta = read_model_metadata(checkpoint_path) or {}
            checkpoint_compressor = meta.get("token_compressor_type")
            model_compressor = getattr(model, "token_compressor_type", None)
            if checkpoint_compressor and model_compressor and checkpoint_compressor != model_compressor:
                raise RuntimeError(
                    f"Checkpoint token_compressor_type mismatch for {checkpoint_path}: "
                    f"checkpoint={checkpoint_compressor}, model={model_compressor}."
                )
        elif expected_arch in {"anymal_v3", "anymal_v4"}:
            meta = read_model_metadata(checkpoint_path) or {}
            model_backbone = getattr(model, "llm_backbone", None)
            checkpoint_backbone = meta.get("llm_backbone")
            if model_backbone and model_backbone != "meta-llama/Meta-Llama-3-8B-Instruct":
                if checkpoint_backbone != model_backbone:
                    raise RuntimeError(
                        f"Checkpoint llm_backbone mismatch for {checkpoint_path}: "
                        f"checkpoint={checkpoint_backbone!r}, model={model_backbone!r}."
                    )
            for key in (
                "connector_type",
                "num_image_tokens",
                "connector_layers",
                "connector_heads",
                "connector_ff_mult",
            ):
                checkpoint_value = meta.get(key)
                model_value = getattr(model, key, None)
                if checkpoint_value is not None and model_value is not None and checkpoint_value != model_value:
                    raise RuntimeError(
                        f"Checkpoint {key} mismatch for {checkpoint_path}: "
                        f"checkpoint={checkpoint_value}, model={model_value}."
                    )
            if expected_arch == "anymal_v4":
                for key in (
                    "num_global_image_tokens",
                    "num_local_image_tokens",
                    "connector_hidden_dim",
                    "connector_output_scale",
                    "connector_output_gate_init",
                    "use_2d_position_features",
                ):
                    checkpoint_value = meta.get(key)
                    model_value = getattr(model, key, None)
                    if checkpoint_value is not None and model_value is not None and checkpoint_value != model_value:
                        raise RuntimeError(
                            f"Checkpoint {key} mismatch for {checkpoint_path}: "
                            f"checkpoint={checkpoint_value}, model={model_value}."
                        )
                if getattr(model, "connector_type", None) == "deepstack_spatial_perceiver_resampler":
                    for key in (
                        "deepstack_num_feature_levels",
                        "deepstack_hidden_state_indices",
                        "vision_feature_layers",
                    ):
                        checkpoint_value = meta.get(key)
                        if key in ("deepstack_hidden_state_indices", "vision_feature_layers"):
                            if checkpoint_value is not None:
                                checkpoint_value = [int(i) for i in checkpoint_value]
                            model_value = list(getattr(model, "deepstack_hidden_state_indices", []))
                        else:
                            model_value = getattr(model, key, None)
                        if (
                            checkpoint_value is not None
                            and model_value is not None
                            and checkpoint_value != model_value
                        ):
                            raise RuntimeError(
                                f"Checkpoint {key} mismatch for {checkpoint_path}: "
                                f"checkpoint={checkpoint_value}, model={model_value}."
                            )

        projector_path = os.path.join(checkpoint_path, "projector.pt")

        if os.path.exists(projector_path):
            print_rank_0(f"Loading projector from {projector_path}")
            state_dict = torch.load(projector_path, map_location="cpu")
            model.projector.load_state_dict(state_dict)
        else:
            print_rank_0(f"WARNING: No projector weights found at {projector_path}")

        if hasattr(model, "load_visual_cross_attention_adapters"):
            model.load_visual_cross_attention_adapters(
                checkpoint_path,
                map_location="cpu",
                allow_missing=False,
            )
        if hasattr(model, "load_vision_adapter"):
            model.load_vision_adapter(
                checkpoint_path,
                map_location="cpu",
            )

        compressor_path = os.path.join(checkpoint_path, "token_compressor.pt")
        if hasattr(model, "token_compressor") and os.path.exists(compressor_path):
            print_rank_0(f"Loading token compressor from {compressor_path}")
            compressor_state = torch.load(compressor_path, map_location="cpu")
            compressor_state = self._adapt_token_compressor_state(
                model.token_compressor,
                compressor_state,
            )
            model.token_compressor.load_state_dict(compressor_state)

    def _adapt_token_compressor_state(self, token_compressor, checkpoint_state):
        """
        Adapt Stage 1 compressor query tables when Stage 2 uses more image tokens.

        V2 currently trains Stage 1 with 256 image tokens and Stage 2 with 384.
        Learned/perceiver query tables are therefore longer in Stage 2, while
        attention/projection weights remain shape-compatible.
        """
        current_state = token_compressor.state_dict()
        adapted_state = {}

        for name, value in checkpoint_state.items():
            if name not in current_state:
                adapted_state[name] = value
                continue

            current_value = current_state[name]
            if current_value.shape == value.shape:
                adapted_state[name] = value
                continue

            can_expand_rows = (
                value.ndim >= 2
                and current_value.ndim == value.ndim
                and current_value.shape[1:] == value.shape[1:]
                and current_value.shape[0] >= value.shape[0]
                and name.endswith("queries")
            )
            if not can_expand_rows:
                raise RuntimeError(
                    f"Token compressor checkpoint shape mismatch for {name}: "
                    f"checkpoint={tuple(value.shape)}, model={tuple(current_value.shape)}"
                )

            expanded = current_value.clone()
            rows = value.shape[0]
            expanded[:rows] = value
            adapted_state[name] = expanded
            print_rank_0(
                f"Expanded token compressor {name}: copied {rows} pretrained rows "
                f"into {current_value.shape[0]} model rows"
            )

        return adapted_state

    def _verify_trainable_params(self, model):
        """Verify that projector + LoRA are trainable."""
        trainable_groups = {
            "adapter": 0,
            "lora": 0,
            "other": 0,
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_adapter_param_name(name):
                trainable_groups["adapter"] += param.numel()
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
        if not self._train_adapter and trainable_groups["adapter"] > 0:
            print_rank_0(
                "WARNING: adapter params are trainable even though "
                "train_adapter=False."
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
        if (
            self._adapter_warmup_active
            and self.global_step >= self._projector_warmup_steps
        ):
            print_rank_0(f"\n[Step {self.global_step}] Multimodal adapter warmup complete")
            self._adapter_warmup_active = False

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

        supervised_tokens = int((labels != -100).sum().item())
        active_tokens = int(attention_mask.sum().item())
        placeholder_id = getattr(self.unwrapped_model, "image_placeholder_token_id", None)
        placeholder_tokens = (
            int((input_ids == placeholder_id).sum().item())
            if placeholder_id is not None else 0
        )
        self._last_batch_metrics = {
            "train/supervised_tokens": supervised_tokens,
            "train/active_tokens": active_tokens,
            "train/supervised_token_ratio": supervised_tokens / max(active_tokens, 1),
            "train/image_placeholder_tokens": placeholder_tokens,
        }

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
            ce_loss = outputs.loss
            raw_loss = ce_loss
            if self._contrastive_answer_suppression:
                negative_images = batch.get("negative_images")
                if negative_images is None:
                    raise ValueError(
                        "contrastive_answer_suppression=True requires "
                        "negative_images in every batch"
                    )
                if negative_images.ndim not in {5, 6}:
                    raise ValueError(
                        "negative_images must have shape [B, K, 3, H, W] or "
                        "[B, K, V, 3, H, W], "
                        f"got {list(negative_images.shape)}"
                    )
                logp_pos = self._answer_logp_from_logits(outputs.logits, labels)
                num_variants = int(negative_images.shape[1])
                contrastive_loss = torch.zeros((), device=labels.device, dtype=ce_loss.dtype)
                contrastive_active_rate = 0.0
                for variant_idx in range(num_variants):
                    neg_outputs = self.model(
                        images=negative_images[:, variant_idx],
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                    )
                    logp_neg = self._answer_logp_from_logits(neg_outputs.logits, labels)
                    hinge = torch.relu(
                        self._contrastive_margin - logp_pos + logp_neg
                    )
                    contrastive_loss = contrastive_loss + hinge.mean() / max(num_variants, 1)
                    contrastive_active_rate += (
                        float((hinge.detach() > 0).float().mean().item())
                        / max(num_variants, 1)
                    )
                raw_loss = ce_loss + self._contrastive_lambda * contrastive_loss
                self._last_batch_metrics["train/ce_loss"] = ce_loss.detach().item()
                self._last_batch_metrics["train/contrastive_loss"] = (
                    contrastive_loss.detach().item()
                )
                self._last_batch_metrics["train/contrastive_active_rate"] = (
                    contrastive_active_rate
                )
            backward_loss = raw_loss * self._loss_scale

            # NaN/Inf check
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                print_rank_0(f"WARNING: loss is {raw_loss.item()} at step {self.global_step}!")

            self._last_batch_metrics["train/raw_loss"] = raw_loss.detach().item()
            self._last_batch_metrics["train/backward_loss"] = backward_loss.detach().item()

            # Scale loss for gradient accumulation
            loss = backward_loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self._adapter_warmup_active:
            self._zero_adapter_grads()

        # Return the unscaled model loss for logging and health monitoring.
        return raw_loss.item()

    def _prepare_eval_batch_for_model(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._contrastive_answer_suppression or "negative_images" not in batch:
            return batch
        return {key: value for key, value in batch.items() if key != "negative_images"}

    @staticmethod
    def _is_adapter_param_name(name: str) -> bool:
        return (
            "projector" in name
            or "token_compressor" in name
            or "visual_cross_attention_adapters" in name
        )

    def _set_adapter_requires_grad(self, model, requires_grad: bool) -> None:
        for name, param in model.named_parameters():
            if self._is_adapter_param_name(name):
                param.requires_grad = requires_grad

    def _zero_adapter_grads(self) -> None:
        for name, param in self.unwrapped_model.named_parameters():
            if self._is_adapter_param_name(name) and param.grad is not None:
                param.grad = None


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
