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

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager

from .trainer import Trainer, TrainerConfig
from .distributed import print_rank_0, is_main_process
from models.projectors import PerceiverResampler, QuestionConditionedPerceiverResampler


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
    connector_scale_only_training: bool = False
    connector_trainable_prefixes: Tuple[str, ...] = ()
    vision_trainable_prefixes: Tuple[str, ...] = ()
    connector_warmup_trainable_prefixes: Tuple[str, ...] = (
        "projector.output_proj",
        "projector.output_gate_logit",
        "projector.trainable_output_log_scale",
    )
    contrastive_answer_suppression: bool = False
    contrastive_lambda: float = 0.1
    contrastive_margin: float = 0.5
    teacher_kl_weight: float = 0.0
    teacher_kl_image_tokens: int = 0
    teacher_kl_temperature: float = 1.0
    teacher_kl_direction: str = "teacher_to_student"
    teacher_kl_checkpoint: str = ""
    teacher_kl_cache_path: str = ""
    teacher_kl_cache_top_k: int = 0

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
        self._connector_scale_only_training = bool(config.connector_scale_only_training)
        self._connector_trainable_prefixes = self._normalize_prefixes(
            config.connector_trainable_prefixes
        )
        self._freeze_connector_for_stage1 = self._is_freeze_prefix_sentinel(
            self._connector_trainable_prefixes
        )
        if (
            self._connector_scale_only_training
            and self._connector_trainable_prefixes
            and not self._freeze_connector_for_stage1
        ):
            raise ValueError(
                "connector_scale_only_training and connector_trainable_prefixes "
                "are mutually exclusive"
            )
        if self._freeze_connector_for_stage1:
            self._enable_connector_frozen_training(model)
        elif self._connector_scale_only_training:
            self._enable_connector_scale_only_training(model)
        elif self._connector_trainable_prefixes:
            self._enable_connector_prefix_only_training(
                model,
                self._connector_trainable_prefixes,
            )
        self._vision_trainable_prefixes = self._normalize_prefixes(
            config.vision_trainable_prefixes
        )
        if self._vision_trainable_prefixes:
            self._enable_vision_prefix_only_training(
                model,
                self._vision_trainable_prefixes,
            )

        self._connector_warmup_active = False
        self._loss_scale = float(config.loss_scale)
        if self._loss_scale <= 0:
            raise ValueError(f"loss_scale must be > 0, got {config.loss_scale}")
        if self._loss_scale != 1.0:
            print_rank_0(f"Stage 1 loss scale: multiplying backward loss by {self._loss_scale:g}")
        self._contrastive_answer_suppression = bool(
            config.contrastive_answer_suppression
        )
        self._contrastive_lambda = float(config.contrastive_lambda)
        self._contrastive_margin = float(config.contrastive_margin)
        self._teacher_kl_weight = float(config.teacher_kl_weight or 0.0)
        self._teacher_kl_temperature = float(config.teacher_kl_temperature or 1.0)
        self._teacher_kl_direction = str(
            config.teacher_kl_direction or "teacher_to_student"
        ).strip().lower().replace("-", "_")
        self._teacher_kl_image_tokens = int(config.teacher_kl_image_tokens or 0)
        self._teacher_kl_checkpoint = str(config.teacher_kl_checkpoint or "").strip()
        self._teacher_kl_cache_path = str(config.teacher_kl_cache_path or "").strip()
        self._teacher_kl_cache_top_k = int(config.teacher_kl_cache_top_k or 0)
        self._teacher_kl_cache_entries = {}
        self._teacher_kl_cache_meta = {}
        self._teacher_kl_mode = "none"
        self._teacher_kl_sidecar_projector = None
        if self._teacher_kl_weight < 0:
            raise ValueError(
                f"teacher_kl_weight must be >= 0, got {config.teacher_kl_weight}"
            )
        if self._teacher_kl_temperature <= 0:
            raise ValueError(
                "teacher_kl_temperature must be > 0, "
                f"got {config.teacher_kl_temperature}"
            )
        if self._teacher_kl_direction not in {
            "teacher_to_student",
            "student_to_teacher",
        }:
            raise ValueError(
                "teacher_kl_direction must be 'teacher_to_student' or "
                f"'student_to_teacher', got {config.teacher_kl_direction!r}"
            )
        self._teacher_kl_enabled = self._teacher_kl_weight > 0
        if self._teacher_kl_enabled:
            base_tokens = int(getattr(model, "v3_base_image_tokens", 0) or 0)
            has_spatial_tail = (
                hasattr(model, "projector")
                and getattr(
                    getattr(model, "projector", None),
                    "v3_spatial_tail_branch",
                    None,
                )
                is not None
            )
            uses_visual_cross_attention = bool(
                hasattr(model, "_uses_visual_cross_attention")
                and model._uses_visual_cross_attention()
            )
            supports_sidecar_teacher = getattr(model, "connector_type", None) in {
                "spatial_grid_projector",
                "mlp_anyres_projector",
            }
            if self._teacher_kl_image_tokens <= 0:
                if has_spatial_tail and base_tokens > 0:
                    self._teacher_kl_image_tokens = base_tokens
                elif self._teacher_kl_checkpoint:
                    self._teacher_kl_image_tokens = (
                        self._infer_teacher_kl_image_tokens_from_checkpoint(
                            self._teacher_kl_checkpoint
                        )
                    )
            if self._teacher_kl_image_tokens <= 0:
                raise ValueError(
                    "teacher_kl_image_tokens must be set when teacher KL is enabled"
                )
            if has_spatial_tail and self._teacher_kl_image_tokens == base_tokens:
                self._teacher_kl_mode = "spatial_tail_self"
            elif (
                uses_visual_cross_attention
                and self._teacher_kl_image_tokens
                == int(getattr(model, "num_image_tokens", 0) or 0)
            ):
                self._teacher_kl_mode = "visual_cross_attention_self"
            elif supports_sidecar_teacher and self._teacher_kl_checkpoint:
                self._teacher_kl_mode = "sidecar_projector"
            else:
                raise ValueError(
                    "Stage 1 teacher KL supports either V3 spatial-tail students "
                    "with teacher_kl_image_tokens equal to model.v3_base_image_tokens "
                    "or V3 visual-cross-attention students with teacher_kl_image_tokens "
                    "equal to model.num_image_tokens "
                    "or sidecar-compatible students (spatial_grid_projector or "
                    "mlp_anyres_projector) with teacher_kl_checkpoint set to a V11 "
                    "Perceiver checkpoint."
                )
            if self._teacher_kl_cache_path:
                self._load_teacher_kl_cache()
                cache_top_k = int(
                    self._teacher_kl_cache_meta.get("top_k")
                    or self._teacher_kl_cache_top_k
                    or 0
                )
                model.stage1_teacher_kl_cache_path = self._teacher_kl_cache_path
                model.stage1_teacher_kl_cache_schema = self._teacher_kl_cache_meta.get(
                    "schema"
                )
                model.stage1_teacher_kl_cache_top_k = cache_top_k
                model.stage1_teacher_kl_cache_entries = len(
                    self._teacher_kl_cache_entries
                )
            if self._teacher_kl_mode == "spatial_tail_self":
                trainable_base = self._v3_teacher_kl_trainable_base_params(model)
                if trainable_base:
                    preview = trainable_base[:10]
                    suffix = "" if len(trainable_base) <= 10 else " ..."
                    raise ValueError(
                        "Stage 1 teacher KL uses the V11 prefix as the teacher, "
                        "so non-tail V3 connector parameters must be frozen. "
                        "Use connector_trainable_prefixes=projector.v3_spatial_tail_branch. "
                        f"Trainable non-tail connector params: {preview}{suffix}"
                    )
            print_rank_0(
                "Stage 1 V11 teacher KL enabled: "
                f"mode={self._teacher_kl_mode}, "
                f"weight={self._teacher_kl_weight:g}, "
                f"teacher_image_tokens={self._teacher_kl_image_tokens}, "
                f"temperature={self._teacher_kl_temperature:g}, "
                f"direction={self._teacher_kl_direction}"
            )
            model.stage1_teacher_kl_mode = self._teacher_kl_mode
            model.stage1_teacher_kl_weight = self._teacher_kl_weight
            model.stage1_teacher_kl_image_tokens = self._teacher_kl_image_tokens
            model.stage1_teacher_kl_temperature = self._teacher_kl_temperature
            model.stage1_teacher_kl_direction = self._teacher_kl_direction
            model.stage1_teacher_kl_checkpoint = self._teacher_kl_checkpoint or None
        if self._contrastive_answer_suppression:
            if self._contrastive_lambda < 0:
                raise ValueError(
                    "contrastive_lambda must be >= 0, "
                    f"got {config.contrastive_lambda}"
                )
            print_rank_0(
                "Stage 1 contrastive answer suppression enabled: "
                f"lambda={self._contrastive_lambda:g}, "
                f"margin={self._contrastive_margin:g}"
            )
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
        self._connector_warmup_trainable_prefixes = self._normalize_prefixes(
            config.connector_warmup_trainable_prefixes
        )
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
        if self._teacher_kl_mode == "sidecar_projector" and not self._teacher_kl_cache_entries:
            self._teacher_kl_sidecar_projector = self._load_teacher_kl_sidecar_projector()
        elif self._teacher_kl_mode == "sidecar_projector":
            print_rank_0(
                "Using cached teacher KL distributions; skipping sidecar V11 "
                "projector load."
            )

    def _enable_connector_scale_only_training(self, model) -> None:
        """Freeze the connector except its opt-in trainable output scale."""
        trainable_scale_params = []
        for name, param in model.named_parameters():
            if not name.startswith("projector."):
                continue
            keep_trainable = name == "projector.trainable_output_log_scale"
            param.requires_grad = keep_trainable
            if keep_trainable:
                trainable_scale_params.append((name, param.numel()))

        if not trainable_scale_params:
            raise ValueError(
                "connector_scale_only_training=True requires a V3 connector with "
                "trainable_scale_mode set to 'global' or 'per_token'."
            )

        total = sum(count for _name, count in trainable_scale_params)
        print_rank_0(
            "Stage 1 connector scale-only training: "
            f"{len(trainable_scale_params)} tensor(s), {total:,} parameter(s)"
        )

    def _enable_connector_prefix_only_training(
        self,
        model,
        prefixes: Tuple[str, ...],
    ) -> None:
        """Freeze connector parameters except those matching explicit prefixes."""
        active_params = []
        frozen_tensors = 0
        for name, param in model.named_parameters():
            if not self._is_connector_param_name(name):
                continue
            keep_trainable = self._matches_trainable_prefix(name, prefixes)
            param.requires_grad = keep_trainable
            if keep_trainable:
                active_params.append((name, param.numel()))
            else:
                frozen_tensors += 1

        if not active_params:
            raise ValueError(
                "connector_trainable_prefixes was requested, but no connector "
                f"parameters matched prefixes={prefixes}"
            )

        total = sum(count for _name, count in active_params)
        print_rank_0(
            "Stage 1 connector prefix-only training: "
            f"prefixes={prefixes}, {len(active_params)} active tensor(s), "
            f"{total:,} parameter(s), {frozen_tensors} frozen tensor(s)"
        )

    def _enable_connector_frozen_training(self, model) -> None:
        frozen_tensors = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if not self._is_connector_param_name(name):
                continue
            param.requires_grad = False
            frozen_tensors += 1
            frozen_params += param.numel()

        print_rank_0(
            "Stage 1 connector frozen by connector_trainable_prefixes sentinel: "
            f"{frozen_tensors} tensor(s), {frozen_params:,} parameter(s)"
        )

    def _enable_vision_prefix_only_training(
        self,
        model,
        prefixes: Tuple[str, ...],
    ) -> None:
        """Unfreeze only explicitly selected vision tower parameters."""
        image_encoder = getattr(model, "image_encoder", None)
        if image_encoder is None:
            raise ValueError("vision_trainable_prefixes requires model.image_encoder")

        active_params = []
        frozen_tensors = 0
        for local_name, param in image_encoder.named_parameters():
            full_name = f"image_encoder.{local_name}"
            keep_trainable = self._matches_trainable_prefix(full_name, prefixes) or (
                self._matches_trainable_prefix(local_name, prefixes)
            )
            param.requires_grad = keep_trainable
            if keep_trainable:
                active_params.append((full_name, param.numel()))
            else:
                frozen_tensors += 1

        if not active_params:
            raise ValueError(
                "vision_trainable_prefixes was requested, but no image encoder "
                f"parameters matched prefixes={prefixes}"
            )

        if hasattr(image_encoder, "model"):
            image_encoder.model.train()
        else:
            image_encoder.train()

        total = sum(count for _name, count in active_params)
        print_rank_0(
            "Stage 1 vision prefix-only training: "
            f"prefixes={prefixes}, {len(active_params)} active tensor(s), "
            f"{total:,} parameter(s), {frozen_tensors} frozen tensor(s)"
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
        visual_cross_attention = getattr(
            model,
            "visual_cross_attention_adapters",
            None,
        )
        if visual_cross_attention is not None:
            adapter_modules.append(visual_cross_attention)
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
        teacher_answer_logits = None
        teacher_answer_labels = None
        use_cached_teacher_kl = self._teacher_kl_enabled and bool(
            self._teacher_kl_cache_entries
        )
        if self._teacher_kl_enabled and not use_cached_teacher_kl:
            teacher_batch = self._make_teacher_prefix_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                teacher_image_tokens=self._teacher_kl_image_tokens,
            )
            teacher_model = self._teacher_kl_model()
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                if self._teacher_kl_mode == "sidecar_projector":
                    teacher_image_tokens = self._encode_sidecar_teacher_image_tokens(images)
                    teacher_context = self._v3_teacher_token_count_context()
                    teacher_forward_kwargs = {
                        "images": None,
                        "input_ids": teacher_batch["input_ids"],
                        "attention_mask": teacher_batch["attention_mask"],
                        "labels": None,
                        "image_tokens": teacher_image_tokens,
                    }
                elif self._teacher_kl_mode == "spatial_tail_self":
                    teacher_context = self._v3_spatial_tail_teacher_context()
                    teacher_forward_kwargs = {
                        "images": images,
                        "input_ids": teacher_batch["input_ids"],
                        "attention_mask": teacher_batch["attention_mask"],
                        "labels": None,
                    }
                elif self._teacher_kl_mode == "visual_cross_attention_self":
                    teacher_context = self._v3_visual_cross_attention_teacher_context()
                    teacher_forward_kwargs = {
                        "images": images,
                        "input_ids": teacher_batch["input_ids"],
                        "attention_mask": teacher_batch["attention_mask"],
                        "labels": None,
                    }
                else:
                    raise RuntimeError(f"Unsupported teacher KL mode: {self._teacher_kl_mode}")
                with teacher_context:
                    with torch.amp.autocast(
                        device_type=device_type,
                        dtype=self.amp_dtype if torch.cuda.is_available() else torch.float32,
                        enabled=self.config.use_amp and torch.cuda.is_available(),
                    ):
                        teacher_outputs = teacher_model(**teacher_forward_kwargs)
                teacher_answer_logits, teacher_answer_labels = (
                    self._select_answer_token_logits(
                        teacher_outputs.logits,
                        teacher_batch["labels"],
                    )
                )
            if was_training:
                self.model.train()

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
            if self._teacher_kl_enabled:
                student_answer_logits, student_answer_labels = (
                    self._select_answer_token_logits(outputs.logits, labels)
                )
                if use_cached_teacher_kl:
                    teacher_kl_loss, teacher_kl_metrics = (
                        self._teacher_kl_loss_from_cache(
                            batch=batch,
                            student_answer_logits=student_answer_logits,
                            student_answer_labels=student_answer_labels,
                        )
                    )
                else:
                    teacher_kl_loss, teacher_kl_metrics = self._teacher_kl_loss(
                        student_answer_logits=student_answer_logits,
                        student_answer_labels=student_answer_labels,
                        teacher_answer_logits=teacher_answer_logits,
                        teacher_answer_labels=teacher_answer_labels,
                    )
                objective_loss = (
                    objective_loss + self._teacher_kl_weight * teacher_kl_loss
                )
                objective_metrics.update(teacher_kl_metrics)
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
                contrastive_loss = torch.zeros(
                    (), device=labels.device, dtype=objective_loss.dtype
                )
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
                    contrastive_loss = (
                        contrastive_loss + hinge.mean() / max(num_variants, 1)
                    )
                    contrastive_active_rate += (
                        float((hinge.detach() > 0).float().mean().item())
                        / max(num_variants, 1)
                    )
                objective_loss = (
                    objective_loss + self._contrastive_lambda * contrastive_loss
                )
                objective_metrics["train/contrastive_loss"] = float(
                    contrastive_loss.detach().item()
                )
                objective_metrics["train/contrastive_active_rate"] = (
                    contrastive_active_rate
                )
                objective_metrics["train/contrastive_variants"] = float(num_variants)
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

    def _make_teacher_prefix_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_image_tokens: int,
    ) -> Dict[str, torch.Tensor]:
        """Collapse the student placeholder block back to the V11 teacher size."""
        model = self._teacher_kl_model()
        placeholder_id = getattr(model, "image_placeholder_token_id", None)
        pad_token_id = getattr(getattr(model, "tokenizer", None), "pad_token_id", 0)
        if placeholder_id is None:
            raise ValueError("teacher KL requires model.image_placeholder_token_id")

        collapsed_ids = []
        collapsed_masks = []
        collapsed_labels = []
        max_len = 0
        for row_ids, row_mask, row_labels in zip(input_ids, attention_mask, labels):
            active_len = int(row_mask.sum().item())
            ids = row_ids[:active_len]
            labs = row_labels[:active_len]
            placeholder_positions = torch.nonzero(
                ids == int(placeholder_id),
                as_tuple=False,
            ).flatten()
            if placeholder_positions.numel() == 0:
                raise ValueError("teacher KL batch has no image placeholder block")
            first = int(placeholder_positions[0].item())
            last = int(placeholder_positions[-1].item())
            expected = torch.arange(
                first,
                last + 1,
                device=placeholder_positions.device,
                dtype=placeholder_positions.dtype,
            )
            if not torch.equal(placeholder_positions, expected):
                raise ValueError(
                    "teacher KL expects one contiguous image placeholder block"
                )
            current_tokens = int(placeholder_positions.numel())
            if current_tokens < teacher_image_tokens:
                raise ValueError(
                    "teacher KL cannot expand placeholder blocks: "
                    f"current={current_tokens}, teacher={teacher_image_tokens}"
                )
            keep_end = first + int(teacher_image_tokens)
            new_ids = torch.cat([ids[:keep_end], ids[last + 1 :]], dim=0)
            new_labels = torch.cat([labs[:keep_end], labs[last + 1 :]], dim=0)
            new_mask = torch.ones_like(new_ids, dtype=row_mask.dtype)
            collapsed_ids.append(new_ids)
            collapsed_masks.append(new_mask)
            collapsed_labels.append(new_labels)
            max_len = max(max_len, int(new_ids.shape[0]))

        padded_ids = []
        padded_masks = []
        padded_labels = []
        for ids, mask, labs in zip(collapsed_ids, collapsed_masks, collapsed_labels):
            pad_len = max_len - int(ids.shape[0])
            padded_ids.append(
                torch.cat(
                    [
                        ids,
                        torch.full(
                            (pad_len,),
                            int(pad_token_id),
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                    ],
                    dim=0,
                )
            )
            padded_masks.append(
                torch.cat(
                    [
                        mask,
                        torch.zeros(pad_len, dtype=mask.dtype, device=mask.device),
                    ],
                    dim=0,
                )
            )
            padded_labels.append(
                torch.cat(
                    [
                        labs,
                        torch.full(
                            (pad_len,),
                            -100,
                            dtype=labs.dtype,
                            device=labs.device,
                        ),
                    ],
                    dim=0,
                )
            )

        return {
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_masks, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
        }

    @staticmethod
    def _read_teacher_kl_checkpoint_meta(checkpoint: str) -> Dict[str, Any]:
        meta_path = os.path.join(checkpoint, "model_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"teacher_kl_checkpoint is missing model_meta.json: {meta_path}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid teacher checkpoint metadata in {meta_path}")
        return meta

    @classmethod
    def _infer_teacher_kl_image_tokens_from_checkpoint(cls, checkpoint: str) -> int:
        meta = cls._read_teacher_kl_checkpoint_meta(checkpoint)
        for key in ("num_image_tokens", "image_tokens", "image_placeholder_count"):
            value = meta.get(key)
            if value is not None:
                return int(value)
        raise ValueError(
            "teacher_kl_image_tokens was not set and could not be inferred from "
            f"{checkpoint}/model_meta.json"
        )

    def _load_teacher_kl_cache(self) -> None:
        """Load cached V11 answer-token top-k distributions for Stage 1 KL."""
        path = self._teacher_kl_cache_path
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"teacher_kl_cache_path does not exist: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid teacher KL cache payload in {path}")
        entries = payload.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"teacher KL cache has no entries: {path}")
        meta = {key: value for key, value in payload.items() if key != "entries"}
        schema = str(meta.get("schema") or "")
        if schema not in {"v14_teacher_kl_topk_v1"}:
            raise ValueError(
                "Unsupported teacher KL cache schema: "
                f"{schema!r} in {path}"
            )
        cache_checkpoint = str(meta.get("teacher_checkpoint") or "")
        if self._teacher_kl_checkpoint and cache_checkpoint:
            if os.path.normpath(cache_checkpoint) != os.path.normpath(
                self._teacher_kl_checkpoint
            ):
                raise ValueError(
                    "teacher KL cache checkpoint mismatch: "
                    f"cache={cache_checkpoint}, requested={self._teacher_kl_checkpoint}"
                )
        top_k = int(meta.get("top_k") or self._teacher_kl_cache_top_k or 0)
        if top_k <= 0:
            raise ValueError(f"teacher KL cache top_k must be positive: {path}")
        self._teacher_kl_cache_entries = entries
        self._teacher_kl_cache_meta = meta
        self._teacher_kl_cache_top_k = top_k
        print_rank_0(
            "Loaded cached V11 teacher KL distributions: "
            f"path={path}, entries={len(entries)}, top_k={top_k}, "
            f"schema={schema}"
        )

    def _load_teacher_kl_sidecar_projector(self) -> torch.nn.Module:
        """Load a frozen V11 projector for sidecar teacher imitation."""
        model = self._teacher_kl_model()
        checkpoint = self._teacher_kl_checkpoint
        meta = self._read_teacher_kl_checkpoint_meta(checkpoint)
        connector_type = str(meta.get("connector_type") or "perceiver_resampler")
        if connector_type == "perceiver_resampler":
            projector_cls = PerceiverResampler
            projector_kwargs = {}
        elif connector_type == "question_conditioned_perceiver_resampler":
            projector_cls = QuestionConditionedPerceiverResampler
            projector_kwargs = {"condition_dim": int(model.llm.hidden_size)}
        else:
            raise ValueError(
                "teacher_kl_checkpoint must contain a V11 Perceiver projector, "
                f"got connector_type={connector_type!r}"
            )

        meta_tokens = int(
            meta.get("num_image_tokens")
            or meta.get("image_tokens")
            or meta.get("image_placeholder_count")
            or 0
        )
        if meta_tokens != int(self._teacher_kl_image_tokens):
            raise ValueError(
                "teacher_kl_image_tokens must match teacher checkpoint metadata: "
                f"requested={self._teacher_kl_image_tokens}, checkpoint={meta_tokens}"
            )
        student_tokens = int(getattr(model, "num_image_tokens", 0) or 0)
        if student_tokens < self._teacher_kl_image_tokens:
            raise ValueError(
                "sidecar teacher KL cannot expand placeholder blocks: "
                f"student={student_tokens}, teacher={self._teacher_kl_image_tokens}"
            )

        projector = projector_cls(
            input_dim=int(model.image_encoder.get_output_dim()),
            output_dim=int(model.llm.hidden_size),
            num_latents=int(self._teacher_kl_image_tokens),
            num_layers=int(meta.get("connector_layers", 6)),
            num_heads=int(meta.get("connector_heads", 16)),
            ff_mult=int(meta.get("connector_ff_mult", 4)),
            output_scale=float(meta.get("connector_output_scale", 1.0)),
            output_gate_init=meta.get("connector_output_gate_init"),
            trainable_scale_mode=str(
                meta.get("connector_trainable_scale_mode") or "none"
            ),
            use_2d_patch_position_features=bool(
                meta.get("use_2d_patch_position_features", False)
            ),
            patch_position_feature_type=meta.get("patch_position_feature_type"),
            patch_position_grid_size=int(meta.get("patch_position_grid_size", 32)),
            patch_position_mlp_hidden_dim=int(
                meta.get("patch_position_mlp_hidden_dim", 128)
            ),
            patch_position_feature_scale=float(
                meta.get("patch_position_feature_scale", 1.0)
            ),
            query_conditioned_visual_scale_mode=str(
                meta.get("query_conditioned_visual_scale_mode") or "none"
            ),
            query_conditioned_visual_scale_min=float(
                meta.get("query_conditioned_visual_scale_min", 0.95)
            ),
            query_conditioned_visual_scale_max=float(
                meta.get("query_conditioned_visual_scale_max", 1.15)
            ),
            query_conditioned_visual_scale_init=meta.get(
                "query_conditioned_visual_scale_init"
            ),
            query_conditioned_patch_selector_mode=str(
                meta.get("query_conditioned_patch_selector_mode") or "none"
            ),
            query_conditioned_patch_selector_hidden_dim=int(
                meta.get("query_conditioned_patch_selector_hidden_dim", 256)
            ),
            query_conditioned_patch_selector_max_residual=float(
                meta.get("query_conditioned_patch_selector_max_residual", 0.25)
            ),
            query_conditioned_patch_selector_normalize_mean=bool(
                meta.get("query_conditioned_patch_selector_normalize_mean", True)
            ),
            **projector_kwargs,
        )
        projector_path = os.path.join(checkpoint, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(
                f"teacher_kl_checkpoint is missing projector.pt: {projector_path}"
            )
        state_dict = torch.load(projector_path, map_location="cpu")
        projector.load_state_dict(state_dict, strict=True)
        for param in projector.parameters():
            param.requires_grad_(False)
        student_projector_param = next(model.projector.parameters())
        projector = projector.to(
            device=self.device,
            dtype=student_projector_param.dtype,
        )
        projector.eval()
        print_rank_0(
            "Loaded sidecar V11 teacher projector for Stage 1 KL from "
            f"{checkpoint} ({self._teacher_kl_image_tokens} image tokens)"
        )
        return projector

    def _encode_sidecar_teacher_image_tokens(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        teacher_projector = self._teacher_kl_sidecar_projector
        if teacher_projector is None:
            raise ValueError("Sidecar teacher projector has not been loaded")
        model = self._teacher_kl_model()
        if images.ndim == 5:
            images = images[:, 0]
        if images.dtype != torch.float32:
            images = images.to(torch.float32)
        vision_features = model.image_encoder(images)
        projector_param = next(teacher_projector.parameters())
        vision_features = vision_features.to(
            device=projector_param.device,
            dtype=projector_param.dtype,
        )
        image_tokens = teacher_projector(vision_features)
        if int(image_tokens.shape[1]) != int(self._teacher_kl_image_tokens):
            raise ValueError(
                "Sidecar teacher projector produced the wrong token count: "
                f"expected={self._teacher_kl_image_tokens}, got={image_tokens.shape[1]}"
            )
        return image_tokens

    @classmethod
    def _v3_teacher_kl_trainable_base_params(cls, model) -> list:
        disallowed = []
        for name, param in model.named_parameters():
            if not param.requires_grad or not cls._is_connector_param_name(name):
                continue
            if name.startswith("projector.v3_spatial_tail_branch."):
                continue
            disallowed.append(name)
        return disallowed

    def _teacher_kl_model(self):
        model = getattr(self, "unwrapped_model", None)
        if model is not None:
            return model
        model = getattr(self, "model", None)
        if model is None:
            raise ValueError("teacher KL helpers require model or unwrapped_model")
        return getattr(model, "module", model)

    @contextmanager
    def _v3_teacher_token_count_context(self):
        """Temporarily run the shared LLM with the teacher placeholder count."""
        model = self._teacher_kl_model()
        original_num = getattr(model, "num_image_tokens", None)
        original_max = getattr(model, "max_image_tokens", None)
        original_min = getattr(model, "min_image_tokens", None)
        try:
            model.num_image_tokens = int(self._teacher_kl_image_tokens)
            model.max_image_tokens = int(self._teacher_kl_image_tokens)
            model.min_image_tokens = int(self._teacher_kl_image_tokens)
            yield
        finally:
            if original_num is not None:
                model.num_image_tokens = original_num
            if original_max is not None:
                model.max_image_tokens = original_max
            if original_min is not None:
                model.min_image_tokens = original_min

    @contextmanager
    def _v3_spatial_tail_teacher_context(self):
        """Run the same warm-started model as the 128-token V11 teacher."""
        model = self._teacher_kl_model()
        projector = getattr(model, "projector", None)
        tail = getattr(projector, "v3_spatial_tail_branch", None)
        if tail is None:
            raise ValueError("teacher KL context requires a V3 spatial tail branch")

        try:
            projector.v3_spatial_tail_branch = None
            with self._v3_teacher_token_count_context():
                yield
        finally:
            projector.v3_spatial_tail_branch = tail

    @contextmanager
    def _v3_visual_cross_attention_teacher_context(self):
        """Run the shared V11 prefix with decoder visual cross-attention disabled."""
        model = self._teacher_kl_model()
        original_mode = getattr(model, "visual_cross_attention_mode", None)
        original_memory = getattr(model, "_active_visual_memory", None)
        try:
            model.visual_cross_attention_mode = "none"
            model._active_visual_memory = None
            with self._v3_teacher_token_count_context():
                yield
        finally:
            if original_mode is not None:
                model.visual_cross_attention_mode = original_mode
            model._active_visual_memory = original_memory

    @staticmethod
    def _select_answer_token_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        selected_logits = []
        selected_labels = []
        for row_logits, row_labels in zip(shift_logits, shift_labels):
            mask = row_labels != -100
            selected_logits.append(row_logits[mask])
            selected_labels.append(row_labels[mask])
        if not selected_logits or sum(x.shape[0] for x in selected_logits) == 0:
            raise ValueError("teacher KL requires at least one supervised answer token")
        return (
            torch.cat(selected_logits, dim=0),
            torch.cat(selected_labels, dim=0),
        )

    def _teacher_kl_loss(
        self,
        student_answer_logits: torch.Tensor,
        student_answer_labels: torch.Tensor,
        teacher_answer_logits: torch.Tensor,
        teacher_answer_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if student_answer_logits.shape != teacher_answer_logits.shape:
            raise ValueError(
                "teacher KL answer-logit shape mismatch: "
                f"student={list(student_answer_logits.shape)}, "
                f"teacher={list(teacher_answer_logits.shape)}"
            )
        if not torch.equal(student_answer_labels, teacher_answer_labels):
            raise ValueError("teacher KL answer-token labels do not align")

        temperature = float(self._teacher_kl_temperature)
        student_logp = F.log_softmax(
            student_answer_logits.float() / temperature,
            dim=-1,
        )
        teacher_logp = F.log_softmax(
            teacher_answer_logits.float() / temperature,
            dim=-1,
        ).detach()
        if self._teacher_kl_direction == "student_to_teacher":
            student_probs = student_logp.exp()
            kl = (student_probs * (student_logp - teacher_logp)).sum(dim=-1).mean()
        else:
            teacher_probs = teacher_logp.exp()
            kl = F.kl_div(
                student_logp,
                teacher_probs,
                reduction="batchmean",
            )
        kl = kl * (temperature ** 2)
        return kl.to(dtype=student_answer_logits.dtype), {
            "train/teacher_kl_loss": float(kl.detach().item()),
            "train/teacher_kl_weight": float(self._teacher_kl_weight),
            "train/teacher_kl_answer_tokens": float(student_answer_labels.numel()),
            "train/teacher_kl_temperature": float(temperature),
        }

    def _cache_entries_for_batch(
        self,
        batch: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            raise ValueError(
                "teacher KL cache requires batch sample_id metadata. "
                "Use an InstructionDataset/InstructionMixtureDataset source."
            )
        entries = []
        missing = []
        for sample_id in sample_ids:
            key = str(sample_id)
            entry = self._teacher_kl_cache_entries.get(key)
            if entry is None:
                missing.append(key)
            else:
                entries.append(entry)
        if missing:
            preview = missing[:5]
            suffix = "" if len(missing) <= 5 else " ..."
            raise KeyError(
                "teacher KL cache is missing batch sample_id(s): "
                f"{preview}{suffix}"
            )
        return entries

    @staticmethod
    def _cache_tensor(entry: Dict[str, Any], key: str) -> torch.Tensor:
        value = entry.get(key)
        if value is None:
            raise ValueError(f"teacher KL cache entry missing {key!r}")
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _teacher_kl_loss_from_cache(
        self,
        batch: Dict[str, Any],
        student_answer_logits: torch.Tensor,
        student_answer_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        entries = self._cache_entries_for_batch(batch)
        labels = torch.cat(
            [self._cache_tensor(entry, "labels").long() for entry in entries],
            dim=0,
        ).to(device=student_answer_labels.device)
        if not torch.equal(labels, student_answer_labels):
            raise ValueError(
                "teacher KL cache labels do not align with student answer tokens"
            )

        topk_ids = torch.cat(
            [self._cache_tensor(entry, "topk_ids").long() for entry in entries],
            dim=0,
        ).to(device=student_answer_logits.device)
        topk_probs = torch.cat(
            [self._cache_tensor(entry, "topk_probs").float() for entry in entries],
            dim=0,
        ).to(device=student_answer_logits.device)
        remainder_probs = torch.cat(
            [self._cache_tensor(entry, "remainder_probs").float() for entry in entries],
            dim=0,
        ).to(device=student_answer_logits.device)

        if topk_ids.ndim != 2 or topk_probs.shape != topk_ids.shape:
            raise ValueError(
                "teacher KL cache top-k tensors must have shape [answer_tokens, top_k]"
            )
        if remainder_probs.ndim != 1 or remainder_probs.shape[0] != topk_ids.shape[0]:
            raise ValueError(
                "teacher KL cache remainder_probs must have shape [answer_tokens]"
            )
        if student_answer_logits.shape[0] != topk_ids.shape[0]:
            raise ValueError(
                "teacher KL cache answer-token count mismatch: "
                f"student={student_answer_logits.shape[0]}, cache={topk_ids.shape[0]}"
            )

        temperature = float(self._teacher_kl_temperature)
        student_logp = F.log_softmax(
            student_answer_logits.float() / temperature,
            dim=-1,
        )
        student_probs = student_logp.exp()
        topk_student_logp = student_logp.gather(dim=-1, index=topk_ids)
        topk_student_probs = student_probs.gather(dim=-1, index=topk_ids)

        eps = torch.finfo(topk_probs.dtype).tiny
        teacher_remainder = remainder_probs.clamp_min(eps)
        student_remainder = (
            1.0 - topk_student_probs.sum(dim=-1)
        ).clamp_min(eps)
        teacher_topk = topk_probs.clamp_min(eps)

        if self._teacher_kl_direction == "student_to_teacher":
            kl_terms = topk_student_probs * (
                topk_student_probs.clamp_min(eps).log() - teacher_topk.log()
            )
            remainder_terms = student_remainder * (
                student_remainder.log() - teacher_remainder.log()
            )
        else:
            kl_terms = teacher_topk * (
                teacher_topk.log() - topk_student_logp
            )
            remainder_terms = teacher_remainder * (
                teacher_remainder.log() - student_remainder.log()
            )

        kl = (kl_terms.sum(dim=-1) + remainder_terms).mean()
        kl = kl * (temperature ** 2)
        student_top1 = student_answer_logits.detach().argmax(dim=-1)
        teacher_top1 = topk_ids[:, 0]
        gold_in_topk = (topk_ids == labels.unsqueeze(-1)).any(dim=-1).float().mean()
        return kl.to(dtype=student_answer_logits.dtype), {
            "train/teacher_kl_loss": float(kl.detach().item()),
            "train/teacher_kl_weight": float(self._teacher_kl_weight),
            "train/teacher_kl_answer_tokens": float(student_answer_labels.numel()),
            "train/teacher_kl_temperature": float(temperature),
            "train/teacher_kl_cache_hits": float(len(entries)),
            "train/teacher_kl_cache_top_k": float(topk_ids.shape[1]),
            "train/teacher_kl_cache_remainder_mass": float(
                remainder_probs.detach().mean().item()
            ),
            "train/student_teacher_top1_token_agreement": float(
                (student_top1 == teacher_top1).float().mean().item()
            ),
            "train/gold_token_in_teacher_topk_rate": float(gold_in_topk.item()),
        }

    @staticmethod
    def _answer_logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~mask, 0)
        token_logp = torch.log_softmax(shift_logits, dim=-1)
        answer_logp = token_logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (answer_logp * mask).sum(dim=1)

    def _prepare_eval_batch_for_model(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        trainer_only = {
            "sample_id",
            "image_ref",
            "question_text",
            "answer_text",
            "mixture_source",
            "source_local_index",
        }
        if self._contrastive_answer_suppression:
            trainer_only.add("negative_images")
        return {key: value for key, value in batch.items() if key not in trainer_only}

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

    @staticmethod
    def _normalize_prefixes(prefixes: Any) -> Tuple[str, ...]:
        if prefixes is None:
            return ()
        if isinstance(prefixes, str):
            prefixes = prefixes.replace(" ", ",").split(",")
        return tuple(str(prefix).strip() for prefix in prefixes if str(prefix).strip())

    @staticmethod
    def _is_freeze_prefix_sentinel(prefixes: Tuple[str, ...]) -> bool:
        return len(prefixes) == 1 and prefixes[0].lower() in {
            "__none__",
            "none",
            "freeze",
            "frozen",
            "off",
        }

    @staticmethod
    def _matches_trainable_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)

    def _is_connector_warmup_trainable_param_name(self, name: str) -> bool:
        return self._matches_trainable_prefix(
            name,
            self._connector_warmup_trainable_prefixes,
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
