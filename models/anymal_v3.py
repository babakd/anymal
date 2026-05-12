"""
AnyMALv3 core architecture.

V3 stack:
- SigLIP2 vision encoder at 384px
- Deep Perceiver resampler projected directly to LLaMA hidden space
- Strict placeholder/token alignment checks inherited from V2
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .anymal_v2 import AnyMALv2
from .encoders import SigLIP2Encoder
from .llm import LlamaWrapper, canonicalize_llm_backbone
from .llm.backbone import CURRENT_LLAMA3_BACKBONE
from .projectors import PerceiverResampler, QuestionConditionedPerceiverResampler
from model_metadata import (
    read_model_metadata,
    validate_checkpoint_architecture,
    validate_checkpoint_metadata_values,
    write_model_metadata,
)


@dataclass
class AnyMALv3Output:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class AnyMALv3(AnyMALv2):
    """AnyMAL v3 multimodal model."""

    architecture = "anymal_v3"

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        llm_backbone: Optional[str] = None,
        vision_encoder_type: str = "siglip2",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        connector_type: str = "perceiver_resampler",
        num_image_tokens: int = 128,
        connector_layers: int = 6,
        connector_heads: int = 16,
        connector_ff_mult: int = 4,
        project_directly_to_llm_dim: bool = True,
        connector_output_scale: float = 1.0,
        connector_output_gate_init: Optional[float] = None,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        use_qlora: bool = True,
        use_lora: Optional[bool] = None,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = True,
        llm_device_map: Union[str, Dict[str, Union[int, str]], None] = "auto",
        llm_torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        self.llm_backbone = canonicalize_llm_backbone(llm_backbone or llm_model_name)

        if num_image_tokens <= 0:
            raise ValueError(f"num_image_tokens must be > 0, got {num_image_tokens}")
        if vision_encoder_type != "siglip2":
            raise ValueError(
                f"Unsupported vision_encoder_type '{vision_encoder_type}'. Expected 'siglip2'."
            )
        supported_connectors = {
            "perceiver_resampler",
            "question_conditioned_perceiver_resampler",
        }
        if connector_type not in supported_connectors:
            raise ValueError(
                "Unsupported connector_type "
                f"'{connector_type}'. Expected one of {sorted(supported_connectors)}."
            )
        if not project_directly_to_llm_dim:
            raise ValueError("AnyMALv3 requires project_directly_to_llm_dim=True.")

        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.vision_encoder_type = vision_encoder_type
        self.connector_type = connector_type
        self.num_image_tokens = num_image_tokens
        # Compatibility with shared dataset/checkpoint helpers.
        self.max_image_tokens = num_image_tokens
        self.min_image_tokens = num_image_tokens
        self.connector_layers = connector_layers
        self.connector_heads = connector_heads
        self.connector_ff_mult = connector_ff_mult
        self.project_directly_to_llm_dim = project_directly_to_llm_dim
        self.connector_output_scale = float(connector_output_scale)
        self.connector_output_gate_init = (
            None if connector_output_gate_init is None else float(connector_output_gate_init)
        )

        self.image_encoder = SigLIP2Encoder(
            model_name=vision_model_name,
            cache_dir=cache_dir,
            freeze=freeze_vision,
        )

        self.llm = LlamaWrapper(
            model_name=llm_model_name,
            use_qlora=use_qlora,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=llm_device_map,
            torch_dtype=llm_torch_dtype,
            use_flash_attention=use_flash_attention,
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
        )

        vision_dim = self.image_encoder.get_output_dim()
        llm_dim = self.llm.hidden_size
        projector_cls = (
            QuestionConditionedPerceiverResampler
            if connector_type == "question_conditioned_perceiver_resampler"
            else PerceiverResampler
        )
        projector_kwargs = {}
        if connector_type == "question_conditioned_perceiver_resampler":
            projector_kwargs["condition_dim"] = llm_dim
        self.projector = projector_cls(
            input_dim=vision_dim,
            output_dim=llm_dim,
            num_latents=num_image_tokens,
            num_layers=connector_layers,
            num_heads=connector_heads,
            ff_mult=connector_ff_mult,
            output_scale=self.connector_output_scale,
            output_gate_init=self.connector_output_gate_init,
            **projector_kwargs,
        )

        if freeze_llm:
            self.llm.freeze_base_model()

        self.tokenizer = self.llm.tokenizer
        self._setup_image_placeholder_token()

    def encode_images(
        self,
        images: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
        question_summary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Encode images with a fixed-size V3 Perceiver visual prefix.

        Returns:
            image_tokens: [B, num_image_tokens, llm_dim]
            image_token_mask: [B, num_image_tokens]
            image_token_counts: [B]
        """
        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        if target_num_tokens is not None:
            requested = target_num_tokens.to(device=images.device, dtype=torch.long)
            expected = torch.full_like(requested, self.num_image_tokens)
            if not torch.equal(requested, expected):
                raise ValueError(
                    "AnyMALv3 uses a fixed visual prefix: "
                    f"expected {self.num_image_tokens} placeholders per sample, "
                    f"got {requested.tolist()}."
                )

        if self.freeze_vision:
            with torch.no_grad():
                vision_features = self.image_encoder(images)
        else:
            vision_features = self.image_encoder(images)

        projector_dtype = next(self.projector.parameters()).dtype
        if vision_features.dtype != projector_dtype:
            vision_features = vision_features.to(projector_dtype)

        if self.connector_type == "question_conditioned_perceiver_resampler":
            self._last_connector_diagnostics = self._compute_connector_diagnostics(
                vision_features
            )
            image_tokens = self.projector(
                vision_features,
                question_summary=question_summary,
            )
        else:
            self._last_connector_diagnostics = self._compute_connector_diagnostics(
                vision_features
            )
            image_tokens = self.projector(vision_features)
        self._last_connector_output_rms_tensor = (
            image_tokens.float().pow(2).mean().sqrt()
        )
        self._record_connector_output_diagnostics(image_tokens)
        token_mask = torch.ones(
            image_tokens.shape[:2],
            dtype=torch.bool,
            device=image_tokens.device,
        )
        token_counts = torch.full(
            (image_tokens.shape[0],),
            self.num_image_tokens,
            dtype=torch.long,
            device=image_tokens.device,
        )
        return image_tokens, token_mask, token_counts

    @staticmethod
    def _tensor_rms(value: torch.Tensor) -> float:
        return float(value.detach().float().pow(2).mean().sqrt().item())

    @torch.no_grad()
    def _compute_connector_diagnostics(self, vision_features: torch.Tensor) -> Dict[str, float]:
        metrics = {
            "train/vision_feature_rms": self._tensor_rms(vision_features),
            "train/vision_feature_mean": float(vision_features.detach().float().mean().item()),
            "train/vision_feature_std": float(vision_features.detach().float().std().item()),
        }
        input_proj = getattr(self.projector, "input_proj", None)
        if input_proj is not None:
            projected = input_proj(vision_features)
            metrics.update(
                {
                    "train/connector_input_proj_rms": self._tensor_rms(projected),
                    "train/connector_input_proj_mean": float(projected.detach().float().mean().item()),
                    "train/connector_input_proj_std": float(projected.detach().float().std().item()),
                }
            )
        return metrics

    @torch.no_grad()
    def _record_connector_output_diagnostics(self, image_tokens: torch.Tensor) -> None:
        metrics = dict(getattr(self, "_last_connector_diagnostics", {}) or {})
        output_multiplier = getattr(self.projector, "_output_multiplier", None)
        if output_multiplier is not None:
            multiplier = output_multiplier()
            metrics["train/connector_output_multiplier"] = float(
                multiplier.detach().float().item()
                if isinstance(multiplier, torch.Tensor)
                else multiplier
            )
        gate_logit = getattr(self.projector, "output_gate_logit", None)
        if gate_logit is not None:
            metrics["train/connector_output_gate"] = float(
                torch.sigmoid(gate_logit.detach().float()).item()
            )
        metrics.update(
            {
                "train/connector_output_rms": self._tensor_rms(image_tokens),
                "train/connector_output_mean": float(image_tokens.detach().float().mean().item()),
                "train/connector_output_std": float(image_tokens.detach().float().std().item()),
                "train/connector_output_tokens": float(image_tokens.shape[1]),
            }
        )
        self._last_connector_diagnostics = metrics

    def _build_question_summary(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.Tensor]:
        """Pool frozen LLM token embeddings over prompt/context tokens only."""
        if self.connector_type != "question_conditioned_perceiver_resampler":
            return None
        if input_ids is None:
            return None

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        mask = mask & (input_ids != self.image_placeholder_token_id)
        if labels is not None:
            mask = mask & (labels == -100)

        fallback_mask = torch.ones_like(mask)
        if attention_mask is not None:
            fallback_mask = fallback_mask & attention_mask.bool()
        fallback_mask = fallback_mask & (input_ids != self.image_placeholder_token_id)
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = torch.where(empty.unsqueeze(1), fallback_mask, mask)

        weights = mask.to(device=text_embeds.device, dtype=text_embeds.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (text_embeds * weights).sum(dim=1) / denom

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ) -> AnyMALv3Output:
        if images is None and image_tokens is None:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return AnyMALv3Output(loss=outputs.loss, logits=outputs.logits)

        if input_ids is None:
            raise ValueError("AnyMALv3 multimodal forward requires input_ids for strict placeholder splice.")

        if image_tokens is None:
            requested_counts = self._extract_placeholder_counts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict=True,
            )
            image_tokens, image_token_mask, _ = self.encode_images(
                images=images,
                target_num_tokens=requested_counts,
                question_summary=self._build_question_summary(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ),
            )
        elif image_token_mask is None:
            image_token_mask = torch.ones(
                image_tokens.shape[:2],
                dtype=torch.bool,
                device=image_tokens.device,
            )
        valid_counts = image_token_mask.sum(dim=1).to(dtype=torch.long)
        expected_counts = torch.full_like(valid_counts, self.num_image_tokens)
        if not torch.equal(valid_counts, expected_counts):
            raise ValueError(
                "AnyMALv3 requires exactly "
                f"{self.num_image_tokens} image tokens per sample, got {valid_counts.tolist()}."
            )

        inputs_embeds, full_attention_mask, full_labels = self._splice_image_tokens_strict(
            input_ids=input_ids,
            image_tokens=image_tokens,
            image_token_mask=image_token_mask,
            attention_mask=attention_mask,
            labels=labels,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        return AnyMALv3Output(loss=outputs.loss, logits=outputs.logits)

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        if images is None and image_tokens is None:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("AnyMALv3 multimodal generate requires input_ids with placeholders.")

        if image_tokens is None:
            requested_counts = self._extract_placeholder_counts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict=True,
            )
            image_tokens, image_token_mask, _ = self.encode_images(
                images=images,
                target_num_tokens=requested_counts,
                question_summary=self._build_question_summary(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                ),
            )
        elif image_token_mask is None:
            image_token_mask = torch.ones(
                image_tokens.shape[:2],
                dtype=torch.bool,
                device=image_tokens.device,
            )
        valid_counts = image_token_mask.sum(dim=1).to(dtype=torch.long)
        expected_counts = torch.full_like(valid_counts, self.num_image_tokens)
        if not torch.equal(valid_counts, expected_counts):
            raise ValueError(
                "AnyMALv3 generation requires exactly "
                f"{self.num_image_tokens} image tokens per sample, got {valid_counts.tolist()}."
            )

        inputs_embeds, full_attention_mask, _ = self._splice_image_tokens_strict(
            input_ids=input_ids,
            image_tokens=image_tokens,
            image_token_mask=image_token_mask,
            attention_mask=attention_mask,
            labels=None,
        )

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )
        if isinstance(generated, torch.Tensor):
            return torch.cat([input_ids, generated], dim=1)
        return generated

    def set_training_stage(self, stage: int) -> None:
        if stage == 1:
            self.image_encoder.freeze()
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            print("Stage 1: Training V3 Perceiver connector only")
        elif stage == 2:
            self.image_encoder.freeze()
            self.llm.freeze_base_model()
            for param in self.projector.parameters():
                param.requires_grad = True
            print("Stage 2: Training V3 Perceiver connector + LoRA")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.print_trainable_parameters()

    def print_trainable_parameters(self) -> None:
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        vision_total, vision_train = count_params(self.image_encoder)
        proj_total, proj_train = count_params(self.projector)
        llm_total, llm_train = count_params(self.llm)

        print("\nTrainable Parameters:")
        print(f"  Vision Encoder:    {vision_train:,} / {vision_total:,}")
        print(f"  V3 Connector:      {proj_train:,} / {proj_total:,}")
        print(f"  LLM:               {llm_train:,} / {llm_total:,}")
        print(
            f"  Total:             {vision_train + proj_train + llm_train:,} / "
            f"{vision_total + proj_total + llm_total:,}"
        )

    def save_pretrained(
        self,
        save_path: str,
        save_llm: bool = True,
        save_llm_base: bool = False,
    ) -> None:
        import os
        import shutil

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))

        llm_save_path = os.path.join(save_path, "llm")
        llm_saved = False
        if save_llm:
            llm_saved = self.llm.save_pretrained(
                llm_save_path,
                save_base_model=save_llm_base,
            )
        if not llm_saved and os.path.isdir(llm_save_path):
            shutil.rmtree(llm_save_path)

        write_model_metadata(
            save_path,
            architecture=self.architecture,
            extra={
                "vision_encoder_type": self.vision_encoder_type,
                "vision_tower": "SigLIP2-So400m-384",
                "connector_type": self.connector_type,
                "num_image_tokens": self.num_image_tokens,
                "max_image_tokens": self.max_image_tokens,
                "min_image_tokens": self.min_image_tokens,
                "image_tokens": self.num_image_tokens,
                "connector_layers": self.connector_layers,
                "connector_heads": self.connector_heads,
                "connector_ff_mult": self.connector_ff_mult,
                "connector_output_scale": self.connector_output_scale,
                "connector_output_gate_init": self.connector_output_gate_init,
                "project_directly_to_llm_dim": self.project_directly_to_llm_dim,
                **(
                    self.llm.get_model_metadata()
                    if hasattr(self.llm, "get_model_metadata")
                    else {"llm_backbone": self.llm_backbone, "llm_model_name": self.llm_backbone}
                ),
                "image_placeholder_token": getattr(self, "image_placeholder_token", None),
                "image_placeholder_token_id": getattr(self, "image_placeholder_token_id", None),
                "image_placeholder_count": self.num_image_tokens,
                "stage1_connector_init": "fresh_random",
                "llm_checkpoint_saved": llm_saved,
                "llm_base_weights_saved": bool(llm_saved and save_llm_base),
                "question_conditioning": (
                    "pooled_prompt_embedding_additive_latent_shift"
                    if self.connector_type == "question_conditioned_perceiver_resampler"
                    else None
                ),
            },
        )
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        llm_backbone: Optional[str] = None,
        **kwargs,
    ) -> "AnyMALv3":
        import os
        from peft import PeftModel

        allow_connector_output_scale_override = bool(
            kwargs.pop("allow_connector_output_scale_override", False)
        )
        allow_missing_model_metadata = bool(
            kwargs.pop("allow_missing_model_metadata", False)
        )
        meta = read_model_metadata(save_path) or {}
        if meta or not allow_missing_model_metadata:
            validate_checkpoint_architecture(save_path, expected_architecture=cls.architecture)
        elif allow_missing_model_metadata:
            print(
                "WARNING: loading AnyMALv3 checkpoint without model_meta.json "
                f"because allow_missing_model_metadata=True: {save_path}"
            )
        if "connector_output_scale" not in kwargs and meta.get("connector_output_scale") is not None:
            kwargs["connector_output_scale"] = float(meta["connector_output_scale"])
        if "connector_output_gate_init" not in kwargs and meta.get("connector_output_gate_init") is not None:
            kwargs["connector_output_gate_init"] = float(meta["connector_output_gate_init"])

        model = cls(llm_model_name=llm_model_name, **kwargs)
        expected_values = {
            "vision_encoder_type": getattr(model, "vision_encoder_type", None),
            "connector_type": getattr(model, "connector_type", None),
            "num_image_tokens": getattr(model, "num_image_tokens", None),
            "max_image_tokens": getattr(model, "max_image_tokens", None),
            "min_image_tokens": getattr(model, "min_image_tokens", None),
            "connector_layers": getattr(model, "connector_layers", None),
            "connector_heads": getattr(model, "connector_heads", None),
            "connector_ff_mult": getattr(model, "connector_ff_mult", None),
            "project_directly_to_llm_dim": getattr(model, "project_directly_to_llm_dim", None),
        }
        if "connector_output_scale" in meta and not allow_connector_output_scale_override:
            expected_values["connector_output_scale"] = getattr(model, "connector_output_scale", None)
        if "connector_output_gate_init" in meta:
            expected_values["connector_output_gate_init"] = getattr(model, "connector_output_gate_init", None)
        if "llm_backbone" in meta or getattr(model, "llm_backbone", None) != CURRENT_LLAMA3_BACKBONE:
            expected_values["llm_backbone"] = getattr(model, "llm_backbone", None)

        if meta or not allow_missing_model_metadata:
            validate_checkpoint_metadata_values(
                save_path,
                expected_architecture=cls.architecture,
                expected_values=expected_values,
            )
        projector_path = os.path.join(save_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Missing V3 connector weights: {projector_path}")

        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            base_model = model.llm.model
            if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
                base_model = base_model.unload()
            model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
        return model
