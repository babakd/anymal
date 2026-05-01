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
from .llm import LlamaWrapper
from .projectors import PerceiverResampler
from model_metadata import (
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
        vision_encoder_type: str = "siglip2",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        connector_type: str = "perceiver_resampler",
        num_image_tokens: int = 128,
        connector_layers: int = 6,
        connector_heads: int = 16,
        connector_ff_mult: int = 4,
        project_directly_to_llm_dim: bool = True,
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

        if num_image_tokens <= 0:
            raise ValueError(f"num_image_tokens must be > 0, got {num_image_tokens}")
        if vision_encoder_type != "siglip2":
            raise ValueError(
                f"Unsupported vision_encoder_type '{vision_encoder_type}'. Expected 'siglip2'."
            )
        if connector_type != "perceiver_resampler":
            raise ValueError(
                f"Unsupported connector_type '{connector_type}'. Expected 'perceiver_resampler'."
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
        self.projector = PerceiverResampler(
            input_dim=vision_dim,
            output_dim=llm_dim,
            num_latents=num_image_tokens,
            num_layers=connector_layers,
            num_heads=connector_heads,
            ff_mult=connector_ff_mult,
        )

        if freeze_llm:
            self.llm.freeze_base_model()

        self.tokenizer = self.llm.tokenizer
        self._setup_image_placeholder_token()

    def encode_images(
        self,
        images: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
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

        image_tokens = self.projector(vision_features)
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
                "connector_type": self.connector_type,
                "num_image_tokens": self.num_image_tokens,
                "max_image_tokens": self.max_image_tokens,
                "min_image_tokens": self.min_image_tokens,
                "connector_layers": self.connector_layers,
                "connector_heads": self.connector_heads,
                "connector_ff_mult": self.connector_ff_mult,
                "project_directly_to_llm_dim": self.project_directly_to_llm_dim,
                "llm_checkpoint_saved": llm_saved,
                "llm_base_weights_saved": bool(llm_saved and save_llm_base),
            },
        )
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs,
    ) -> "AnyMALv3":
        import os
        from peft import PeftModel

        validate_checkpoint_architecture(save_path, expected_architecture=cls.architecture)

        model = cls(llm_model_name=llm_model_name, **kwargs)
        validate_checkpoint_metadata_values(
            save_path,
            expected_architecture=cls.architecture,
            expected_values={
                "vision_encoder_type": getattr(model, "vision_encoder_type", None),
                "connector_type": getattr(model, "connector_type", None),
                "num_image_tokens": getattr(model, "num_image_tokens", None),
                "max_image_tokens": getattr(model, "max_image_tokens", None),
                "min_image_tokens": getattr(model, "min_image_tokens", None),
                "connector_layers": getattr(model, "connector_layers", None),
                "connector_heads": getattr(model, "connector_heads", None),
                "connector_ff_mult": getattr(model, "connector_ff_mult", None),
                "project_directly_to_llm_dim": getattr(model, "project_directly_to_llm_dim", None),
            },
        )
        projector_path = os.path.join(save_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path))
        else:
            raise FileNotFoundError(f"Missing V3 connector weights: {projector_path}")

        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            base_model = model.llm.model
            if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
                base_model = base_model.unload()
            model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
        return model
