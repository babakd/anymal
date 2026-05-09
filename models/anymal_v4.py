"""
AnyMALv4 core architecture.

V4 stack:
- SigLIP2 vision encoder at 384px
- Spatial global/local Perceiver connector projected directly to LLaMA space
- Strict placeholder/token alignment checks inherited from V2
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .anymal_v3 import AnyMALv3
from .encoders import SigLIP2Encoder
from .llm import LlamaWrapper
from .projectors import DeepStackSpatialPerceiverResampler, SpatialPerceiverResampler
from model_metadata import (
    validate_checkpoint_architecture,
    validate_checkpoint_metadata_values,
    write_model_metadata,
)


class AnyMALv4(AnyMALv3):
    """AnyMAL v4 multimodal model with a spatial global/local connector."""

    architecture = "anymal_v4"

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        vision_encoder_type: str = "siglip2",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        connector_type: str = "spatial_perceiver_resampler",
        num_global_image_tokens: int = 64,
        num_local_image_tokens: int = 64,
        num_image_tokens: Optional[int] = None,
        connector_layers: int = 6,
        connector_heads: int = 16,
        connector_ff_mult: int = 4,
        connector_hidden_dim: Optional[int] = None,
        connector_output_scale: float = 1.0,
        connector_output_gate_init: Optional[float] = None,
        use_2d_position_features: bool = True,
        deepstack_num_feature_levels: int = 3,
        deepstack_hidden_state_indices: Optional[Sequence[int]] = None,
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

        total_tokens = num_global_image_tokens + num_local_image_tokens
        if num_image_tokens is not None and num_image_tokens != total_tokens:
            raise ValueError(
                "num_image_tokens must equal num_global_image_tokens + "
                f"num_local_image_tokens ({total_tokens}), got {num_image_tokens}."
            )
        if total_tokens <= 0:
            raise ValueError("AnyMALv4 requires at least one image token.")
        if vision_encoder_type != "siglip2":
            raise ValueError(
                f"Unsupported vision_encoder_type '{vision_encoder_type}'. Expected 'siglip2'."
            )
        supported_connectors = {
            "spatial_perceiver_resampler",
            "deepstack_spatial_perceiver_resampler",
        }
        if connector_type not in supported_connectors:
            raise ValueError(
                "Unsupported connector_type "
                f"'{connector_type}'. Expected one of {sorted(supported_connectors)}."
            )
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.vision_encoder_type = vision_encoder_type
        self.connector_type = connector_type
        self.num_global_image_tokens = num_global_image_tokens
        self.num_local_image_tokens = num_local_image_tokens
        self.num_image_tokens = total_tokens
        self.max_image_tokens = total_tokens
        self.min_image_tokens = total_tokens
        self.connector_layers = connector_layers
        self.connector_heads = connector_heads
        self.connector_ff_mult = connector_ff_mult
        self.use_2d_position_features = use_2d_position_features
        self.deepstack_num_feature_levels = int(deepstack_num_feature_levels)
        if self.deepstack_num_feature_levels <= 0:
            raise ValueError(
                f"deepstack_num_feature_levels must be > 0, got {deepstack_num_feature_levels}"
            )
        if deepstack_hidden_state_indices is None:
            deepstack_hidden_state_indices = tuple(
                range(-self.deepstack_num_feature_levels, 0)
            )
        self.deepstack_hidden_state_indices = tuple(int(i) for i in deepstack_hidden_state_indices)
        if len(self.deepstack_hidden_state_indices) != self.deepstack_num_feature_levels:
            raise ValueError(
                "deepstack_hidden_state_indices length must match "
                f"deepstack_num_feature_levels ({self.deepstack_num_feature_levels}), "
                f"got {self.deepstack_hidden_state_indices}."
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
        if connector_hidden_dim is None:
            connector_hidden_dim = llm_dim
        connector_hidden_dim = int(connector_hidden_dim)
        if connector_hidden_dim <= 0:
            raise ValueError(f"connector_hidden_dim must be > 0, got {connector_hidden_dim}")
        if project_directly_to_llm_dim and connector_hidden_dim != llm_dim:
            raise ValueError(
                "project_directly_to_llm_dim=True requires connector_hidden_dim to "
                f"equal the LLM hidden size ({llm_dim}), got {connector_hidden_dim}."
            )
        self.connector_hidden_dim = connector_hidden_dim
        self.connector_output_scale = float(connector_output_scale)
        self.connector_output_gate_init = (
            None if connector_output_gate_init is None else float(connector_output_gate_init)
        )
        self.project_directly_to_llm_dim = bool(connector_hidden_dim == llm_dim)
        projector_cls = (
            DeepStackSpatialPerceiverResampler
            if connector_type == "deepstack_spatial_perceiver_resampler"
            else SpatialPerceiverResampler
        )
        projector_kwargs = {}
        if connector_type == "deepstack_spatial_perceiver_resampler":
            projector_kwargs["num_feature_levels"] = self.deepstack_num_feature_levels
        self.projector = projector_cls(
            input_dim=vision_dim,
            output_dim=llm_dim,
            connector_dim=connector_hidden_dim,
            num_global_latents=num_global_image_tokens,
            num_local_latents=num_local_image_tokens,
            num_layers=connector_layers,
            num_heads=connector_heads,
            ff_mult=connector_ff_mult,
            use_2d_position_features=use_2d_position_features,
            output_scale=self.connector_output_scale,
            output_gate_init=self.connector_output_gate_init,
            **projector_kwargs,
        )

        if freeze_llm:
            self.llm.freeze_base_model()

        self.tokenizer = self.llm.tokenizer
        self._setup_image_placeholder_token()

    def set_training_stage(self, stage: int) -> None:
        if stage == 1:
            self.image_encoder.freeze()
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            print("Stage 1: Training V4 spatial connector only")
        elif stage == 2:
            self.image_encoder.freeze()
            self.llm.freeze_base_model()
            for param in self.projector.parameters():
                param.requires_grad = True
            print("Stage 2: Training V4 spatial connector + LoRA")
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
        print(f"  V4 Connector:      {proj_train:,} / {proj_total:,}")
        print(f"  LLM:               {llm_train:,} / {llm_total:,}")
        print(
            f"  Total:             {vision_train + proj_train + llm_train:,} / "
            f"{vision_total + proj_total + llm_total:,}"
        )

    def encode_images(
        self,
        images: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Encode images with the fixed-size V4 visual prefix.

        DeepStack-lite connectors request multiple hidden-state levels from the
        frozen SigLIP2 tower; default spatial connectors keep the single final
        feature level used by the base V4 path.
        """
        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        if target_num_tokens is not None:
            requested = target_num_tokens.to(device=images.device, dtype=torch.long)
            expected = torch.full_like(requested, self.num_image_tokens)
            if not torch.equal(requested, expected):
                raise ValueError(
                    "AnyMALv4 uses a fixed visual prefix: "
                    f"expected {self.num_image_tokens} placeholders per sample, "
                    f"got {requested.tolist()}."
                )

        use_deepstack = self.connector_type == "deepstack_spatial_perceiver_resampler"
        with torch.no_grad() if self.freeze_vision else torch.enable_grad():
            if use_deepstack:
                vision_features = self.image_encoder(
                    images,
                    output_hidden_states=True,
                    hidden_state_indices=self.deepstack_hidden_state_indices,
                )
            else:
                vision_features = self.image_encoder(images)

        projector_dtype = next(self.projector.parameters()).dtype
        if isinstance(vision_features, (list, tuple)):
            vision_features = tuple(
                feature.to(projector_dtype) if feature.dtype != projector_dtype else feature
                for feature in vision_features
            )
        elif vision_features.dtype != projector_dtype:
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

    def _metadata_expected_values(self) -> Dict[str, object]:
        values = {
            "vision_encoder_type": self.vision_encoder_type,
            "connector_type": self.connector_type,
            "num_image_tokens": self.num_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "min_image_tokens": self.min_image_tokens,
            "num_global_image_tokens": self.num_global_image_tokens,
            "num_local_image_tokens": self.num_local_image_tokens,
            "connector_layers": self.connector_layers,
            "connector_heads": self.connector_heads,
            "connector_ff_mult": self.connector_ff_mult,
            "connector_hidden_dim": self.connector_hidden_dim,
            "connector_output_scale": self.connector_output_scale,
            "connector_output_gate_init": self.connector_output_gate_init,
            "use_2d_position_features": self.use_2d_position_features,
            "project_directly_to_llm_dim": self.project_directly_to_llm_dim,
        }
        if self.connector_type == "deepstack_spatial_perceiver_resampler":
            values.update(
                {
                    "vision_feature_strategy": "deepstack_lite",
                    "vision_feature_layers": list(self.deepstack_hidden_state_indices),
                    "num_vision_feature_levels": self.deepstack_num_feature_levels,
                    "deepstack_num_feature_levels": self.deepstack_num_feature_levels,
                    "deepstack_hidden_state_indices": list(
                        self.deepstack_hidden_state_indices
                    ),
                }
            )
        return values

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

        extra = self._metadata_expected_values()
        extra.update(
            {
                "llm_checkpoint_saved": llm_saved,
                "llm_base_weights_saved": bool(llm_saved and save_llm_base),
            }
        )
        write_model_metadata(save_path, architecture=self.architecture, extra=extra)
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs,
    ) -> "AnyMALv4":
        import os
        from peft import PeftModel

        validate_checkpoint_architecture(save_path, expected_architecture=cls.architecture)

        model = cls(llm_model_name=llm_model_name, **kwargs)
        validate_checkpoint_metadata_values(
            save_path,
            expected_architecture=cls.architecture,
            expected_values=model._metadata_expected_values(),
        )

        projector_path = os.path.join(save_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Missing V4 connector weights: {projector_path}")

        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            base_model = model.llm.model
            if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
                base_model = base_model.unload()
            model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
        return model
