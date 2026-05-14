"""
AnyMALv3 core architecture.

V3 stack:
- SigLIP2 vision encoder at 384px
- Deep Perceiver resampler projected directly to LLaMA hidden space
- Strict placeholder/token alignment checks inherited from V2
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anymal_v2 import AnyMALv2
from .encoders import SigLIP2Encoder
from .llm import LlamaWrapper, canonicalize_llm_backbone
from .llm.backbone import CURRENT_LLAMA3_BACKBONE
from .projectors import (
    AnyResMLPProjector,
    PerceiverResampler,
    QuestionConditionedPerceiverResampler,
    SpatialGridProjector,
)
from .visual_cross_attention import GatedVisualCrossAttentionAdapter
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


class V3SpatialResidualBranch(nn.Module):
    """Small gated spatial residual over frozen V3 image tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int,
        hidden_dim: int = 128,
        grid_size: int = 32,
        gate_init: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_latents = int(num_latents)
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        if self.input_dim <= 0 or self.output_dim <= 0 or self.num_latents <= 0:
            raise ValueError("V3 spatial residual dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"spatial_residual_hidden_dim must be > 0, got {hidden_dim}")
        if self.grid_size <= 0:
            raise ValueError(f"spatial_residual_grid_size must be > 0, got {grid_size}")
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError(
                "spatial_residual_gate_init must be between 0 and 1, "
                f"got {gate_init}"
            )

        self.row_embedding = nn.Parameter(torch.randn(self.grid_size, self.input_dim) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(self.grid_size, self.input_dim) * 0.02)
        self.latent_queries = nn.Parameter(torch.randn(self.num_latents, self.hidden_dim) * 0.02)
        self.key_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        self.gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init), dtype=torch.float32))
        )

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> Tuple[int, Optional[int], Optional[int]]:
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return 0, side, side
        side = int(max(num_tokens - 1, 0) ** 0.5)
        if side * side == num_tokens - 1:
            return 1, side, side
        return 0, None, None

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit.detach().float()).item())

    def _position_features(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        if height is None or width is None:
            idx = torch.linspace(
                0,
                self.grid_size - 1,
                num_tokens,
                device=device,
                dtype=torch.float32,
            ).round().long().clamp(0, self.grid_size - 1)
            positions = self.row_embedding.to(device=device)[idx]
        else:
            row_idx = torch.linspace(
                0,
                self.grid_size - 1,
                height,
                device=device,
                dtype=torch.float32,
            ).round().long().clamp(0, self.grid_size - 1)
            col_idx = torch.linspace(
                0,
                self.grid_size - 1,
                width,
                device=device,
                dtype=torch.float32,
            ).round().long().clamp(0, self.grid_size - 1)
            rows = self.row_embedding.to(device=device)[row_idx][:, None, :]
            cols = self.col_embedding.to(device=device)[col_idx][None, :, :]
            positions = (rows + cols).reshape(height * width, self.input_dim)
            if prefix_tokens:
                prefix = torch.zeros(
                    prefix_tokens,
                    self.input_dim,
                    device=device,
                    dtype=positions.dtype,
                )
                positions = torch.cat([prefix, positions], dim=0)
        return positions.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = vision_features.shape
        branch_dtype = self.key_proj.weight.dtype
        x = vision_features.to(dtype=branch_dtype)
        x = x + self._position_features(
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=vision_features.device,
            dtype=branch_dtype,
        )
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.latent_queries.to(device=vision_features.device, dtype=branch_dtype)
        attn = torch.softmax(
            torch.einsum("lh,bph->blp", queries, keys) / (self.hidden_dim ** 0.5),
            dim=-1,
        )
        pooled = torch.einsum("blp,bph->blh", attn, values)
        residual = self.norm(self.out_proj(pooled))
        return residual.to(dtype=vision_features.dtype) * torch.sigmoid(
            self.gate_logit.to(device=vision_features.device, dtype=vision_features.dtype)
        )


class V3SpatialTailBranch(nn.Module):
    """Spatial patch-token tail appended after a warm-started V3 prefix."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tail_tokens: int,
        hidden_dim: Optional[int] = None,
        output_scale: float = 1.0,
        gate_init: float = 1e-4,
        use_2d_position_features: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tail_tokens = int(num_tail_tokens)
        self.hidden_dim = int(hidden_dim or output_dim)
        self.output_scale = float(output_scale)
        self.use_2d_position_features = bool(use_2d_position_features)
        if self.input_dim <= 0 or self.output_dim <= 0 or self.num_tail_tokens <= 0:
            raise ValueError("V3 spatial tail dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"spatial_tail_hidden_dim must be > 0, got {hidden_dim}")
        if self.output_scale <= 0:
            raise ValueError(f"spatial_tail_output_scale must be > 0, got {output_scale}")
        if not 0.0 <= float(gate_init) < 1.0:
            raise ValueError(
                "spatial_tail_gate_init must be in [0, 1), "
                f"got {gate_init}"
            )

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        if self.use_2d_position_features:
            self.position_mlp = nn.Sequential(
                nn.Linear(2, self.input_dim),
                nn.GELU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
            nn.init.zeros_(self.position_mlp[-1].weight)
            nn.init.zeros_(self.position_mlp[-1].bias)
        else:
            self.position_mlp = None
        if float(gate_init) == 0.0:
            gate_logit = torch.tensor(-20.0, dtype=torch.float32)
        else:
            gate_logit = torch.logit(torch.tensor(float(gate_init), dtype=torch.float32))
        self.gate_logit = nn.Parameter(gate_logit)
        self._last_tail_rms = None

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> Tuple[int, Optional[int], Optional[int]]:
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return 0, side, side
        side = int(math.isqrt(max(num_tokens - 1, 0)))
        if side * side == num_tokens - 1:
            return 1, side, side
        return 0, None, None

    @staticmethod
    def _target_grid(num_tokens: int) -> Tuple[int, int]:
        root = max(1, int(math.isqrt(num_tokens)))
        for height in range(root, 0, -1):
            if num_tokens % height == 0:
                return height, num_tokens // height
        height = max(1, root)
        width = max(1, int(math.ceil(num_tokens / height)))
        return height, width

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit.detach().float()).item())

    def _pool_vision_tokens(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = vision_features.shape
        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        target_h, target_w = self._target_grid(self.num_tail_tokens)
        if height is not None and width is not None:
            patches = vision_features[:, prefix_tokens:, :]
            patches = patches.reshape(batch_size, height, width, channels)
            patches = patches.permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(patches, output_size=(target_h, target_w))
            pooled = pooled.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        else:
            x = vision_features.transpose(1, 2)
            pooled = F.interpolate(
                x,
                size=self.num_tail_tokens,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)

        if pooled.shape[1] > self.num_tail_tokens:
            pooled = pooled[:, : self.num_tail_tokens, :]
        elif pooled.shape[1] < self.num_tail_tokens:
            pad = pooled[:, -1:, :].expand(
                batch_size,
                self.num_tail_tokens - pooled.shape[1],
                channels,
            )
            pooled = torch.cat([pooled, pad], dim=1)
        return pooled

    def _position_features(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.position_mlp is None:
            raise RuntimeError("Position features requested without position_mlp")
        target_h, target_w = self._target_grid(self.num_tail_tokens)
        ys = torch.linspace(-1.0, 1.0, target_h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, target_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        coords = coords[: self.num_tail_tokens]
        mlp_dtype = next(self.position_mlp.parameters()).dtype
        features = self.position_mlp(coords.to(dtype=mlp_dtype))
        return features.to(device=device, dtype=dtype).unsqueeze(0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        branch_dtype = self.input_proj.weight.dtype
        pooled = self._pool_vision_tokens(vision_features).to(dtype=branch_dtype)
        if self.position_mlp is not None:
            pooled = pooled + self._position_features(
                device=vision_features.device,
                dtype=branch_dtype,
            )
        hidden = F.gelu(self.input_proj(self.input_norm(pooled)))
        tail = self.output_norm(self.output_proj(hidden))
        multiplier = self.output_scale * torch.sigmoid(
            self.gate_logit.to(device=vision_features.device, dtype=tail.dtype)
        )
        tail = tail * multiplier
        self._last_tail_rms = tail.detach().float().pow(2).mean().sqrt()
        return tail.to(dtype=vision_features.dtype)


class AnyMALv3(AnyMALv2):
    """AnyMAL v3 multimodal model."""

    architecture = "anymal_v3"
    visual_cross_attention_state_file = "visual_cross_attention_adapters.pt"
    vision_adapter_state_file = "vision_adapter.pt"

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        llm_backbone: Optional[str] = None,
        vision_encoder_type: str = "siglip2",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        vision_image_size: int = 384,
        connector_type: str = "perceiver_resampler",
        num_image_tokens: int = 128,
        connector_layers: int = 6,
        connector_heads: int = 16,
        connector_ff_mult: int = 4,
        project_directly_to_llm_dim: bool = True,
        connector_output_scale: float = 1.0,
        connector_output_gate_init: Optional[float] = None,
        connector_trainable_scale_mode: str = "none",
        use_2d_patch_position_features: bool = False,
        patch_position_feature_type: Optional[str] = None,
        patch_position_grid_size: int = 32,
        patch_position_mlp_hidden_dim: int = 128,
        patch_position_feature_scale: float = 1.0,
        query_conditioned_visual_scale_mode: str = "none",
        query_conditioned_visual_scale_min: float = 0.95,
        query_conditioned_visual_scale_max: float = 1.15,
        query_conditioned_visual_scale_init: Optional[float] = None,
        query_conditioned_patch_selector_mode: str = "none",
        query_conditioned_patch_selector_hidden_dim: int = 256,
        query_conditioned_patch_selector_max_residual: float = 0.25,
        query_conditioned_patch_selector_normalize_mean: bool = True,
        spatial_residual_mode: str = "none",
        spatial_residual_hidden_dim: int = 128,
        spatial_residual_grid_size: int = 32,
        spatial_residual_gate_init: float = 1e-4,
        spatial_tail_mode: str = "none",
        spatial_tail_tokens: int = 0,
        spatial_tail_hidden_dim: Optional[int] = None,
        spatial_tail_output_scale: float = 1.0,
        spatial_tail_gate_init: float = 1e-4,
        spatial_tail_use_2d_position_features: bool = True,
        anyres_tokens_per_view: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        anyres_crop_mode: str = "global_two_crops",
        anyres_use_view_embeddings: bool = True,
        visual_cross_attention_mode: str = "none",
        visual_cross_attention_layers: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        visual_cross_attention_num_heads: int = 16,
        visual_cross_attention_adapter_dim: Optional[int] = None,
        visual_cross_attention_gate_init: float = 0.0,
        visual_cross_attention_dropout: float = 0.0,
        visual_cross_attention_freeze_connector: bool = False,
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
        if int(vision_image_size) <= 0:
            raise ValueError(f"vision_image_size must be > 0, got {vision_image_size}")
        if vision_encoder_type != "siglip2":
            raise ValueError(
                f"Unsupported vision_encoder_type '{vision_encoder_type}'. Expected 'siglip2'."
            )
        supported_connectors = {
            "mlp_anyres_projector",
            "perceiver_resampler",
            "question_conditioned_perceiver_resampler",
            "spatial_grid_projector",
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
        self.vision_image_size = int(vision_image_size)
        self.connector_type = connector_type
        self.spatial_tail_mode = self._normalize_spatial_tail_mode(spatial_tail_mode)
        self.spatial_tail_tokens = (
            int(spatial_tail_tokens) if self.spatial_tail_mode != "none" else 0
        )
        if self.spatial_tail_mode != "none" and self.spatial_tail_tokens <= 0:
            raise ValueError(
                "spatial_tail_tokens must be > 0 when spatial_tail_mode is enabled"
            )
        if self.spatial_tail_tokens >= int(num_image_tokens):
            raise ValueError(
                "spatial_tail_tokens must be smaller than num_image_tokens so the "
                "warm-started V3 prefix remains non-empty"
            )
        if self.spatial_tail_mode != "none" and connector_type == "spatial_grid_projector":
            raise ValueError(
                "spatial_tail_mode is a Track 2 hybrid option and cannot be "
                "combined with connector_type='spatial_grid_projector'"
            )
        if self.spatial_tail_mode != "none" and connector_type == "mlp_anyres_projector":
            raise ValueError(
                "spatial_tail_mode is a Track 2 hybrid option and cannot be "
                "combined with connector_type='mlp_anyres_projector'"
            )
        if (
            self._normalize_spatial_residual_mode(spatial_residual_mode) != "none"
            and connector_type == "mlp_anyres_projector"
        ):
            raise ValueError(
                "spatial_residual_mode cannot be combined with "
                "connector_type='mlp_anyres_projector'"
            )
        self.v3_base_image_tokens = int(num_image_tokens) - self.spatial_tail_tokens
        self.num_image_tokens = int(num_image_tokens)
        # Compatibility with shared dataset/checkpoint helpers.
        self.max_image_tokens = self.num_image_tokens
        self.min_image_tokens = self.num_image_tokens
        self.connector_layers = connector_layers
        self.connector_heads = connector_heads
        self.connector_ff_mult = connector_ff_mult
        self.project_directly_to_llm_dim = project_directly_to_llm_dim
        self.connector_output_scale = float(connector_output_scale)
        self.connector_output_gate_init = (
            None if connector_output_gate_init is None else float(connector_output_gate_init)
        )
        self.connector_trainable_scale_mode = str(
            connector_trainable_scale_mode or "none"
        ).strip().lower()
        self.use_2d_patch_position_features = bool(use_2d_patch_position_features)
        self.patch_position_feature_type = patch_position_feature_type
        self.patch_position_grid_size = int(patch_position_grid_size)
        self.patch_position_mlp_hidden_dim = int(patch_position_mlp_hidden_dim)
        self.patch_position_feature_scale = float(patch_position_feature_scale)
        self.query_conditioned_visual_scale_mode = str(
            query_conditioned_visual_scale_mode or "none"
        ).strip().lower()
        self.query_conditioned_visual_scale_min = float(
            query_conditioned_visual_scale_min
        )
        self.query_conditioned_visual_scale_max = float(
            query_conditioned_visual_scale_max
        )
        self.query_conditioned_visual_scale_init = (
            None
            if query_conditioned_visual_scale_init is None
            else float(query_conditioned_visual_scale_init)
        )
        self.query_conditioned_patch_selector_mode = str(
            query_conditioned_patch_selector_mode or "none"
        ).strip().lower()
        self.query_conditioned_patch_selector_hidden_dim = int(
            query_conditioned_patch_selector_hidden_dim
        )
        self.query_conditioned_patch_selector_max_residual = float(
            query_conditioned_patch_selector_max_residual
        )
        self.query_conditioned_patch_selector_normalize_mean = bool(
            query_conditioned_patch_selector_normalize_mean
        )
        self.spatial_residual_mode = self._normalize_spatial_residual_mode(
            spatial_residual_mode
        )
        self.spatial_residual_hidden_dim = int(spatial_residual_hidden_dim)
        self.spatial_residual_grid_size = int(spatial_residual_grid_size)
        self.spatial_residual_gate_init = float(spatial_residual_gate_init)
        self.spatial_tail_hidden_dim = (
            None if spatial_tail_hidden_dim is None else int(spatial_tail_hidden_dim)
        )
        self.spatial_tail_output_scale = float(spatial_tail_output_scale)
        self.spatial_tail_gate_init = float(spatial_tail_gate_init)
        self.spatial_tail_use_2d_position_features = bool(
            spatial_tail_use_2d_position_features
        )
        self.anyres_tokens_per_view = (
            self._resolve_anyres_tokens_per_view(
                anyres_tokens_per_view,
                total_tokens=self.v3_base_image_tokens,
            )
            if self.connector_type == "mlp_anyres_projector"
            else None
        )
        self.anyres_num_views = (
            len(self.anyres_tokens_per_view)
            if self.anyres_tokens_per_view is not None
            else None
        )
        self.anyres_crop_mode = str(anyres_crop_mode or "global_two_crops")
        self.anyres_use_view_embeddings = bool(anyres_use_view_embeddings)
        self.image_view_mode = (
            "anyres_global_two_crops"
            if self.connector_type == "mlp_anyres_projector"
            else "single"
        )
        if self._uses_spatial_tail():
            self.spatial_tail_grid_height, self.spatial_tail_grid_width = (
                V3SpatialTailBranch._target_grid(self.spatial_tail_tokens)
            )
        else:
            self.spatial_tail_grid_height, self.spatial_tail_grid_width = (None, None)
        self.visual_cross_attention_mode = self._normalize_visual_cross_attention_mode(
            visual_cross_attention_mode
        )
        self.visual_cross_attention_layers = self._normalize_visual_cross_attention_layers(
            visual_cross_attention_layers
        )
        self.visual_cross_attention_num_heads = int(visual_cross_attention_num_heads)
        self.visual_cross_attention_adapter_dim = (
            None
            if visual_cross_attention_adapter_dim is None
            else int(visual_cross_attention_adapter_dim)
        )
        self.visual_cross_attention_gate_init = float(visual_cross_attention_gate_init)
        self.visual_cross_attention_dropout = float(visual_cross_attention_dropout)
        self.visual_cross_attention_freeze_connector = bool(
            visual_cross_attention_freeze_connector
        )
        self.visual_cross_attention_adapters = nn.ModuleDict()
        self._visual_cross_attention_hook_handles = []
        self._active_visual_memory = None
        self._last_visual_cross_attention_diagnostics: Dict[str, float] = {}

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
        if connector_type == "mlp_anyres_projector":
            if self.query_conditioned_visual_scale_mode != "none":
                raise ValueError(
                    "mlp_anyres_projector does not support "
                    "query_conditioned_visual_scale_mode yet"
                )
            if self.query_conditioned_patch_selector_mode != "none":
                raise ValueError(
                    "mlp_anyres_projector does not support "
                    "query_conditioned_patch_selector_mode yet"
                )
            self.projector = AnyResMLPProjector(
                input_dim=vision_dim,
                output_dim=llm_dim,
                tokens_per_view=self.anyres_tokens_per_view,
                output_scale=self.connector_output_scale,
                output_gate_init=self.connector_output_gate_init,
                trainable_scale_mode=self.connector_trainable_scale_mode,
                use_2d_patch_position_features=self.use_2d_patch_position_features,
                patch_position_feature_type=self.patch_position_feature_type,
                patch_position_grid_size=self.patch_position_grid_size,
                patch_position_mlp_hidden_dim=self.patch_position_mlp_hidden_dim,
                patch_position_feature_scale=self.patch_position_feature_scale,
                use_view_embeddings=self.anyres_use_view_embeddings,
            )
        elif connector_type == "spatial_grid_projector":
            if self.query_conditioned_visual_scale_mode != "none":
                raise ValueError(
                    "spatial_grid_projector does not support "
                    "query_conditioned_visual_scale_mode yet"
                )
            if self.query_conditioned_patch_selector_mode != "none":
                raise ValueError(
                    "spatial_grid_projector does not support "
                    "query_conditioned_patch_selector_mode yet"
                )
            self.projector = SpatialGridProjector(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_grid_tokens=self.v3_base_image_tokens,
                output_scale=self.connector_output_scale,
                output_gate_init=self.connector_output_gate_init,
                trainable_scale_mode=self.connector_trainable_scale_mode,
                use_2d_patch_position_features=self.use_2d_patch_position_features,
                patch_position_feature_type=self.patch_position_feature_type,
                patch_position_grid_size=self.patch_position_grid_size,
                patch_position_mlp_hidden_dim=self.patch_position_mlp_hidden_dim,
                patch_position_feature_scale=self.patch_position_feature_scale,
            )
        else:
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
                num_latents=self.v3_base_image_tokens,
                num_layers=connector_layers,
                num_heads=connector_heads,
                ff_mult=connector_ff_mult,
                output_scale=self.connector_output_scale,
                output_gate_init=self.connector_output_gate_init,
                trainable_scale_mode=self.connector_trainable_scale_mode,
                use_2d_patch_position_features=self.use_2d_patch_position_features,
                patch_position_feature_type=self.patch_position_feature_type,
                patch_position_grid_size=self.patch_position_grid_size,
                patch_position_mlp_hidden_dim=self.patch_position_mlp_hidden_dim,
                patch_position_feature_scale=self.patch_position_feature_scale,
                query_conditioned_visual_scale_mode=self.query_conditioned_visual_scale_mode,
                query_conditioned_visual_scale_min=self.query_conditioned_visual_scale_min,
                query_conditioned_visual_scale_max=self.query_conditioned_visual_scale_max,
                query_conditioned_visual_scale_init=self.query_conditioned_visual_scale_init,
                query_conditioned_patch_selector_mode=self.query_conditioned_patch_selector_mode,
                query_conditioned_patch_selector_hidden_dim=self.query_conditioned_patch_selector_hidden_dim,
                query_conditioned_patch_selector_max_residual=self.query_conditioned_patch_selector_max_residual,
                query_conditioned_patch_selector_normalize_mean=self.query_conditioned_patch_selector_normalize_mean,
                **projector_kwargs,
            )
        self.connector_trainable_scale_mode = getattr(
            self.projector,
            "trainable_scale_mode",
            self.connector_trainable_scale_mode,
        )
        self.patch_position_feature_type = getattr(
            self.projector,
            "patch_position_feature_type",
            "learned_table" if self.use_2d_patch_position_features else "none",
        )
        self.use_2d_patch_position_features = getattr(
            self.projector,
            "use_2d_patch_position_features",
            self.use_2d_patch_position_features,
        )
        self.patch_position_mlp_hidden_dim = getattr(
            self.projector,
            "patch_position_mlp_hidden_dim",
            self.patch_position_mlp_hidden_dim,
        )
        self.patch_position_feature_scale = getattr(
            self.projector,
            "patch_position_feature_scale",
            self.patch_position_feature_scale,
        )
        self.query_conditioned_visual_scale_mode = getattr(
            self.projector,
            "query_conditioned_visual_scale_mode",
            self.query_conditioned_visual_scale_mode,
        )
        self.query_conditioned_visual_scale_min = getattr(
            self.projector,
            "query_conditioned_visual_scale_min",
            self.query_conditioned_visual_scale_min,
        )
        self.query_conditioned_visual_scale_max = getattr(
            self.projector,
            "query_conditioned_visual_scale_max",
            self.query_conditioned_visual_scale_max,
        )
        self.query_conditioned_visual_scale_init = getattr(
            self.projector,
            "query_conditioned_visual_scale_init",
            self.query_conditioned_visual_scale_init,
        )
        self.query_conditioned_patch_selector_mode = getattr(
            self.projector,
            "query_conditioned_patch_selector_mode",
            self.query_conditioned_patch_selector_mode,
        )
        self.query_conditioned_patch_selector_hidden_dim = getattr(
            self.projector,
            "query_conditioned_patch_selector_hidden_dim",
            self.query_conditioned_patch_selector_hidden_dim,
        )
        self.query_conditioned_patch_selector_max_residual = getattr(
            self.projector,
            "query_conditioned_patch_selector_max_residual",
            self.query_conditioned_patch_selector_max_residual,
        )
        self.query_conditioned_patch_selector_normalize_mean = getattr(
            self.projector,
            "query_conditioned_patch_selector_normalize_mean",
            self.query_conditioned_patch_selector_normalize_mean,
        )
        self.spatial_grid_tokens = (
            self.v3_base_image_tokens
            if self.connector_type == "spatial_grid_projector"
            else None
        )
        self.spatial_grid_height = getattr(self.projector, "grid_height", None)
        self.spatial_grid_width = getattr(self.projector, "grid_width", None)
        self.spatial_grid_pooling_mode = getattr(self.projector, "pooling_mode", None)
        self.spatial_grid_parameterization = (
            "pooled_siglip_patch_grid_mlp_to_qwen_tokens"
            if self.connector_type == "spatial_grid_projector"
            else None
        )
        self.anyres_grid_shapes = (
            [list(shape) for shape in getattr(self.projector, "grid_shapes", [])]
            if self.connector_type == "mlp_anyres_projector"
            else None
        )
        self.anyres_pooling_mode = (
            getattr(self.projector, "pooling_mode", None)
            if self.connector_type == "mlp_anyres_projector"
            else None
        )
        self.anyres_parameterization = (
            "global_plus_two_local_crop_mlp_pass_through"
            if self.connector_type == "mlp_anyres_projector"
            else None
        )
        if self._uses_spatial_residual():
            self.projector.v3_spatial_residual_branch = V3SpatialResidualBranch(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_latents=self.v3_base_image_tokens,
                hidden_dim=self.spatial_residual_hidden_dim,
                grid_size=self.spatial_residual_grid_size,
                gate_init=self.spatial_residual_gate_init,
            )
        if self._uses_spatial_tail():
            self.projector.v3_spatial_tail_branch = V3SpatialTailBranch(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_tail_tokens=self.spatial_tail_tokens,
                hidden_dim=self.spatial_tail_hidden_dim,
                output_scale=self.spatial_tail_output_scale,
                gate_init=self.spatial_tail_gate_init,
                use_2d_position_features=self.spatial_tail_use_2d_position_features,
            )
        if self._uses_visual_cross_attention():
            self._setup_visual_cross_attention_adapters()

        if freeze_llm:
            self.llm.freeze_base_model()

        self.tokenizer = self.llm.tokenizer
        self._setup_image_placeholder_token()

    @staticmethod
    def _normalize_visual_cross_attention_mode(value: Optional[str]) -> str:
        text = str(value or "none").strip().lower().replace("-", "_")
        if text in {"", "none", "off", "false", "0"}:
            return "none"
        if text in {"on", "true", "1", "gated", "gated_cross_attention"}:
            return "gated"
        raise ValueError(
            "visual_cross_attention_mode must be one of: none, gated"
        )

    @staticmethod
    def _normalize_visual_cross_attention_layers(
        value: Optional[Union[str, List[int], Tuple[int, ...]]],
    ) -> List[int]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            parts = text.replace(" ", ",").split(",")
        else:
            parts = list(value)
        layers = []
        for part in parts:
            if part == "":
                continue
            layer_idx = int(part)
            if layer_idx < 0:
                raise ValueError(
                    f"visual_cross_attention_layers must be non-negative, got {layer_idx}"
                )
            layers.append(layer_idx)
        return sorted(dict.fromkeys(layers))

    def _uses_visual_cross_attention(self) -> bool:
        return self.visual_cross_attention_mode != "none"

    @staticmethod
    def _normalize_spatial_residual_mode(value: Optional[str]) -> str:
        text = str(value or "none").strip().lower().replace("-", "_")
        if text in {"", "none", "off", "false", "0"}:
            return "none"
        if text in {"on", "true", "1", "separable", "separable_table"}:
            return "separable_table"
        raise ValueError(
            "spatial_residual_mode must be one of: none, separable_table"
        )

    def _uses_spatial_residual(self) -> bool:
        return self.spatial_residual_mode != "none"

    @staticmethod
    def _normalize_spatial_tail_mode(value: Optional[str]) -> str:
        text = str(value or "none").strip().lower().replace("-", "_")
        if text in {"", "none", "off", "false", "0"}:
            return "none"
        if text in {"on", "true", "1", "grid", "grid_mlp", "pooled_grid_mlp"}:
            return "pooled_grid_mlp"
        raise ValueError(
            "spatial_tail_mode must be one of: none, pooled_grid_mlp"
        )

    def _uses_spatial_tail(self) -> bool:
        return self.spatial_tail_mode != "none"

    @staticmethod
    def _resolve_anyres_tokens_per_view(
        value: Optional[Union[str, List[int], Tuple[int, ...]]],
        total_tokens: int,
    ) -> Tuple[int, ...]:
        total_tokens = int(total_tokens)
        if total_tokens <= 0:
            raise ValueError(f"AnyRes total_tokens must be > 0, got {total_tokens}")
        if value is None:
            if total_tokens == 256:
                tokens = (128, 64, 64)
            else:
                global_tokens = max(1, total_tokens // 2)
                remaining = total_tokens - global_tokens
                first_local = max(1, remaining // 2)
                second_local = remaining - first_local
                if second_local <= 0:
                    raise ValueError(
                        "AnyRes default split requires at least 3 tokens; "
                        f"got total_tokens={total_tokens}"
                    )
                tokens = (global_tokens, first_local, second_local)
        elif isinstance(value, str):
            parts = [part for part in value.replace(",", " ").split() if part]
            tokens = tuple(int(part) for part in parts)
        else:
            tokens = tuple(int(part) for part in value)
        if not tokens or any(token <= 0 for token in tokens):
            raise ValueError(f"AnyRes tokens_per_view must be positive, got {tokens}")
        if sum(tokens) != total_tokens:
            raise ValueError(
                "AnyRes tokens_per_view must sum to num_image_tokens: "
                f"tokens_per_view={tokens}, total={sum(tokens)}, "
                f"num_image_tokens={total_tokens}"
            )
        return tokens

    def _resolve_decoder_layers(self):
        candidates = [
            ("model.layers", self.llm.model),
            ("model.model.layers", self.llm.model),
            ("base_model.model.model.layers", self.llm.model),
            ("base_model.model.layers", self.llm.model),
        ]
        for path, root in candidates:
            value = root
            found = True
            for part in path.split("."):
                if not hasattr(value, part):
                    found = False
                    break
                value = getattr(value, part)
            if found and isinstance(value, nn.ModuleList):
                return value
        raise RuntimeError(
            "Could not locate decoder layers for visual cross-attention adapters."
        )

    def _setup_visual_cross_attention_adapters(self) -> None:
        decoder_layers = self._resolve_decoder_layers()
        if not self.visual_cross_attention_layers:
            self.visual_cross_attention_layers = [
                idx for idx in (12, 18, 24, 30) if idx < len(decoder_layers)
            ]
        if not self.visual_cross_attention_layers:
            raise ValueError("visual_cross_attention_layers resolved to an empty list")

        for layer_idx in self.visual_cross_attention_layers:
            if layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"visual_cross_attention layer {layer_idx} is out of range for "
                    f"decoder with {len(decoder_layers)} layers"
                )
            key = str(layer_idx)
            self.visual_cross_attention_adapters[key] = GatedVisualCrossAttentionAdapter(
                hidden_size=self.llm.hidden_size,
                num_heads=self.visual_cross_attention_num_heads,
                adapter_dim=self.visual_cross_attention_adapter_dim,
                gate_init=self.visual_cross_attention_gate_init,
                dropout=self.visual_cross_attention_dropout,
            )
            handle = decoder_layers[layer_idx].register_forward_hook(
                self._make_visual_cross_attention_hook(key)
            )
            self._visual_cross_attention_hook_handles.append(handle)

    def _make_visual_cross_attention_hook(self, layer_key: str):
        def hook(_module, _inputs, output):
            visual_memory = self._active_visual_memory
            if visual_memory is None:
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states is None:
                return output
            adapter = self.visual_cross_attention_adapters[layer_key]
            adapted = adapter(hidden_states, visual_memory)
            self._last_visual_cross_attention_diagnostics[
                f"train/visual_cross_attention_layer_{layer_key}_gate"
            ] = adapter.gate_value()
            if isinstance(output, tuple):
                return (adapted,) + output[1:]
            return adapted

        return hook

    @contextmanager
    def _visual_cross_attention_context(self, visual_memory: Optional[torch.Tensor]):
        previous = self._active_visual_memory
        self._active_visual_memory = (
            visual_memory if self._uses_visual_cross_attention() else None
        )
        try:
            yield
        finally:
            self._active_visual_memory = previous

    def encode_images(
        self,
        images: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
        question_summary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Encode images with a fixed-size V3 visual prefix.

        Returns:
            image_tokens: [B, num_image_tokens, llm_dim]
            image_token_mask: [B, num_image_tokens]
            image_token_counts: [B]
        """
        if images.dtype != torch.float32:
            images = images.to(torch.float32)
        if images.ndim == 5:
            if self.connector_type != "mlp_anyres_projector":
                raise ValueError(
                    "Multi-view image tensors [B, V, 3, H, W] are only supported "
                    "by connector_type='mlp_anyres_projector'"
                )
            batch_size, num_views = int(images.shape[0]), int(images.shape[1])
            if self.anyres_num_views is not None and num_views != int(self.anyres_num_views):
                raise ValueError(
                    f"Expected {self.anyres_num_views} AnyRes image views, got {num_views}"
                )
            encoder_images = images.reshape(
                batch_size * num_views,
                *images.shape[2:],
            )
        elif images.ndim == 4:
            if self.connector_type == "mlp_anyres_projector":
                raise ValueError(
                    "mlp_anyres_projector expects image tensors shaped "
                    "[B, V, 3, H, W]"
                )
            batch_size, num_views = int(images.shape[0]), 1
            encoder_images = images
        else:
            raise ValueError(
                "images must have shape [B, 3, H, W] or [B, V, 3, H, W], "
                f"got {list(images.shape)}"
            )

        if target_num_tokens is not None:
            requested = target_num_tokens.to(device=images.device, dtype=torch.long)
            expected = torch.full_like(requested, self.num_image_tokens)
            if not torch.equal(requested, expected):
                raise ValueError(
                    "AnyMALv3 uses a fixed visual prefix: "
                    f"expected {self.num_image_tokens} placeholders per sample, "
                    f"got {requested.tolist()}."
                )

        train_vision = (not self.freeze_vision) or any(
            param.requires_grad for param in self.image_encoder.parameters()
        )
        if train_vision:
            vision_features = self.image_encoder(encoder_images)
        else:
            with torch.no_grad():
                vision_features = self.image_encoder(encoder_images)

        projector_dtype = next(self.projector.parameters()).dtype
        if vision_features.dtype != projector_dtype:
            vision_features = vision_features.to(projector_dtype)
        if self.connector_type == "mlp_anyres_projector":
            vision_features_for_projector = vision_features.reshape(
                batch_size,
                num_views,
                vision_features.shape[1],
                vision_features.shape[2],
            )
            diagnostics_features = vision_features
        else:
            vision_features_for_projector = vision_features
            diagnostics_features = vision_features

        if self._uses_question_summary():
            self._last_connector_diagnostics = self._compute_connector_diagnostics(
                diagnostics_features
            )
            image_tokens = self.projector(
                vision_features_for_projector,
                question_summary=question_summary,
            )
        else:
            self._last_connector_diagnostics = self._compute_connector_diagnostics(
                diagnostics_features
            )
            image_tokens = self.projector(vision_features_for_projector)
        spatial_branch = getattr(self.projector, "v3_spatial_residual_branch", None)
        if spatial_branch is not None:
            image_tokens = image_tokens + spatial_branch(vision_features).to(
                dtype=image_tokens.dtype,
                device=image_tokens.device,
            )
        spatial_tail = getattr(self.projector, "v3_spatial_tail_branch", None)
        if spatial_tail is not None:
            tail_tokens = spatial_tail(vision_features).to(
                dtype=image_tokens.dtype,
                device=image_tokens.device,
            )
            image_tokens = torch.cat([image_tokens, tail_tokens], dim=1)
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
            if isinstance(multiplier, torch.Tensor):
                multiplier_tensor = multiplier.detach().float()
                metrics["train/connector_output_multiplier_mean"] = float(
                    multiplier_tensor.mean().item()
                )
                metrics["train/connector_output_multiplier_min"] = float(
                    multiplier_tensor.min().item()
                )
                metrics["train/connector_output_multiplier_max"] = float(
                    multiplier_tensor.max().item()
                )
                if multiplier_tensor.numel() == 1:
                    metrics["train/connector_output_multiplier"] = float(
                        multiplier_tensor.item()
                    )
            else:
                metrics["train/connector_output_multiplier"] = float(multiplier)
                metrics["train/connector_output_multiplier_mean"] = float(multiplier)
                metrics["train/connector_output_multiplier_min"] = float(multiplier)
                metrics["train/connector_output_multiplier_max"] = float(multiplier)
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
        query_scale_diagnostics = getattr(
            self.projector,
            "get_query_visual_scale_diagnostics",
            None,
        )
        if query_scale_diagnostics is not None:
            metrics.update(query_scale_diagnostics())
        query_patch_diagnostics = getattr(
            self.projector,
            "get_query_patch_selector_diagnostics",
            None,
        )
        if query_patch_diagnostics is not None:
            metrics.update(query_patch_diagnostics())
        if self._uses_visual_cross_attention():
            metrics.update(
                {
                    f"train/visual_cross_attention_layer_{key}_gate": adapter.gate_value()
                    for key, adapter in self.visual_cross_attention_adapters.items()
                }
            )
            metrics.update(self._last_visual_cross_attention_diagnostics)
        spatial_branch = getattr(self.projector, "v3_spatial_residual_branch", None)
        if spatial_branch is not None:
            metrics["train/v3_spatial_residual_gate"] = spatial_branch.gate_value()
        spatial_tail = getattr(self.projector, "v3_spatial_tail_branch", None)
        if spatial_tail is not None:
            metrics["train/v3_spatial_tail_gate"] = spatial_tail.gate_value()
            metrics["train/v3_spatial_tail_tokens"] = float(
                spatial_tail.num_tail_tokens
            )
            tail_rms = getattr(spatial_tail, "_last_tail_rms", None)
            if tail_rms is not None:
                metrics["train/v3_spatial_tail_rms"] = float(tail_rms.item())
            metrics["train/v3_base_image_tokens"] = float(self.v3_base_image_tokens)
        self._last_connector_diagnostics = metrics

    def _uses_question_summary(self) -> bool:
        return (
            self.connector_type == "question_conditioned_perceiver_resampler"
            or self.query_conditioned_visual_scale_mode != "none"
            or self.query_conditioned_patch_selector_mode != "none"
        )

    def _build_question_summary(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.Tensor]:
        """Pool frozen LLM token embeddings over prompt/context tokens only."""
        if not self._uses_question_summary():
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
        if (
            self._uses_visual_cross_attention()
            and self.visual_cross_attention_freeze_connector
            and not inputs_embeds.requires_grad
        ):
            # HF gradient checkpointing skips autograd if all layer inputs are
            # frozen. E1 trains only decoder-side adapters, so use a detached
            # input leaf to carry gradients to hooks without updating the
            # frozen connector or token embeddings.
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        with self._visual_cross_attention_context(image_tokens):
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

        with self._visual_cross_attention_context(image_tokens):
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
            train_connector = not (
                self._uses_visual_cross_attention()
                and self.visual_cross_attention_freeze_connector
            )
            for param in self.projector.parameters():
                param.requires_grad = train_connector
            for param in self.visual_cross_attention_adapters.parameters():
                param.requires_grad = self._uses_visual_cross_attention()
            if self._uses_visual_cross_attention():
                connector_note = "frozen connector" if not train_connector else "connector"
                print(
                    "Stage 1: Training V3 "
                    f"{connector_note} + gated visual cross-attention adapters"
                )
            else:
                print(f"Stage 1: Training V3 {self.connector_type} connector only")
        elif stage == 2:
            self.image_encoder.freeze()
            self.llm.freeze_base_model()
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.visual_cross_attention_adapters.parameters():
                param.requires_grad = self._uses_visual_cross_attention()
            if self._uses_visual_cross_attention():
                print(
                    f"Stage 2: Training V3 {self.connector_type} connector + LoRA + "
                    "gated visual cross-attention adapters"
                )
            else:
                print(f"Stage 2: Training V3 {self.connector_type} connector + LoRA")
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
        vx_total, vx_train = count_params(self.visual_cross_attention_adapters)

        print("\nTrainable Parameters:")
        print(f"  Vision Encoder:    {vision_train:,} / {vision_total:,}")
        print(f"  V3 Connector:      {proj_train:,} / {proj_total:,}")
        print(f"  Visual Cross-Attn: {vx_train:,} / {vx_total:,}")
        print(f"  LLM:               {llm_train:,} / {llm_total:,}")
        print(
            f"  Total:             {vision_train + proj_train + vx_train + llm_train:,} / "
            f"{vision_total + proj_total + vx_total + llm_total:,}"
        )

    def save_visual_cross_attention_adapters(self, save_path: str) -> None:
        if not self._uses_visual_cross_attention():
            return
        import os

        torch.save(
            self.visual_cross_attention_adapters.state_dict(),
            os.path.join(save_path, self.visual_cross_attention_state_file),
        )

    def load_visual_cross_attention_adapters(
        self,
        checkpoint_path: str,
        map_location="cpu",
        strict: bool = True,
        allow_missing: bool = False,
    ) -> None:
        if not self._uses_visual_cross_attention():
            return
        import os

        adapter_path = os.path.join(
            checkpoint_path,
            self.visual_cross_attention_state_file,
        )
        if not os.path.exists(adapter_path):
            if allow_missing:
                print(
                    "Initialized new V3 visual cross-attention adapter parameter(s) "
                    f"while warm-starting from {checkpoint_path}"
                )
                return
            raise FileNotFoundError(
                f"Missing visual cross-attention adapter weights: {adapter_path}"
            )
        state_dict = torch.load(adapter_path, map_location=map_location)
        self.visual_cross_attention_adapters.load_state_dict(
            state_dict,
            strict=strict,
        )

    def save_vision_adapter(self, save_path: str) -> list[str]:
        """Save trainable vision-tower deltas for opt-in E3-style adapters."""
        import os

        state_dict = {
            name: param.detach().cpu()
            for name, param in self.image_encoder.named_parameters()
            if param.requires_grad
        }
        if not state_dict:
            return []

        torch.save(
            {"state_dict": state_dict, "parameter_names": sorted(state_dict)},
            os.path.join(save_path, self.vision_adapter_state_file),
        )
        return sorted(state_dict)

    def load_vision_adapter(
        self,
        checkpoint_path: str,
        map_location="cpu",
        strict: bool = True,
    ) -> list[str]:
        import os

        adapter_path = os.path.join(checkpoint_path, self.vision_adapter_state_file)
        if not os.path.exists(adapter_path):
            return []

        payload = torch.load(adapter_path, map_location=map_location)
        state_dict = payload.get("state_dict", payload)
        params = dict(self.image_encoder.named_parameters())
        missing = sorted(set(state_dict) - set(params))
        if strict and missing:
            raise RuntimeError(
                "Vision adapter checkpoint contains unknown parameter(s): "
                f"{missing}"
            )

        loaded = []
        with torch.no_grad():
            for name, tensor in state_dict.items():
                param = params.get(name)
                if param is None:
                    continue
                if tuple(param.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"Vision adapter shape mismatch for {name}: "
                        f"checkpoint={tuple(tensor.shape)} model={tuple(param.shape)}"
                    )
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(name)
        print(
            "Loaded V3 vision adapter parameter(s) from "
            f"{adapter_path}: {len(loaded)} tensor(s)"
        )
        return loaded

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
        self.save_visual_cross_attention_adapters(save_path)
        vision_adapter_names = self.save_vision_adapter(save_path)

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
                "vision_image_size": self.vision_image_size,
                "vision_adapter_saved": bool(vision_adapter_names),
                "vision_adapter_parameter_names": vision_adapter_names,
                "connector_type": self.connector_type,
                "num_image_tokens": self.num_image_tokens,
                "v3_base_image_tokens": self.v3_base_image_tokens,
                "max_image_tokens": self.max_image_tokens,
                "min_image_tokens": self.min_image_tokens,
                "image_tokens": self.num_image_tokens,
                "connector_layers": self.connector_layers,
                "connector_heads": self.connector_heads,
                "connector_ff_mult": self.connector_ff_mult,
                "connector_output_scale": self.connector_output_scale,
                "connector_output_gate_init": self.connector_output_gate_init,
                "connector_trainable_scale_mode": self.connector_trainable_scale_mode,
                "connector_trainable_scale_parameterization": (
                    "log_scale"
                    if self.connector_trainable_scale_mode != "none"
                    else None
                ),
                "use_2d_patch_position_features": self.use_2d_patch_position_features,
                "patch_position_feature_type": self.patch_position_feature_type,
                "patch_position_grid_size": self.patch_position_grid_size,
                "patch_position_mlp_hidden_dim": self.patch_position_mlp_hidden_dim,
                "patch_position_feature_scale": self.patch_position_feature_scale,
                "spatial_grid_tokens": self.spatial_grid_tokens,
                "spatial_grid_height": self.spatial_grid_height,
                "spatial_grid_width": self.spatial_grid_width,
                "spatial_grid_pooling_mode": self.spatial_grid_pooling_mode,
                "spatial_grid_output_norm": (
                    "per_token_rms"
                    if self.connector_type == "spatial_grid_projector"
                    else None
                ),
                "spatial_grid_parameterization": self.spatial_grid_parameterization,
                "anyres_num_views": self.anyres_num_views,
                "anyres_tokens_per_view": (
                    list(self.anyres_tokens_per_view)
                    if self.anyres_tokens_per_view is not None
                    else None
                ),
                "anyres_crop_mode": (
                    self.anyres_crop_mode
                    if self.connector_type == "mlp_anyres_projector"
                    else None
                ),
                "anyres_image_view_mode": (
                    self.image_view_mode
                    if self.connector_type == "mlp_anyres_projector"
                    else None
                ),
                "image_view_mode": self.image_view_mode,
                "anyres_use_view_embeddings": (
                    self.anyres_use_view_embeddings
                    if self.connector_type == "mlp_anyres_projector"
                    else None
                ),
                "anyres_grid_shapes": self.anyres_grid_shapes,
                "anyres_pooling_mode": self.anyres_pooling_mode,
                "anyres_parameterization": self.anyres_parameterization,
                "query_conditioned_visual_scale_mode": self.query_conditioned_visual_scale_mode,
                "query_conditioned_visual_scale_min": self.query_conditioned_visual_scale_min,
                "query_conditioned_visual_scale_max": self.query_conditioned_visual_scale_max,
                "query_conditioned_visual_scale_init": self.query_conditioned_visual_scale_init,
                "query_conditioned_visual_scale_parameterization": (
                    "bounded_sigmoid_absolute_scale"
                    if self.query_conditioned_visual_scale_mode != "none"
                    else None
                ),
                "query_conditioned_patch_selector_mode": self.query_conditioned_patch_selector_mode,
                "query_conditioned_patch_selector_hidden_dim": self.query_conditioned_patch_selector_hidden_dim,
                "query_conditioned_patch_selector_max_residual": self.query_conditioned_patch_selector_max_residual,
                "query_conditioned_patch_selector_normalize_mean": self.query_conditioned_patch_selector_normalize_mean,
                "query_conditioned_patch_selector_parameterization": (
                    "bounded_residual_mlp_patch_weights"
                    if self.query_conditioned_patch_selector_mode != "none"
                    else None
                ),
                "spatial_residual_mode": self.spatial_residual_mode,
                "spatial_residual_hidden_dim": self.spatial_residual_hidden_dim,
                "spatial_residual_grid_size": self.spatial_residual_grid_size,
                "spatial_residual_gate_init": self.spatial_residual_gate_init,
                "spatial_residual_parameterization": (
                    "separable_row_col_patch_attention_residual"
                    if self._uses_spatial_residual()
                    else None
                ),
                "spatial_tail_mode": self.spatial_tail_mode,
                "spatial_tail_tokens": self.spatial_tail_tokens,
                "spatial_tail_hidden_dim": self.spatial_tail_hidden_dim,
                "spatial_tail_output_scale": self.spatial_tail_output_scale,
                "spatial_tail_gate_init": self.spatial_tail_gate_init,
                "spatial_tail_use_2d_position_features": self.spatial_tail_use_2d_position_features,
                "spatial_tail_grid_height": self.spatial_tail_grid_height,
                "spatial_tail_grid_width": self.spatial_tail_grid_width,
                "spatial_tail_pooling_mode": (
                    "adaptive_avg_pool2d_or_linear_interpolate"
                    if self._uses_spatial_tail()
                    else None
                ),
                "spatial_tail_parameterization": (
                    "v11_prefix_plus_gated_pooled_siglip_patch_mlp_tail"
                    if self._uses_spatial_tail()
                    else None
                ),
                "visual_cross_attention_mode": self.visual_cross_attention_mode,
                "visual_cross_attention_layers": list(self.visual_cross_attention_layers),
                "visual_cross_attention_num_heads": self.visual_cross_attention_num_heads,
                "visual_cross_attention_adapter_dim": self.visual_cross_attention_adapter_dim,
                "visual_cross_attention_gate_init": self.visual_cross_attention_gate_init,
                "visual_cross_attention_dropout": self.visual_cross_attention_dropout,
                "visual_cross_attention_freeze_connector": self.visual_cross_attention_freeze_connector,
                "visual_cross_attention_parameterization": (
                    "decoder_layer_gated_cross_attention_to_visual_memory"
                    if self._uses_visual_cross_attention()
                    else None
                ),
                "project_directly_to_llm_dim": self.project_directly_to_llm_dim,
                **(
                    self.llm.get_model_metadata()
                    if hasattr(self.llm, "get_model_metadata")
                    else {"llm_backbone": self.llm_backbone, "llm_model_name": self.llm_backbone}
                ),
                "image_placeholder_token": getattr(self, "image_placeholder_token", None),
                "image_placeholder_token_id": getattr(self, "image_placeholder_token_id", None),
                "image_placeholder_count": self.num_image_tokens,
                "stage1_connector_init": getattr(
                    self,
                    "stage1_connector_init",
                    "fresh_random",
                ),
                "stage1_connector_source_checkpoint": getattr(
                    self,
                    "stage1_connector_source_checkpoint",
                    None,
                ),
                "stage1_teacher_kl_mode": getattr(
                    self,
                    "stage1_teacher_kl_mode",
                    None,
                ),
                "stage1_teacher_kl_weight": getattr(
                    self,
                    "stage1_teacher_kl_weight",
                    None,
                ),
                "stage1_teacher_kl_image_tokens": getattr(
                    self,
                    "stage1_teacher_kl_image_tokens",
                    None,
                ),
                "stage1_teacher_kl_temperature": getattr(
                    self,
                    "stage1_teacher_kl_temperature",
                    None,
                ),
                "stage1_teacher_kl_direction": getattr(
                    self,
                    "stage1_teacher_kl_direction",
                    None,
                ),
                "stage1_teacher_kl_checkpoint": getattr(
                    self,
                    "stage1_teacher_kl_checkpoint",
                    None,
                ),
                "stage1_teacher_kl_cache_path": getattr(
                    self,
                    "stage1_teacher_kl_cache_path",
                    None,
                ),
                "stage1_teacher_kl_cache_schema": getattr(
                    self,
                    "stage1_teacher_kl_cache_schema",
                    None,
                ),
                "stage1_teacher_kl_cache_top_k": getattr(
                    self,
                    "stage1_teacher_kl_cache_top_k",
                    None,
                ),
                "stage1_teacher_kl_cache_entries": getattr(
                    self,
                    "stage1_teacher_kl_cache_entries",
                    None,
                ),
                "llm_checkpoint_saved": llm_saved,
                "llm_base_weights_saved": bool(llm_saved and save_llm_base),
                "question_conditioning": (
                    "pooled_prompt_embedding_additive_latent_shift"
                    if self.connector_type == "question_conditioned_perceiver_resampler"
                    else (
                        "pooled_prompt_embedding_bounded_visual_scale"
                        if self.query_conditioned_visual_scale_mode != "none"
                        else (
                        "pooled_prompt_embedding_bounded_residual_patch_selector"
                        if self.query_conditioned_patch_selector_mode != "none"
                        else (
                            "spatial_preserving_compressed_grid"
                            if self.connector_type == "spatial_grid_projector"
                            else (
                                "gated_spatial_residual_branch"
                                if self._uses_spatial_residual()
                                else (
                                    "hybrid_v11_prefix_spatial_tail"
                                    if self._uses_spatial_tail()
                                    else (
                                        "decoder_layer_gated_visual_cross_attention"
                                        if self._uses_visual_cross_attention()
                                        else None
                                    )
                                )
                            )
                        )
                    )
                    )
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
        if "connector_type" not in kwargs and meta.get("connector_type") is not None:
            kwargs["connector_type"] = meta["connector_type"]
        if "num_image_tokens" not in kwargs:
            meta_image_tokens = (
                meta.get("num_image_tokens")
                or meta.get("image_tokens")
                or meta.get("image_placeholder_count")
            )
            if meta_image_tokens is not None:
                kwargs["num_image_tokens"] = int(meta_image_tokens)
        if "connector_layers" not in kwargs and meta.get("connector_layers") is not None:
            kwargs["connector_layers"] = int(meta["connector_layers"])
        if "connector_heads" not in kwargs and meta.get("connector_heads") is not None:
            kwargs["connector_heads"] = int(meta["connector_heads"])
        if "connector_ff_mult" not in kwargs and meta.get("connector_ff_mult") is not None:
            kwargs["connector_ff_mult"] = int(meta["connector_ff_mult"])
        if (
            "project_directly_to_llm_dim" not in kwargs
            and meta.get("project_directly_to_llm_dim") is not None
        ):
            kwargs["project_directly_to_llm_dim"] = bool(
                meta["project_directly_to_llm_dim"]
            )
        if llm_backbone is not None and "llm_backbone" not in kwargs:
            kwargs["llm_backbone"] = llm_backbone
        elif "llm_backbone" not in kwargs and meta.get("llm_backbone") is not None:
            kwargs["llm_backbone"] = meta["llm_backbone"]
        if "vision_image_size" not in kwargs and meta.get("vision_image_size") is not None:
            kwargs["vision_image_size"] = int(meta["vision_image_size"])
        if "connector_output_scale" not in kwargs and meta.get("connector_output_scale") is not None:
            kwargs["connector_output_scale"] = float(meta["connector_output_scale"])
        if "connector_output_gate_init" not in kwargs and meta.get("connector_output_gate_init") is not None:
            kwargs["connector_output_gate_init"] = float(meta["connector_output_gate_init"])
        if (
            "connector_trainable_scale_mode" not in kwargs
            and meta.get("connector_trainable_scale_mode") is not None
        ):
            kwargs["connector_trainable_scale_mode"] = meta["connector_trainable_scale_mode"]
        if (
            "use_2d_patch_position_features" not in kwargs
            and meta.get("use_2d_patch_position_features") is not None
        ):
            kwargs["use_2d_patch_position_features"] = bool(
                meta["use_2d_patch_position_features"]
            )
        if (
            "patch_position_feature_type" not in kwargs
            and meta.get("patch_position_feature_type") is not None
        ):
            kwargs["patch_position_feature_type"] = meta["patch_position_feature_type"]
        if (
            "patch_position_grid_size" not in kwargs
            and meta.get("patch_position_grid_size") is not None
        ):
            kwargs["patch_position_grid_size"] = int(meta["patch_position_grid_size"])
        if (
            "patch_position_mlp_hidden_dim" not in kwargs
            and meta.get("patch_position_mlp_hidden_dim") is not None
        ):
            kwargs["patch_position_mlp_hidden_dim"] = int(
                meta["patch_position_mlp_hidden_dim"]
            )
        if (
            "patch_position_feature_scale" not in kwargs
            and meta.get("patch_position_feature_scale") is not None
        ):
            kwargs["patch_position_feature_scale"] = float(
                meta["patch_position_feature_scale"]
            )
        if (
            "query_conditioned_visual_scale_mode" not in kwargs
            and meta.get("query_conditioned_visual_scale_mode") is not None
        ):
            kwargs["query_conditioned_visual_scale_mode"] = meta[
                "query_conditioned_visual_scale_mode"
            ]
        if (
            "query_conditioned_visual_scale_min" not in kwargs
            and meta.get("query_conditioned_visual_scale_min") is not None
        ):
            kwargs["query_conditioned_visual_scale_min"] = float(
                meta["query_conditioned_visual_scale_min"]
            )
        if (
            "query_conditioned_visual_scale_max" not in kwargs
            and meta.get("query_conditioned_visual_scale_max") is not None
        ):
            kwargs["query_conditioned_visual_scale_max"] = float(
                meta["query_conditioned_visual_scale_max"]
            )
        if (
            "query_conditioned_visual_scale_init" not in kwargs
            and meta.get("query_conditioned_visual_scale_init") is not None
        ):
            kwargs["query_conditioned_visual_scale_init"] = float(
                meta["query_conditioned_visual_scale_init"]
            )
        if (
            "query_conditioned_patch_selector_mode" not in kwargs
            and meta.get("query_conditioned_patch_selector_mode") is not None
        ):
            kwargs["query_conditioned_patch_selector_mode"] = meta[
                "query_conditioned_patch_selector_mode"
            ]
        if (
            "query_conditioned_patch_selector_hidden_dim" not in kwargs
            and meta.get("query_conditioned_patch_selector_hidden_dim") is not None
        ):
            kwargs["query_conditioned_patch_selector_hidden_dim"] = int(
                meta["query_conditioned_patch_selector_hidden_dim"]
            )
        if (
            "query_conditioned_patch_selector_max_residual" not in kwargs
            and meta.get("query_conditioned_patch_selector_max_residual") is not None
        ):
            kwargs["query_conditioned_patch_selector_max_residual"] = float(
                meta["query_conditioned_patch_selector_max_residual"]
            )
        if (
            "query_conditioned_patch_selector_normalize_mean" not in kwargs
            and meta.get("query_conditioned_patch_selector_normalize_mean") is not None
        ):
            kwargs["query_conditioned_patch_selector_normalize_mean"] = bool(
                meta["query_conditioned_patch_selector_normalize_mean"]
            )
        if (
            "visual_cross_attention_mode" not in kwargs
            and meta.get("visual_cross_attention_mode") is not None
        ):
            kwargs["visual_cross_attention_mode"] = meta["visual_cross_attention_mode"]
        if (
            "visual_cross_attention_layers" not in kwargs
            and meta.get("visual_cross_attention_layers") is not None
        ):
            kwargs["visual_cross_attention_layers"] = meta["visual_cross_attention_layers"]
        if (
            "visual_cross_attention_num_heads" not in kwargs
            and meta.get("visual_cross_attention_num_heads") is not None
        ):
            kwargs["visual_cross_attention_num_heads"] = int(
                meta["visual_cross_attention_num_heads"]
            )
        if (
            "visual_cross_attention_adapter_dim" not in kwargs
            and meta.get("visual_cross_attention_adapter_dim") is not None
        ):
            kwargs["visual_cross_attention_adapter_dim"] = int(
                meta["visual_cross_attention_adapter_dim"]
            )
        if (
            "visual_cross_attention_gate_init" not in kwargs
            and meta.get("visual_cross_attention_gate_init") is not None
        ):
            kwargs["visual_cross_attention_gate_init"] = float(
                meta["visual_cross_attention_gate_init"]
            )
        if (
            "visual_cross_attention_dropout" not in kwargs
            and meta.get("visual_cross_attention_dropout") is not None
        ):
            kwargs["visual_cross_attention_dropout"] = float(
                meta["visual_cross_attention_dropout"]
            )
        if (
            "visual_cross_attention_freeze_connector" not in kwargs
            and meta.get("visual_cross_attention_freeze_connector") is not None
        ):
            kwargs["visual_cross_attention_freeze_connector"] = bool(
                meta["visual_cross_attention_freeze_connector"]
            )
        if (
            "spatial_residual_mode" not in kwargs
            and meta.get("spatial_residual_mode") is not None
        ):
            kwargs["spatial_residual_mode"] = meta["spatial_residual_mode"]
        if (
            "spatial_residual_hidden_dim" not in kwargs
            and meta.get("spatial_residual_hidden_dim") is not None
        ):
            kwargs["spatial_residual_hidden_dim"] = int(
                meta["spatial_residual_hidden_dim"]
            )
        if (
            "spatial_residual_grid_size" not in kwargs
            and meta.get("spatial_residual_grid_size") is not None
        ):
            kwargs["spatial_residual_grid_size"] = int(
                meta["spatial_residual_grid_size"]
            )
        if (
            "spatial_residual_gate_init" not in kwargs
            and meta.get("spatial_residual_gate_init") is not None
        ):
            kwargs["spatial_residual_gate_init"] = float(
                meta["spatial_residual_gate_init"]
            )
        if (
            "spatial_tail_mode" not in kwargs
            and meta.get("spatial_tail_mode") is not None
        ):
            kwargs["spatial_tail_mode"] = meta["spatial_tail_mode"]
        if (
            "spatial_tail_tokens" not in kwargs
            and meta.get("spatial_tail_tokens") is not None
        ):
            kwargs["spatial_tail_tokens"] = int(meta["spatial_tail_tokens"])
        if (
            "spatial_tail_hidden_dim" not in kwargs
            and meta.get("spatial_tail_hidden_dim") is not None
        ):
            kwargs["spatial_tail_hidden_dim"] = int(meta["spatial_tail_hidden_dim"])
        if (
            "spatial_tail_output_scale" not in kwargs
            and meta.get("spatial_tail_output_scale") is not None
        ):
            kwargs["spatial_tail_output_scale"] = float(
                meta["spatial_tail_output_scale"]
            )
        if (
            "spatial_tail_gate_init" not in kwargs
            and meta.get("spatial_tail_gate_init") is not None
        ):
            kwargs["spatial_tail_gate_init"] = float(meta["spatial_tail_gate_init"])
        if (
            "spatial_tail_use_2d_position_features" not in kwargs
            and meta.get("spatial_tail_use_2d_position_features") is not None
        ):
            kwargs["spatial_tail_use_2d_position_features"] = bool(
                meta["spatial_tail_use_2d_position_features"]
            )
        if (
            "anyres_tokens_per_view" not in kwargs
            and meta.get("anyres_tokens_per_view") is not None
        ):
            kwargs["anyres_tokens_per_view"] = meta["anyres_tokens_per_view"]
        if "anyres_crop_mode" not in kwargs and meta.get("anyres_crop_mode") is not None:
            kwargs["anyres_crop_mode"] = meta["anyres_crop_mode"]
        if (
            "anyres_use_view_embeddings" not in kwargs
            and meta.get("anyres_use_view_embeddings") is not None
        ):
            kwargs["anyres_use_view_embeddings"] = bool(
                meta["anyres_use_view_embeddings"]
            )

        model = cls(llm_model_name=llm_model_name, **kwargs)
        checkpoint_spatial_tail_mode = cls._normalize_spatial_tail_mode(
            meta.get("spatial_tail_mode")
        )
        checkpoint_has_spatial_tail = checkpoint_spatial_tail_mode != "none"
        expected_image_tokens = getattr(model, "num_image_tokens", None)
        if model._uses_spatial_tail() and not checkpoint_has_spatial_tail:
            expected_image_tokens = getattr(model, "v3_base_image_tokens", None)
        expected_values = {
            "vision_encoder_type": getattr(model, "vision_encoder_type", None),
            "connector_type": getattr(model, "connector_type", None),
            "num_image_tokens": expected_image_tokens,
            "max_image_tokens": expected_image_tokens,
            "min_image_tokens": expected_image_tokens,
            "connector_layers": getattr(model, "connector_layers", None),
            "connector_heads": getattr(model, "connector_heads", None),
            "connector_ff_mult": getattr(model, "connector_ff_mult", None),
            "project_directly_to_llm_dim": getattr(model, "project_directly_to_llm_dim", None),
        }
        if "vision_image_size" in meta:
            expected_values["vision_image_size"] = getattr(model, "vision_image_size", None)
        if "connector_output_scale" in meta and not allow_connector_output_scale_override:
            expected_values["connector_output_scale"] = getattr(model, "connector_output_scale", None)
        if "connector_output_gate_init" in meta:
            expected_values["connector_output_gate_init"] = getattr(model, "connector_output_gate_init", None)
        if "connector_trainable_scale_mode" in meta:
            expected_values["connector_trainable_scale_mode"] = getattr(
                model,
                "connector_trainable_scale_mode",
                None,
            )
        if "use_2d_patch_position_features" in meta:
            expected_values["use_2d_patch_position_features"] = getattr(
                model,
                "use_2d_patch_position_features",
                None,
            )
        if "patch_position_feature_type" in meta:
            expected_values["patch_position_feature_type"] = getattr(
                model,
                "patch_position_feature_type",
                None,
            )
        if "patch_position_grid_size" in meta:
            expected_values["patch_position_grid_size"] = getattr(
                model,
                "patch_position_grid_size",
                None,
            )
        if "patch_position_mlp_hidden_dim" in meta:
            expected_values["patch_position_mlp_hidden_dim"] = getattr(
                model,
                "patch_position_mlp_hidden_dim",
                None,
            )
        if "patch_position_feature_scale" in meta:
            expected_values["patch_position_feature_scale"] = getattr(
                model,
                "patch_position_feature_scale",
                None,
            )
        if "spatial_grid_tokens" in meta:
            expected_values["spatial_grid_tokens"] = getattr(
                model,
                "spatial_grid_tokens",
                None,
            )
        if "spatial_grid_height" in meta:
            expected_values["spatial_grid_height"] = getattr(
                model,
                "spatial_grid_height",
                None,
            )
        if "spatial_grid_width" in meta:
            expected_values["spatial_grid_width"] = getattr(
                model,
                "spatial_grid_width",
                None,
            )
        if "spatial_grid_pooling_mode" in meta:
            expected_values["spatial_grid_pooling_mode"] = getattr(
                model,
                "spatial_grid_pooling_mode",
                None,
            )
        if meta.get("anyres_num_views") is not None:
            expected_values["anyres_num_views"] = getattr(
                model,
                "anyres_num_views",
                None,
            )
        if meta.get("anyres_tokens_per_view") is not None:
            tokens_per_view = getattr(model, "anyres_tokens_per_view", None)
            expected_values["anyres_tokens_per_view"] = (
                list(tokens_per_view) if tokens_per_view is not None else None
            )
        if meta.get("anyres_crop_mode") is not None:
            expected_values["anyres_crop_mode"] = getattr(
                model,
                "anyres_crop_mode",
                None,
            )
        if meta.get("anyres_image_view_mode") is not None:
            expected_values["anyres_image_view_mode"] = getattr(
                model,
                "image_view_mode",
                None,
            )
        if "image_view_mode" in meta:
            expected_values["image_view_mode"] = getattr(
                model,
                "image_view_mode",
                None,
            )
        if meta.get("anyres_use_view_embeddings") is not None:
            expected_values["anyres_use_view_embeddings"] = getattr(
                model,
                "anyres_use_view_embeddings",
                None,
            )
        if meta.get("anyres_grid_shapes") is not None:
            expected_values["anyres_grid_shapes"] = getattr(
                model,
                "anyres_grid_shapes",
                None,
            )
        if "anyres_pooling_mode" in meta:
            expected_values["anyres_pooling_mode"] = getattr(
                model,
                "anyres_pooling_mode",
                None,
            )
        if "query_conditioned_visual_scale_mode" in meta:
            expected_values["query_conditioned_visual_scale_mode"] = getattr(
                model,
                "query_conditioned_visual_scale_mode",
                None,
            )
        if "query_conditioned_visual_scale_min" in meta:
            expected_values["query_conditioned_visual_scale_min"] = getattr(
                model,
                "query_conditioned_visual_scale_min",
                None,
            )
        if "query_conditioned_visual_scale_max" in meta:
            expected_values["query_conditioned_visual_scale_max"] = getattr(
                model,
                "query_conditioned_visual_scale_max",
                None,
            )
        if "query_conditioned_visual_scale_init" in meta:
            expected_values["query_conditioned_visual_scale_init"] = getattr(
                model,
                "query_conditioned_visual_scale_init",
                None,
            )
        if "query_conditioned_patch_selector_mode" in meta:
            expected_values["query_conditioned_patch_selector_mode"] = getattr(
                model,
                "query_conditioned_patch_selector_mode",
                None,
            )
        if "query_conditioned_patch_selector_hidden_dim" in meta:
            expected_values["query_conditioned_patch_selector_hidden_dim"] = getattr(
                model,
                "query_conditioned_patch_selector_hidden_dim",
                None,
            )
        if "query_conditioned_patch_selector_max_residual" in meta:
            expected_values["query_conditioned_patch_selector_max_residual"] = getattr(
                model,
                "query_conditioned_patch_selector_max_residual",
                None,
            )
        if "query_conditioned_patch_selector_normalize_mean" in meta:
            expected_values["query_conditioned_patch_selector_normalize_mean"] = getattr(
                model,
                "query_conditioned_patch_selector_normalize_mean",
                None,
            )
        if "visual_cross_attention_mode" in meta:
            expected_values["visual_cross_attention_mode"] = getattr(
                model,
                "visual_cross_attention_mode",
                None,
            )
        if "visual_cross_attention_layers" in meta:
            expected_values["visual_cross_attention_layers"] = list(
                getattr(model, "visual_cross_attention_layers", [])
            )
        if "visual_cross_attention_num_heads" in meta:
            expected_values["visual_cross_attention_num_heads"] = getattr(
                model,
                "visual_cross_attention_num_heads",
                None,
            )
        if "visual_cross_attention_adapter_dim" in meta:
            expected_values["visual_cross_attention_adapter_dim"] = getattr(
                model,
                "visual_cross_attention_adapter_dim",
                None,
            )
        if "visual_cross_attention_gate_init" in meta:
            expected_values["visual_cross_attention_gate_init"] = getattr(
                model,
                "visual_cross_attention_gate_init",
                None,
            )
        if "visual_cross_attention_dropout" in meta:
            expected_values["visual_cross_attention_dropout"] = getattr(
                model,
                "visual_cross_attention_dropout",
                None,
            )
        if "visual_cross_attention_freeze_connector" in meta:
            expected_values["visual_cross_attention_freeze_connector"] = getattr(
                model,
                "visual_cross_attention_freeze_connector",
                None,
            )
        if "spatial_residual_mode" in meta:
            expected_values["spatial_residual_mode"] = getattr(
                model,
                "spatial_residual_mode",
                None,
            )
        if "spatial_residual_hidden_dim" in meta:
            expected_values["spatial_residual_hidden_dim"] = getattr(
                model,
                "spatial_residual_hidden_dim",
                None,
            )
        if "spatial_residual_grid_size" in meta:
            expected_values["spatial_residual_grid_size"] = getattr(
                model,
                "spatial_residual_grid_size",
                None,
            )
        if "spatial_residual_gate_init" in meta:
            expected_values["spatial_residual_gate_init"] = getattr(
                model,
                "spatial_residual_gate_init",
                None,
            )
        if "spatial_tail_mode" in meta:
            expected_values["spatial_tail_mode"] = getattr(
                model,
                "spatial_tail_mode",
                None,
            )
        if "spatial_tail_tokens" in meta:
            expected_values["spatial_tail_tokens"] = getattr(
                model,
                "spatial_tail_tokens",
                None,
            )
        if "spatial_tail_hidden_dim" in meta:
            expected_values["spatial_tail_hidden_dim"] = getattr(
                model,
                "spatial_tail_hidden_dim",
                None,
            )
        if "spatial_tail_output_scale" in meta:
            expected_values["spatial_tail_output_scale"] = getattr(
                model,
                "spatial_tail_output_scale",
                None,
            )
        if "spatial_tail_gate_init" in meta:
            expected_values["spatial_tail_gate_init"] = getattr(
                model,
                "spatial_tail_gate_init",
                None,
            )
        if "spatial_tail_use_2d_position_features" in meta:
            expected_values["spatial_tail_use_2d_position_features"] = getattr(
                model,
                "spatial_tail_use_2d_position_features",
                None,
            )
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
            projector_state = torch.load(projector_path, map_location="cpu")
            allowed_missing = set()
            allowed_missing_prefixes = set()
            if getattr(model.projector, "trainable_output_log_scale", None) is not None:
                allowed_missing.add("trainable_output_log_scale")
            if getattr(model.projector, "patch_position_embedding", None) is not None:
                allowed_missing.add("patch_position_embedding")
            if getattr(model.projector, "patch_position_mlp", None) is not None:
                allowed_missing_prefixes.add("patch_position_mlp.")
            if getattr(model.projector, "query_visual_scale", None) is not None:
                allowed_missing_prefixes.add("query_visual_scale.")
            if getattr(model.projector, "query_patch_selector", None) is not None:
                allowed_missing_prefixes.add("query_patch_selector.")
            if getattr(model.projector, "v3_spatial_residual_branch", None) is not None:
                allowed_missing_prefixes.add("v3_spatial_residual_branch.")
            if getattr(model.projector, "v3_spatial_tail_branch", None) is not None:
                allowed_missing_prefixes.add("v3_spatial_tail_branch.")
            if allowed_missing or allowed_missing_prefixes:
                incompatible = model.projector.load_state_dict(
                    projector_state,
                    strict=False,
                )
                missing = set(incompatible.missing_keys)
                unexpected = set(incompatible.unexpected_keys)
                disallowed_missing = {
                    key
                    for key in missing
                    if key not in allowed_missing
                    and not any(
                        key.startswith(prefix)
                        for prefix in allowed_missing_prefixes
                    )
                }
                if disallowed_missing or unexpected:
                    raise RuntimeError(
                        "V3 projector warm-start only allows missing "
                        f"{sorted(allowed_missing)} and prefixes "
                        f"{sorted(allowed_missing_prefixes)}; "
                        f"missing={sorted(missing)}, "
                        f"unexpected={sorted(unexpected)}"
                    )
                if missing:
                    print(
                        "Initialized new V3 projector parameter(s) while "
                        f"warm-starting from {save_path}: {sorted(missing)}"
                    )
            else:
                model.projector.load_state_dict(projector_state)
        else:
            raise FileNotFoundError(f"Missing V3 connector weights: {projector_path}")

        model.load_vision_adapter(save_path, strict=True)

        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            base_model = model.llm.model
            if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
                base_model = base_model.unload()
            model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
            if model._uses_visual_cross_attention():
                model._visual_cross_attention_hook_handles = []
                model._setup_visual_cross_attention_adapters()
        if model._uses_visual_cross_attention():
            checkpoint_mode = model._normalize_visual_cross_attention_mode(
                meta.get("visual_cross_attention_mode")
            )
            model.load_visual_cross_attention_adapters(
                save_path,
                allow_missing=checkpoint_mode == "none",
            )
        return model
