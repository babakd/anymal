"""Spatial-grid projector for V13 Qwen substrate experiments."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGridProjector(nn.Module):
    """Project spatially pooled SigLIP patch grids directly into LLM tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grid_tokens: int,
        hidden_dim: Optional[int] = None,
        output_scale: float = 1.0,
        output_gate_init: Optional[float] = None,
        trainable_scale_mode: str = "none",
        use_2d_patch_position_features: bool = True,
        patch_position_feature_type: Optional[str] = None,
        patch_position_grid_size: int = 32,
        patch_position_mlp_hidden_dim: int = 128,
        patch_position_feature_scale: float = 1.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_grid_tokens = int(num_grid_tokens)
        self.num_latents = self.num_grid_tokens
        self.hidden_dim = int(hidden_dim or output_dim)
        self.output_scale = float(output_scale)
        self.patch_position_grid_size = int(patch_position_grid_size)
        self.patch_position_mlp_hidden_dim = int(patch_position_mlp_hidden_dim)
        self.patch_position_feature_scale = float(patch_position_feature_scale)
        self.rms_norm_eps = float(rms_norm_eps)
        if self.input_dim <= 0 or self.output_dim <= 0 or self.num_grid_tokens <= 0:
            raise ValueError("SpatialGridProjector dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"spatial_grid hidden_dim must be > 0, got {hidden_dim}")
        if self.output_scale <= 0:
            raise ValueError(f"output_scale must be > 0, got {output_scale}")
        if self.patch_position_grid_size <= 0:
            raise ValueError(
                "patch_position_grid_size must be > 0, "
                f"got {patch_position_grid_size}"
            )
        if self.patch_position_mlp_hidden_dim <= 0:
            raise ValueError(
                "patch_position_mlp_hidden_dim must be > 0, "
                f"got {patch_position_mlp_hidden_dim}"
            )
        if self.patch_position_feature_scale < 0:
            raise ValueError(
                "patch_position_feature_scale must be >= 0, "
                f"got {patch_position_feature_scale}"
            )

        self.grid_height, self.grid_width = self._target_grid(self.num_grid_tokens)
        self.pooling_mode = "adaptive_avg_pool2d_or_linear_interpolate"
        self.patch_position_feature_type = self._normalize_patch_position_feature_type(
            patch_position_feature_type,
            use_2d_patch_position_features=use_2d_patch_position_features,
        )
        self.use_2d_patch_position_features = self.patch_position_feature_type != "none"
        self.trainable_scale_mode = str(trainable_scale_mode or "none").strip().lower()
        if self.trainable_scale_mode in {"off", "false", "0"}:
            self.trainable_scale_mode = "none"
        if self.trainable_scale_mode not in {"none", "global", "per_token"}:
            raise ValueError(
                "trainable_scale_mode must be one of 'none', 'global', or "
                f"'per_token', got {trainable_scale_mode!r}"
            )
        if self.trainable_scale_mode == "global":
            self.trainable_output_log_scale = nn.Parameter(torch.zeros(()))
        elif self.trainable_scale_mode == "per_token":
            self.trainable_output_log_scale = nn.Parameter(
                torch.zeros(self.num_grid_tokens, 1)
            )
        else:
            self.register_parameter("trainable_output_log_scale", None)

        self.output_gate_init = None if output_gate_init is None else float(output_gate_init)
        if self.output_gate_init is not None:
            if not 0.0 < self.output_gate_init < 1.0:
                raise ValueError(
                    "output_gate_init must be between 0 and 1 when set, "
                    f"got {output_gate_init}"
                )
            gate_logit = torch.logit(torch.tensor(self.output_gate_init, dtype=torch.float32))
            self.output_gate_logit = nn.Parameter(gate_logit)
        else:
            self.output_gate_logit = None

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        if self.patch_position_feature_type == "learned_table":
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(
                    self.patch_position_grid_size,
                    self.patch_position_grid_size,
                    self.input_dim,
                )
            )
        else:
            self.register_parameter("patch_position_embedding", None)
        if self.patch_position_feature_type == "coord_mlp":
            self.patch_position_mlp = nn.Sequential(
                nn.Linear(5, self.patch_position_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(self.patch_position_mlp_hidden_dim, self.input_dim),
            )
            nn.init.zeros_(self.patch_position_mlp[-1].weight)
            nn.init.zeros_(self.patch_position_mlp[-1].bias)
        else:
            self.patch_position_mlp = None
        self._last_output_rms = None

    @staticmethod
    def _normalize_patch_position_feature_type(
        feature_type: Optional[str],
        use_2d_patch_position_features: bool,
    ) -> str:
        if feature_type is None:
            return "learned_table" if use_2d_patch_position_features else "none"
        value = str(feature_type).strip().lower().replace("-", "_")
        if value in {"", "auto"}:
            return "learned_table" if use_2d_patch_position_features else "none"
        if value in {"none", "off", "false", "0"}:
            return "none"
        if value in {"table", "learned", "learned_table", "embedding"}:
            return "learned_table"
        if value in {"coord", "coords", "coordinate", "coordinates", "coord_mlp"}:
            return "coord_mlp"
        raise ValueError(
            "patch_position_feature_type must be one of 'none', "
            "'learned_table', or 'coord_mlp', got "
            f"{feature_type!r}"
        )

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

    def _pool_vision_tokens(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = vision_features.shape
        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        if height is not None and width is not None:
            patches = vision_features[:, prefix_tokens:, :]
            patches = patches.reshape(batch_size, height, width, channels)
            patches = patches.permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(
                patches,
                output_size=(self.grid_height, self.grid_width),
            )
            pooled = pooled.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        else:
            pooled = F.interpolate(
                vision_features.transpose(1, 2),
                size=self.num_grid_tokens,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        if pooled.shape[1] > self.num_grid_tokens:
            pooled = pooled[:, : self.num_grid_tokens, :]
        elif pooled.shape[1] < self.num_grid_tokens:
            pad = pooled[:, -1:, :].expand(
                batch_size,
                self.num_grid_tokens - pooled.shape[1],
                channels,
            )
            pooled = torch.cat([pooled, pad], dim=1)
        return pooled

    @staticmethod
    def _coordinate_features_for_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy, xx.square(), yy.square(), xx * yy], dim=-1)
        return coords.reshape(1, height * width, 5)

    def _learned_patch_position_features(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.patch_position_embedding is None:
            raise RuntimeError("learned position features requested without table")
        table = self.patch_position_embedding.to(device=device)
        table_4d = table.permute(2, 0, 1).unsqueeze(0)
        if (
            self.grid_height != self.patch_position_grid_size
            or self.grid_width != self.patch_position_grid_size
        ):
            table_4d = F.interpolate(
                table_4d,
                size=(self.grid_height, self.grid_width),
                mode="bilinear",
                align_corners=True,
            )
        positions = (
            table_4d.squeeze(0)
            .permute(1, 2, 0)
            .reshape(1, self.grid_height * self.grid_width, self.input_dim)
        )
        return positions[:, : self.num_grid_tokens, :].to(dtype=dtype)

    def _coord_mlp_patch_position_features(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.patch_position_mlp is None:
            raise RuntimeError("coordinate position features requested without MLP")
        coords = self._coordinate_features_for_grid(
            self.grid_height,
            self.grid_width,
            device=device,
            dtype=dtype,
        )
        coords = coords[:, : self.num_grid_tokens, :]
        mlp_dtype = self.patch_position_mlp[0].weight.dtype
        features = self.patch_position_mlp(coords.to(dtype=mlp_dtype))
        return features.to(device=device, dtype=dtype)

    def _add_patch_position_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_position_feature_type == "none":
            return x
        if self.patch_position_feature_type == "learned_table":
            positions = self._learned_patch_position_features(x.device, x.dtype)
        elif self.patch_position_feature_type == "coord_mlp":
            positions = self._coord_mlp_patch_position_features(x.device, x.dtype)
        else:
            raise RuntimeError(
                f"Unsupported patch_position_feature_type={self.patch_position_feature_type!r}"
            )
        return x + positions * self.patch_position_feature_scale

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(self.rms_norm_eps).sqrt()
        return x * torch.rsqrt(rms.to(dtype=x.dtype).square())

    def _output_multiplier(self) -> torch.Tensor:
        multiplier = self.output_scale
        if self.output_gate_logit is not None:
            multiplier = multiplier * torch.sigmoid(self.output_gate_logit)
        if self.trainable_output_log_scale is not None:
            multiplier = multiplier * torch.exp(self.trainable_output_log_scale)
        return multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_dtype = self.input_proj.weight.dtype
        pooled = self._pool_vision_tokens(x).to(dtype=branch_dtype)
        pooled = self._add_patch_position_features(pooled)
        hidden = F.gelu(self.input_proj(self.input_norm(pooled)))
        output = self._rms_normalize(self.output_proj(hidden))
        multiplier = self._output_multiplier()
        if torch.is_tensor(multiplier):
            multiplier = multiplier.to(device=output.device, dtype=output.dtype)
        else:
            multiplier = output.new_tensor(multiplier)
        output = output * multiplier
        self._last_output_rms = output.detach().float().pow(2).mean().sqrt()
        return output.to(dtype=x.dtype)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
