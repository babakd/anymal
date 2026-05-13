"""
Perceiver Resampler for AnyMAL

Projects variable-length vision features to fixed-length latent tokens
using cross-attention, following the Flamingo architecture.

Educational Notes:
-----------------
The Perceiver Resampler solves a key problem: how do we convert a variable
number of image patches (257 for ViT-L) into a fixed number of tokens that
the LLM can process efficiently?

Key Insight from Flamingo:
The resampler uses learnable "latent queries" that attend to vision features
via cross-attention. Think of it like DETR's object queries, but for
extracting a compressed representation of the image.

Architecture:
1. Learnable latent queries: [num_latents, output_dim] (e.g., 64 queries)
2. For each layer:
   a. Cross-attention: latents attend to vision features
   b. Self-attention: latents attend to each other
   c. Feed-forward network: transform features
3. Output: [batch, num_latents, output_dim]

Why cross-attention?
- Learns to extract relevant information from any number of patches
- Fixed output size regardless of input resolution
- Gradients flow through to train the projection

Memory comparison (LLaMA 3 8B, batch=1):
- 257 tokens: ~1.0 GB attention KV cache
- 64 tokens: ~0.25 GB attention KV cache
- 4x memory reduction allows larger batch sizes!

Mathematical view:
Let Q = latent queries, K = V = vision features
Cross-attn(Q, K, V) = softmax(QK^T / sqrt(d)) V
This "queries" the image for information relevant to each latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler that projects vision features to LLM embedding space.

    This is the KEY TRAINABLE COMPONENT during alignment pretraining.
    It learns to compress and project image information into tokens
    that the frozen LLM can understand.

    Args:
        input_dim: Dimension of input vision features (e.g., 1024 for ViT-L)
        output_dim: Dimension of output tokens (e.g., 4096 for LLaMA 3 8B)
        num_latents: Number of output tokens (default: 64)
        num_layers: Number of resampler layers (default: 6)
        num_heads: Number of attention heads (default: 16)
        ff_mult: Feed-forward hidden dim multiplier (default: 4)
        dropout: Dropout rate (default: 0.0)

    Example:
        >>> resampler = PerceiverResampler(
        ...     input_dim=1024,   # CLIP ViT-L output
        ...     output_dim=4096,  # LLaMA 3 8B hidden dim
        ...     num_latents=64,
        ...     num_layers=6
        ... )
        >>> vision_features = torch.randn(4, 257, 1024)  # From CLIP
        >>> image_tokens = resampler(vision_features)
        >>> print(image_tokens.shape)  # [4, 64, 4096]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.0,
        output_scale: float = 1.0,
        output_gate_init: Optional[float] = None,
        trainable_scale_mode: str = "none",
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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_layers = num_layers
        self.patch_position_feature_type = self._normalize_patch_position_feature_type(
            patch_position_feature_type,
            use_2d_patch_position_features=use_2d_patch_position_features,
        )
        self.use_2d_patch_position_features = self.patch_position_feature_type != "none"
        self.patch_position_grid_size = int(patch_position_grid_size)
        if self.patch_position_grid_size <= 0:
            raise ValueError(
                "patch_position_grid_size must be > 0, "
                f"got {patch_position_grid_size}"
            )
        self.patch_position_mlp_hidden_dim = int(patch_position_mlp_hidden_dim)
        if self.patch_position_mlp_hidden_dim <= 0:
            raise ValueError(
                "patch_position_mlp_hidden_dim must be > 0, "
                f"got {patch_position_mlp_hidden_dim}"
            )
        self.patch_position_feature_scale = float(patch_position_feature_scale)
        if self.patch_position_feature_scale < 0.0:
            raise ValueError(
                "patch_position_feature_scale must be >= 0, "
                f"got {patch_position_feature_scale}"
            )
        self.output_scale = float(output_scale)
        if self.output_scale <= 0:
            raise ValueError(f"output_scale must be > 0, got {output_scale}")
        self.query_conditioned_visual_scale_mode = (
            str(query_conditioned_visual_scale_mode or "none").strip().lower()
        )
        if self.query_conditioned_visual_scale_mode in {"off", "false", "0"}:
            self.query_conditioned_visual_scale_mode = "none"
        if self.query_conditioned_visual_scale_mode not in {"none", "scalar", "per_token"}:
            raise ValueError(
                "query_conditioned_visual_scale_mode must be one of 'none', "
                f"'scalar', or 'per_token', got {query_conditioned_visual_scale_mode!r}"
            )
        self.query_conditioned_patch_selector_mode = (
            str(query_conditioned_patch_selector_mode or "none").strip().lower().replace("-", "_")
        )
        if self.query_conditioned_patch_selector_mode in {"off", "false", "0"}:
            self.query_conditioned_patch_selector_mode = "none"
        if self.query_conditioned_patch_selector_mode in {"on", "true", "1", "residual"}:
            self.query_conditioned_patch_selector_mode = "residual_mlp"
        if self.query_conditioned_patch_selector_mode not in {"none", "residual_mlp"}:
            raise ValueError(
                "query_conditioned_patch_selector_mode must be one of 'none' "
                f"or 'residual_mlp', got {query_conditioned_patch_selector_mode!r}"
            )
        self.query_conditioned_patch_selector_hidden_dim = int(
            query_conditioned_patch_selector_hidden_dim
        )
        if self.query_conditioned_patch_selector_hidden_dim <= 0:
            raise ValueError(
                "query_conditioned_patch_selector_hidden_dim must be > 0, "
                f"got {query_conditioned_patch_selector_hidden_dim}"
            )
        self.query_conditioned_patch_selector_max_residual = float(
            query_conditioned_patch_selector_max_residual
        )
        if not 0.0 < self.query_conditioned_patch_selector_max_residual < 1.0:
            raise ValueError(
                "query_conditioned_patch_selector_max_residual must be between "
                f"0 and 1, got {query_conditioned_patch_selector_max_residual}"
            )
        self.query_conditioned_patch_selector_normalize_mean = bool(
            query_conditioned_patch_selector_normalize_mean
        )
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
                torch.zeros(num_latents, 1)
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

        if self.patch_position_feature_type == "learned_table":
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(
                    self.patch_position_grid_size,
                    self.patch_position_grid_size,
                    input_dim,
                )
            )
        else:
            self.register_parameter("patch_position_embedding", None)
        if self.patch_position_feature_type == "coord_mlp":
            self.patch_position_mlp = nn.Sequential(
                nn.Linear(5, self.patch_position_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(self.patch_position_mlp_hidden_dim, input_dim),
            )
            nn.init.zeros_(self.patch_position_mlp[-1].weight)
            nn.init.zeros_(self.patch_position_mlp[-1].bias)
        else:
            self.patch_position_mlp = None
        if self.query_conditioned_visual_scale_mode != "none":
            self.query_visual_scale = QueryConditionedVisualScale(
                condition_dim=output_dim,
                num_latents=num_latents,
                mode=self.query_conditioned_visual_scale_mode,
                min_scale=query_conditioned_visual_scale_min,
                max_scale=query_conditioned_visual_scale_max,
                init_scale=(
                    self.output_scale
                    if query_conditioned_visual_scale_init is None
                    else float(query_conditioned_visual_scale_init)
                ),
            )
        else:
            self.query_visual_scale = None
        self.query_conditioned_visual_scale_min = (
            None
            if self.query_visual_scale is None
            else self.query_visual_scale.min_scale
        )
        self.query_conditioned_visual_scale_max = (
            None
            if self.query_visual_scale is None
            else self.query_visual_scale.max_scale
        )
        self.query_conditioned_visual_scale_init = (
            None
            if self.query_visual_scale is None
            else self.query_visual_scale.init_scale
        )
        self._last_query_visual_scale = None
        if self.query_conditioned_patch_selector_mode != "none":
            self.query_patch_selector = QueryConditionedPatchSelector(
                condition_dim=output_dim,
                input_dim=input_dim,
                hidden_dim=self.query_conditioned_patch_selector_hidden_dim,
                max_residual=self.query_conditioned_patch_selector_max_residual,
                normalize_mean=self.query_conditioned_patch_selector_normalize_mean,
            )
        else:
            self.query_patch_selector = None

        # Learnable latent queries
        # These are the "questions" we ask about the image
        # Initialized with scaled normal distribution for stability
        self.latents = nn.Parameter(
            torch.randn(num_latents, output_dim) * 0.02
        )

        # Project input to output dimension for cross-attention
        # This aligns the vision features with the latent dimension
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Stack of Perceiver blocks
        self.layers = nn.ModuleList([
            PerceiverResamplerBlock(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm for output stability
        self.norm = nn.LayerNorm(output_dim)

    def _output_multiplier(self) -> torch.Tensor:
        multiplier = self.output_scale
        if self.output_gate_logit is not None:
            multiplier = multiplier * torch.sigmoid(self.output_gate_logit)
        if self.trainable_output_log_scale is not None:
            multiplier = multiplier * torch.exp(self.trainable_output_log_scale)
        return multiplier

    @staticmethod
    def _normalize_patch_position_feature_type(
        feature_type: Optional[str],
        use_2d_patch_position_features: bool = False,
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
        if value in {"coord", "coords", "coordinate", "coordinates", "coord_mlp", "coordinate_mlp"}:
            return "coord_mlp"
        raise ValueError(
            "patch_position_feature_type must be one of 'none', "
            "'learned_table', or 'coord_mlp', got "
            f"{feature_type!r}"
        )

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> tuple[int, Optional[int], Optional[int]]:
        """Return (prefix_tokens, height, width) for square ViT patch layouts."""
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return 0, side, side

        side = int(math.isqrt(max(num_tokens - 1, 0)))
        if side * side == num_tokens - 1:
            return 1, side, side

        return 0, None, None

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

    def _coordinate_features_for_tokens(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        if height is not None and width is not None:
            coords = self._coordinate_features_for_grid(height, width, device, dtype)
            if prefix_tokens:
                prefix = torch.zeros(
                    1,
                    prefix_tokens,
                    5,
                    device=device,
                    dtype=dtype,
                )
                coords = torch.cat([prefix, coords], dim=1)
            return coords

        x = torch.linspace(-1.0, 1.0, num_tokens, device=device, dtype=dtype)
        y = torch.zeros_like(x)
        coords = torch.stack([x, y, x.square(), y.square(), x * y], dim=-1)
        return coords.unsqueeze(0)

    def _learned_patch_position_features(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.patch_position_embedding is None:
            raise RuntimeError(
                "Patch position features requested without a position table"
            )

        table = self.patch_position_embedding.to(device=device)
        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        if height is not None and width is not None:
            table_4d = table.permute(2, 0, 1).unsqueeze(0)
            if (
                height != self.patch_position_grid_size
                or width != self.patch_position_grid_size
            ):
                table_4d = F.interpolate(
                    table_4d,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
            positions = (
                table_4d.squeeze(0)
                .permute(1, 2, 0)
                .reshape(1, height * width, self.input_dim)
            )
            if prefix_tokens:
                prefix = torch.zeros(
                    1,
                    prefix_tokens,
                    self.input_dim,
                    device=device,
                    dtype=positions.dtype,
                )
                positions = torch.cat([prefix, positions], dim=1)
        else:
            flat = table.reshape(-1, self.input_dim).transpose(0, 1).unsqueeze(0)
            flat = F.interpolate(
                flat,
                size=num_tokens,
                mode="linear",
                align_corners=True,
            )
            positions = flat.squeeze(0).transpose(0, 1).unsqueeze(0)

        return positions.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _coord_mlp_patch_position_features(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.patch_position_mlp is None:
            raise RuntimeError(
                "Coordinate patch position features requested without a position MLP"
            )
        coords = self._coordinate_features_for_tokens(
            num_tokens=num_tokens,
            device=device,
            dtype=dtype,
        )
        mlp_dtype = self.patch_position_mlp[0].weight.dtype
        features = self.patch_position_mlp(coords.to(dtype=mlp_dtype))
        return features.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _add_patch_position_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_position_feature_type == "none":
            return x
        kwargs = {
            "batch_size": x.shape[0],
            "num_tokens": x.shape[1],
            "device": x.device,
            "dtype": x.dtype,
        }
        if self.patch_position_feature_type == "learned_table":
            features = self._learned_patch_position_features(**kwargs)
            return x + features * self.patch_position_feature_scale
        if self.patch_position_feature_type == "coord_mlp":
            features = self._coord_mlp_patch_position_features(**kwargs)
            return x + features * self.patch_position_feature_scale
        raise RuntimeError(
            f"Unsupported patch_position_feature_type={self.patch_position_feature_type!r}"
        )

    def _apply_query_patch_selector(
        self,
        patches: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        question_summary: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.query_patch_selector is None:
            return patches
        return self.query_patch_selector(
            patches,
            question_summary=question_summary,
            attention_mask=attention_mask,
        )

    def _apply_query_visual_scale(
        self,
        image_tokens: torch.Tensor,
        question_summary: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._last_query_visual_scale = None
        if self.query_visual_scale is None:
            return image_tokens

        scale = self.query_visual_scale(
            question_summary,
            batch_size=image_tokens.shape[0],
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )
        self._last_query_visual_scale = scale.detach()
        return image_tokens * (scale / self.output_scale)

    def get_query_visual_scale_diagnostics(self) -> dict[str, float]:
        scale = self._last_query_visual_scale
        if scale is None:
            return {}
        scale = scale.detach().float()
        return {
            "train/query_visual_scale_mean": float(scale.mean().item()),
            "train/query_visual_scale_min": float(scale.min().item()),
            "train/query_visual_scale_max": float(scale.max().item()),
            "train/effective_connector_output_scale_mean": float(scale.mean().item()),
            "train/effective_connector_output_scale_min": float(scale.min().item()),
            "train/effective_connector_output_scale_max": float(scale.max().item()),
        }

    def get_query_patch_selector_diagnostics(self) -> dict[str, float]:
        if self.query_patch_selector is None:
            return {}
        return self.query_patch_selector.get_diagnostics()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        question_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project vision features to LLM token space.

        Args:
            x: Vision features [B, num_patches, input_dim]
            attention_mask: Optional mask for vision features [B, num_patches]

        Returns:
            Image tokens [B, num_latents, output_dim]

        Process:
        1. Project vision features to output dimension
        2. Expand latent queries for batch
        3. Pass through resampler layers (cross-attn + self-attn + FFN)
        4. Apply final layer norm
        """
        batch_size = x.shape[0]

        x = self._apply_query_patch_selector(x, attention_mask, question_summary)

        # Project input features to output dimension
        # [B, num_patches, input_dim] -> [B, num_patches, output_dim]
        context = self.input_proj(self._add_patch_position_features(x))

        # Expand learnable latents for the batch
        # [num_latents, output_dim] -> [B, num_latents, output_dim]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through resampler layers
        for layer in self.layers:
            latents = layer(latents, context, attention_mask)

        # Final normalization
        image_tokens = self.norm(latents) * self._output_multiplier()
        return self._apply_query_visual_scale(image_tokens, question_summary)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QuestionConditionedPerceiverResampler(PerceiverResampler):
    """
    Perceiver Resampler with additive question-conditioned latent shifts.

    The base image-token contract stays identical to ``PerceiverResampler``.
    When a pooled question/context embedding is provided, a small conditioning
    MLP produces a bounded shift that is added to the learned latent queries
    before the normal image cross-attention stack.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.0,
        output_scale: float = 1.0,
        output_gate_init: Optional[float] = None,
        trainable_scale_mode: str = "none",
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
        condition_dim: Optional[int] = None,
        condition_scale_init: float = 0.02,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_latents=num_latents,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            output_scale=output_scale,
            output_gate_init=output_gate_init,
            trainable_scale_mode=trainable_scale_mode,
            use_2d_patch_position_features=use_2d_patch_position_features,
            patch_position_feature_type=patch_position_feature_type,
            patch_position_grid_size=patch_position_grid_size,
            patch_position_mlp_hidden_dim=patch_position_mlp_hidden_dim,
            patch_position_feature_scale=patch_position_feature_scale,
            query_conditioned_visual_scale_mode=query_conditioned_visual_scale_mode,
            query_conditioned_visual_scale_min=query_conditioned_visual_scale_min,
            query_conditioned_visual_scale_max=query_conditioned_visual_scale_max,
            query_conditioned_visual_scale_init=query_conditioned_visual_scale_init,
            query_conditioned_patch_selector_mode=query_conditioned_patch_selector_mode,
            query_conditioned_patch_selector_hidden_dim=query_conditioned_patch_selector_hidden_dim,
            query_conditioned_patch_selector_max_residual=query_conditioned_patch_selector_max_residual,
            query_conditioned_patch_selector_normalize_mean=query_conditioned_patch_selector_normalize_mean,
        )
        self.condition_dim = int(condition_dim or output_dim)
        self.condition_norm = nn.LayerNorm(self.condition_dim)
        self.condition_proj = (
            nn.Identity()
            if self.condition_dim == output_dim
            else nn.Linear(self.condition_dim, output_dim)
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.latent_condition_gates = nn.Parameter(torch.ones(num_latents, output_dim))
        self.condition_scale = nn.Parameter(torch.tensor(float(condition_scale_init)))

    def _condition_latents(
        self,
        latents: torch.Tensor,
        question_summary: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if question_summary is None:
            return latents

        question_summary = question_summary.to(device=latents.device, dtype=latents.dtype)
        summary = self.condition_norm(question_summary)
        summary = self.condition_proj(summary)
        shift = torch.tanh(self.condition_mlp(summary))
        shift = shift.unsqueeze(1) * self.latent_condition_gates.unsqueeze(0)
        return latents + (self.condition_scale * shift)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        question_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project vision features to LLM token space.

        Args:
            x: Vision features [B, num_patches, input_dim]
            attention_mask: Optional mask for vision features [B, num_patches]
            question_summary: Optional pooled prompt/context embedding [B, H]
        """
        batch_size = x.shape[0]
        x = self._apply_query_patch_selector(x, attention_mask, question_summary)
        context = self.input_proj(self._add_patch_position_features(x))
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents = self._condition_latents(latents, question_summary)

        for layer in self.layers:
            latents = layer(latents, context, attention_mask)

        image_tokens = self.norm(latents) * self._output_multiplier()
        return self._apply_query_visual_scale(image_tokens, question_summary)


class QueryConditionedVisualScale(nn.Module):
    """Bounded query-conditioned absolute visual scale for D1/D2."""

    def __init__(
        self,
        condition_dim: int,
        num_latents: int,
        mode: str,
        min_scale: float,
        max_scale: float,
        init_scale: float,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"scalar", "per_token"}:
            raise ValueError(f"Unsupported query visual scale mode: {mode!r}")
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.init_scale = float(init_scale)
        if not self.min_scale < self.max_scale:
            raise ValueError(
                "query visual scale bounds must satisfy min < max, got "
                f"{self.min_scale} >= {self.max_scale}"
            )
        if not self.min_scale < self.init_scale < self.max_scale:
            raise ValueError(
                "query visual scale init must be strictly inside bounds, got "
                f"init={self.init_scale}, bounds=({self.min_scale}, {self.max_scale})"
            )

        output_dim = 1 if self.mode == "scalar" else int(num_latents)
        self.norm = nn.LayerNorm(condition_dim)
        self.proj = nn.Linear(condition_dim, output_dim)
        nn.init.zeros_(self.proj.weight)
        ratio = (self.init_scale - self.min_scale) / (self.max_scale - self.min_scale)
        init_logit = torch.logit(torch.tensor(ratio, dtype=torch.float32))
        nn.init.constant_(self.proj.bias, float(init_logit.item()))

    def forward(
        self,
        question_summary: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if question_summary is None:
            shape = (batch_size, 1, 1) if self.mode == "scalar" else (batch_size, self.proj.out_features, 1)
            return torch.full(
                shape,
                self.init_scale,
                device=device,
                dtype=dtype,
            )

        summary = question_summary.to(
            device=device,
            dtype=self.norm.weight.dtype,
        )
        logits = self.proj(self.norm(summary))
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(logits)
        if self.mode == "scalar":
            scale = scale.view(scale.shape[0], 1, 1)
        else:
            scale = scale.unsqueeze(-1)
        return scale.to(device=device, dtype=dtype)


class QueryConditionedPatchSelector(nn.Module):
    """Neutral bounded query-conditioned residual patch weighting for D3."""

    def __init__(
        self,
        condition_dim: int,
        input_dim: int,
        hidden_dim: int,
        max_residual: float,
        normalize_mean: bool = True,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)
        self.normalize_mean = bool(normalize_mean)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if not 0.0 < self.max_residual < 1.0:
            raise ValueError(
                f"max_residual must be between 0 and 1, got {max_residual}"
            )

        self.patch_norm = nn.LayerNorm(self.input_dim)
        self.condition_norm = nn.LayerNorm(self.condition_dim)
        self.condition_proj = nn.Linear(self.condition_dim, self.input_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)
        self._last_patch_weights = None

    def forward(
        self,
        patches: torch.Tensor,
        question_summary: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._last_patch_weights = None
        if question_summary is None:
            return patches

        module_dtype = self.patch_norm.weight.dtype
        patch_features = self.patch_norm(patches.to(dtype=module_dtype))
        summary = question_summary.to(device=patches.device, dtype=module_dtype)
        condition = self.condition_proj(self.condition_norm(summary)).unsqueeze(1)
        logits = self.score_mlp(torch.tanh(patch_features + condition)).squeeze(-1)
        weights = 1.0 + self.max_residual * torch.tanh(logits)

        if attention_mask is not None:
            valid = attention_mask.to(device=patches.device, dtype=torch.bool)
            weights = torch.where(valid, weights, torch.ones_like(weights))
        else:
            valid = None

        if self.normalize_mean:
            if valid is None:
                mean = weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                valid_f = valid.to(dtype=weights.dtype)
                denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean = (weights * valid_f).sum(dim=1, keepdim=True) / denom
                mean = mean.clamp_min(1e-6)
            normalized = weights / mean
            if valid is not None:
                weights = torch.where(valid, normalized, torch.ones_like(weights))
            else:
                weights = normalized

        weights = weights.to(device=patches.device, dtype=patches.dtype).unsqueeze(-1)
        self._last_patch_weights = weights.detach()
        return patches * weights

    def get_diagnostics(self) -> dict[str, float]:
        weights = self._last_patch_weights
        if weights is None:
            return {}
        weights = weights.detach().float()
        return {
            "train/query_patch_weight_mean": float(weights.mean().item()),
            "train/query_patch_weight_min": float(weights.min().item()),
            "train/query_patch_weight_max": float(weights.max().item()),
            "train/query_patch_weight_std": float(weights.std().item()),
        }


class PerceiverResamplerBlock(nn.Module):
    """
    Single Perceiver Resampler block.

    Architecture:
    1. Cross-attention: latents attend to vision context
       - Query: latent tokens
       - Key/Value: vision features
       - Allows latents to "look at" the image

    2. Self-attention: latents attend to each other
       - Query/Key/Value: latent tokens
       - Allows latents to share and combine information

    3. Feed-forward network: transform features
       - Two linear layers with GELU activation
       - Expands dimension by ff_mult, then projects back

    All sub-layers use pre-norm (LayerNorm before attention/FFN)
    and residual connections for stable training.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Cross-attention: latents query vision features
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention: latents attend to each other
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through one resampler block.

        Args:
            latents: Latent queries [B, num_latents, dim]
            context: Vision features [B, num_patches, dim]
            context_mask: Optional mask for context [B, num_patches]

        Returns:
            Updated latents [B, num_latents, dim]
        """
        # Cross-attention: latents attend to vision features
        # Pre-norm on both latents and context for stable training
        normed_latents = self.cross_attn_norm(latents)
        normed_context = self.context_norm(context)

        # Convert boolean mask to attention mask if provided
        # MultiheadAttention expects: True = ignore, False = attend
        attn_mask = None
        if context_mask is not None:
            # context_mask: True = valid, False = padding
            # Convert to: True = ignore (for padding positions)
            attn_mask = ~context_mask

        cross_attn_out, _ = self.cross_attn(
            query=normed_latents,
            key=normed_context,
            value=normed_context,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        latents = latents + cross_attn_out

        # Self-attention: latents attend to each other
        normed_latents = self.self_attn_norm(latents)
        self_attn_out, _ = self.self_attn(
            query=normed_latents,
            key=normed_latents,
            value=normed_latents,
            need_weights=False,
        )
        latents = latents + self_attn_out

        # Feed-forward network
        latents = latents + self.ff(self.ff_norm(latents))

        return latents


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) for transformer blocks.

    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout

    The hidden dimension is expanded by `mult` factor, allowing
    the network to learn complex transformations.
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = dim * mult

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_perceiver_resampler(
    input_dim: int = 1024,
    output_dim: int = 4096,
    num_latents: int = 64,
    num_layers: int = 6,
    **kwargs,
) -> PerceiverResampler:
    """
    Factory function to create Perceiver Resampler.

    Default configuration matches the paper:
    - 64 latent tokens
    - 6 layers
    - Projects from CLIP ViT-L (1024) to LLaMA 3 8B (4096)

    Args:
        input_dim: Vision encoder output dimension
        output_dim: LLM embedding dimension
        num_latents: Number of output image tokens
        num_layers: Depth of resampler
        **kwargs: Additional arguments

    Returns:
        Configured PerceiverResampler instance
    """
    return PerceiverResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        num_latents=num_latents,
        num_layers=num_layers,
        **kwargs,
    )
