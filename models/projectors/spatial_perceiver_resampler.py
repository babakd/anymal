"""
Spatial global/local Perceiver connector for AnyMAL v4.

The connector keeps the compact visual-token contract that made V3 viable, but
splits the learned queries into global summary tokens and position-aware local
tokens. The output is still one contiguous image-token block for the LLM.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .perceiver_resampler import PerceiverResamplerBlock


class SpatialPerceiverResampler(nn.Module):
    """
    Perceiver resampler with separate global and local/spatial latent queries.

    Args:
        input_dim: Dimension of input vision features.
        output_dim: Dimension of output LLM embeddings.
        connector_dim: Internal Perceiver width. Defaults to output_dim for the
            original direct-to-LLM connector; set lower for bottlenecked V4
            ablations.
        output_scale: Constant multiplier applied after the final projection.
            Values below 1.0 keep random visual tokens closer to the frozen LLM's
            text manifold and scale connector gradients during early alignment.
        num_global_latents: Image-level summary tokens.
        num_local_latents: Position-aware local tokens.
        num_layers: Number of cross/self-attention blocks per branch.
        num_heads: Attention heads in each branch.
        ff_mult: Feed-forward expansion multiplier.
        use_2d_position_features: Add learned 2D features to patch context and
            initialize local queries with a regular spatial grid.
        output_gate_init: Optional trainable sigmoid gate initialization. Values
            near zero create a gentle handoff into the frozen LLM while still
            allowing the gate to grow during alignment.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        connector_dim: Optional[int] = None,
        num_global_latents: int = 64,
        num_local_latents: int = 64,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_2d_position_features: bool = True,
        output_scale: float = 1.0,
        output_gate_init: Optional[float] = None,
    ):
        super().__init__()
        if num_global_latents < 0 or num_local_latents < 0:
            raise ValueError("num_global_latents and num_local_latents must be >= 0")
        if num_global_latents + num_local_latents <= 0:
            raise ValueError("At least one visual latent is required")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.connector_dim = int(connector_dim or output_dim)
        self.num_global_latents = num_global_latents
        self.num_local_latents = num_local_latents
        self.num_latents = num_global_latents + num_local_latents
        self.num_layers = num_layers
        self.use_2d_position_features = use_2d_position_features
        self.output_scale = float(output_scale)
        if self.output_scale <= 0:
            raise ValueError(f"output_scale must be > 0, got {output_scale}")
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

        self.input_proj = nn.Linear(input_dim, self.connector_dim)
        self.global_latents = nn.Parameter(
            torch.randn(num_global_latents, self.connector_dim) * 0.02
        )
        self.local_latents = nn.Parameter(
            torch.randn(num_local_latents, self.connector_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    dim=self.connector_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        if use_2d_position_features:
            self.position_mlp = nn.Sequential(
                nn.Linear(2, self.connector_dim),
                nn.GELU(),
                nn.Linear(self.connector_dim, self.connector_dim),
            )
            query_coords = self._make_query_grid(num_local_latents)
            self.register_buffer("local_query_coords", query_coords, persistent=False)
        else:
            self.position_mlp = None
            self.register_buffer(
                "local_query_coords",
                torch.empty(num_local_latents, 2),
                persistent=False,
            )

        self.type_embeddings = nn.Parameter(torch.randn(2, self.connector_dim) * 0.02)
        self.norm = nn.LayerNorm(self.connector_dim)
        self.output_proj = (
            nn.Identity()
            if self.connector_dim == output_dim
            else nn.Linear(self.connector_dim, output_dim)
        )

    def _output_multiplier(self) -> torch.Tensor:
        multiplier = self.output_scale
        if self.output_gate_logit is not None:
            multiplier = multiplier * torch.sigmoid(self.output_gate_logit)
        return multiplier

    @staticmethod
    def _make_query_grid(num_tokens: int) -> torch.Tensor:
        if num_tokens <= 0:
            return torch.empty(0, 2)

        height = int(math.floor(math.sqrt(num_tokens)))
        width = int(math.ceil(num_tokens / max(height, 1)))
        coords = []
        for idx in range(num_tokens):
            row = idx // width
            col = idx % width
            y = 0.0 if height <= 1 else (row / (height - 1)) * 2.0 - 1.0
            x = 0.0 if width <= 1 else (col / (width - 1)) * 2.0 - 1.0
            coords.append((x, y))
        return torch.tensor(coords, dtype=torch.float32)

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> Tuple[int, Optional[int], Optional[int]]:
        """Return (prefix_tokens, height, width) for square ViT patch layouts."""
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return 0, side, side

        side = int(math.isqrt(max(num_tokens - 1, 0)))
        if side * side == num_tokens - 1:
            return 1, side, side

        return 0, None, None

    def _position_features(self, batch_size: int, num_tokens: int, device, dtype) -> torch.Tensor:
        if self.position_mlp is None:
            raise RuntimeError("Position features requested without a position_mlp")

        prefix_tokens, height, width = self._infer_square_grid(num_tokens)
        if height is None or width is None:
            # Fallback for non-square/unknown token layouts. This preserves token
            # order without inventing a false square grid.
            x = torch.linspace(-1.0, 1.0, num_tokens, device=device, dtype=dtype)
            coords = torch.stack([x, torch.zeros_like(x)], dim=-1)
        else:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            if prefix_tokens:
                prefix = torch.zeros(prefix_tokens, 2, device=device, dtype=dtype)
                coords = torch.cat([prefix, coords], dim=0)

        pos = self.position_mlp(coords.to(dtype=next(self.position_mlp.parameters()).dtype))
        pos = pos.to(device=device, dtype=dtype)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        context = self.input_proj(x)

        if self.use_2d_position_features:
            spatial_context = context + self._position_features(
                batch_size=batch_size,
                num_tokens=num_tokens,
                device=context.device,
                dtype=context.dtype,
            )
            local_query_pos = self.position_mlp(
                self.local_query_coords.to(
                    device=context.device,
                    dtype=next(self.position_mlp.parameters()).dtype,
                )
            ).to(dtype=context.dtype)
        else:
            spatial_context = context
            local_query_pos = torch.zeros_like(self.local_latents)

        latents = []
        if self.num_global_latents:
            global_latents = self.global_latents.unsqueeze(0).expand(batch_size, -1, -1)
            global_latents = global_latents + self.type_embeddings[0].to(dtype=context.dtype)
            latents.append(global_latents)

        if self.num_local_latents:
            local_latents = self.local_latents + local_query_pos
            local_latents = local_latents.unsqueeze(0).expand(batch_size, -1, -1)
            local_latents = local_latents + self.type_embeddings[1].to(dtype=context.dtype)
            latents.append(local_latents)

        latents = torch.cat(latents, dim=1)
        for layer in self.layers:
            latents = layer(latents, spatial_context, attention_mask)

        latents = self.norm(latents)
        return self.output_proj(latents) * self._output_multiplier()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
