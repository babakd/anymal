"""
DeepStack-lite spatial Perceiver connector for AnyMAL v4.

This ablation keeps the same fixed visual-token output contract as the default
V4 connector, but exposes multiple SigLIP2 hidden-state levels to the Perceiver
context. Each level receives a learned embedding before the shared global/local
query tower attends over the stacked patch tokens.
"""

from typing import Optional, Sequence, Union

import torch

from .spatial_perceiver_resampler import SpatialPerceiverResampler


class DeepStackSpatialPerceiverResampler(SpatialPerceiverResampler):
    """
    Spatial Perceiver over a small stack of vision hidden-state levels.

    Args mirror :class:`SpatialPerceiverResampler`, with `num_feature_levels`
    controlling how many selected vision levels the connector expects.
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
        num_feature_levels: int = 3,
    ):
        if int(num_feature_levels) <= 0:
            raise ValueError(f"num_feature_levels must be > 0, got {num_feature_levels}")
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            connector_dim=connector_dim,
            num_global_latents=num_global_latents,
            num_local_latents=num_local_latents,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            use_2d_position_features=use_2d_position_features,
            output_scale=output_scale,
            output_gate_init=output_gate_init,
        )
        self.num_feature_levels = int(num_feature_levels)
        self.level_embeddings = torch.nn.Parameter(
            torch.randn(self.num_feature_levels, self.connector_dim) * 0.02
        )

    def _stack_context(
        self,
        features: Sequence[torch.Tensor],
        attention_mask: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ):
        contexts = []
        masks = []
        for level_idx, x in enumerate(features):
            batch_size, num_tokens, _ = x.shape
            context = self.input_proj(x)
            context = context + self.level_embeddings[level_idx].to(
                device=context.device,
                dtype=context.dtype,
            )
            if self.use_2d_position_features:
                context = context + self._position_features(
                    batch_size=batch_size,
                    num_tokens=num_tokens,
                    device=context.device,
                    dtype=context.dtype,
                )
            contexts.append(context)

            if isinstance(attention_mask, (list, tuple)):
                masks.append(attention_mask[level_idx])

        stacked_context = torch.cat(contexts, dim=1)
        if isinstance(attention_mask, (list, tuple)):
            stacked_mask = torch.cat(masks, dim=1)
        else:
            stacked_mask = attention_mask
        return stacked_context, stacked_mask

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        attention_mask: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            features = (x,)
        else:
            features = tuple(x)
        if not features:
            raise ValueError("DeepStack connector requires at least one feature level")
        if len(features) != self.num_feature_levels:
            raise ValueError(
                "DeepStack connector received "
                f"{len(features)} feature levels, expected {self.num_feature_levels}."
            )

        batch_size = features[0].shape[0]
        dtype = features[0].dtype
        device = features[0].device
        for idx, feature in enumerate(features):
            if feature.ndim != 3:
                raise ValueError(f"Feature level {idx} must be [B, T, C], got {feature.shape}")
            if feature.shape[0] != batch_size:
                raise ValueError("All DeepStack feature levels must share the same batch size")

        spatial_context, stacked_mask = self._stack_context(features, attention_mask)

        if self.use_2d_position_features:
            local_query_pos = self.position_mlp(
                self.local_query_coords.to(
                    device=device,
                    dtype=next(self.position_mlp.parameters()).dtype,
                )
            ).to(dtype=dtype)
        else:
            local_query_pos = torch.zeros_like(self.local_latents)

        latents = []
        if self.num_global_latents:
            global_latents = self.global_latents.unsqueeze(0).expand(batch_size, -1, -1)
            global_latents = global_latents + self.type_embeddings[0].to(dtype=dtype)
            latents.append(global_latents)

        if self.num_local_latents:
            local_latents = self.local_latents + local_query_pos
            local_latents = local_latents.unsqueeze(0).expand(batch_size, -1, -1)
            local_latents = local_latents + self.type_embeddings[1].to(dtype=dtype)
            latents.append(local_latents)

        latents = torch.cat(latents, dim=1)
        for layer in self.layers:
            latents = layer(latents, spatial_context, stacked_mask)

        latents = self.norm(latents)
        return self.output_proj(latents) * self._output_multiplier()
