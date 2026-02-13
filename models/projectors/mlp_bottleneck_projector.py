"""
MLP bottleneck projector for AnyMALv2.
"""

import torch
import torch.nn as nn


class MLPBottleneckProjector(nn.Module):
    """
    Lightweight projector: input_dim -> bottleneck_dim -> output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
