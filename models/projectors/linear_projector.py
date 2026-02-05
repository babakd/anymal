"""
Linear Projector for AnyMAL

Simple linear projection baseline for comparison with Perceiver Resampler.

Educational Notes:
-----------------
This is a simpler alternative to the Perceiver Resampler. Instead of using
cross-attention to compress visual features, it:
1. Applies layer norm to vision features
2. Projects each patch token independently via a linear layer
3. Optionally pools to reduce sequence length

Comparison with Perceiver Resampler:
- Faster: No attention computation
- More memory: Keeps all 257 tokens (vs 64 for resampler)
- Less expressive: No learned compression or inter-token communication

When to use:
- Baseline experiments
- When memory is less constrained
- When you want to preserve maximum spatial information

The paper finds Perceiver Resampler works better, likely because:
1. Cross-attention learns task-relevant compression
2. Self-attention allows tokens to share information
3. Reduced sequence length enables larger batch sizes
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearProjector(nn.Module):
    """
    Simple linear projection from vision features to LLM embedding space.

    This is a baseline that projects each vision token independently,
    without the learned compression of the Perceiver Resampler.

    Args:
        input_dim: Dimension of input vision features (e.g., 1024 for ViT-L)
        output_dim: Dimension of output tokens (e.g., 4096 for LLaMA 3 8B)
        num_layers: Number of MLP layers (1 = linear, 2 = MLP)
        pool_type: Pooling to reduce sequence length
            - None: Keep all tokens
            - "avg": Average pool to num_output_tokens
            - "learned": Learn to select tokens (attention pooling)
        num_output_tokens: Number of output tokens if pooling (default: 64)

    Example:
        >>> projector = LinearProjector(
        ...     input_dim=1024,
        ...     output_dim=4096,
        ...     num_layers=2,
        ... )
        >>> vision_features = torch.randn(4, 257, 1024)
        >>> image_tokens = projector(vision_features)
        >>> print(image_tokens.shape)  # [4, 257, 4096]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        pool_type: Optional[str] = None,
        num_output_tokens: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_type = pool_type
        self.num_output_tokens = num_output_tokens

        # Build projection layers
        if num_layers == 1:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        elif num_layers == 2:
            # Two-layer MLP with GELU activation
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        # Optional pooling to reduce sequence length
        if pool_type == "learned":
            # Learned attention pooling
            self.pool_queries = nn.Parameter(
                torch.randn(num_output_tokens, output_dim) * 0.02
            )
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True,
            )
            self.pool_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project vision features to LLM token space.

        Args:
            x: Vision features [B, num_patches, input_dim]
            attention_mask: Optional mask for vision features [B, num_patches]

        Returns:
            Image tokens [B, num_tokens, output_dim]
            - num_tokens = num_patches if no pooling
            - num_tokens = num_output_tokens if pooling
        """
        batch_size, seq_len, _ = x.shape

        # Project to output dimension
        x = self.proj(x)  # [B, seq_len, output_dim]

        # Apply pooling if specified
        if self.pool_type == "avg":
            # Simple average pooling to reduce sequence length
            # Reshape: [B, seq_len, dim] -> [B, num_output, pool_size, dim]
            pool_size = seq_len // self.num_output_tokens
            x_reshaped = x[:, :self.num_output_tokens * pool_size]
            x_reshaped = x_reshaped.view(
                batch_size, self.num_output_tokens, pool_size, -1
            )
            x = x_reshaped.mean(dim=2)  # [B, num_output_tokens, dim]

        elif self.pool_type == "learned":
            # Attention pooling with learned queries
            queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
            x_normed = self.pool_norm(x)

            x, _ = self.pool_attn(
                query=queries,
                key=x_normed,
                value=x_normed,
                need_weights=False,
            )  # [B, num_output_tokens, dim]

        return x

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_linear_projector(
    input_dim: int = 1024,
    output_dim: int = 4096,
    num_layers: int = 2,
    pool_type: Optional[str] = None,
    **kwargs,
) -> LinearProjector:
    """
    Factory function to create Linear Projector.

    Args:
        input_dim: Vision encoder output dimension
        output_dim: LLM embedding dimension
        num_layers: Number of MLP layers
        pool_type: Pooling type (None, "avg", "learned")
        **kwargs: Additional arguments

    Returns:
        Configured LinearProjector instance
    """
    return LinearProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        pool_type=pool_type,
        **kwargs,
    )
