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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_layers = num_layers

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
            Image tokens [B, num_latents, output_dim]

        Process:
        1. Project vision features to output dimension
        2. Expand latent queries for batch
        3. Pass through resampler layers (cross-attn + self-attn + FFN)
        4. Apply final layer norm
        """
        batch_size = x.shape[0]

        # Project input features to output dimension
        # [B, num_patches, input_dim] -> [B, num_patches, output_dim]
        context = self.input_proj(x)

        # Expand learnable latents for the batch
        # [num_latents, output_dim] -> [B, num_latents, output_dim]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through resampler layers
        for layer in self.layers:
            latents = layer(latents, context, attention_mask)

        # Final normalization
        return self.norm(latents)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
