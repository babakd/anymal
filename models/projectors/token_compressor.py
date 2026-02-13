"""
Token compressor for AnyMALv2.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompressor(nn.Module):
    """
    Compress variable-length vision tokens to a bounded token budget.

    Supports:
    - learned: learned query cross-attention pooling (default)
    - avg: interpolation-based pooling
    """

    def __init__(
        self,
        input_dim: int,
        max_tokens: int = 256,
        compressor_type: str = "learned",
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        self.input_dim = input_dim
        self.max_tokens = max_tokens
        self.compressor_type = compressor_type

        if compressor_type == "learned":
            if input_dim % num_heads != 0:
                raise ValueError(
                    f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
                )
            self.pool_queries = nn.Parameter(torch.randn(max_tokens, input_dim) * 0.02)
            self.pool_norm = nn.LayerNorm(input_dim)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        elif compressor_type == "avg":
            self.pool_queries = None
            self.pool_norm = None
            self.pool_attn = None
        else:
            raise ValueError(
                f"Unknown compressor_type '{compressor_type}'. "
                "Expected one of ['learned', 'avg']."
            )

    def _normalize_target_counts(
        self,
        batch_size: int,
        device: torch.device,
        target_num_tokens: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        if target_num_tokens is None:
            counts = torch.full(
                (batch_size,),
                self.max_tokens,
                device=device,
                dtype=torch.long,
            )
        else:
            counts = target_num_tokens.to(device=device, dtype=torch.long).view(-1)
            if counts.numel() != batch_size:
                raise ValueError(
                    f"target_num_tokens size mismatch: got {counts.numel()}, expected {batch_size}"
                )
            counts = counts.clamp(min=1, max=self.max_tokens)
        return counts

    def forward(
        self,
        x: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Args:
            x: [B, in_tokens, input_dim]
            target_num_tokens: per-sample requested output token counts [B]
            attention_mask: optional valid-token mask for x [B, in_tokens]

        Returns:
            tokens: [B, max_tokens, input_dim]
            token_mask: [B, max_tokens] (True for valid tokens)
            token_counts: [B]
        """
        batch_size, _, _ = x.shape
        token_counts = self._normalize_target_counts(
            batch_size=batch_size,
            device=x.device,
            target_num_tokens=target_num_tokens,
        )

        if self.compressor_type == "learned":
            queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
            context = self.pool_norm(x)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            pooled, _ = self.pool_attn(
                query=queries,
                key=context,
                value=context,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            pooled = F.interpolate(
                x.transpose(1, 2),
                size=self.max_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        index = torch.arange(self.max_tokens, device=x.device).unsqueeze(0)
        token_mask = index < token_counts.unsqueeze(1)

        # Zero invalid pooled tokens to avoid accidental use.
        pooled = pooled * token_mask.unsqueeze(-1).to(dtype=pooled.dtype)
        return pooled, token_mask, token_counts
