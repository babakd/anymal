"""Small gated visual cross-attention adapters for decoder layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedVisualCrossAttentionAdapter(nn.Module):
    """Cross-attend decoder hidden states to fixed visual memory with a zero gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        adapter_dim: int = None,
        gate_init: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        adapter_dim = int(adapter_dim or hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be > 0, got {adapter_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if adapter_dim % num_heads != 0:
            raise ValueError(
                f"adapter_dim ({adapter_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads
        self.dropout = float(dropout)

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, adapter_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, adapter_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, adapter_dim, bias=False)
        self.o_proj = nn.Linear(adapter_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _hidden = value.shape
        return value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_memory: torch.Tensor,
    ) -> torch.Tensor:
        if visual_memory is None:
            return hidden_states
        if hidden_states.ndim != 3 or visual_memory.ndim != 3:
            raise ValueError(
                "GatedVisualCrossAttentionAdapter expects hidden_states and "
                f"visual_memory with shape [B, T, H], got "
                f"{tuple(hidden_states.shape)} and {tuple(visual_memory.shape)}"
            )
        if hidden_states.shape[0] != visual_memory.shape[0]:
            raise ValueError(
                "hidden_states and visual_memory batch sizes must match: "
                f"{hidden_states.shape[0]} != {visual_memory.shape[0]}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        adapter_dtype = self.hidden_norm.weight.dtype
        adapter_hidden = hidden_states.to(dtype=adapter_dtype)
        visual_memory = visual_memory.to(
            device=hidden_states.device,
            dtype=adapter_dtype,
        )
        query = self._split_heads(self.q_proj(self.hidden_norm(adapter_hidden)))
        key = self._split_heads(self.k_proj(self.visual_norm(visual_memory)))
        value = self._split_heads(self.v_proj(self.visual_norm(visual_memory)))
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attn = attn.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.adapter_dim,
        )
        residual = self.o_proj(attn).to(dtype=hidden_states.dtype)
        return hidden_states + self.gate.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ) * residual

    def gate_value(self) -> float:
        return float(self.gate.detach().float().item())
