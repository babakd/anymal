"""
LLM Wrappers

Provides wrappers for language models with multimodal input support.

Available:
- LlamaWrapper: decoder-only causal LM wrapper with QLoRA support
"""

from .llama_wrapper import LlamaWrapper
from .backbone import (
    CURRENT_LLAMA3_BACKBONE,
    QWEN3_8B_BACKBONE,
    canonicalize_llm_backbone,
)

__all__ = [
    "LlamaWrapper",
    "CURRENT_LLAMA3_BACKBONE",
    "QWEN3_8B_BACKBONE",
    "canonicalize_llm_backbone",
]
