"""
LLM Wrappers

Provides wrappers for language models with multimodal input support.

Available:
- LlamaWrapper: LLaMA-3-8B with QLoRA support
"""

from .llama_wrapper import LlamaWrapper

__all__ = ["LlamaWrapper"]
