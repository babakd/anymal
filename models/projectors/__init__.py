"""
Modality Projectors

Projects encoder outputs to LLM embedding space.

Available projectors:
- PerceiverResampler: Cross-attention based (from Flamingo)
- LinearProjector: Simple linear projection baseline
"""

from .perceiver_resampler import PerceiverResampler
from .linear_projector import LinearProjector

__all__ = ["PerceiverResampler", "LinearProjector"]
