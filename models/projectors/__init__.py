"""
Modality Projectors

Projects encoder outputs to LLM embedding space.

Available projectors:
- PerceiverResampler: Cross-attention based (from Flamingo)
- LinearProjector: Simple linear projection baseline
"""

from .perceiver_resampler import PerceiverResampler, QuestionConditionedPerceiverResampler
from .spatial_perceiver_resampler import SpatialPerceiverResampler
from .deepstack_spatial_perceiver_resampler import DeepStackSpatialPerceiverResampler
from .linear_projector import LinearProjector
from .mlp_bottleneck_projector import MLPBottleneckProjector
from .token_compressor import TokenCompressor

__all__ = [
    "PerceiverResampler",
    "QuestionConditionedPerceiverResampler",
    "SpatialPerceiverResampler",
    "DeepStackSpatialPerceiverResampler",
    "LinearProjector",
    "MLPBottleneckProjector",
    "TokenCompressor",
]
