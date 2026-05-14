"""
Modality Projectors

Projects encoder outputs to LLM embedding space.

Available projectors:
- PerceiverResampler: Cross-attention based (from Flamingo)
- LinearProjector: Simple linear projection baseline
"""

from .perceiver_resampler import (
    PerceiverResampler,
    QuestionConditionedPerceiverResampler,
    QueryConditionedPatchSelector,
)
from .spatial_perceiver_resampler import SpatialPerceiverResampler
from .deepstack_spatial_perceiver_resampler import DeepStackSpatialPerceiverResampler
from .linear_projector import LinearProjector
from .mlp_bottleneck_projector import MLPBottleneckProjector
from .anyres_mlp_projector import AnyResMLPProjector
from .spatial_grid_projector import SpatialGridProjector
from .token_compressor import TokenCompressor

__all__ = [
    "PerceiverResampler",
    "QuestionConditionedPerceiverResampler",
    "QueryConditionedPatchSelector",
    "SpatialPerceiverResampler",
    "DeepStackSpatialPerceiverResampler",
    "LinearProjector",
    "MLPBottleneckProjector",
    "AnyResMLPProjector",
    "SpatialGridProjector",
    "TokenCompressor",
]
