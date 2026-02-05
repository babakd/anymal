"""
AnyMAL Evaluation Module

Provides evaluation utilities for:
- VQA benchmarks (VQAv2, TextVQA, OKVQA, ScienceQA)
- Captioning benchmarks (COCO)
"""

from .vqa_eval import VQAEvaluator
from .captioning_eval import CaptioningEvaluator

__all__ = ["VQAEvaluator", "CaptioningEvaluator"]
