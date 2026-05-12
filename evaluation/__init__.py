"""Evaluation utilities.

Imports are intentionally lazy so Modal entrypoint discovery can load lightweight
wrapper modules on machines that do not have torch installed.
"""

__all__ = ["VQAEvaluator", "CaptioningEvaluator"]


def __getattr__(name):
    if name == "VQAEvaluator":
        from .vqa_eval import VQAEvaluator

        return VQAEvaluator
    if name == "CaptioningEvaluator":
        from .captioning_eval import CaptioningEvaluator

        return CaptioningEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
