"""Compatibility wrapper for the Modal GQA checkpoint evaluator."""

from evaluation.checkpoint_eval.gqa_checkpoint_eval import app, evaluate_gqa, main

__all__ = ["app", "evaluate_gqa", "main"]

