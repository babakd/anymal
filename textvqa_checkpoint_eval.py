"""Compatibility wrapper for the Modal TextVQA checkpoint evaluator."""

from evaluation.checkpoint_eval.textvqa_checkpoint_eval import app, evaluate_textvqa, main

__all__ = ["app", "evaluate_textvqa", "main"]
