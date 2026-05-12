"""Compatibility wrapper for the AnyMAL Modal training entrypoint."""

from scripts.modal.train import app, main

__all__ = ["app", "main"]
