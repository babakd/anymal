"""
Model factory for architecture selection.
"""

import inspect
from typing import Any, Dict, Optional

from .anymal import AnyMAL
from .anymal_v2 import AnyMALv2
from model_metadata import normalize_architecture_name


def _filter_kwargs_for_constructor(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unknown kwargs so shared config can include arch-specific fields."""
    signature = inspect.signature(cls.__init__)
    valid = set(signature.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid}


def create_model(
    architecture: str = "anymal_v1",
    **kwargs,
):
    """Create model from architecture name + kwargs."""
    architecture = normalize_architecture_name(architecture)
    if architecture == "anymal_v1":
        return AnyMAL(**_filter_kwargs_for_constructor(AnyMAL, kwargs))
    if architecture == "anymal_v2":
        return AnyMALv2(**_filter_kwargs_for_constructor(AnyMALv2, kwargs))
    raise ValueError(f"Unsupported architecture: {architecture}")


def create_model_from_config(
    config: Dict[str, Any],
    model_overrides: Optional[Dict[str, Any]] = None,
    **runtime_overrides,
):
    """
    Create model from config dict with optional overrides.

    Args:
        config: Top-level config dictionary.
        model_overrides: Dict merged into config["model"] before creation.
        **runtime_overrides: Final kwargs override (e.g. dtype/device map).
    """
    model_cfg = dict(config.get("model", {}))
    if model_overrides:
        model_cfg.update(model_overrides)
    model_cfg.update(runtime_overrides)

    architecture = model_cfg.pop("architecture", "anymal_v1")
    return create_model(architecture=architecture, **model_cfg)
