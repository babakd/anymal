"""
Checkpoint metadata helpers for architecture compatibility.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple


MODEL_META_FILENAME = "model_meta.json"
DEFAULT_LEGACY_ARCHITECTURE = "anymal_v1"
VALID_ARCHITECTURES = {"anymal_v1", "anymal_v2"}


def normalize_architecture_name(name: Optional[str]) -> str:
    """Normalize architecture aliases and validate."""
    if not name:
        return DEFAULT_LEGACY_ARCHITECTURE

    normalized = str(name).strip().lower()
    aliases = {
        "anymal": "anymal_v1",
        "v1": "anymal_v1",
        "anymal_v1": "anymal_v1",
        "anymalv1": "anymal_v1",
        "v2": "anymal_v2",
        "anymal_v2": "anymal_v2",
        "anymalv2": "anymal_v2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{name}'. "
            f"Expected one of: {sorted(VALID_ARCHITECTURES)}"
        )
    return normalized


def write_model_metadata(
    save_dir: str,
    architecture: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write checkpoint metadata and return payload."""
    payload = {
        "architecture": normalize_architecture_name(architecture),
    }
    if extra:
        payload.update(extra)

    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, MODEL_META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def read_model_metadata(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Read metadata if present, else return None."""
    meta_path = os.path.join(checkpoint_dir, MODEL_META_FILENAME)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_architecture(checkpoint_dir: str) -> Tuple[str, bool]:
    """
    Resolve checkpoint architecture.

    Returns:
        (architecture, has_metadata)
    """
    meta = read_model_metadata(checkpoint_dir)
    if meta is None:
        return DEFAULT_LEGACY_ARCHITECTURE, False
    return normalize_architecture_name(meta.get("architecture")), True


def validate_checkpoint_architecture(
    checkpoint_dir: str,
    expected_architecture: str,
) -> str:
    """
    Validate checkpoint architecture against model architecture.

    Legacy checkpoints without metadata are treated as v1-only.
    """
    expected = normalize_architecture_name(expected_architecture)
    found, has_metadata = resolve_checkpoint_architecture(checkpoint_dir)

    if found != expected:
        if not has_metadata and expected != DEFAULT_LEGACY_ARCHITECTURE:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_dir} has no {MODEL_META_FILENAME}; "
                "legacy checkpoints are treated as anymal_v1-only. "
                f"Refusing to load into {expected}."
            )
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {checkpoint_dir}: "
            f"checkpoint={found}, model={expected}."
        )
    return found
