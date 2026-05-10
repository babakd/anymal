"""Backbone metadata helpers for text decoders used by AnyMAL."""

from __future__ import annotations

import os
from typing import Any, Optional


CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
QWEN3_8B_BACKBONE = "Qwen/Qwen3-8B"

SUPPORTED_V8_BACKBONES = {
    CURRENT_LLAMA3_BACKBONE,
    QWEN3_8B_BACKBONE,
}

MODAL_BACKBONE_CACHE_DIRS = {
    CURRENT_LLAMA3_BACKBONE: "/checkpoints/llama3-8b-instruct",
    QWEN3_8B_BACKBONE: "/checkpoints/qwen3-8b",
}


def _as_str(value: Optional[str]) -> str:
    return str(value or "").strip()


def canonicalize_llm_backbone(value: Optional[str]) -> str:
    """Return the canonical backbone ID when the value is a known alias/path."""
    raw = _as_str(value) or CURRENT_LLAMA3_BACKBONE
    lowered = raw.lower()
    base = os.path.basename(raw.rstrip("/")).lower()

    if "llama-3.1" in lowered or "llama3.1" in lowered:
        raise ValueError(
            "Llama 3.1 is intentionally excluded from the V8 option space. "
            "Use the current LLaMA-3-8B incumbent path or Qwen/Qwen3-8B."
        )
    if lowered in {
        CURRENT_LLAMA3_BACKBONE.lower(),
        "meta-llama/meta-llama-3-8b-instruct",
        "llama3",
        "llama-3",
        "llama3-8b-instruct",
        "meta-llama-3-8b-instruct",
    } or base in {"llama3-8b-instruct", "meta-llama-3-8b-instruct"}:
        return CURRENT_LLAMA3_BACKBONE
    if lowered in {
        QWEN3_8B_BACKBONE.lower(),
        "qwen3",
        "qwen3-8b",
        "qwen/qwen3-8b",
    } or base in {"qwen3-8b", "qwen3_8b"}:
        return QWEN3_8B_BACKBONE

    return raw


def is_qwen3_backbone(value: Optional[str], model_type: Optional[str] = None) -> bool:
    canonical = canonicalize_llm_backbone(value)
    return str(model_type or "").lower() == "qwen3" or canonical == QWEN3_8B_BACKBONE


def is_current_llama_backbone(value: Optional[str], model_type: Optional[str] = None) -> bool:
    canonical = canonicalize_llm_backbone(value)
    model_type = str(model_type or "").lower()
    return canonical == CURRENT_LLAMA3_BACKBONE or model_type in {"llama", "llama3"}


def infer_chat_template_family(
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    tokenizer: Any = None,
) -> str:
    """Infer the chat-template family from model config/name."""
    if model_type is None and tokenizer is not None:
        model_type = getattr(getattr(tokenizer, "init_kwargs", {}), "get", lambda _k, _d=None: None)(
            "model_type",
            None,
        )
    model_type_s = str(model_type or "").lower()
    name_s = str(model_name or "").lower()
    if model_type_s == "qwen3" or "qwen3" in name_s:
        return "qwen3_non_thinking"
    return "llama3"


def modal_cache_dir_for_backbone(backbone: Optional[str]) -> Optional[str]:
    canonical = canonicalize_llm_backbone(backbone)
    return MODAL_BACKBONE_CACHE_DIRS.get(canonical)


def assert_supported_v8_backbone(backbone: Optional[str]) -> str:
    canonical = canonicalize_llm_backbone(backbone)
    if os.path.isabs(canonical):
        return canonical
    if canonical not in SUPPORTED_V8_BACKBONES:
        raise ValueError(
            f"Unsupported V8 llm_backbone={backbone!r}. "
            f"Supported values are {sorted(SUPPORTED_V8_BACKBONES)}. "
            "Llama 3.1 is intentionally excluded."
        )
    return canonical
