"""
Shared multimodal input helpers for placeholder-aware text construction.
"""

from typing import Dict, Optional

import torch

from .data_utils import TextProcessor


DEFAULT_MULTIMODAL_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)

IMAGE_SENTINEL = "<|image_sentinel|>"


def resolve_image_placeholder_token_id(tokenizer) -> Optional[int]:
    """Resolve the tokenizer ID used for image placeholders."""
    vocab = tokenizer.get_vocab()
    if "<|reserved_special_token_0|>" in vocab:
        return vocab["<|reserved_special_token_0|>"]
    if "<|image|>" in vocab:
        return vocab["<|image|>"]
    return None


def build_image_placeholder_block(
    image_placeholder_token_id: int,
    num_image_tokens: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a fixed-length placeholder token block."""
    return torch.full(
        (num_image_tokens,),
        image_placeholder_token_id,
        dtype=dtype,
        device=device,
    )


def build_multimodal_input_ids(
    tokenizer,
    before_text: str,
    after_text: str,
    image_placeholder_token_id: int,
    num_image_tokens: int,
) -> Dict[str, torch.Tensor]:
    """Build placeholder-aware input IDs from text fragments around the image slot."""
    before_ids = tokenizer(
        before_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    after_ids = tokenizer(
        after_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    placeholder_block = build_image_placeholder_block(
        image_placeholder_token_id=image_placeholder_token_id,
        num_image_tokens=num_image_tokens,
        dtype=before_ids.dtype,
    )
    input_ids = torch.cat([before_ids, placeholder_block, after_ids], dim=0)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def build_multimodal_chat_input(
    tokenizer,
    user_text: str,
    image_placeholder_token_id: int,
    num_image_tokens: int,
    system_prompt: str = DEFAULT_MULTIMODAL_SYSTEM_PROMPT,
) -> Dict[str, torch.Tensor]:
    """Build a single-turn LLaMA-3 chat prompt with an explicit image placeholder block."""
    before_image = (
        f"<|begin_of_text|>"
        f"{TextProcessor.SYSTEM_HEADER}"
        f"{system_prompt}"
        f"{TextProcessor.END_TURN}"
        f"{TextProcessor.USER_HEADER}"
    )
    after_image = (
        f"\n{user_text}"
        f"{TextProcessor.END_TURN}"
        f"{TextProcessor.ASSISTANT_HEADER}"
    )
    return build_multimodal_input_ids(
        tokenizer=tokenizer,
        before_text=before_image,
        after_text=after_image,
        image_placeholder_token_id=image_placeholder_token_id,
        num_image_tokens=num_image_tokens,
    )


def replace_sentinel_with_placeholder_block(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
    text: str,
    image_sentinel: str,
    image_placeholder_token_id: Optional[int],
    num_image_tokens: int,
    max_length: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Replace a tokenized sentinel span with a placeholder block."""
    if (
        not image_sentinel
        or image_placeholder_token_id is None
        or image_sentinel not in text
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }

    sentinel_start = text.find(image_sentinel)
    sentinel_end = sentinel_start + len(image_sentinel)

    sentinel_token_indices = []
    for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
        if token_end > sentinel_start and token_start < sentinel_end:
            sentinel_token_indices.append(i)

    if not sentinel_token_indices:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }

    first_idx = sentinel_token_indices[0]
    num_sentinel_tokens = len(sentinel_token_indices)

    before = input_ids[:first_idx]
    after = input_ids[first_idx + num_sentinel_tokens :]
    placeholder_block = build_image_placeholder_block(
        image_placeholder_token_id=image_placeholder_token_id,
        num_image_tokens=num_image_tokens,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([before, placeholder_block, after])

    mask_before = attention_mask[:first_idx]
    mask_after = attention_mask[first_idx + num_sentinel_tokens :]
    mask_placeholder = torch.ones(num_image_tokens, dtype=attention_mask.dtype)
    attention_mask = torch.cat([mask_before, mask_placeholder, mask_after])

    offsets_before = offset_mapping[:first_idx]
    offsets_after = offset_mapping[first_idx + num_sentinel_tokens :]
    offsets_placeholder = torch.zeros(num_image_tokens, 2, dtype=offset_mapping.dtype)
    offset_mapping = torch.cat([offsets_before, offsets_placeholder, offsets_after])

    if input_ids.shape[0] > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        offset_mapping = offset_mapping[:max_length]
    elif input_ids.shape[0] < max_length:
        pad_len = max_length - input_ids.shape[0]
        input_ids = torch.cat(
            [input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)]
        )
        attention_mask = torch.cat(
            [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)]
        )
        offset_mapping = torch.cat(
            [offset_mapping, torch.zeros(pad_len, 2, dtype=offset_mapping.dtype)]
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }
