"""Small chat-template helpers shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)

DIRECT_ANSWER_SYSTEM_PROMPT = (
    "Answer with only the final answer. Do not include role labels, "
    "explanations, or the word assistant. End after the answer."
)

IMAGE_SENTINEL = "<|image_sentinel|>"


@dataclass(frozen=True)
class ChatTemplateSpec:
    family: str
    system_header: str
    user_header: str
    assistant_header: str
    end_turn: str
    assistant_prefill: str = ""


LLAMA3_TEMPLATE = ChatTemplateSpec(
    family="llama3",
    system_header="<|start_header_id|>system<|end_header_id|>\n\n",
    user_header="<|start_header_id|>user<|end_header_id|>\n\n",
    assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    end_turn="<|eot_id|>",
)

QWEN3_NON_THINKING_TEMPLATE = ChatTemplateSpec(
    family="qwen3_non_thinking",
    system_header="<|im_start|>system\n",
    user_header="<|im_start|>user\n",
    assistant_header="<|im_start|>assistant\n",
    end_turn="<|im_end|>\n",
    assistant_prefill="<think>\n\n</think>\n\n",
)


def get_chat_template_spec(family: Optional[str]) -> ChatTemplateSpec:
    family = str(family or "llama3").strip().lower()
    if family == "qwen3":
        family = "qwen3_non_thinking"
    if family == "qwen3_non_thinking":
        return QWEN3_NON_THINKING_TEMPLATE
    if family == "llama3":
        return LLAMA3_TEMPLATE
    raise ValueError(f"Unsupported chat_template_family: {family!r}")


def resolve_chat_template_spec(
    tokenizer,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    family: Optional[str] = None,
) -> ChatTemplateSpec:
    if family is None:
        model_type_s = str(model_type or "").lower()
        model_name_s = str(model_name or getattr(tokenizer, "name_or_path", "") or "").lower()
        family = "qwen3_non_thinking" if model_type_s == "qwen3" or "qwen3" in model_name_s else "llama3"
    return get_chat_template_spec(family)


def build_generation_prompt_text(
    tokenizer,
    question: str,
    system_prompt: Optional[str] = None,
    image_sentinel: Optional[str] = IMAGE_SENTINEL,
    chat_template_family: Optional[str] = None,
) -> str:
    """Build the single-turn prompt used by VQA-style evals."""
    spec = resolve_chat_template_spec(
        tokenizer,
        model_name=getattr(tokenizer, "name_or_path", None),
        family=chat_template_family,
    )
    user_text = str(question or "").strip()
    if image_sentinel:
        user_text = f"{image_sentinel}\n{user_text}"
    return (
        f"{spec.system_header}{system_prompt or TRAINING_SYSTEM_PROMPT}{spec.end_turn}"
        f"{spec.user_header}{user_text}{spec.end_turn}"
        f"{spec.assistant_header}{spec.assistant_prefill}"
    )
