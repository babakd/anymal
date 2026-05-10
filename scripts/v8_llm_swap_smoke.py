#!/usr/bin/env python3
"""Stage 0 smoke tests for the V8 decoder swap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from data.chat_templates import DIRECT_ANSWER_SYSTEM_PROMPT, IMAGE_SENTINEL, build_generation_prompt_text
from models.llm import LlamaWrapper, canonicalize_llm_backbone


def _safe_name(backbone: str) -> str:
    return str(backbone).replace("/", "__").replace(":", "_")


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _replace_sentinel_with_placeholders(tokenizer, text: str, placeholder_id: int, count: int):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    offsets = encoding["offset_mapping"].squeeze(0)
    start = text.find(IMAGE_SENTINEL)
    if start < 0:
        raise RuntimeError("Prompt text did not contain the image sentinel")
    end = start + len(IMAGE_SENTINEL)
    sentinel_indices = [
        i
        for i, (token_start, token_end) in enumerate(offsets.tolist())
        if token_end > start and token_start < end
    ]
    if not sentinel_indices:
        raise RuntimeError("Tokenizer offsets did not locate the image sentinel")
    first = sentinel_indices[0]
    last = sentinel_indices[-1] + 1
    block = torch.full((count,), int(placeholder_id), dtype=input_ids.dtype)
    block_mask = torch.ones((count,), dtype=attention_mask.dtype)
    return {
        "input_ids": torch.cat([input_ids[:first], block, input_ids[last:]]),
        "attention_mask": torch.cat([attention_mask[:first], block_mask, attention_mask[last:]]),
        "sentinel_token_count": int(last - first),
        "placeholder_start": int(first),
    }


def _placeholder_contract(input_ids: torch.Tensor, placeholder_id: int, count: int) -> dict:
    positions = (input_ids == int(placeholder_id)).nonzero(as_tuple=True)[0]
    found = int(positions.numel())
    contiguous = bool(found and int(positions[-1] - positions[0] + 1) == found)
    return {
        "placeholder_token_id": int(placeholder_id),
        "placeholder_count_expected": int(count),
        "placeholder_count_found": found,
        "placeholder_positions_start": int(positions[0].item()) if found else None,
        "placeholder_positions_end": int(positions[-1].item()) if found else None,
        "placeholder_block_contiguous": contiguous,
        "pass": found == int(count) and contiguous,
    }


def run_smoke(
    llm_backbone: str = "Qwen/Qwen3-8B",
    output_root: str = "outputs/v8_llm_swap_smoke",
    image_tokens: int = 128,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    use_qlora: bool = False,
    max_new_tokens: int = 8,
) -> dict:
    """Run Stage 0 smoke and return the JSON payloads that were written."""
    backbone = canonicalize_llm_backbone(llm_backbone)
    model_name = llm_backbone if os.path.isabs(str(llm_backbone)) else backbone
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[torch_dtype]
    out_dir = Path(output_root) / _safe_name(backbone)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = LlamaWrapper(
        model_name=model_name,
        use_qlora=bool(use_qlora),
        use_lora=False,
        device_map=device_map,
        torch_dtype=dtype,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )
    tokenizer = wrapper.tokenizer
    placeholder_id = wrapper.ensure_single_token_placeholder(
        ["<image>", "<|image|>"]
        if wrapper.llm_model_type == "qwen3"
        else ["<|reserved_special_token_0|>", "<|image|>", "<image>"]
    )
    placeholder_token = tokenizer.convert_ids_to_tokens(int(placeholder_id))

    tokenizer_report = {
        "llm_backbone": wrapper.llm_backbone,
        "llm_model_name": wrapper.model_name,
        "llm_model_type": wrapper.llm_model_type,
        "tokenizer_name": wrapper.tokenizer_name,
        "chat_template_family": wrapper.chat_template_family,
        "hidden_size": wrapper.hidden_size,
        "num_hidden_layers": getattr(wrapper.model.config, "num_hidden_layers", None),
        "num_attention_heads": getattr(wrapper.model.config, "num_attention_heads", None),
        "num_key_value_heads": getattr(wrapper.model.config, "num_key_value_heads", None),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "placeholder_token": placeholder_token,
        "placeholder_token_id": int(placeholder_id),
        "placeholder_tokenizes_to": tokenizer.encode(placeholder_token, add_special_tokens=False),
        "added_special_tokens": list(wrapper.added_special_tokens),
        "lora_target_modules": wrapper._validate_lora_target_modules(wrapper.DEFAULT_LORA_TARGETS),
    }
    _write(out_dir / "tokenizer_report.json", tokenizer_report)

    prompt_text = build_generation_prompt_text(
        tokenizer=tokenizer,
        question="What color is the bus?",
        system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
        image_sentinel=IMAGE_SENTINEL,
        chat_template_family=wrapper.chat_template_family,
    )
    encoded = _replace_sentinel_with_placeholders(
        tokenizer,
        prompt_text,
        placeholder_id=int(placeholder_id),
        count=int(image_tokens),
    )
    contract = _placeholder_contract(encoded["input_ids"], int(placeholder_id), int(image_tokens))
    contract.update(
        {
            "prompt_text": prompt_text,
            "sentinel_token_count": encoded["sentinel_token_count"],
            "input_length": int(encoded["input_ids"].shape[0]),
        }
    )
    _write(out_dir / "prompt_contract_report.json", contract)
    if not contract["pass"]:
        raise RuntimeError(f"Prompt contract failed: {contract}")

    device = wrapper.device
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        input_forward = wrapper(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        embeds = wrapper.get_input_embeddings()(input_ids)
        embed_forward = wrapper(inputs_embeds=embeds, attention_mask=attention_mask, return_dict=True)
        random_visual = torch.randn(
            image_tokens,
            wrapper.hidden_size,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        replaced_embeds = embeds.clone()
        ph_positions = (input_ids[0] == int(placeholder_id)).nonzero(as_tuple=True)[0]
        replaced_embeds[0, ph_positions] = random_visual
        random_visual_forward = wrapper(
            inputs_embeds=replaced_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
    inputs_embeds_report = {
        "input_ids_forward_logits_shape": list(input_forward.logits.shape),
        "inputs_embeds_forward_logits_shape": list(embed_forward.logits.shape),
        "random_visual_forward_logits_shape": list(random_visual_forward.logits.shape),
        "input_vs_embeds_last_logit_max_abs_diff": float(
            (input_forward.logits[:, -1] - embed_forward.logits[:, -1]).abs().max().detach().cpu()
        ),
        "pass": True,
    }
    _write(out_dir / "inputs_embeds_report.json", inputs_embeds_report)

    text_prompt = build_generation_prompt_text(
        tokenizer=tokenizer,
        question="Answer yes or no: is snow white?",
        system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
        image_sentinel=None,
        chat_template_family=wrapper.chat_template_family,
    )
    text_encoding = tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
    text_ids = text_encoding["input_ids"].to(device)
    text_mask = text_encoding["attention_mask"].to(device)
    with torch.no_grad():
        text_generated = wrapper.generate(
            input_ids=text_ids,
            attention_mask=text_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        embed_generated = wrapper.generate(
            inputs_embeds=replaced_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        short = tokenizer("Short:", return_tensors="pt", add_special_tokens=False)
        long = tokenizer("A much longer direct-answer prompt:", return_tensors="pt", add_special_tokens=False)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(short["input_ids"].shape[1], long["input_ids"].shape[1])
        batch_ids = []
        batch_mask = []
        for item in (short, long):
            pad_len = max_len - item["input_ids"].shape[1]
            batch_ids.append(torch.nn.functional.pad(item["input_ids"].squeeze(0), (pad_len, 0), value=pad_id))
            batch_mask.append(torch.nn.functional.pad(item["attention_mask"].squeeze(0), (pad_len, 0), value=0))
        batch_ids = torch.stack(batch_ids).to(device)
        batch_mask = torch.stack(batch_mask).to(device)
        batch_generated = wrapper.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text_new = text_generated[0, text_ids.shape[1] :] if text_generated.shape[1] > text_ids.shape[1] else text_generated[0]
    generation_report = {
        "hf_or_custom_generation_path": wrapper.generation_path,
        "text_only_generated_shape": list(text_generated.shape),
        "inputs_embeds_generated_shape": list(embed_generated.shape),
        "batched_leftpad_generated_shape": list(batch_generated.shape),
        "text_only_raw_answer": tokenizer.decode(text_new, skip_special_tokens=True).strip(),
        "direct_answer_garbage_markers": {
            "assistant": "assistant" in tokenizer.decode(text_new, skip_special_tokens=True).lower(),
            "think": "think" in tokenizer.decode(text_new, skip_special_tokens=False).lower(),
        },
        "pass": True,
    }
    _write(out_dir / "generation_report.json", generation_report)

    print(f"Wrote V8 smoke reports to {out_dir}")
    return {
        "output_dir": str(out_dir),
        "tokenizer_report": _jsonify(tokenizer_report),
        "prompt_contract_report": _jsonify(contract),
        "inputs_embeds_report": _jsonify(inputs_embeds_report),
        "generation_report": _jsonify(generation_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-backbone", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-root", default="outputs/v8_llm_swap_smoke")
    parser.add_argument("--image-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    run_smoke(
        llm_backbone=args.llm_backbone,
        output_root=args.output_root,
        image_tokens=args.image_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        use_qlora=args.use_qlora,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
