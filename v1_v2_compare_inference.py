"""Compare a legacy V1 checkpoint with a V2 checkpoint under one prompt path.

This is the user-facing sanity check for repair candidates: V2 should not just
look better than an earlier V2 run, it should beat the best available V1
baseline on the same deterministic examples and chat-formatted prompt.
"""

import json
import os
from pathlib import Path

import modal


app = modal.App("anymal-v1-v2-compare")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
PROJECT_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.53.0,<5.0.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "einops>=0.7.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)
STRICT_SYSTEM_PROMPT = (
    "Answer directly and briefly. Do not greet the user. Do not ask follow-up questions."
)
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>"
IMAGE_SENTINEL = "<|image_sentinel|>"


def _load_val_examples(tokenizer, num_examples: int):
    import sys

    sys.path.insert(0, "/root/anymal")
    from data.dataset_splitter import deterministic_train_val_split
    from data.instruction_dataset import InstructionDataset

    dataset = InstructionDataset(
        data_path="/checkpoints/llava_data/llava_instruct_150k.json",
        image_dir="/checkpoints/coco_images",
        tokenizer=tokenizer,
        image_size=384,
        max_length=2304,
        filter_to_available_images=True,
    )
    _, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    stride = max(1, len(val_dataset) // num_examples)
    indices = [i * stride for i in range(num_examples) if i * stride < len(val_dataset)]

    examples = []
    for local_idx in indices:
        original_idx = val_dataset.indices[local_idx]
        sample = dataset.samples[original_idx]
        question = ""
        answer = ""
        for turn in sample.get("conversations", []):
            role = turn.get("from", "")
            content = turn.get("value", "").replace("<image>", "").replace("\n", " ").strip()
            if role == "human" and not question:
                question = content
            elif role == "gpt" and not answer:
                answer = content
        examples.append(
            {
                "image": sample["image"],
                "question": question,
                "ground_truth": answer,
            }
        )
    return examples


def _build_prompt_input_ids(
    tokenizer,
    placeholder_id: int,
    num_image_tokens: int,
    question: str,
    system_prompt: str,
    max_length: int,
):
    import torch

    text = (
        f"{SYSTEM_HEADER}{system_prompt}{END_TURN}"
        f"{USER_HEADER}{IMAGE_SENTINEL}\n{question.strip()}{END_TURN}"
        f"{ASSISTANT_HEADER}"
    )
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    offsets = encoding["offset_mapping"].squeeze(0)

    sentinel_start = text.find(IMAGE_SENTINEL)
    sentinel_end = sentinel_start + len(IMAGE_SENTINEL)
    sentinel_indices = [
        i
        for i, (token_start, token_end) in enumerate(offsets.tolist())
        if token_end > sentinel_start and token_start < sentinel_end
    ]
    if not sentinel_indices:
        raise RuntimeError("Could not locate image sentinel after tokenization")

    first = sentinel_indices[0]
    last = sentinel_indices[-1] + 1
    placeholder_block = torch.full(
        (num_image_tokens,),
        placeholder_id,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([input_ids[:first], placeholder_block, input_ids[last:]])
    attention_mask = torch.cat(
        [
            attention_mask[:first],
            torch.ones(num_image_tokens, dtype=attention_mask.dtype),
            attention_mask[last:],
        ]
    )
    if input_ids.shape[0] > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)


def _load_v1_model(checkpoint: str, llama_path: str, device):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from model_metadata import validate_checkpoint_metadata_values
    from models.anymal import AnyMAL
    from peft import PeftModel

    validate_checkpoint_metadata_values(checkpoint, expected_architecture="anymal_v1")
    model = AnyMAL(
        llm_model_name=llama_path,
        use_qlora=True,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
    )
    projector_path = os.path.join(checkpoint, "projector.pt")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    llm_path = os.path.join(checkpoint, "llm")
    if os.path.isdir(llm_path):
        base_model = model.llm.model
        if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
            base_model = base_model.unload()
        model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
    model.eval()
    model.to(device)
    return model


def _load_v2_model(checkpoint: str, llama_path: str, device):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from model_metadata import validate_checkpoint_metadata_values
    from models.anymal_v2 import AnyMALv2
    from peft import PeftModel

    validate_checkpoint_metadata_values(
        checkpoint,
        expected_architecture="anymal_v2",
        expected_values={
            "vision_encoder_type": "siglip2",
            "token_compressor_type": "learned",
            "max_image_tokens": 384,
            "min_image_tokens": 384,
        },
    )
    model = AnyMALv2(
        llm_model_name=llama_path,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        token_compressor_type="learned",
        bottleneck_dim=2048,
        max_image_tokens=384,
        min_image_tokens=384,
        use_qlora=True,
        use_lora=False,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
        llm_device_map="auto",
        llm_torch_dtype=torch.bfloat16,
    )
    model.projector.load_state_dict(
        torch.load(os.path.join(checkpoint, "projector.pt"), map_location="cpu")
    )
    model.token_compressor.load_state_dict(
        torch.load(os.path.join(checkpoint, "token_compressor.pt"), map_location="cpu")
    )
    llm_path = os.path.join(checkpoint, "llm")
    if os.path.isdir(llm_path):
        base_model = model.llm.model
        if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
            base_model = base_model.unload()
        model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
    model.eval()
    model.image_encoder.to(device)
    model.token_compressor.to(device)
    model.projector.to(device)
    return model


def _generate(
    model,
    architecture: str,
    example,
    image_dir: str,
    device,
    max_new_tokens: int,
    system_prompt: str,
):
    import sys
    import torch
    from PIL import Image

    sys.path.insert(0, "/root/anymal")
    from data.data_utils import get_image_transform, get_vision_transform

    if architecture == "v2":
        transform = get_vision_transform(
            vision_encoder_type="siglip2",
            vision_model_name="google/siglip2-so400m-patch14-384",
            image_size=384,
            is_train=False,
            use_augmentation=False,
        )
        max_length = 2304
    else:
        transform = get_image_transform(image_size=224, is_train=False, use_augmentation=False)
        max_length = 1024

    pil = Image.open(os.path.join(image_dir, example["image"])).convert("RGB")
    image_tensor = transform(pil).unsqueeze(0).to(device)
    input_ids, attention_mask = _build_prompt_input_ids(
        tokenizer=model.tokenizer,
        placeholder_id=model.image_placeholder_token_id,
        num_image_tokens=model.num_image_tokens,
        question=example["question"],
        system_prompt=system_prompt,
        max_length=max_length,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        generated = model.generate(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = generated[0, input_ids.shape[1] :]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def compare_v1_v2(
    v1_checkpoint: str,
    v2_checkpoint: str,
    v2_label: str,
    num_examples: int = 24,
    max_new_tokens: int = 96,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
):
    import gc
    import sys
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root/anymal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama_path = "/checkpoints/llama3-8b-instruct"
    image_dir = "/checkpoints/coco_images"

    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _load_val_examples(tokenizer, num_examples)
    results = [dict(example) for example in examples]

    runs = [
        {
            "key": "v1_baseline",
            "label": "V1 ablation-F mix_665k checkpoint-500",
            "checkpoint": v1_checkpoint,
            "architecture": "v1",
        },
        {
            "key": "v2_candidate",
            "label": v2_label,
            "checkpoint": v2_checkpoint,
            "architecture": "v2",
        },
    ]

    for run in runs:
        print(f"Loading {run['label']} from {run['checkpoint']}")
        if run["architecture"] == "v1":
            model = _load_v1_model(run["checkpoint"], llama_path, device)
        else:
            model = _load_v2_model(run["checkpoint"], llama_path, device)
        for i, example in enumerate(examples):
            try:
                response = _generate(
                    model,
                    run["architecture"],
                    example,
                    image_dir,
                    device,
                    max_new_tokens,
                    system_prompt,
                )
            except Exception as exc:
                response = f"<ERROR: {exc}>"
            results[i][run["key"]] = response
            print(f"[{i + 1}/{len(examples)}] {run['key']}: {response[:120]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "runs": runs,
        "num_examples": len(results),
        "max_new_tokens": max_new_tokens,
        "system_prompt": system_prompt,
        "results": results,
    }


@app.local_entrypoint()
def main(
    v1_checkpoint: str = "/checkpoints/finetune-output/ablation-F/checkpoint-500",
    v2_checkpoint: str = "/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000",
    v2_label: str = "V2 candidate",
    num_examples: int = 24,
    max_new_tokens: int = 96,
    prompt_mode: str = "training",
    output: str = "v1_v2_compare_predictions.json",
):
    if prompt_mode == "training":
        system_prompt = TRAINING_SYSTEM_PROMPT
    elif prompt_mode == "strict":
        system_prompt = STRICT_SYSTEM_PROMPT
    else:
        system_prompt = prompt_mode

    result = compare_v1_v2.remote(
        v1_checkpoint=v1_checkpoint,
        v2_checkpoint=v2_checkpoint,
        v2_label=v2_label,
        num_examples=num_examples,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
