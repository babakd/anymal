"""
Minimal side-by-side inference script for comparing two finetune checkpoints.

Usage:
    modal run compare_inference.py --checkpoint-a /checkpoints/finetune-output/ablation-A/checkpoint-500 \\
                                   --checkpoint-b /checkpoints/finetune-output/ablation-F/checkpoint-500 \\
                                   --num-examples 20

Writes compare_predictions.json with {image, question, ground_truth, response_a, response_b}.
"""

import modal
import json
import os
from pathlib import Path

app = modal.App("anymal-compare")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
PROJECT_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


def _load_val_examples(tokenizer, num_examples: int):
    """Pick deterministic val-split examples using the same splitter as training."""
    import sys
    sys.path.insert(0, "/root/anymal")
    from data.instruction_dataset import InstructionDataset
    from data.dataset_splitter import deterministic_train_val_split

    json_path = "/checkpoints/llava_data/llava_instruct_150k.json"
    image_dir = "/checkpoints/coco_images"
    dataset = InstructionDataset(
        data_path=json_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        image_size=224,
        max_length=512,
        filter_to_available_images=True,
    )
    _, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    val_size = len(val_dataset)
    # Uniform stride so we cover the whole val split, not just the start
    stride = max(1, val_size // num_examples)
    indices = [i * stride for i in range(num_examples) if i * stride < val_size]

    examples = []
    for local_idx in indices:
        original_idx = val_dataset.indices[local_idx]
        sample = dataset.samples[original_idx]
        # Extract user question + ground-truth GPT answer
        question, answer = "", ""
        for turn in sample.get("conversations", []):
            role = turn.get("from", "")
            content = turn.get("value", "").replace("<image>", "").replace("\n", " ").strip()
            if role == "human" and not question:
                question = content
            elif role == "gpt" and not answer:
                answer = content
        examples.append({
            "image": sample["image"],
            "question": question,
            "ground_truth": answer,
        })
    return examples


def _load_model(llama_path: str, checkpoint_dir: str, device):
    """Load AnyMAL + projector + LoRA from a specific checkpoint."""
    import torch
    from models.anymal import AnyMAL

    model = AnyMAL(
        llm_model_name=llama_path,
        use_qlora=True,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
    )

    projector_path = os.path.join(checkpoint_dir, "projector.pt")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    print(f"  Loaded projector from {projector_path}")

    llm_path = os.path.join(checkpoint_dir, "llm")
    if os.path.exists(llm_path):
        from peft import PeftModel
        base = model.llm.model
        if hasattr(base, "base_model"):
            base = base.base_model
        model.llm.model = PeftModel.from_pretrained(base, llm_path)
        print(f"  Loaded LoRA from {llm_path}")

    model.eval()
    model.to(device)
    return model


def _generate(model, example, image_dir, device, max_new_tokens=384):
    import torch
    from PIL import Image
    from data.data_utils import get_image_transform

    transform = get_image_transform(image_size=224, is_train=False, use_augmentation=False)

    pil = Image.open(os.path.join(image_dir, example["image"])).convert("RGB")
    image_tensor = transform(pil).unsqueeze(0).to(device)

    placeholder_id = model.image_placeholder_token_id
    num_image_tokens = model.num_image_tokens
    tokenizer = model.tokenizer

    placeholder_str = "".join(
        [tokenizer.convert_ids_to_tokens([placeholder_id])[0]] * num_image_tokens
    )
    full_prompt = f"{placeholder_str}{example['question']}"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated = model.generate(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def compare(checkpoint_a: str, checkpoint_b: str, num_examples: int = 20):
    import sys
    sys.path.insert(0, "/root/anymal")
    import gc
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama_path = "/checkpoints/llama3-8b-instruct"
    image_dir = "/checkpoints/coco_images"

    # Load tokenizer once (shared)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Picking {num_examples} deterministic val examples...")
    examples = _load_val_examples(tokenizer, num_examples)
    print(f"Selected {len(examples)} val examples")

    results = [dict(e) for e in examples]

    for label, ckpt in [("response_a", checkpoint_a), ("response_b", checkpoint_b)]:
        print(f"\n=== Loading {label} from {ckpt} ===")
        model = _load_model(llama_path, ckpt, device)
        for i, ex in enumerate(examples):
            try:
                out = _generate(model, ex, image_dir, device)
            except Exception as e:
                out = f"<ERROR: {e}>"
            results[i][label] = out
            print(f"  [{i+1}/{len(examples)}] {label}: {out[:80]}...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "num_examples": len(examples),
        "results": results,
    }


@app.local_entrypoint()
def main(
    checkpoint_a: str = "/checkpoints/finetune-output/ablation-A/checkpoint-500",
    checkpoint_b: str = "/checkpoints/finetune-output/ablation-F/checkpoint-500",
    num_examples: int = 20,
    output: str = "compare_predictions.json",
):
    print(f"Comparing:\n  A: {checkpoint_a}\n  B: {checkpoint_b}\n  N: {num_examples}")
    result = compare.remote(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        num_examples=num_examples,
    )
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {output} with {result['num_examples']} comparisons")
