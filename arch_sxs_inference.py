"""Run side-by-side inference for AnyMAL architecture candidates V1 through V4.

Usage:
    modal run arch_sxs_inference.py --num-examples 12

Writes a local JSON file and, by default, updates the Modal volume artifact used
by ``modal_viewer.py``:
    /checkpoints/arch_sxs_predictions.json
"""

import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("anymal-arch-sxs-inference")
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

LLAMA_PATH = "/checkpoints/llama3-8b-instruct"
VQA_QUESTIONS = "/checkpoints/vqa_data/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS = "/checkpoints/vqa_data/v2_mscoco_val2014_annotations.json"
VQA_IMAGE_DIR = "/checkpoints/coco_val2014"
LLAVA_DATA = "/checkpoints/llava_data/llava_instruct_150k.json"
LLAVA_IMAGE_DIR = "/checkpoints/coco_images"

DIRECT_SYSTEM_PROMPT = "Answer the image question directly and briefly. End after the answer."
TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>"
IMAGE_SENTINEL = "<|image_sentinel|>"
ASSISTANT_ROLE_PREFIX_RE = re.compile(r"^assistant\s*[:\n\r ]+\s*", re.IGNORECASE)

DEFAULT_RUNS = [
    {
        "key": "v1_ablation_f",
        "label": "V1 ablation-F ckpt500",
        "architecture": "v1",
        "checkpoint": "/checkpoints/finetune-output/ablation-F/checkpoint-500",
    },
    {
        "key": "v2_balanced_mix",
        "label": "V2 balanced-mix ckpt3000",
        "architecture": "v2",
        "checkpoint": "/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000",
    },
    {
        "key": "v3_direct_calibration",
        "label": "V3 direct-calibration ckpt100",
        "architecture": "v3",
        "checkpoint": "/checkpoints/finetune-output/v3-direct-calibration-loraonly-300-20260508-codex/checkpoint-100",
    },
    {
        "key": "v4_semantic_calibration",
        "label": "V4 semantic-calibration ckpt100",
        "architecture": "v4",
        "checkpoint": "/checkpoints/finetune-output/v4-stage2a-semanticcal-bs4-lossscale003-lora1e5-from-stage1b248-20260509-codex/checkpoint-100",
    },
]


def _load_vqa_examples(num_examples: int, seed: int):
    with open(VQA_QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(VQA_ANNOTATIONS, encoding="utf-8") as f:
        annotations = {
            ann["question_id"]: ann
            for ann in json.load(f)["annotations"]
        }

    available = {
        name
        for name in os.listdir(VQA_IMAGE_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    filtered = [
        q for q in questions
        if f"COCO_val2014_{int(q['image_id']):012d}.jpg" in available
    ]
    rng = random.Random(int(seed))
    rng.shuffle(filtered)
    picked = filtered[: int(num_examples)]

    examples = []
    for q in picked:
        ann = annotations.get(q["question_id"], {})
        answers = [a.get("answer", "") for a in ann.get("answers", [])]
        majority = Counter(answers).most_common(1)
        examples.append(
            {
                "image": f"COCO_val2014_{int(q['image_id']):012d}.jpg",
                "question": q["question"],
                "ground_truth": majority[0][0] if majority else "",
                "answers": answers,
                "answer_type": ann.get("answer_type", ""),
                "question_type": ann.get("question_type", ""),
            }
        )
    return examples, VQA_IMAGE_DIR, "vqa_val2014"


def _load_llava_examples(tokenizer, num_examples: int):
    import sys

    sys.path.insert(0, "/root/anymal")
    from data.dataset_splitter import deterministic_train_val_split
    from data.instruction_dataset import InstructionDataset

    dataset = InstructionDataset(
        data_path=LLAVA_DATA,
        image_dir=LLAVA_IMAGE_DIR,
        tokenizer=tokenizer,
        image_size=384,
        max_length=2304,
        filter_to_available_images=True,
    )
    _, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    stride = max(1, len(val_dataset) // int(num_examples))
    indices = [i * stride for i in range(int(num_examples)) if i * stride < len(val_dataset)]

    examples = []
    for local_idx in indices:
        sample = dataset.samples[val_dataset.indices[local_idx]]
        question = ""
        answer = ""
        for turn in sample.get("conversations", []):
            role = turn.get("from", "")
            content = turn.get("value", "").replace("<image>", "").replace("\n", " ").strip()
            if role == "human" and not question:
                question = content
            elif role == "gpt" and not answer:
                answer = content
        examples.append({"image": sample["image"], "question": question, "ground_truth": answer})
    return examples, LLAVA_IMAGE_DIR, "llava_val"


def _build_prompt_input_ids(tokenizer, placeholder_id, num_image_tokens, question, system_prompt, max_length):
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
    placeholders = torch.full((int(num_image_tokens),), int(placeholder_id), dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids[:first], placeholders, input_ids[last:]])
    attention_mask = torch.cat(
        [
            attention_mask[:first],
            torch.ones(int(num_image_tokens), dtype=attention_mask.dtype),
            attention_mask[last:],
        ]
    )
    if input_ids.shape[0] > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)


def _stop_token_ids(tokenizer):
    ids = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.append(int(tokenizer.eos_token_id))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        eot_id = get_vocab().get(END_TURN)
        if eot_id is not None:
            ids.append(int(eot_id))
    return list(dict.fromkeys(ids))


def _trim_at_stop(token_ids, stop_ids):
    if token_ids.numel() == 0 or not stop_ids:
        return token_ids
    stops = set(stop_ids)
    for idx, token_id in enumerate(token_ids.tolist()):
        if int(token_id) in stops:
            return token_ids[:idx]
    return token_ids


def _load_v3_model(checkpoint: str, llama_path: str, device):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from models.anymal_v3 import AnyMALv3

    model = AnyMALv3.from_pretrained(
        checkpoint,
        llm_model_name=llama_path,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        connector_type="perceiver_resampler",
        num_image_tokens=128,
        connector_layers=6,
        connector_heads=16,
        connector_ff_mult=4,
        project_directly_to_llm_dim=True,
        use_qlora=True,
        use_lora=False,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
        llm_device_map="auto",
        llm_torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.image_encoder.to(device)
    model.projector.to(device)
    return model


def _load_v4_model(checkpoint: str, llama_path: str, device):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from model_metadata import read_model_metadata
    from models.anymal_v4 import AnyMALv4

    meta = read_model_metadata(checkpoint) or {}
    connector_type = meta.get("connector_type", "spatial_perceiver_resampler")
    deepstack_layers = meta.get("deepstack_hidden_state_indices") or meta.get("vision_feature_layers")
    deepstack_kwargs = {}
    if connector_type == "deepstack_spatial_perceiver_resampler":
        deepstack_kwargs = {
            "deepstack_num_feature_levels": int(
                meta.get("deepstack_num_feature_levels") or len(deepstack_layers or []) or 3
            ),
            "deepstack_hidden_state_indices": deepstack_layers,
        }

    model = AnyMALv4.from_pretrained(
        checkpoint,
        llm_model_name=llama_path,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        connector_type=connector_type,
        num_global_image_tokens=int(meta.get("num_global_image_tokens", 64)),
        num_local_image_tokens=int(meta.get("num_local_image_tokens", 64)),
        num_image_tokens=int(meta.get("num_image_tokens", 128)),
        connector_layers=int(meta.get("connector_layers", 6)),
        connector_heads=int(meta.get("connector_heads", 16)),
        connector_ff_mult=int(meta.get("connector_ff_mult", 4)),
        connector_hidden_dim=(
            int(meta["connector_hidden_dim"]) if meta.get("connector_hidden_dim") is not None else None
        ),
        connector_output_scale=float(meta.get("connector_output_scale", 1.0)),
        connector_output_gate_init=(
            float(meta["connector_output_gate_init"])
            if meta.get("connector_output_gate_init") is not None
            else None
        ),
        use_2d_position_features=bool(meta.get("use_2d_position_features", True)),
        project_directly_to_llm_dim=bool(meta.get("project_directly_to_llm_dim", True)),
        use_qlora=True,
        use_lora=False,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
        llm_device_map="auto",
        llm_torch_dtype=torch.bfloat16,
        **deepstack_kwargs,
    )
    model.eval()
    model.image_encoder.to(device)
    model.projector.to(device)
    return model


def _load_model(run, llama_path: str, device):
    import sys

    sys.path.insert(0, "/root/anymal")
    from v1_v2_compare_inference import _load_v1_model, _load_v2_model

    if run["architecture"] == "v1":
        return _load_v1_model(run["checkpoint"], llama_path, device)
    if run["architecture"] == "v2":
        return _load_v2_model(run["checkpoint"], llama_path, device)
    if run["architecture"] == "v3":
        return _load_v3_model(run["checkpoint"], llama_path, device)
    if run["architecture"] == "v4":
        return _load_v4_model(run["checkpoint"], llama_path, device)
    raise ValueError(f"Unknown architecture: {run['architecture']}")


def _generate(model, architecture: str, example, image_dir: str, device, max_new_tokens: int, system_prompt: str):
    import sys
    import torch
    from PIL import Image

    sys.path.insert(0, "/root/anymal")
    from data.data_utils import get_image_transform, get_vision_transform

    if architecture in {"v2", "v3", "v4"}:
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
    seq = generated[0]
    new_tokens = seq[input_ids.shape[1] :] if seq.shape[0] > input_ids.shape[1] else seq
    new_tokens = _trim_at_stop(new_tokens, _stop_token_ids(model.tokenizer))
    raw = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    cleaned = ASSISTANT_ROLE_PREFIX_RE.sub("", raw).strip()
    return {"text": cleaned, "raw_text": raw, "generated_tokens": int(new_tokens.shape[0])}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def compare_architectures(
    num_examples: int = 12,
    max_new_tokens: int = 32,
    seed: int = 42,
    example_source: str = "vqa",
    system_prompt: str = DIRECT_SYSTEM_PROMPT,
    runs=None,
    remote_output_path: str = "/checkpoints/arch_sxs_predictions.json",
):
    import gc
    import sys
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root/anymal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if example_source == "vqa":
        examples, image_dir, source_label = _load_vqa_examples(num_examples, seed)
    elif example_source == "llava":
        examples, image_dir, source_label = _load_llava_examples(tokenizer, num_examples)
    else:
        raise ValueError("example_source must be 'vqa' or 'llava'")

    runs = runs or DEFAULT_RUNS
    results = [dict(example, responses={}) for example in examples]

    for run in runs:
        print(f"\n=== {run['label']} ===")
        model = _load_model(run, LLAMA_PATH, device)
        for i, example in enumerate(examples):
            try:
                response = _generate(
                    model=model,
                    architecture=run["architecture"],
                    example=example,
                    image_dir=image_dir,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                response = {"text": f"<ERROR: {exc}>", "raw_text": "", "generated_tokens": 0}
            results[i]["responses"][run["key"]] = response
            print(f"[{i + 1}/{len(examples)}] {run['key']}: {response['text'][:120]}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "title": "AnyMAL architecture side-by-side",
        "description": "Greedy inference on the same examples for V1, V2, V3, and V4 checkpoints.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example_source": source_label,
        "image_dir": image_dir,
        "num_examples": len(results),
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed),
        "system_prompt": system_prompt,
        "runs": runs,
        "results": results,
    }

    if remote_output_path:
        os.makedirs(os.path.dirname(remote_output_path), exist_ok=True)
        with open(remote_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        volume.commit()
        print(f"Saved remote side-by-side predictions to {remote_output_path}")
    return output


@app.local_entrypoint()
def main(
    num_examples: int = 12,
    max_new_tokens: int = 32,
    seed: int = 42,
    example_source: str = "vqa",
    prompt_mode: str = "direct",
    output: str = "arch_sxs_predictions.json",
    remote_output_path: str = "/checkpoints/arch_sxs_predictions.json",
):
    if prompt_mode == "direct":
        system_prompt = DIRECT_SYSTEM_PROMPT
    elif prompt_mode == "training":
        system_prompt = TRAINING_SYSTEM_PROMPT
    else:
        system_prompt = prompt_mode

    result = compare_architectures.remote(
        num_examples=num_examples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        example_source=example_source,
        system_prompt=system_prompt,
        remote_output_path=remote_output_path,
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
