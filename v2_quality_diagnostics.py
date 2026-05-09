"""
Modal diagnostics for the completed AnyMAL V2 learned-compressor run.

Usage:
    modal run v2_quality_diagnostics.py --mode audit
    modal run v2_quality_diagnostics.py --mode probe --max-new-tokens 32,64,128,192
"""

import json
import os
from collections import Counter
from pathlib import Path

import modal


app = modal.App("anymal-v2-quality-diagnostics")
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

STAGE1_CKPT = "/checkpoints/pretrain-output/v2-stage1-learned-2500-20260428/checkpoint-2500"
STAGE2_CKPT = "/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000"
LLAMA_PATH = "/checkpoints/llama3-8b-instruct"
COCO_IMAGE_DIR = "/checkpoints/coco_images"

TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)
STRICT_SYSTEM_PROMPT = (
    "Answer directly and briefly. Do not greet the user. Do not ask follow-up questions."
)

CHATTY_PHRASES = (
    "i'd be happy",
    "i would be happy",
    "great question",
    "sure",
    "certainly",
    "of course",
    "let me",
)

TARGETED_PROBES = [
    {
        "image": "000000291841.jpg",
        "question": "What vehicle is shown?",
        "expected": "bus",
    },
    {
        "image": "000000291841.jpg",
        "question": "What object is this?",
        "expected": "bus",
    },
    {
        "image": "000000145989.jpg",
        "question": "What color is the train?",
        "expected": "color",
    },
    {
        "image": "000000258823.jpg",
        "question": "How many giraffes are visible in the image?",
        "expected": "three",
    },
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_summary(checkpoint_dir):
    import torch

    summary = {
        "path": checkpoint_dir,
        "exists": os.path.isdir(checkpoint_dir),
        "files": {},
    }
    for rel in ("model_meta.json", "projector.pt", "token_compressor.pt", "trainer_state.pt"):
        path = os.path.join(checkpoint_dir, rel)
        summary["files"][rel] = os.path.exists(path)

    meta_path = os.path.join(checkpoint_dir, "model_meta.json")
    if os.path.exists(meta_path):
        summary["metadata"] = _read_json(meta_path)

    for rel in ("projector.pt", "token_compressor.pt"):
        path = os.path.join(checkpoint_dir, rel)
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location="cpu")
        summary[rel] = {
            "num_tensors": len(state),
            "shapes": {key: list(value.shape) for key, value in state.items()},
        }

    llm_dir = os.path.join(checkpoint_dir, "llm")
    summary["llm"] = {
        "exists": os.path.isdir(llm_dir),
        "adapter_model": os.path.exists(os.path.join(llm_dir, "adapter_model.safetensors")),
        "adapter_config": os.path.exists(os.path.join(llm_dir, "adapter_config.json")),
    }
    adapter_config = os.path.join(llm_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        summary["llm"]["adapter_config_json"] = _read_json(adapter_config)
    return summary


def _first_human_and_gpt(sample):
    question = ""
    answer = ""
    for turn in sample.get("conversations", []):
        role = turn.get("from", turn.get("role", ""))
        text = turn.get("value", turn.get("content", "")).replace("<image>", "").strip()
        if role in {"human", "user"} and not question:
            question = text
        elif role in {"gpt", "assistant"} and not answer:
            answer = text
    return question, answer


def _task_type(question):
    q = " ".join(question.lower().split())
    if "how many" in q or q.startswith("count"):
        return "count"
    if "what color" in q or "colour" in q:
        return "color"
    if any(term in q for term in ("left", "right", "behind", "front", "next to", "where")):
        return "spatial"
    if any(term in q for term in ("describe", "caption", "detailed", "image")):
        return "caption_or_description"
    if any(term in q for term in ("why", "how", "reason", "likely", "contribute")):
        return "reasoning"
    if any(term in q for term in ("what is", "what object", "what vehicle", "which")):
        return "short_object"
    return "other"


def _dataset_stats(name, samples):
    answer_lengths = []
    chatty = 0
    short_answers = 0
    task_counts = Counter()
    empty_question = 0
    empty_answer = 0

    for sample in samples:
        question, answer = _first_human_and_gpt(sample)
        if not question:
            empty_question += 1
        if not answer:
            empty_answer += 1
        words = answer.split()
        answer_lengths.append(len(words))
        short_answers += int(len(words) <= 3 and bool(words))
        answer_l = answer.lower()
        chatty += int(any(phrase in answer_l for phrase in CHATTY_PHRASES))
        task_counts[_task_type(question)] += 1

    answer_lengths.sort()
    total = len(samples)

    def percentile(p):
        if not answer_lengths:
            return 0
        idx = min(len(answer_lengths) - 1, int(round((len(answer_lengths) - 1) * p)))
        return answer_lengths[idx]

    return {
        "name": name,
        "samples": total,
        "avg_answer_words": sum(answer_lengths) / max(total, 1),
        "p50_answer_words": percentile(0.50),
        "p90_answer_words": percentile(0.90),
        "short_answer_count_le_3_words": short_answers,
        "short_answer_rate": short_answers / max(total, 1),
        "chatty_phrase_count": chatty,
        "chatty_phrase_rate": chatty / max(total, 1),
        "task_counts": dict(task_counts),
        "empty_question": empty_question,
        "empty_answer": empty_answer,
    }


def _stage1_data_audit():
    import zipfile

    pretrain_json = "/checkpoints/llava_data/blip_laion_cc_sbu_558k.json"
    manifest = "/checkpoints/llava_pretrain/manifest.json"
    images_zip = "/checkpoints/llava_pretrain/images.zip"
    result = {
        "pretrain_json_exists": os.path.exists(pretrain_json),
        "manifest_exists": os.path.exists(manifest),
        "images_zip_exists": os.path.exists(images_zip),
    }
    if os.path.exists(manifest):
        result["manifest"] = _read_json(manifest)
    if os.path.exists(images_zip):
        with zipfile.ZipFile(images_zip) as zf:
            names = zf.namelist()
            result["zip_member_count"] = len(names)
            result["zip_image_member_count"] = sum(
                name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                for name in names
            )
            result["zip_first_members"] = names[:5]
    if os.path.exists(pretrain_json):
        records = _read_json(pretrain_json)
        result["annotation_count"] = len(records)
        result["first_annotations"] = records[:5]
    return result


def _stage2_data_audit():
    instruct_path = "/checkpoints/llava_data/llava_instruct_150k.json"
    mix_raw_path = "/checkpoints/llava_data/llava_v1_5_mix665k.json"
    mix_filtered_path = "/checkpoints/llava_data/mix665k_filtered.json"
    image_dir = COCO_IMAGE_DIR
    available = set(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    instruct = _read_json(instruct_path)
    mix_raw = _read_json(mix_raw_path)
    if os.path.exists(mix_filtered_path):
        mix_filtered = _read_json(mix_filtered_path)
    else:
        mix_filtered = []
        for sample in mix_raw:
            img = os.path.basename(sample.get("image", ""))
            if img in available:
                item = dict(sample)
                item["image"] = img
                mix_filtered.append(item)

    return {
        "cached_coco_images": len(available),
        "source_counts": {
            "instruct_150k_raw": len(instruct),
            "instruct_150k_with_cached_images": sum(
                1 for sample in instruct if sample.get("image") in available
            ),
            "mix_665k_raw": len(mix_raw),
            "mix_665k_with_cached_coco_images": len(mix_filtered),
            "balanced_mix_epoch_length": max(len(instruct), len(mix_filtered)) * 2,
        },
        "instruct_150k": _dataset_stats("instruct_150k", instruct),
        "mix_665k_filtered": _dataset_stats("mix_665k_filtered", mix_filtered),
    }


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def audit_v2():
    return {
        "checkpoints": {
            "stage1": _checkpoint_summary(STAGE1_CKPT),
            "stage2": _checkpoint_summary(STAGE2_CKPT),
        },
        "stage1_data": _stage1_data_audit(),
        "stage2_data": _stage2_data_audit(),
    }


def _load_v2_finetune_model(checkpoint_dir, device):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from models.anymal_v2 import AnyMALv2
    from peft import PeftModel

    meta = {}
    meta_path = os.path.join(checkpoint_dir, "model_meta.json")
    if os.path.exists(meta_path):
        meta = _read_json(meta_path)
    token_compressor_type = meta.get("token_compressor_type", "learned")
    max_image_tokens = int(meta.get("max_image_tokens", 384))
    min_image_tokens = int(meta.get("min_image_tokens", max_image_tokens))

    model = AnyMALv2(
        llm_model_name=LLAMA_PATH,
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        token_compressor_type=token_compressor_type,
        bottleneck_dim=2048,
        max_image_tokens=max_image_tokens,
        min_image_tokens=min_image_tokens,
        use_qlora=True,
        use_lora=False,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=False,
        use_flash_attention=False,
        llm_device_map="auto",
        llm_torch_dtype=torch.bfloat16,
    )
    model.projector.load_state_dict(torch.load(os.path.join(checkpoint_dir, "projector.pt"), map_location="cpu"))
    model.token_compressor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "token_compressor.pt"), map_location="cpu"))
    llm_dir = os.path.join(checkpoint_dir, "llm")
    if os.path.isdir(llm_dir):
        model.llm.model = PeftModel.from_pretrained(model.llm.model, llm_dir)
    model.image_encoder.to(device)
    model.token_compressor.to(device)
    model.projector.to(device)
    model.eval()
    return model


def _prompt_inputs(model, question, system_prompt, include_image=True):
    from v2_compare_inference import _build_prompt_input_ids

    if include_image:
        return _build_prompt_input_ids(
            tokenizer=model.tokenizer,
            placeholder_id=model.image_placeholder_token_id,
            num_image_tokens=model.num_image_tokens,
            question=question,
            system_prompt=system_prompt,
        )

    text = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{question.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    encoding = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=2304)
    return encoding["input_ids"], encoding["attention_mask"]


def _image_tensor(image_name, transform, device, variant):
    import torch
    from PIL import Image

    if variant == "blank":
        pil = Image.new("RGB", (384, 384), color=(128, 128, 128))
    else:
        pil = Image.open(os.path.join(COCO_IMAGE_DIR, image_name)).convert("RGB")
    return transform(pil).unsqueeze(0).to(device)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe_stage2(max_new_tokens_values=None, include_strict: bool = True):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from data.data_utils import get_vision_transform

    if max_new_tokens_values is None:
        max_new_tokens_values = [32, 64, 128, 192]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_v2_finetune_model(STAGE2_CKPT, device)
    transform = get_vision_transform(
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        image_size=384,
        is_train=False,
        use_augmentation=False,
    )

    probes = [
        probe for probe in TARGETED_PROBES
        if os.path.exists(os.path.join(COCO_IMAGE_DIR, probe["image"]))
    ]
    prompts = {"training": TRAINING_SYSTEM_PROMPT}
    if include_strict:
        prompts["strict"] = STRICT_SYSTEM_PROMPT

    rows = []
    for probe in probes:
        wrong_image = next(
            (other["image"] for other in probes if other["image"] != probe["image"]),
            probe["image"],
        )
        for prompt_name, system_prompt in prompts.items():
            for max_new_tokens in max_new_tokens_values:
                for variant, image_name, include_image in (
                    ("correct_image", probe["image"], True),
                    ("wrong_image", wrong_image, True),
                    ("blank_image", probe["image"], True),
                    ("text_only", probe["image"], False),
                ):
                    input_ids, attention_mask = _prompt_inputs(
                        model,
                        probe["question"],
                        system_prompt,
                        include_image=include_image,
                    )
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    images = None
                    if include_image:
                        images = _image_tensor(
                            image_name,
                            transform,
                            device,
                            "blank" if variant == "blank_image" else "normal",
                        )
                    with torch.no_grad():
                        generated = model.generate(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=int(max_new_tokens),
                            do_sample=False,
                        )
                    new_tokens = generated[0, input_ids.shape[1]:]
                    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    rows.append(
                        {
                            **probe,
                            "prompt": prompt_name,
                            "variant": variant,
                            "max_new_tokens": int(max_new_tokens),
                            "response": text,
                            "generated_tokens": int(new_tokens.shape[0]),
                        }
                    )
                    print(
                        f"{probe['image']} {prompt_name} {variant} "
                        f"{max_new_tokens}: {text[:100]}"
                    )
    return {"checkpoint": STAGE2_CKPT, "results": rows}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe_checkpoint(
    checkpoint: str,
    label: str = "checkpoint",
    max_new_tokens_values=None,
    include_strict: bool = True,
):
    import sys
    import torch

    sys.path.insert(0, "/root/anymal")
    from data.data_utils import get_vision_transform

    if max_new_tokens_values is None:
        max_new_tokens_values = [32, 64, 128, 192]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_v2_finetune_model(checkpoint, device)
    transform = get_vision_transform(
        vision_encoder_type="siglip2",
        vision_model_name="google/siglip2-so400m-patch14-384",
        image_size=384,
        is_train=False,
        use_augmentation=False,
    )

    probes = [
        probe for probe in TARGETED_PROBES
        if os.path.exists(os.path.join(COCO_IMAGE_DIR, probe["image"]))
    ]
    prompts = {"training": TRAINING_SYSTEM_PROMPT}
    if include_strict:
        prompts["strict"] = STRICT_SYSTEM_PROMPT

    rows = []
    for probe in probes:
        wrong_image = next(
            (other["image"] for other in probes if other["image"] != probe["image"]),
            probe["image"],
        )
        for prompt_name, system_prompt in prompts.items():
            for max_new_tokens in max_new_tokens_values:
                for variant, image_name, include_image in (
                    ("correct_image", probe["image"], True),
                    ("wrong_image", wrong_image, True),
                    ("blank_image", probe["image"], True),
                    ("text_only", probe["image"], False),
                ):
                    input_ids, attention_mask = _prompt_inputs(
                        model,
                        probe["question"],
                        system_prompt,
                        include_image=include_image,
                    )
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    images = None
                    if include_image:
                        images = _image_tensor(
                            image_name,
                            transform,
                            device,
                            "blank" if variant == "blank_image" else "normal",
                        )
                    with torch.no_grad():
                        generated = model.generate(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=int(max_new_tokens),
                            do_sample=False,
                        )
                    new_tokens = generated[0, input_ids.shape[1]:]
                    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    rows.append(
                        {
                            **probe,
                            "label": label,
                            "prompt": prompt_name,
                            "variant": variant,
                            "max_new_tokens": int(max_new_tokens),
                            "response": text,
                            "generated_tokens": int(new_tokens.shape[0]),
                        }
                    )
                    print(
                        f"{label} {probe['image']} {prompt_name} {variant} "
                        f"{max_new_tokens}: {text[:100]}"
                    )
    return {"checkpoint": checkpoint, "label": label, "results": rows}


@app.local_entrypoint()
def main(
    mode: str = "audit",
    output: str = "v2_quality_diagnostics.json",
    max_new_tokens: str = "32,64,128,192",
    include_strict: bool = True,
    checkpoint: str = STAGE2_CKPT,
    label: str = "checkpoint",
):
    if mode == "audit":
        result = audit_v2.remote()
    elif mode == "probe":
        values = [int(part) for part in max_new_tokens.split(",") if part.strip()]
        result = probe_stage2.remote(
            max_new_tokens_values=values,
            include_strict=include_strict,
        )
    elif mode == "probe-checkpoint":
        values = [int(part) for part in max_new_tokens.split(",") if part.strip()]
        result = probe_checkpoint.remote(
            checkpoint=checkpoint,
            label=label,
            max_new_tokens_values=values,
            include_strict=include_strict,
        )
    else:
        raise ValueError("mode must be 'audit' or 'probe'")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
