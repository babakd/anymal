"""Held-out VQAv2 evaluation for AnyMAL checkpoints.

This is intentionally separate from the tiny qualitative probes. It evaluates
checkpoints on a deterministic VQAv2 val2014 subset with answer-type breakdowns
so checkpoint selection is not driven by hand-picked examples.
"""

import json
import os
import random
from pathlib import Path

import modal


app = modal.App("anymal-vqa-checkpoint-eval")
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
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "einops>=0.7.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)

LLAMA_PATH = "/checkpoints/llama3-8b-instruct"
V1_CKPT = "/checkpoints/finetune-output/ablation-F/checkpoint-500"
V2_FINAL_CKPT = "/checkpoints/finetune-output/v2-stage2-balanced-mix-3000-20260428/checkpoint-3000"
VQA_QUESTIONS = "/checkpoints/vqa_data/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS = "/checkpoints/vqa_data/v2_mscoco_val2014_annotations.json"
DEFAULT_IMAGE_DIR = "/checkpoints/coco_val2014"
IMAGE_PERTURBATIONS = {
    "none",
    "resize_up",
    "mild_blur",
    "center_crop_90",
    "translate_5pct",
}


def _default_runs(
    candidate_checkpoint=None,
    candidate_label=None,
    candidate_architecture="v2",
    include_baselines=True,
):
    runs = []
    if include_baselines:
        runs.extend(
            [
                {
                    "key": "v1_ablation_f",
                    "label": "V1 ablation-F checkpoint 500",
                    "architecture": "v1",
                    "checkpoint": V1_CKPT,
                },
                {
                    "key": "v2_final",
                    "label": "V2 final balanced-mix checkpoint 3000",
                    "architecture": "v2",
                    "checkpoint": V2_FINAL_CKPT,
                },
            ]
        )
    if candidate_checkpoint:
        candidate_architecture = str(candidate_architecture).lower()
        architecture_aliases = {
            "anymal_v2": "v2",
            "anymal_v3": "v3",
            "anymal_v4": "v4",
        }
        candidate_architecture = architecture_aliases.get(
            candidate_architecture,
            candidate_architecture,
        )
        if candidate_architecture not in {"v2", "v3", "v4"}:
            raise ValueError(
                f"Unsupported candidate_architecture '{candidate_architecture}'. "
                "Expected 'v2', 'v3', or 'v4'."
            )
        runs.append(
            {
                "key": f"{candidate_architecture}_candidate",
                "label": candidate_label or f"{candidate_architecture.upper()} candidate",
                "architecture": candidate_architecture,
                "checkpoint": candidate_checkpoint,
            }
        )
    if not runs:
        raise ValueError(
            "No VQA eval runs requested. Provide --candidate-checkpoint "
            "or keep include_baselines=True."
        )
    return runs


def _ensure_vqa_images(questions_file, image_dir, num_images, image_sample_seed):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests

    os.makedirs(image_dir, exist_ok=True)
    manifest_path = os.path.join(image_dir, f"manifest_seed{image_sample_seed}_n{num_images}.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("requested", 0) >= num_images:
            print(f"Using existing confirmatory VQA image cache manifest: {manifest_path}")
            return

    with open(questions_file) as f:
        questions = json.load(f)["questions"]

    image_ids = sorted({int(q["image_id"]) for q in questions})
    rng = random.Random(int(image_sample_seed))
    rng.shuffle(image_ids)
    image_ids = image_ids[: int(num_images)]

    coco_val_url = "http://images.cocodataset.org/val2014"

    def download_one(img_id):
        filename = f"COCO_val2014_{img_id:012d}.jpg"
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            return filename, "skip"
        try:
            resp = requests.get(f"{coco_val_url}/{filename}", timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            return filename, "ok"
        except Exception as e:
            return filename, f"fail: {e}"

    print(
        f"Ensuring {len(image_ids)} deterministic VQA images in {image_dir} "
        f"(image_sample_seed={image_sample_seed})"
    )
    ok_count, skip_count, fail_count = 0, 0, 0
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(download_one, img_id) for img_id in image_ids]
        for i, future in enumerate(as_completed(futures)):
            _, status = future.result()
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
            if (i + 1) % 250 == 0:
                print(
                    f"  VQA images: {i + 1}/{len(image_ids)} "
                    f"(ok={ok_count}, skip={skip_count}, fail={fail_count})"
                )

    manifest = {
        "requested": int(num_images),
        "image_sample_seed": int(image_sample_seed),
        "downloaded": ok_count,
        "skipped": skip_count,
        "failed": fail_count,
        "image_ids": image_ids,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()


def _require_vqa_files(min_images, image_dir=DEFAULT_IMAGE_DIR, ensure_num_images=0, image_sample_seed=20260429):
    questions = VQA_QUESTIONS
    annotations = VQA_ANNOTATIONS
    required_paths = [questions, annotations]
    if not ensure_num_images:
        required_paths.append(image_dir)
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"Missing VQA eval files: {missing}")
    if ensure_num_images:
        _ensure_vqa_images(
            questions_file=questions,
            image_dir=image_dir,
            num_images=int(ensure_num_images),
            image_sample_seed=int(image_sample_seed),
        )
    available = [
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(available) < min_images:
        raise RuntimeError(
            f"Only {len(available)} VQA images are cached in {image_dir}; "
            f"need at least {min_images}."
        )
    return questions, annotations, image_dir


def _make_image_transform(base_transform, image_perturbation):
    image_perturbation = str(image_perturbation or "none").lower()
    if image_perturbation not in IMAGE_PERTURBATIONS:
        raise ValueError(
            f"Unknown image_perturbation '{image_perturbation}'. "
            f"Expected one of {sorted(IMAGE_PERTURBATIONS)}."
        )
    if image_perturbation == "none":
        return base_transform

    def transform(image):
        from PIL import Image, ImageFilter

        width, height = image.size
        if image_perturbation == "resize_up":
            image = image.resize(
                (max(1, int(width * 1.25)), max(1, int(height * 1.25))),
                resample=Image.Resampling.BICUBIC,
            ).resize((width, height), resample=Image.Resampling.BICUBIC)
        elif image_perturbation == "mild_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif image_perturbation == "center_crop_90":
            crop_w = max(1, int(width * 0.90))
            crop_h = max(1, int(height * 0.90))
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            image = image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height),
                resample=Image.Resampling.BICUBIC,
            )
        elif image_perturbation == "translate_5pct":
            dx = max(1, int(width * 0.05))
            dy = max(1, int(height * 0.05))
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0, 0, 0),
            )
        return base_transform(image)

    return transform


def _build_vqa_dataloader(
    model,
    architecture,
    questions,
    annotations,
    image_dir,
    max_samples,
    seed,
    batch_size,
    prompt_style,
    image_perturbation="none",
    system_prompt=None,
):
    import random
    import sys

    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, "/root/anymal")
    from data.data_utils import get_image_transform, get_vision_transform
    from evaluation.vqa_eval import VQADataset, vqa_collate_fn

    if architecture in {"v2", "v3", "v4"}:
        transform = get_vision_transform(
            vision_encoder_type="siglip2",
            vision_model_name="google/siglip2-so400m-patch14-384",
            image_size=384,
            is_train=False,
            use_augmentation=False,
        )
        placeholder_id = getattr(model, "image_placeholder_token_id", None)
        num_image_tokens = getattr(model, "num_image_tokens", 0)
    else:
        transform = get_image_transform(image_size=224, is_train=False, use_augmentation=False)
        placeholder_id = None
        num_image_tokens = 0
    transform = _make_image_transform(transform, image_perturbation)

    dataset = VQADataset(
        questions_file=questions,
        annotations_file=annotations,
        image_dir=image_dir,
        transform=transform,
        tokenizer=model.tokenizer,
        filter_to_available_images=True,
        image_placeholder_token_id=placeholder_id,
        num_image_tokens=num_image_tokens,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
    )
    if len(dataset) < max_samples:
        raise RuntimeError(f"VQA dataset has only {len(dataset)} available samples; requested {max_samples}")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    dataset = Subset(dataset, indices[:max_samples])

    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: vqa_collate_fn(b, pad_token_id),
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_vqa(
    v2_checkpoint=None,
    v2_label=None,
    candidate_checkpoint=None,
    candidate_label=None,
    candidate_architecture="v2",
    max_samples=1000,
    seed=42,
    batch_size=8,
    include_baselines=True,
    image_dir=DEFAULT_IMAGE_DIR,
    ensure_num_images=0,
    image_sample_seed=20260429,
    prompt_style="training_chat",
    image_perturbation="none",
    remote_output_path=None,
    prediction_samples=0,
    system_prompt=None,
):
    import gc
    import sys

    import torch

    sys.path.insert(0, "/root/anymal")
    from evaluation.vqa_eval import VQAEvaluator
    from v1_v2_compare_inference import _load_v1_model, _load_v2_model
    from models.anymal_v3 import AnyMALv3
    from models.anymal_v4 import AnyMALv4

    questions, annotations, image_dir = _require_vqa_files(
        min_images=max_samples,
        image_dir=image_dir,
        ensure_num_images=ensure_num_images,
        image_sample_seed=image_sample_seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    if v2_checkpoint and not candidate_checkpoint:
        candidate_checkpoint = v2_checkpoint
        candidate_label = v2_label
        candidate_architecture = "v2"

    for run in _default_runs(
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        include_baselines=bool(include_baselines),
    ):
        print(f"Evaluating {run['label']} from {run['checkpoint']}")
        if run["architecture"] == "v1":
            model = _load_v1_model(run["checkpoint"], LLAMA_PATH, device)
        elif run["architecture"] == "v2":
            model = _load_v2_model(run["checkpoint"], LLAMA_PATH, device)
        elif run["architecture"] == "v3":
            model = AnyMALv3.from_pretrained(
                run["checkpoint"],
                llm_model_name=LLAMA_PATH,
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
        elif run["architecture"] == "v4":
            from model_metadata import read_model_metadata

            meta = read_model_metadata(run["checkpoint"]) or {}
            connector_type = meta.get("connector_type", "spatial_perceiver_resampler")
            deepstack_layers = (
                meta.get("deepstack_hidden_state_indices")
                or meta.get("vision_feature_layers")
            )
            deepstack_kwargs = {}
            if connector_type == "deepstack_spatial_perceiver_resampler":
                deepstack_kwargs = {
                    "deepstack_num_feature_levels": int(
                        meta.get("deepstack_num_feature_levels")
                        or len(deepstack_layers or [])
                        or 3
                    ),
                    "deepstack_hidden_state_indices": deepstack_layers,
                }
            model = AnyMALv4.from_pretrained(
                run["checkpoint"],
                llm_model_name=LLAMA_PATH,
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
                    int(meta["connector_hidden_dim"])
                    if meta.get("connector_hidden_dim") is not None
                    else None
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
        else:
            raise ValueError(f"Unknown architecture: {run['architecture']}")

        dataloader = _build_vqa_dataloader(
            model=model,
            architecture=run["architecture"],
            questions=questions,
            annotations=annotations,
            image_dir=image_dir,
            max_samples=int(max_samples),
            seed=int(seed),
            batch_size=int(batch_size),
            prompt_style=str(prompt_style),
            image_perturbation=str(image_perturbation),
            system_prompt=system_prompt,
        )
        evaluator = VQAEvaluator(model, device=device, max_new_tokens=32)
        prediction_output = f"/tmp/{run['key']}_vqa_predictions.json" if int(prediction_samples or 0) > 0 else None
        metrics = evaluator.evaluate(dataloader, output_file=prediction_output)
        print(f"{run['key']}: {metrics}")
        result_entry = {**run, "metrics": metrics}
        if prediction_output:
            with open(prediction_output) as f:
                result_entry["prediction_samples"] = json.load(f)[: int(prediction_samples)]
        results.append(result_entry)

        del evaluator, dataloader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "max_samples": int(max_samples),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "image_dir": image_dir,
        "ensure_num_images": int(ensure_num_images),
        "image_sample_seed": int(image_sample_seed),
        "prompt_style": str(prompt_style),
        "system_prompt": system_prompt,
        "image_perturbation": str(image_perturbation),
        "runs": results,
    }
    if remote_output_path:
        remote_output_path = str(remote_output_path)
        os.makedirs(os.path.dirname(remote_output_path), exist_ok=True)
        with open(remote_output_path, "w") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        print(f"Saved remote eval result to {remote_output_path}")
    return result


@app.local_entrypoint()
def main(
    v2_checkpoint: str = None,
    v2_label: str = None,
    candidate_checkpoint: str = None,
    candidate_label: str = None,
    candidate_architecture: str = "v2",
    max_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 8,
    include_baselines: bool = True,
    image_dir: str = DEFAULT_IMAGE_DIR,
    ensure_num_images: int = 0,
    image_sample_seed: int = 20260429,
    prompt_style: str = "training_chat",
    image_perturbation: str = "none",
    output: str = "vqa_checkpoint_eval.json",
    remote_output_path: str = None,
    prediction_samples: int = 0,
    system_prompt: str = None,
):
    # Spawn + get keeps the remote call durable under `modal run --detach`
    # without relying on a foreground .remote() call from the local entrypoint.
    result = evaluate_vqa.spawn(
        v2_checkpoint=v2_checkpoint,
        v2_label=v2_label,
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=candidate_label,
        candidate_architecture=candidate_architecture,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
        include_baselines=include_baselines,
        image_dir=image_dir,
        ensure_num_images=ensure_num_images,
        image_sample_seed=image_sample_seed,
        prompt_style=prompt_style,
        image_perturbation=image_perturbation,
        remote_output_path=remote_output_path,
        prediction_samples=prediction_samples,
        system_prompt=system_prompt,
    ).get()
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")
