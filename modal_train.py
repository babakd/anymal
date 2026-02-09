"""
Modal Training Script for AnyMAL

Usage:
    1. Install Modal: pip install modal
    2. Setup Modal: modal setup
    3. Add HuggingFace secret: modal secret create huggingface HF_TOKEN=<your-token>
    4. Run training: modal run modal_train.py

Options:
    modal run modal_train.py --max-steps 100    # Quick test
    modal run modal_train.py --max-steps 1000   # Longer run
    modal run modal_train.py --stage pretrain   # Stage 1 pretraining
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("anymal-training")

# Create a volume to persist model weights between runs
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)

# Get the local project directory
PROJECT_DIR = Path(__file__).parent

# Define the container image with all dependencies + local code
# Layer order optimized: stable layers first (apt, pip), then local code mounted at runtime
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Stable - rarely changes
    .pip_install(  # Stable - changes weekly at most
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "datasets>=2.15.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    # Mount local code - changes frequently but won't invalidate pip cache
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


@app.cls(
    image=image,
    gpu="A100-80GB",  # Use A100 80GB for large models
    timeout=7200,  # 2 hour timeout
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
class Trainer:
    """
    AnyMAL Trainer class with lifecycle hooks for efficient Modal execution.

    Uses @modal.enter() to load model once per container, avoiding redundant
    model initialization on subsequent calls to the same warm container.
    """

    @modal.enter()
    def setup(self):
        """
        Called once when container starts. Downloads weights and initializes model.
        Subsequent calls to train() reuse the loaded model.
        """
        import sys
        sys.path.insert(0, "/root/anymal")

        print("=" * 60)
        print("Container startup - loading model (runs once per container)")
        print("=" * 60)

        # Download LLaMA weights if not cached in volume
        self.llama_path = "/checkpoints/llama3-8b-instruct"
        if not os.path.exists(os.path.join(self.llama_path, "config.json")):
            print("Downloading LLaMA-3-8B-Instruct weights...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                local_dir=self.llama_path,
                local_dir_use_symlinks=False,
            )
            volume.commit()
            print(f"LLaMA weights saved to {self.llama_path}")
        else:
            print(f"Using cached LLaMA weights from {self.llama_path}")

        # Download LLaVA dataset JSON if not cached
        self._ensure_llava_data_cached()

        # Download VQA evaluation data if not cached (non-fatal if it fails)
        try:
            self._ensure_vqa_data_cached()
        except Exception as e:
            print(f"Warning: VQA data download failed: {e}")
            print("Training will proceed without VQA evaluation.")

        # Pre-import heavy modules
        print("Pre-importing modules...")
        from models import AnyMAL
        from data import build_dataloader, ImageTextCollator
        from training import FinetuneTrainer, PretrainTrainer

        # Store model class for later instantiation
        # (We don't create the model here because finetune vs pretrain have different configs)
        self.AnyMAL = AnyMAL
        self._model_loaded = True
        print("Container setup complete!")

    def _ensure_llava_data_cached(self):
        """Download LLaVA dataset JSON files to volume if not already cached."""
        from huggingface_hub import hf_hub_download

        cache_dir = "/checkpoints/llava_data"
        os.makedirs(cache_dir, exist_ok=True)

        # LLaVA-Instruct-150K for Stage 2
        instruct_json = os.path.join(cache_dir, "llava_instruct_150k.json")
        if not os.path.exists(instruct_json):
            print("Downloading LLaVA-Instruct-150K JSON...")
            hf_hub_download(
                repo_id="liuhaotian/LLaVA-Instruct-150K",
                filename="llava_instruct_150k.json",
                repo_type="dataset",
                local_dir=cache_dir,
            )
            volume.commit()
            print(f"Saved to {instruct_json}")
        else:
            print(f"Using cached LLaVA-Instruct JSON from {instruct_json}")

        # LLaVA-Pretrain for Stage 1 (CC3M subset)
        pretrain_json = os.path.join(cache_dir, "blip_laion_cc_sbu_558k.json")
        if not os.path.exists(pretrain_json):
            print("Downloading LLaVA-Pretrain JSON...")
            try:
                hf_hub_download(
                    repo_id="liuhaotian/LLaVA-Pretrain",
                    filename="blip_laion_cc_sbu_558k.json",
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
                volume.commit()
                print(f"Saved to {pretrain_json}")
            except Exception as e:
                print(f"Warning: Could not download pretrain JSON: {e}")
                print("Pretrain will use dummy data if real data requested.")
        else:
            print(f"Using cached LLaVA-Pretrain JSON from {pretrain_json}")

        # Download COCO images subset for real training
        self._ensure_coco_images_cached(instruct_json)

    def _ensure_coco_images_cached(self, json_path: str, num_images: int = 100000):
        """Download a subset of COCO images for training with real data."""
        import json
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        image_dir = "/checkpoints/coco_images"
        manifest_path = os.path.join(image_dir, "manifest.json")

        # Check if we already have enough images
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            existing = manifest.get("downloaded", 0) + manifest.get("skipped", 0)
            if existing >= num_images:
                print(f"Using cached COCO images ({existing} images in {image_dir})")
                return

        os.makedirs(image_dir, exist_ok=True)
        print(f"Downloading {num_images} COCO images...")

        # Get image list from LLaVA JSON
        with open(json_path) as f:
            data = json.load(f)

        seen = set()
        images = []
        for sample in data:
            img = sample.get("image")
            if img and img not in seen:
                seen.add(img)
                images.append(img)
                if len(images) >= num_images:
                    break

        COCO_BASE_URL = "http://images.cocodataset.org/train2017"

        def download_one(img_name):
            path = os.path.join(image_dir, img_name)
            if os.path.exists(path):
                return (img_name, "skip")
            try:
                resp = requests.get(f"{COCO_BASE_URL}/{img_name}", timeout=30)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                return (img_name, "ok")
            except Exception as e:
                return (img_name, f"fail: {e}")

        # Download in parallel
        ok_count, skip_count, fail_count = 0, 0, 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_one, img) for img in images]
            for i, future in enumerate(as_completed(futures)):
                img_name, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 500 == 0:
                    print(f"  Progress: {i+1}/{len(images)} (ok={ok_count}, skip={skip_count}, fail={fail_count})")

        print(f"COCO images: downloaded={ok_count}, cached={skip_count}, failed={fail_count}")

        # Save manifest
        manifest = {
            "downloaded": ok_count,
            "skipped": skip_count,
            "failed": fail_count,
            "total": len(images),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        volume.commit()
        print(f"COCO images cached to {image_dir}")

    def _ensure_vqa_data_cached(self, num_val_images: int = 500):
        """Download VQAv2 evaluation data and a subset of COCO val2014 images."""
        import json
        import zipfile
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        vqa_dir = "/checkpoints/vqa_data"
        val_image_dir = "/checkpoints/coco_val2014"
        os.makedirs(vqa_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)

        # Download VQAv2 questions
        questions_path = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_val2014_questions.json")
        if not os.path.exists(questions_path):
            print("Downloading VQAv2 val2014 questions...")
            url = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                zip_path = os.path.join(vqa_dir, "questions.zip")
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(vqa_dir)
                os.remove(zip_path)
                volume.commit()
                print(f"VQA questions saved to {vqa_dir}")
            except Exception as e:
                print(f"Warning: Could not download VQA questions: {e}")
        else:
            print(f"Using cached VQA questions from {vqa_dir}")

        # Download VQAv2 annotations
        annotations_path = os.path.join(vqa_dir, "v2_mscoco_val2014_annotations.json")
        if not os.path.exists(annotations_path):
            print("Downloading VQAv2 val2014 annotations...")
            url = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                zip_path = os.path.join(vqa_dir, "annotations.zip")
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(vqa_dir)
                os.remove(zip_path)
                volume.commit()
                print(f"VQA annotations saved to {vqa_dir}")
            except Exception as e:
                print(f"Warning: Could not download VQA annotations: {e}")
        else:
            print(f"Using cached VQA annotations from {vqa_dir}")

        # Download a subset of COCO val2014 images for VQA evaluation
        manifest_path = os.path.join(val_image_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("downloaded", 0) + manifest.get("skipped", 0) >= num_val_images:
                print(f"Using cached COCO val2014 images ({manifest.get('downloaded', 0) + manifest.get('skipped', 0)} images)")
                return

        # Get image IDs from VQA questions to know which images to download
        if not os.path.exists(questions_path):
            print("Warning: VQA questions not available, skipping val image download")
            return

        with open(questions_path) as f:
            questions = json.load(f)

        # Get unique image IDs
        seen_ids = set()
        image_ids = []
        for q in questions["questions"]:
            img_id = q["image_id"]
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                image_ids.append(img_id)
                if len(image_ids) >= num_val_images:
                    break

        COCO_VAL_URL = "http://images.cocodataset.org/val2014"

        def download_one(img_id):
            filename = f"COCO_val2014_{img_id:012d}.jpg"
            path = os.path.join(val_image_dir, filename)
            if os.path.exists(path):
                return (filename, "skip")
            try:
                resp = requests.get(f"{COCO_VAL_URL}/{filename}", timeout=30)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                return (filename, "ok")
            except Exception as e:
                return (filename, f"fail: {e}")

        print(f"Downloading {len(image_ids)} COCO val2014 images for VQA eval...")
        ok_count, skip_count, fail_count = 0, 0, 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_one, img_id) for img_id in image_ids]
            for i, future in enumerate(as_completed(futures)):
                _, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Val images: {i+1}/{len(image_ids)} (ok={ok_count}, skip={skip_count}, fail={fail_count})")

        print(f"COCO val2014: downloaded={ok_count}, cached={skip_count}, failed={fail_count}")

        manifest = {"downloaded": ok_count, "skipped": skip_count, "failed": fail_count, "total": len(image_ids)}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        volume.commit()

    @modal.method()
    def train(
        self,
        max_steps: int = 100,
        stage: str = "finetune",
        learning_rate: float = None,
        batch_size: int = 4,
        use_wandb: bool = False,
        use_dummy_data: bool = False,
        wandb_api_key: str = None,
        track_per_layer_grad_norms: bool = True,
        run_eval_benchmarks: bool = True,
        pretrain_checkpoint: str = None,
    ):
        """
        Run AnyMAL training on Modal.

        Args:
            max_steps: Number of training steps
            stage: "pretrain" for Stage 1, "finetune" for Stage 2
            learning_rate: Learning rate (default: 1e-5 for finetune, 2e-4 for pretrain)
            batch_size: Per-device batch size
            use_wandb: Enable Weights & Biases logging
            use_dummy_data: Use dummy data instead of real dataset (for testing)
            wandb_api_key: Weights & Biases API key (optional, pass directly)
            pretrain_checkpoint: Path to Stage 1 checkpoint for Stage 2 (auto-discovered if None)
        """
        import sys
        sys.path.insert(0, "/root/anymal")

        # Setup W&B if requested
        if use_wandb:
            import wandb
            api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
                print("Weights & Biases enabled!")
            else:
                print("WARNING: use_wandb=True but no WANDB_API_KEY found. Disabling W&B.")
                use_wandb = False

        from training.distributed import print_rank_0

        print_rank_0("=" * 60)
        print_rank_0(f"AnyMAL Training on Modal")
        print_rank_0(f"Stage: {stage}")
        print_rank_0(f"Max steps: {max_steps}")
        print_rank_0(f"Batch size: {batch_size}")
        print_rank_0(f"Data: {'dummy' if use_dummy_data else 'LLaVA'}")
        print_rank_0("=" * 60)

        if stage == "finetune":
            lr = learning_rate or 1e-5
            run_finetune(
                llama_path=self.llama_path,
                max_steps=max_steps,
                learning_rate=lr,
                batch_size=batch_size,
                use_wandb=use_wandb,
                use_dummy_data=use_dummy_data,
                track_per_layer_grad_norms=track_per_layer_grad_norms,
                run_eval_benchmarks=run_eval_benchmarks,
                pretrain_checkpoint=pretrain_checkpoint,
            )
        else:
            lr = learning_rate or 2e-4
            run_pretrain(
                llama_path=self.llama_path,
                max_steps=max_steps,
                learning_rate=lr,
                batch_size=batch_size,
                use_wandb=use_wandb,
                use_dummy_data=use_dummy_data,
            )

        # Save outputs to volume
        volume.commit()
        print("Training complete! Outputs saved to volume.")


def _diagnose_model(model):
    """Print diagnostic info about model configuration."""
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)

    # Tokenizer / pad token
    tok = model.tokenizer
    print(f"  Tokenizer vocab size: {len(tok)}")
    print(f"  pad_token: {repr(tok.pad_token)} (id={tok.pad_token_id})")
    print(f"  eos_token: {repr(tok.eos_token)} (id={tok.eos_token_id})")
    print(f"  pad_token == eos_token? {tok.pad_token_id == tok.eos_token_id}")
    if tok.pad_token_id == tok.eos_token_id:
        print("  WARNING: pad_token equals eos_token - EOS labels will be masked!")

    # Image placeholder token
    placeholder_id = getattr(model, "image_placeholder_token_id", None)
    if placeholder_id is not None:
        placeholder_str = tok.decode([placeholder_id])
        print(f"  image_placeholder_token_id: {placeholder_id} ({repr(placeholder_str)})")
    else:
        print("  image_placeholder_token_id: None (will prepend image tokens)")

    # Parameter counts by component
    groups = {"projector": [0, 0], "lora": [0, 0], "vision": [0, 0], "other": [0, 0]}
    for name, param in model.named_parameters():
        total = param.numel()
        trainable = total if param.requires_grad else 0
        if "projector" in name:
            groups["projector"][0] += total
            groups["projector"][1] += trainable
        elif "lora" in name.lower():
            groups["lora"][0] += total
            groups["lora"][1] += trainable
        elif "image_encoder" in name:
            groups["vision"][0] += total
            groups["vision"][1] += trainable
        else:
            groups["other"][0] += total
            groups["other"][1] += trainable

    print("\n  Parameter counts (total / trainable):")
    for g, (t, tr) in groups.items():
        print(f"    {g:12s}: {t:>12,} / {tr:>12,}")

    print("=" * 60 + "\n")


def _diagnose_dataset_sample(dataset, tokenizer, num_samples=3):
    """Sample a few items from the dataset and log diagnostics."""
    import torch
    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTICS")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img = sample["image"]
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        labels = sample["labels"]

        total_tokens = mask.sum().item()
        supervised = (labels != -100).sum().item()
        pad_count = (ids == tokenizer.pad_token_id).sum().item()

        # Check if any image placeholder tokens exist
        placeholder_id = getattr(dataset, "image_placeholder_token_id", None)
        if placeholder_id is None:
            # try the tokenizer vocab
            vocab = tokenizer.get_vocab()
            for c in ["<|reserved_special_token_0|>", "<|image|>"]:
                if c in vocab:
                    placeholder_id = vocab[c]
                    break
        has_placeholder = (ids == placeholder_id).sum().item() if placeholder_id else 0

        print(f"\n  Sample {i}:")
        print(f"    image shape: {list(img.shape)}, range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"    input_ids shape: {list(ids.shape)}, non-pad: {total_tokens}, pad: {pad_count}")
        print(f"    labels: {supervised} supervised / {total_tokens} non-pad tokens ({100*supervised/max(total_tokens,1):.1f}%)")
        print(f"    image placeholder tokens: {has_placeholder}")

        # Decode a small window of supervised tokens to sanity-check
        supervised_mask = labels != -100
        if supervised_mask.any():
            first_sup_idx = supervised_mask.nonzero(as_tuple=True)[0][0].item()
            supervised_window = ids[first_sup_idx:first_sup_idx+20]
            decoded = tokenizer.decode(supervised_window, skip_special_tokens=False)
            print(f"    first supervised tokens: {repr(decoded[:100])}")

        # Check that eos_token is NOT masked (should be supervised if at end of response)
        eos_id = tokenizer.eos_token_id
        eot_vocab = tokenizer.get_vocab()
        eot_id = eot_vocab.get("<|eot_id|>", None)
        if eot_id is not None:
            eot_positions = (ids == eot_id).nonzero(as_tuple=True)[0]
            if len(eot_positions) > 0:
                eot_supervised = sum(1 for pos in eot_positions if labels[pos] != -100)
                print(f"    <|eot_id|> tokens: {len(eot_positions)} total, {eot_supervised} supervised")

    print("=" * 60 + "\n")


def _diagnose_batch(batch, tokenizer, step_name="first batch"):
    """Log diagnostics for a collated batch."""
    print(f"\n--- Batch diagnostics ({step_name}) ---")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}, device={val.device}")
    if "labels" in batch:
        labels = batch["labels"]
        supervised = (labels != -100).sum().item()
        total = labels.numel()
        print(f"  labels: {supervised} supervised / {total} total ({100*supervised/max(total,1):.1f}%)")
    if "images" in batch:
        imgs = batch["images"]
        print(f"  images range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print("---\n")


def run_finetune(llama_path, max_steps, learning_rate, batch_size, use_wandb, use_dummy_data,
                  track_per_layer_grad_norms=True, run_eval_benchmarks=True,
                  pretrain_checkpoint=None):
    """Run Stage 2 fine-tuning with real COCO images."""
    import torch
    from models import AnyMAL
    from data import build_dataloader, ImageTextCollator
    from data.dataset_splitter import deterministic_train_val_split
    from training import FinetuneTrainer
    from training.finetune import FinetuneConfig

    if use_dummy_data:
        print("WARNING: --use-dummy-data was passed but all training must use real images.")
        print("Ignoring --use-dummy-data flag and loading real COCO images.")

    # Initialize model
    print("Initializing model...")
    model = AnyMAL(
        llm_model_name=llama_path,
        vision_model_name="ViT-L-14",
        vision_pretrained="openai",
        projector_type="perceiver",
        num_image_tokens=64,
        use_qlora=True,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=True,
        use_flash_attention=False,  # Skip flash-attn, use SDPA instead
    )

    # Diagnose model
    _diagnose_model(model)

    # Always load real images - never use dummy data
    print("Loading dataset with real COCO images...")
    dataset = load_llava_instruct_dataset(model.tokenizer)

    # Split into train/val
    train_dataset, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    print(f"Dataset split: {len(train_dataset):,} train / {len(val_dataset):,} val")

    # Diagnose dataset
    _diagnose_dataset_sample(train_dataset, model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=512,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        distributed=False,
        collate_fn=collator,
    )

    eval_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        distributed=False,
        collate_fn=collator,
    )

    # Diagnose one collated batch
    print("Sampling one batch for diagnostics...")
    diag_iter = iter(train_dataloader)
    diag_batch = next(diag_iter)
    _diagnose_batch(diag_batch, model.tokenizer, "pre-training sample")

    # Auto-discover pretrain checkpoint if not explicitly provided
    if pretrain_checkpoint is None:
        pretrain_dir = "/checkpoints/pretrain-output"
        if os.path.exists(pretrain_dir):
            # Find the latest checkpoint
            candidates = []
            for entry in os.listdir(pretrain_dir):
                if entry.startswith("checkpoint-"):
                    ckpt_path = os.path.join(pretrain_dir, entry)
                    projector_path = os.path.join(ckpt_path, "projector.pt")
                    if os.path.exists(projector_path):
                        step_str = entry.split("-", 1)[1]
                        try:
                            candidates.append((int(step_str), ckpt_path))
                        except ValueError:
                            continue
            if candidates:
                candidates.sort(key=lambda x: x[0])
                pretrain_checkpoint = candidates[-1][1]
                print(f"Auto-discovered pretrain checkpoint: {pretrain_checkpoint}")

    if pretrain_checkpoint:
        print(f"Will load Stage 1 projector from: {pretrain_checkpoint}")
    else:
        print("WARNING: No pretrain checkpoint found. Perceiver resampler has random weights.")
        print("Stage 1 pretraining is strongly recommended before Stage 2.")

    # Create trainer config
    eval_steps = max(50, max_steps // 10)  # ~10 eval points during training
    config = FinetuneConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=min(100, max_steps // 10),
        weight_decay=0.01,
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=1,  # Log every step for close monitoring
        save_steps=max(100, max_steps // 4),
        eval_steps=eval_steps,
        max_eval_batches=200,  # Clip eval to 200 batches (~55s) during training
        output_dir="/checkpoints/finetune-output",
        use_wandb=use_wandb,
        wandb_project="anymal-finetune",
        track_per_layer_grad_norms=track_per_layer_grad_norms,
        pretrain_checkpoint=pretrain_checkpoint,
    )

    # Train
    trainer = FinetuneTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Set up eval runner for VQA benchmarks
    eval_runner = None
    if run_eval_benchmarks:
        try:
            from evaluation.eval_runner import EvalRunner
            vqa_questions = "/checkpoints/vqa_data/v2_OpenEnded_mscoco_val2014_questions.json"
            vqa_annotations = "/checkpoints/vqa_data/v2_mscoco_val2014_annotations.json"
            vqa_image_dir = "/checkpoints/coco_val2014"
            if os.path.exists(vqa_questions) and os.path.exists(vqa_annotations) and os.path.exists(vqa_image_dir):
                eval_runner = EvalRunner(
                    model=model,
                    vqa_questions_file=vqa_questions,
                    vqa_annotations_file=vqa_annotations,
                    vqa_image_dir=vqa_image_dir,
                )
                print(f"VQA eval runner initialized (will run every {eval_steps} steps)")
            else:
                print("VQA data not found, skipping benchmark evaluation")
        except Exception as e:
            print(f"Could not initialize eval runner: {e}")

    metrics = trainer.train()

    # Run final VQA eval
    if eval_runner is not None:
        print("\nRunning final VQA evaluation...")
        try:
            vqa_metrics = eval_runner.run(["vqa"])
            if vqa_metrics and trainer.logger is not None:
                trainer.logger.log(vqa_metrics)
            print(f"Final VQA metrics: {vqa_metrics}")
        except Exception as e:
            print(f"Final VQA eval failed: {e}")

    print(f"\nTraining complete! Final metrics: {metrics}")


def run_pretrain(llama_path, max_steps, learning_rate, batch_size, use_wandb,
                  use_dummy_data=False, distributed=False, resume_checkpoint=None):
    """Run Stage 1 pretraining with real COCO images."""
    import torch
    from models import AnyMAL
    from data import build_dataloader, ImageTextCollator
    from data.dataset_splitter import deterministic_train_val_split
    from training import PretrainTrainer
    from training.pretrain import PretrainConfig

    if use_dummy_data:
        print("WARNING: --use-dummy-data was passed but Stage 1 must use real images.")
        print("Ignoring --use-dummy-data flag and loading real COCO images.")

    # Initialize model (no LoRA for pretraining)
    print("Initializing model...")
    # For DDP, disable device_map so each process places model on its own GPU
    device_map = None if distributed else "auto"
    model = AnyMAL(
        llm_model_name=llama_path,
        vision_model_name="ViT-L-14",
        vision_pretrained="openai",
        projector_type="perceiver",
        num_image_tokens=64,
        use_qlora=False,  # No LoRA for Stage 1
        gradient_checkpointing=True,
        use_flash_attention=False,  # Skip flash-attn, use SDPA instead
        llm_device_map=device_map,
    )

    # Configure for Stage 1: only train projector
    model.set_training_stage(1)

    # Diagnose model (rank 0 only)
    from training.distributed import is_main_process as _is_main
    if _is_main():
        _diagnose_model(model)

    # Always load real images
    print("Loading dataset with real COCO images...")
    dataset = load_llava_pretrain_dataset(model.tokenizer)

    # Split into train/val
    train_dataset, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    print(f"Dataset split: {len(train_dataset):,} train / {len(val_dataset):,} val")

    # Diagnose dataset (rank 0 only)
    if _is_main():
        _diagnose_dataset_sample(train_dataset, model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=256,
    )

    # When using mp.spawn for DDP, num_workers>0 causes nested fork issues
    dl_workers = 0 if distributed else 4

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dl_workers,
        distributed=distributed,
        collate_fn=collator,
    )

    eval_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if distributed else 2,
        distributed=False,
        collate_fn=collator,
    )

    # Diagnose one collated batch (only on rank 0)
    from training.distributed import is_main_process
    if is_main_process():
        print("Sampling one batch for diagnostics...")
        diag_iter = iter(train_dataloader)
        diag_batch = next(diag_iter)
        _diagnose_batch(diag_batch, model.tokenizer, "pre-training sample")

    # Create trainer config
    eval_steps = max(50, max_steps // 10)
    config = PretrainConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        warmup_steps=min(100, max_steps // 10),
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=10,
        save_steps=250,
        save_total_limit=5,
        eval_steps=eval_steps,
        max_eval_batches=200,
        output_dir="/checkpoints/pretrain-output",
        use_wandb=use_wandb,
        wandb_project="anymal-pretrain",
        resume_from_checkpoint=resume_checkpoint,
    )

    # Train
    trainer = PretrainTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    metrics = trainer.train()
    print(f"Training complete! Final metrics: {metrics}")


def load_llava_instruct_dataset(tokenizer):
    """
    Load LLaVA-Instruct-150K dataset using cached JSON from volume.

    Uses the existing InstructionDataset class with image_dir=None
    (dummy images) since downloading all COCO images is impractical.

    The JSON is pre-cached during container setup via @modal.enter().
    """
    from data.instruction_dataset import InstructionDataset

    # JSON should already be cached by Trainer.setup()
    json_path = "/checkpoints/llava_data/llava_instruct_150k.json"

    if not os.path.exists(json_path):
        # Fallback: download now if not cached (shouldn't happen with lifecycle hooks)
        print("Warning: LLaVA JSON not pre-cached, downloading now...")
        from huggingface_hub import hf_hub_download
        cache_dir = "/checkpoints/llava_data"
        os.makedirs(cache_dir, exist_ok=True)
        hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
            local_dir=cache_dir,
        )

    print(f"Loading LLaVA-Instruct-150K from {json_path}")

    # Require real COCO images - no dummy image fallback
    image_dir = "/checkpoints/coco_images"
    if not os.path.exists(image_dir):
        raise RuntimeError(
            f"COCO images directory not found at {image_dir}. "
            "Real images are required for training. "
            "The container setup should have downloaded them via _ensure_coco_images_cached()."
        )

    num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"Found {num_images} real COCO images in {image_dir}")
    if num_images == 0:
        raise RuntimeError(f"No JPEG images found in {image_dir}. Cannot train without real images.")

    print("Filtering dataset to only samples with real images")

    dataset = InstructionDataset(
        data_path=json_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        image_size=224,
        max_length=512,
        filter_to_available_images=True,  # Only use samples with real images
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset is empty after filtering. No LLaVA samples matched the "
            f"{num_images} available COCO images. Check image filenames."
        )

    print(f"Loaded {len(dataset)} instruction samples with real images")
    return dataset


class COCOCaptionDataset:
    """Caption dataset with real COCO images for Stage 1 pretraining.

    Defined at module level so it can be pickled by DataLoader workers.
    """

    def __init__(self, samples, image_dir, tokenizer):
        from data.data_utils import get_image_transform, TextProcessor
        self.samples = samples
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = get_image_transform(image_size=224, is_train=True)
        self.text_processor = TextProcessor(tokenizer, max_length=256)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        item = self.samples[idx]

        # Load real image
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = item["caption"]
        encoding = self.text_processor.encode_text(caption)

        # For captioning, predict all non-pad tokens
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "image": image,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def load_llava_pretrain_dataset(tokenizer):
    """
    Load captioning dataset for Stage 1 alignment using real COCO images.

    Uses llava_instruct_150k.json (which references COCO images we have cached)
    and extracts the first GPT response as a caption. Filters to samples whose
    images exist in /checkpoints/coco_images.
    """
    import json

    json_path = "/checkpoints/llava_data/llava_instruct_150k.json"
    image_dir = "/checkpoints/coco_images"

    if not os.path.exists(json_path):
        raise RuntimeError(
            f"LLaVA JSON not found at {json_path}. "
            "Run training setup first to cache data."
        )
    if not os.path.exists(image_dir):
        raise RuntimeError(
            f"COCO images not found at {image_dir}. "
            "Run training setup first to cache images."
        )

    print(f"Loading pretrain captions from {json_path}")

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Build set of available images
    available_images = set(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    print(f"Found {len(available_images)} COCO images in {image_dir}")

    # Filter to samples with available images and extract captions
    samples = []
    for item in raw_data:
        img_name = item.get("image", "")
        if img_name not in available_images:
            continue
        # Extract first GPT response as caption
        conversations = item.get("conversations", [])
        caption = ""
        for conv in conversations:
            if conv.get("from") == "gpt":
                caption = conv.get("value", "")
                break
        if not caption:
            continue
        samples.append({"image": img_name, "caption": caption})

    print(f"Filtered to {len(samples)} samples with real images")
    if len(samples) == 0:
        raise RuntimeError("No pretrain samples matched available COCO images.")

    dataset = COCOCaptionDataset(samples, image_dir, tokenizer)
    print(f"Loaded {len(dataset)} pretrain samples with real COCO images")
    return dataset


def create_dummy_instruction_dataset(tokenizer, num_samples=1000):
    """Create a dummy instruction dataset for testing."""
    import torch
    from torch.utils.data import Dataset
    from data.data_utils import get_image_transform, TextProcessor

    class DummyInstructionDataset(Dataset):
        def __init__(self, tokenizer, num_samples):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.transform = get_image_transform(image_size=224, is_train=True)
            self.text_processor = TextProcessor(tokenizer, max_length=512)
            from data.data_utils import CLIP_MEAN, CLIP_STD
            from torchvision import transforms as T
            self._clip_normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = self._clip_normalize(torch.rand(3, 224, 224))
            conversations = [
                {"role": "user", "content": f"What do you see in this image? (sample {idx})"},
                {"role": "assistant", "content": "I see various objects in this image."},
            ]
            text, response_start = self.text_processor.format_conversation(conversations)
            encoding = self.text_processor.encode_for_training(text, response_start)
            return {
                "image": image,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["labels"],
            }

    return DummyInstructionDataset(tokenizer, num_samples)


def create_dummy_caption_dataset(tokenizer, num_samples=1000):
    """Create a dummy caption dataset for testing."""
    import torch
    from torch.utils.data import Dataset
    from data.data_utils import get_image_transform, TextProcessor

    class DummyCaptionDataset(Dataset):
        def __init__(self, tokenizer, num_samples):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.text_processor = TextProcessor(tokenizer, max_length=256)
            from data.data_utils import CLIP_MEAN, CLIP_STD
            from torchvision import transforms as T
            self._clip_normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = self._clip_normalize(torch.rand(3, 224, 224))
            caption = f"A photograph showing various objects, sample {idx}."
            encoding = self.text_processor.encode_text(caption)
            labels = encoding["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "image": image,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }

    return DummyCaptionDataset(tokenizer, num_samples)


def _pretrain_worker(local_rank, world_size, config):
    """Worker function for distributed Stage 1 pretraining."""
    import sys
    sys.path.insert(0, "/root/anymal")

    # Set environment variables for distributed training
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    import torch
    torch.cuda.set_device(local_rank)

    from training.distributed import setup_distributed, cleanup_distributed

    setup_distributed()

    try:
        run_pretrain(
            llama_path=config["llama_path"],
            max_steps=config["max_steps"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            use_wandb=config["use_wandb"] and local_rank == 0,  # Only rank 0 logs
            distributed=True,
            resume_checkpoint=config.get("resume_checkpoint"),
        )
    finally:
        cleanup_distributed()


@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=14400,  # 4 hour timeout
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def pretrain_distributed(max_steps, learning_rate, batch_size, use_wandb, wandb_api_key=None, resume_checkpoint=None):
    """Run Stage 1 pretraining on 4 GPUs using DDP."""
    import sys
    sys.path.insert(0, "/root/anymal")
    import torch
    import torch.multiprocessing as mp

    # Setup W&B
    if use_wandb:
        import wandb
        api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        else:
            print("WARNING: use_wandb=True but no WANDB_API_KEY found. Disabling W&B.")
            use_wandb = False

    # Ensure data is cached
    llama_path = "/checkpoints/llama3-8b-instruct"
    if not os.path.exists(os.path.join(llama_path, "config.json")):
        print("Downloading LLaMA-3-8B-Instruct weights...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            local_dir=llama_path,
            local_dir_use_symlinks=False,
        )
        volume.commit()

    num_gpus = torch.cuda.device_count()
    print(f"Starting distributed pretraining on {num_gpus} GPUs")

    config = {
        "llama_path": llama_path,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "use_wandb": use_wandb,
        "resume_checkpoint": resume_checkpoint,
    }

    mp.spawn(_pretrain_worker, nprocs=num_gpus, args=(num_gpus, config))

    volume.commit()
    print("Distributed pretraining complete! Outputs saved to volume.")


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    stage: str = "finetune",
    learning_rate: float = None,
    batch_size: int = 4,
    use_wandb: bool = False,
    use_dummy_data: bool = False,
    wandb_api_key: str = None,
    track_per_layer_grad_norms: bool = True,
    run_eval_benchmarks: bool = True,
    pretrain_checkpoint: str = None,
    resume_checkpoint: str = None,
):
    """
    Entry point for Modal training.

    Examples:
        modal run modal_train.py                              # Quick test with LLaVA data
        modal run modal_train.py --use-dummy-data             # Test with dummy data
        modal run modal_train.py --max-steps 500              # Longer run
        modal run modal_train.py --stage pretrain             # Stage 1 (4 GPUs)
        modal run modal_train.py --stage finetune             # Stage 2 (auto-discovers pretrain ckpt)
        modal run modal_train.py --use-wandb --wandb-api-key YOUR_KEY  # With W&B
        modal run modal_train.py --stage pretrain --resume-checkpoint /checkpoints/pretrain-output/checkpoint-250
    """
    print(f"Starting AnyMAL training on Modal...")
    print(f"  Stage: {stage}")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {'dummy' if use_dummy_data else 'LLaVA'}")
    print(f"  W&B: {'enabled' if use_wandb else 'disabled'}")
    print(f"  Per-layer grad norms: {track_per_layer_grad_norms}")
    print(f"  Eval benchmarks: {run_eval_benchmarks}")
    if pretrain_checkpoint:
        print(f"  Pretrain checkpoint: {pretrain_checkpoint}")
    if resume_checkpoint:
        print(f"  Resume from: {resume_checkpoint}")

    if stage == "pretrain":
        # Stage 1 uses multi-GPU distributed pretraining
        lr = learning_rate or 2e-4
        pretrain_distributed.remote(
            max_steps=max_steps,
            learning_rate=lr,
            batch_size=batch_size,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            resume_checkpoint=resume_checkpoint,
        )
    else:
        # Stage 2 uses single-GPU with QLoRA
        trainer = Trainer()
        trainer.train.remote(
            max_steps=max_steps,
            stage=stage,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_wandb=use_wandb,
            use_dummy_data=use_dummy_data,
            wandb_api_key=wandb_api_key,
            track_per_layer_grad_norms=track_per_layer_grad_norms,
            run_eval_benchmarks=run_eval_benchmarks,
            pretrain_checkpoint=pretrain_checkpoint,
        )
