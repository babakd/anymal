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

        COCO_BASE_URL = "https://images.cocodataset.org/train2017"

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


def run_finetune(llama_path, max_steps, learning_rate, batch_size, use_wandb, use_dummy_data):
    """Run Stage 2 fine-tuning."""
    import torch
    from models import AnyMAL
    from data import build_dataloader, ImageTextCollator
    from training import FinetuneTrainer
    from training.finetune import FinetuneConfig

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

    # Load dataset
    print("Loading dataset...")
    if use_dummy_data:
        dataset = create_dummy_instruction_dataset(
            model.tokenizer,
            num_samples=max_steps * batch_size * 2
        )
    else:
        dataset = load_llava_instruct_dataset(model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=512,
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        distributed=False,
        collate_fn=collator,
    )

    print(f"Dataset size: {len(dataset):,}")

    # Create trainer config
    config = FinetuneConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=min(100, max_steps // 10),
        weight_decay=0.01,
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=10,
        save_steps=max(100, max_steps // 4),
        output_dir="/checkpoints/finetune-output",
        use_wandb=use_wandb,
        wandb_project="anymal-finetune",
    )

    # Train
    trainer = FinetuneTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
    )

    metrics = trainer.train()
    print(f"Training complete! Final metrics: {metrics}")


def run_pretrain(llama_path, max_steps, learning_rate, batch_size, use_wandb, use_dummy_data):
    """Run Stage 1 pretraining."""
    import torch
    from models import AnyMAL
    from data import build_dataloader, ImageTextCollator
    from training import PretrainTrainer
    from training.pretrain import PretrainConfig

    # Initialize model (no LoRA for pretraining)
    print("Initializing model...")
    model = AnyMAL(
        llm_model_name=llama_path,
        vision_model_name="ViT-L-14",
        vision_pretrained="openai",
        projector_type="perceiver",
        num_image_tokens=64,
        use_qlora=False,  # No LoRA for Stage 1
        gradient_checkpointing=True,
        use_flash_attention=False,  # Skip flash-attn, use SDPA instead
    )

    # Configure for Stage 1: only train projector
    model.set_training_stage(1)

    # Load dataset
    print("Loading dataset...")
    if use_dummy_data:
        dataset = create_dummy_caption_dataset(
            model.tokenizer,
            num_samples=max_steps * batch_size * 2
        )
    else:
        dataset = load_llava_pretrain_dataset(model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=256,
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        distributed=False,
        collate_fn=collator,
    )

    print(f"Dataset size: {len(dataset):,}")

    # Create trainer config
    config = PretrainConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        warmup_steps=min(100, max_steps // 10),
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=10,
        save_steps=max(100, max_steps // 4),
        output_dir="/checkpoints/pretrain-output",
        use_wandb=use_wandb,
        wandb_project="anymal-pretrain",
    )

    # Train
    trainer = PretrainTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
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

    # Use real COCO images - filter dataset to only samples with available images
    image_dir = "/checkpoints/coco_images"
    if os.path.exists(image_dir):
        num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        print(f"Found {num_images} real COCO images in {image_dir}")
        print("Filtering dataset to only samples with real images (no dummy images)")

        dataset = InstructionDataset(
            data_path=json_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            image_size=224,
            max_length=512,
            filter_to_available_images=True,  # Only use samples with real images
        )
    else:
        print("WARNING: No COCO images found, using dummy images")
        dataset = InstructionDataset(
            data_path=json_path,
            image_dir=None,
            tokenizer=tokenizer,
            image_size=224,
            max_length=512,
        )

    print(f"Loaded {len(dataset)} instruction samples")
    return dataset


def load_llava_pretrain_dataset(tokenizer):
    """
    Load LLaVA-Pretrain (CC3M/LAION subset) for Stage 1 alignment.

    Uses cached JSON from volume. Falls back to dummy data if JSON
    is not available (pretrain JSON is larger and may fail to download).
    """
    import torch
    from torch.utils.data import Dataset
    import json
    from data.data_utils import get_image_transform, TextProcessor

    # JSON should be cached by Trainer.setup()
    json_path = "/checkpoints/llava_data/blip_laion_cc_sbu_558k.json"

    if not os.path.exists(json_path):
        print("Warning: LLaVA-Pretrain JSON not available, using dummy data")
        # Return dummy dataset for testing
        return create_dummy_caption_dataset(tokenizer, num_samples=10000)

    print(f"Loading LLaVA-Pretrain from {json_path}")

    class LLaVAPretrainDataset(Dataset):
        """Simple pretrain dataset that loads from cached JSON with dummy images."""

        def __init__(self, json_path, tokenizer):
            with open(json_path, "r") as f:
                self.data = json.load(f)
            self.tokenizer = tokenizer
            self.transform = get_image_transform(image_size=224, is_train=True)
            self.text_processor = TextProcessor(tokenizer, max_length=256)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # Get caption from conversations
            conversations = item.get("conversations", [])
            caption = ""
            for conv in conversations:
                if conv.get("from") == "gpt":
                    caption = conv.get("value", "")
                    break

            if not caption:
                caption = "A sample image."

            # Use dummy image (downloading millions of images is impractical)
            image = torch.randn(3, 224, 224)

            encoding = self.text_processor.encode_text(caption)

            # For captioning, predict all tokens
            labels = encoding["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "image": image,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }

    dataset = LLaVAPretrainDataset(json_path, tokenizer)
    print(f"Loaded {len(dataset)} pretrain samples (using dummy images)")
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

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
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

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
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


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    stage: str = "finetune",
    learning_rate: float = None,
    batch_size: int = 4,
    use_wandb: bool = False,
    use_dummy_data: bool = False,
    wandb_api_key: str = None,
):
    """
    Entry point for Modal training.

    Examples:
        modal run modal_train.py                              # Quick test with LLaVA data
        modal run modal_train.py --use-dummy-data             # Test with dummy data
        modal run modal_train.py --max-steps 500              # Longer run
        modal run modal_train.py --stage pretrain             # Stage 1
        modal run modal_train.py --use-wandb --wandb-api-key YOUR_KEY  # With W&B
    """
    print(f"Starting AnyMAL training on Modal...")
    print(f"  Stage: {stage}")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {'dummy' if use_dummy_data else 'LLaVA'}")
    print(f"  W&B: {'enabled' if use_wandb else 'disabled'}")

    # Use the Trainer class - model loading happens once in @modal.enter()
    trainer = Trainer()
    trainer.train.remote(
        max_steps=max_steps,
        stage=stage,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        use_dummy_data=use_dummy_data,
        wandb_api_key=wandb_api_key,
    )
