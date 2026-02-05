#!/usr/bin/env python3
"""
Stage 2: Instruction Fine-tuning Script for AnyMAL

Fine-tunes the model on instruction-following data with LoRA adapters.

Usage:
    # Single GPU
    python scripts/train_finetune.py --config configs/finetune.yaml

    # Multi-GPU with torchrun (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train_finetune.py \
        --config configs/finetune.yaml

    # With Stage 1 checkpoint
    torchrun --nproc_per_node=8 scripts/train_finetune.py \
        --config configs/finetune.yaml \
        --pretrain_checkpoint ./outputs/pretrain/checkpoint-100000
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import AnyMAL
from data import InstructionDataset, build_dataloader, ImageTextCollator
from training import FinetuneTrainer
from training.finetune import FinetuneConfig
from training.distributed import setup_distributed, cleanup_distributed, print_rank_0


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with inheritance support."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "defaults" in config:
        defaults = config.pop("defaults")
        merged = {}
        for default in defaults:
            base_name = default if isinstance(default, str) else list(default.values())[0]
            base_path = os.path.join(config_dir, f"{base_name}.yaml")
            if os.path.exists(base_path):
                merged = deep_merge(merged, load_config(base_path))
        config = deep_merge(merged, config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Instruction Fine-tuning for AnyMAL"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to configuration file",
    )

    # Model overrides
    parser.add_argument("--llm_model_name", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Data overrides
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # Training overrides
    parser.add_argument("--pretrain_checkpoint", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up distributed training
    rank, local_rank, world_size = setup_distributed()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.llm_model_name:
        config["model"]["llm_model_name"] = args.llm_model_name
    if args.lora_r:
        config["model"]["lora_r"] = args.lora_r
    if args.lora_alpha:
        config["model"]["lora_alpha"] = args.lora_alpha
    if args.train_data_path:
        config["data"]["train_data_path"] = args.train_data_path
    if args.image_dir:
        config["data"]["image_dir"] = args.image_dir
    if args.per_device_batch_size:
        config["data"]["per_device_batch_size"] = args.per_device_batch_size
    if args.gradient_accumulation_steps:
        config["data"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.pretrain_checkpoint:
        config["training"]["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir
    if args.use_wandb:
        config["logging"]["use_wandb"] = True
    if args.wandb_project:
        config["logging"]["wandb_project"] = args.wandb_project

    # Debug mode
    if args.debug:
        config["training"]["max_steps"] = 50
        config["logging"]["logging_steps"] = 1
        config["checkpointing"]["save_steps"] = 25

    print_rank_0("=" * 60)
    print_rank_0("AnyMAL Stage 2: Instruction Fine-tuning")
    print_rank_0("=" * 60)

    # Initialize model with LoRA
    print_rank_0("\nInitializing model...")
    amp_dtype_name = config.get("training", {}).get("amp_dtype", "bfloat16")
    if torch.cuda.is_available():
        llm_torch_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        # Avoid `device_map="auto"` spanning multiple GPUs under DDP.
        llm_device_map = {"": local_rank}
    else:
        llm_torch_dtype = torch.float32
        llm_device_map = None

    model = AnyMAL(
        llm_model_name=config["model"]["llm_model_name"],
        vision_model_name=config["model"].get("vision_model_name", "ViT-L-14"),
        vision_pretrained=config["model"].get("vision_pretrained", "openai"),
        projector_type=config["model"].get("projector_type", "perceiver"),
        num_image_tokens=config["model"].get("num_image_tokens", 64),
        projector_layers=config["model"].get("projector_layers", 6),
        projector_heads=config["model"].get("projector_heads", 16),
        projector_ff_mult=config["model"].get("projector_ff_mult", 4),
        use_qlora=config["model"].get("use_qlora", True),
        lora_r=config["model"].get("lora_r", 64),
        lora_alpha=config["model"].get("lora_alpha", 16),
        lora_dropout=config["model"].get("lora_dropout", 0.05),
        lora_target_modules=config["model"].get("lora_target_modules"),
        use_flash_attention=config["model"].get("use_flash_attention", True),
        gradient_checkpointing=config["model"].get("gradient_checkpointing", True),
        llm_device_map=llm_device_map,
        llm_torch_dtype=llm_torch_dtype,
    )

    # Initialize dataset
    print_rank_0("\nLoading dataset...")
    train_dataset = InstructionDataset(
        data_path=config["data"]["train_data_path"],
        image_dir=config["data"]["image_dir"],
        tokenizer=model.tokenizer,
        image_size=config["data"].get("image_size", 224),
        max_length=config["data"].get("max_length", 2048),
        system_prompt=config["data"].get("system_prompt"),
    )

    print_rank_0(f"Dataset size: {len(train_dataset):,}")

    # Create data loader
    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=config["data"].get("max_length", 2048),
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=config["data"].get("per_device_batch_size", 16),
        shuffle=True,
        num_workers=config["data"].get("dataloader_num_workers", 4),
        distributed=(world_size > 1),
        collate_fn=collator,
    )

    # Create trainer config
    # Check both config locations for pretrain_checkpoint (backward compatibility)
    pretrain_checkpoint = (
        config.get("training", {}).get("pretrain_checkpoint") or
        config.get("checkpoint", {}).get("pretrain_checkpoint")
    )

    trainer_config = FinetuneConfig(
        # Training
        max_steps=config["training"]["max_steps"],
        gradient_accumulation_steps=config["data"].get("gradient_accumulation_steps", 2),
        pretrain_checkpoint=pretrain_checkpoint,

        # Optimization
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"].get("warmup_steps", 100),
        weight_decay=config["training"].get("weight_decay", 0.01),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),

        # LoRA
        lora_r=config["model"].get("lora_r", 64),
        lora_alpha=config["model"].get("lora_alpha", 16),

        # Mixed precision
        use_amp=config["training"].get("use_amp", True),
        amp_dtype=config["training"].get("amp_dtype", "bfloat16"),

        # Logging
        logging_steps=config["logging"].get("logging_steps", 10),
        use_wandb=config["logging"].get("use_wandb", False),
        wandb_project=config["logging"].get("wandb_project", "anymal-finetune"),

        # Checkpointing
        save_steps=config["checkpointing"].get("save_steps", 500),
        save_total_limit=config["checkpointing"].get("save_total_limit", 5),
        output_dir=config["checkpointing"]["output_dir"],
    )

    # Initialize trainer
    print_rank_0("\nInitializing trainer...")
    trainer = FinetuneTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_dataloader,
    )

    # Print training info
    print_rank_0("\nTraining configuration:")
    print_rank_0(f"  World size: {world_size}")
    print_rank_0(f"  Per-device batch size: {config['data'].get('per_device_batch_size', 16)}")
    print_rank_0(f"  Gradient accumulation: {trainer_config.gradient_accumulation_steps}")
    print_rank_0(f"  Effective batch size: {trainer._get_effective_batch_size()}")
    print_rank_0(f"  Max steps: {trainer_config.max_steps}")
    print_rank_0(f"  Learning rate: {trainer_config.learning_rate}")
    print_rank_0(f"  LoRA rank: {trainer_config.lora_r}")
    if trainer_config.pretrain_checkpoint:
        print_rank_0(f"  Stage 1 checkpoint: {trainer_config.pretrain_checkpoint}")

    # Train
    print_rank_0("\nStarting training...")
    metrics = trainer.train()

    print_rank_0("\nTraining completed!")
    print_rank_0(f"Final metrics: {metrics}")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
