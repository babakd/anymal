#!/usr/bin/env python3
"""
Stage 1: Alignment Pretraining Script for AnyMAL

Trains the Perceiver Resampler to align image features with LLM space.

Usage:
    # Single GPU
    python scripts/train_pretrain.py --config configs/pretrain_image.yaml

    # Multi-GPU with torchrun (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train_pretrain.py \
        --config configs/pretrain_image.yaml

    # With overrides
    torchrun --nproc_per_node=8 scripts/train_pretrain.py \
        --config configs/pretrain_image.yaml \
        --per_device_batch_size 64 \
        --max_steps 100000 \
        --learning_rate 2e-4
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

from models import create_model_from_config
from model_metadata import normalize_architecture_name
from data import create_laion_dataset, build_dataloader, ImageTextCollator
from data.dataset_splitter import deterministic_train_val_split
from training import PretrainTrainer
from training.pretrain import PretrainConfig
from training.distributed import setup_distributed, cleanup_distributed, print_rank_0


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with base config merging."""
    config_dir = Path(config_path).parent

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle defaults/inheritance (simplified version of OmegaConf's defaults)
    if "defaults" in config:
        merged = {}
        for default in config["defaults"]:
            base_path = config_dir / f"{default}.yaml"
            if base_path.exists():
                merged = deep_merge(merged, load_config(str(base_path)))
        del config["defaults"]
        config = deep_merge(merged, config)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Alignment Pretraining for AnyMAL"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_image.yaml",
        help="Path to configuration file",
    )

    # Model overrides
    parser.add_argument("--llm_model_name", type=str, default=None)
    parser.add_argument("--llm-backbone", dest="llm_backbone", type=str, default=None)
    parser.add_argument("--vision_model_name", type=str, default=None)
    parser.add_argument("--num_image_tokens", type=int, default=None)
    parser.add_argument("--architecture", type=str, default=None)

    # Data overrides
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # Training overrides
    parser.add_argument("--pretrain_checkpoint", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=None,
        help="Multiply Stage 1 backward loss while logging the unscaled loss.",
    )
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default=None,
        choices=["mean", "supervised_token_target"],
        help=(
            "Stage 1 objective normalization. 'mean' keeps the HF token-mean loss; "
            "'supervised_token_target' scales that mean by supervised_tokens/target "
            "for the backward objective while logging raw loss separately."
        ),
    )
    parser.add_argument(
        "--loss_normalization_target_tokens",
        type=float,
        default=None,
        help="Target supervised-token count for --loss_normalization supervised_token_target.",
    )
    parser.add_argument(
        "--connector_warmup_steps",
        type=int,
        default=None,
        help="Stage 1 optimizer steps that update only connector output projection/gate grads.",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Hardware options
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="Disable flash attention (for non-CUDA systems)")

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
    if args.llm_backbone:
        config["model"]["llm_model_name"] = args.llm_backbone
        config["model"]["llm_backbone"] = args.llm_backbone
    if args.vision_model_name:
        config["model"]["vision_model_name"] = args.vision_model_name
    if args.architecture:
        config["model"]["architecture"] = args.architecture
    if args.num_image_tokens:
        config["model"]["num_image_tokens"] = args.num_image_tokens
        config["model"]["max_image_tokens"] = args.num_image_tokens
    if args.train_data_path:
        config["data"]["train_data_path"] = args.train_data_path
    if args.pretrain_checkpoint:
        config.setdefault("checkpoint", {})["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.per_device_batch_size:
        config["data"]["per_device_batch_size"] = args.per_device_batch_size
    if args.gradient_accumulation_steps:
        config["data"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.warmup_steps:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.loss_scale is not None:
        config["training"]["loss_scale"] = args.loss_scale
    if args.loss_normalization is not None:
        config["training"]["loss_normalization"] = args.loss_normalization
    if args.loss_normalization_target_tokens is not None:
        config["training"]["loss_normalization_target_tokens"] = args.loss_normalization_target_tokens
    if args.connector_warmup_steps is not None:
        config["training"]["connector_warmup_steps"] = args.connector_warmup_steps
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir
    if args.use_wandb:
        config["logging"]["use_wandb"] = True
    if args.wandb_project:
        config["logging"]["wandb_project"] = args.wandb_project

    # Debug mode
    if args.debug:
        config["training"]["max_steps"] = 100
        config["logging"]["logging_steps"] = 1
        config["checkpointing"]["save_steps"] = 50

    # Flash attention override
    if args.no_flash_attention:
        config["model"]["use_flash_attention"] = False

    print_rank_0("=" * 60)
    print_rank_0("AnyMAL Stage 1: Alignment Pretraining")
    print_rank_0("=" * 60)

    # Initialize model
    print_rank_0("\nInitializing model...")
    amp_dtype_name = config.get("training", {}).get("amp_dtype", "bfloat16")
    if torch.cuda.is_available():
        llm_torch_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        # Avoid `device_map="auto"` spanning multiple GPUs under DDP.
        llm_device_map = {"": local_rank}
    else:
        llm_torch_dtype = torch.float32
        llm_device_map = None

    model = create_model_from_config(
        config,
        model_overrides={
            # Stage 1: quantization optional, LoRA always disabled.
            "use_lora": False,
        },
        llm_device_map=llm_device_map,
        llm_torch_dtype=llm_torch_dtype,
    )
    architecture = normalize_architecture_name(config["model"].get("architecture", "anymal_v1"))
    print_rank_0(f"Model architecture: {architecture}")

    pretrain_checkpoint = config.get("checkpoint", {}).get("pretrain_checkpoint")
    if pretrain_checkpoint and config.get("checkpoint", {}).get("load_projector", True):
        from model_metadata import validate_checkpoint_metadata_values

        expected_values = {}
        if architecture == "anymal_v2":
            expected_values = {
                "vision_encoder_type": getattr(model, "vision_encoder_type", None),
                "token_compressor_type": getattr(model, "token_compressor_type", None),
                "max_image_tokens": getattr(model, "max_image_tokens", None),
                "min_image_tokens": getattr(model, "min_image_tokens", None),
            }
        elif architecture in {"anymal_v3", "anymal_v4"}:
            expected_values = {
                "vision_encoder_type": getattr(model, "vision_encoder_type", None),
                "connector_type": getattr(model, "connector_type", None),
                "num_image_tokens": getattr(model, "num_image_tokens", None),
                    "connector_layers": getattr(model, "connector_layers", None),
                    "connector_heads": getattr(model, "connector_heads", None),
                    "connector_ff_mult": getattr(model, "connector_ff_mult", None),
                    "connector_hidden_dim": getattr(model, "connector_hidden_dim", None),
                    "connector_output_scale": getattr(model, "connector_output_scale", None),
                    "connector_output_gate_init": getattr(model, "connector_output_gate_init", None),
                }
            if architecture == "anymal_v4":
                expected_values.update(
                    {
                        "num_global_image_tokens": getattr(model, "num_global_image_tokens", None),
                        "num_local_image_tokens": getattr(model, "num_local_image_tokens", None),
                        "use_2d_position_features": getattr(model, "use_2d_position_features", None),
                    }
                )
                if getattr(model, "connector_type", None) == "deepstack_spatial_perceiver_resampler":
                    expected_values.update(
                        {
                            "deepstack_num_feature_levels": getattr(
                                model,
                                "deepstack_num_feature_levels",
                                None,
                            ),
                            "deepstack_hidden_state_indices": list(
                                getattr(model, "deepstack_hidden_state_indices", [])
                            ),
                            "vision_feature_layers": list(
                                getattr(model, "deepstack_hidden_state_indices", [])
                            ),
                        }
                    )
        validate_checkpoint_metadata_values(
            checkpoint_dir=pretrain_checkpoint,
            expected_architecture=architecture,
            expected_values=expected_values,
        )
        projector_path = Path(pretrain_checkpoint) / "projector.pt"
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing projector weights: {projector_path}")
        print_rank_0(f"Loading Stage 1 adapter from {projector_path}")
        model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

        compressor_path = Path(pretrain_checkpoint) / "token_compressor.pt"
        if hasattr(model, "token_compressor") and compressor_path.exists():
            print_rank_0(f"Loading token compressor from {compressor_path}")
            model.token_compressor.load_state_dict(
                torch.load(compressor_path, map_location="cpu")
            )

    # Initialize dataset
    print_rank_0("\nLoading dataset...")
    streaming = config["data"].get("streaming", False)
    dataset_kwargs = {
        "image_size": config["data"].get("image_size", 224),
        "max_length": config["data"].get("max_length", 256),
        "caption_prompt": config["data"].get("caption_prompt", "A photo of"),
        "vision_encoder_type": "siglip2" if architecture in {"anymal_v2", "anymal_v3", "anymal_v4"} else "clip",
        "vision_model_name": config["model"].get("vision_model_name"),
    }
    if not streaming:
        dataset_kwargs.update(
            {
                "image_dir": config["data"].get("image_dir"),
                "filter_to_available_images": config["data"].get("filter_to_available_images", False),
                "min_caption_chars": config["data"].get("min_caption_chars", 1),
                "deduplicate_captions": config["data"].get("deduplicate_captions", False),
            }
        )
    if architecture in {"anymal_v2", "anymal_v3", "anymal_v4"}:
        dataset_kwargs["insert_image_placeholders"] = True
        dataset_kwargs["num_image_tokens"] = config["model"].get(
            "max_image_tokens",
            config["model"].get("num_image_tokens", 256),
        )
    if streaming:
        dataset_kwargs["buffer_size"] = config["data"].get("shuffle_buffer_size", 10000)

    train_dataset = create_laion_dataset(
        data_path=config["data"]["train_data_path"],
        tokenizer=model.tokenizer,
        streaming=streaming,
        dataset_type=config["data"].get("dataset_type", "laion"),
        **dataset_kwargs,
    )

    if hasattr(train_dataset, '__len__'):
        print_rank_0(f"Dataset size: {len(train_dataset):,}")
    else:
        print_rank_0("Dataset: streaming (size unknown)")

    # Create data loader
    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=config["data"].get("max_length", 256),
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=config["data"].get("per_device_batch_size", 64),
        shuffle=True,
        num_workers=config["data"].get("dataloader_num_workers", 4),
        distributed=(world_size > 1),
        collate_fn=collator,
    )

    # Read evaluation config
    eval_config = config.get("evaluation", {})

    # Create trainer config
    trainer_config = PretrainConfig(
        # Training
        max_steps=config["training"]["max_steps"],
        gradient_accumulation_steps=config["data"].get("gradient_accumulation_steps", 4),

        # Optimization
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        loss_scale=config["training"].get("loss_scale", 1.0),
        loss_normalization=config["training"].get("loss_normalization", "mean"),
        loss_normalization_target_tokens=config["training"].get(
            "loss_normalization_target_tokens",
            8.0,
        ),
        loss_normalization_min_multiplier=config["training"].get(
            "loss_normalization_min_multiplier",
            0.05,
        ),
        loss_normalization_max_multiplier=config["training"].get(
            "loss_normalization_max_multiplier",
            4.0,
        ),
        connector_warmup_steps=config["training"].get("connector_warmup_steps", 0),
        connector_warmup_trainable_prefixes=tuple(
            config["training"].get(
                "connector_warmup_trainable_prefixes",
                ("projector.output_proj", "projector.output_gate_logit"),
            )
        ),

        # Mixed precision
        use_amp=config["training"].get("use_amp", True),
        amp_dtype=config["training"].get("amp_dtype", "bfloat16"),

        # Logging
        logging_steps=config["logging"].get("logging_steps", 10),
        use_wandb=config["logging"].get("use_wandb", False),
        wandb_project=config["logging"].get("wandb_project", "anymal-pretrain"),

        # Checkpointing
        save_steps=config["checkpointing"].get("save_steps", 5000),
        save_total_limit=config["checkpointing"].get("save_total_limit", 5),
        output_dir=config["checkpointing"]["output_dir"],

        # Evaluation
        eval_steps=eval_config.get("eval_steps"),
        eval_strategy=eval_config.get("eval_strategy", "steps"),
        max_eval_batches=eval_config.get("max_eval_batches"),
    )

    # Create eval dataloader if eval config is present and dataset supports splitting
    eval_dataloader = None
    if trainer_config.eval_steps and not streaming:
        eval_data_path = config["data"].get("eval_data_path")
        if eval_data_path:
            print_rank_0(f"\nLoading eval dataset from {eval_data_path}...")
            eval_dataset = create_laion_dataset(
                data_path=eval_data_path,
                tokenizer=model.tokenizer,
                streaming=False,
                dataset_type=config["data"].get(
                    "eval_dataset_type",
                    config["data"].get("dataset_type", "laion"),
                ),
                **dataset_kwargs,
            )
        elif hasattr(train_dataset, '__len__'):
            # Split train dataset into train/val
            val_fraction = 0.05
            print_rank_0(f"\nSplitting dataset: {1-val_fraction:.0%} train / {val_fraction:.0%} val")
            train_dataset, eval_dataset = deterministic_train_val_split(
                train_dataset, val_fraction=val_fraction
            )
            # Recreate train dataloader with the split subset
            train_dataloader = build_dataloader(
                dataset=train_dataset,
                batch_size=config["data"].get("per_device_batch_size", 64),
                shuffle=True,
                num_workers=config["data"].get("dataloader_num_workers", 4),
                distributed=(world_size > 1),
                collate_fn=collator,
            )
        else:
            eval_dataset = None

        if eval_dataset is not None:
            print_rank_0(f"Eval dataset size: {len(eval_dataset):,}")
            eval_dataloader = build_dataloader(
                dataset=eval_dataset,
                batch_size=config["data"].get("per_device_batch_size", 64),
                shuffle=False,
                num_workers=config["data"].get("dataloader_num_workers", 4),
                distributed=(world_size > 1),
                collate_fn=collator,
            )

    # Initialize trainer
    print_rank_0("\nInitializing trainer...")
    trainer = PretrainTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Print training info
    print_rank_0("\nTraining configuration:")
    print_rank_0(f"  World size: {world_size}")
    print_rank_0(f"  Per-device batch size: {config['data'].get('per_device_batch_size', 64)}")
    print_rank_0(f"  Gradient accumulation: {trainer_config.gradient_accumulation_steps}")
    print_rank_0(f"  Effective batch size: {trainer._get_effective_batch_size()}")
    print_rank_0(f"  Max steps: {trainer_config.max_steps}")
    print_rank_0(f"  Learning rate: {trainer_config.learning_rate}")

    # Train
    print_rank_0("\nStarting training...")
    metrics = trainer.train()

    print_rank_0("\nTraining completed!")
    print_rank_0(f"Final metrics: {metrics}")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
