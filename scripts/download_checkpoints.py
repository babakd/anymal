#!/usr/bin/env python3
"""
Download Pre-trained Checkpoints for AnyMAL

Downloads required model weights:
1. LLaMA-3-8B-Instruct (requires Meta approval)
2. CLIP ViT-L/14 (OpenAI or LAION)

Usage:
    python scripts/download_checkpoints.py --all
    python scripts/download_checkpoints.py --llama
    python scripts/download_checkpoints.py --clip

Note: LLaMA requires approval from Meta via HuggingFace.
      Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
"""

import os
import argparse
from pathlib import Path


def download_llama(output_dir: str = "./checkpoints/llama3-8b-instruct"):
    """
    Download LLaMA-3-8B-Instruct from HuggingFace.

    Requires:
    1. HuggingFace account
    2. Accepted Meta's license on HuggingFace
    3. huggingface-cli login

    Instructions:
    1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    2. Accept the license
    3. Run: huggingface-cli login
    4. Then run this script
    """
    print("=" * 60)
    print("Downloading LLaMA-3-8B-Instruct")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download

        # Check if user is logged in
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token is None:
            print("\nERROR: Not logged in to HuggingFace!")
            print("Please run: huggingface-cli login")
            print("\nMake sure you have:")
            print("1. A HuggingFace account")
            print("2. Accepted Meta's license at:")
            print("   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            return False

        print(f"\nDownloading to: {output_dir}")
        print("This may take a while (~15GB)...")

        snapshot_download(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print(f"\nLLaMA downloaded to: {output_dir}")
        return True

    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"ERROR downloading LLaMA: {e}")
        return False


def download_clip(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    output_dir: str = "./checkpoints/clip",
):
    """
    Download CLIP model weights.

    Args:
        model_name: CLIP model variant (ViT-L-14, ViT-G-14, etc.)
        pretrained: Pretrained weights (openai, laion2b_s34b_b88k)
        output_dir: Output directory
    """
    print("=" * 60)
    print(f"Downloading CLIP {model_name} ({pretrained})")
    print("=" * 60)

    try:
        import open_clip

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nDownloading {model_name} with {pretrained} weights...")
        print("This will cache the model for future use.")

        # This downloads and caches the model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=output_dir,
        )

        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nCLIP model loaded:")
        print(f"  Model: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Parameters: {num_params / 1e6:.1f}M")
        print(f"  Cache dir: {output_dir}")

        return True

    except ImportError:
        print("ERROR: open_clip not installed")
        print("Run: pip install open_clip_torch")
        return False
    except Exception as e:
        print(f"ERROR downloading CLIP: {e}")
        return False


def list_available_clip_models():
    """List available CLIP models in open_clip."""
    try:
        import open_clip
        pretrained = open_clip.list_pretrained()

        print("\nAvailable CLIP models:")
        print("-" * 40)

        # Group by model
        models = {}
        for model, weights in pretrained:
            if model not in models:
                models[model] = []
            models[model].append(weights)

        for model in sorted(models.keys()):
            print(f"\n{model}:")
            for weights in models[model]:
                print(f"  - {weights}")

    except ImportError:
        print("open_clip not installed")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained checkpoints for AnyMAL"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all required checkpoints",
    )
    parser.add_argument(
        "--llama",
        action="store_true",
        help="Download LLaMA-3-8B-Instruct",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Download CLIP ViT-L/14",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L-14",
        help="CLIP model variant (default: ViT-L-14)",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="openai",
        help="CLIP pretrained weights (default: openai)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--list-clip",
        action="store_true",
        help="List available CLIP models",
    )

    args = parser.parse_args()

    if args.list_clip:
        list_available_clip_models()
        return

    if not (args.all or args.llama or args.clip):
        parser.print_help()
        return

    success = True

    if args.all or args.llama:
        llama_dir = os.path.join(args.output_dir, "llama3-8b-instruct")
        success = download_llama(llama_dir) and success

    if args.all or args.clip:
        clip_dir = os.path.join(args.output_dir, "clip")
        success = download_clip(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            output_dir=clip_dir,
        ) and success

    if success:
        print("\n" + "=" * 60)
        print("All downloads completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some downloads failed. Check errors above.")
        print("=" * 60)


if __name__ == "__main__":
    main()
