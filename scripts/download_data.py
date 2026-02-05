#!/usr/bin/env python3
"""
Download Training Data for AnyMAL

Downloads required datasets:
1. LAION subset for alignment pretraining
2. LLaVA-Instruct-150K for instruction tuning
3. COCO images (for LLaVA evaluation)

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --laion --samples 1000000
    python scripts/download_data.py --llava
    python scripts/download_data.py --coco

Note: Full LAION-2B is very large. Start with a small subset for testing.
"""

import os
import argparse
from pathlib import Path


def download_laion_subset(
    output_dir: str = "./data/laion",
    num_samples: int = 1000000,
):
    """
    Download a subset of LAION-2B for alignment pretraining.

    Uses the research-safe filtered version from HuggingFace.

    Args:
        output_dir: Output directory
        num_samples: Number of samples to download (default: 1M)
    """
    print("=" * 60)
    print(f"Downloading LAION subset ({num_samples:,} samples)")
    print("=" * 60)

    try:
        from datasets import load_dataset
        import json

        os.makedirs(output_dir, exist_ok=True)

        print("\nLoading LAION-2B-en-research-safe from HuggingFace...")
        print("This is a streaming dataset, downloading samples...")

        # Use streaming to avoid downloading the full dataset
        dataset = load_dataset(
            "laion/relaion2B-en-research-safe",
            split="train",
            streaming=True,
        )

        # Download samples
        samples = []
        print(f"\nDownloading {num_samples:,} samples...")

        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            samples.append({
                "url": sample["url"],
                "caption": sample["caption"],
                "width": sample.get("width"),
                "height": sample.get("height"),
            })

            if (i + 1) % 10000 == 0:
                print(f"  Downloaded {i + 1:,} samples...")

        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.jsonl")
        with open(metadata_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"\nSaved {len(samples):,} samples to {metadata_path}")
        print("\nNOTE: This only saves metadata (URLs + captions).")
        print("Images need to be downloaded separately using the URLs.")
        print("Consider using img2dataset for efficient image downloading:")
        print("  pip install img2dataset")
        print("  img2dataset --url_list metadata.jsonl --output_folder images/")

        return True

    except ImportError:
        print("ERROR: datasets not installed")
        print("Run: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_llava_instruct(output_dir: str = "./data/llava"):
    """
    Download LLaVA-Instruct-150K for instruction tuning.

    This is a relatively small dataset (~150K conversations).
    """
    print("=" * 60)
    print("Downloading LLaVA-Instruct-150K")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
        import json

        os.makedirs(output_dir, exist_ok=True)

        print("\nDownloading from HuggingFace...")

        # Download the instruction data
        file_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
            local_dir=output_dir,
        )

        print(f"\nDownloaded to: {file_path}")

        # Count samples
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Total samples: {len(data):,}")

        print("\nNOTE: This dataset uses COCO images.")
        print("You'll need to download COCO train2017 images separately.")
        print("Run: python scripts/download_data.py --coco")

        return True

    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_coco(
    output_dir: str = "./data/coco",
    split: str = "train2017",
):
    """
    Download COCO dataset images.

    Args:
        output_dir: Output directory
        split: Dataset split (train2017, val2017)
    """
    print("=" * 60)
    print(f"Downloading COCO {split}")
    print("=" * 60)

    import urllib.request
    import zipfile

    os.makedirs(output_dir, exist_ok=True)

    # COCO URLs
    urls = {
        "train2017": "https://images.cocodataset.org/zips/train2017.zip",
        "val2017": "https://images.cocodataset.org/zips/val2017.zip",
        "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    if split not in urls:
        print(f"Unknown split: {split}")
        print(f"Available: {list(urls.keys())}")
        return False

    url = urls[split]
    filename = os.path.basename(url)
    zip_path = os.path.join(output_dir, filename)

    try:
        # Download
        if not os.path.exists(zip_path):
            print(f"\nDownloading {url}")
            print("This is a large file (~18GB for train2017)")

            def progress_hook(count, block_size, total_size):
                percent = count * block_size * 100 // total_size
                print(f"\rProgress: {percent}%", end="")

            urllib.request.urlretrieve(url, zip_path, progress_hook)
            print("\nDownload complete!")
        else:
            print(f"\nZip file already exists: {zip_path}")

        # Extract
        extract_dir = os.path.join(output_dir, split)
        if not os.path.exists(extract_dir):
            print(f"\nExtracting to {output_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            print("Extraction complete!")
        else:
            print(f"\nAlready extracted: {extract_dir}")

        # Count images
        images = list(Path(extract_dir).glob("*.jpg"))
        print(f"\nTotal images: {len(images):,}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def create_sample_dataset(output_dir: str = "./data/sample"):
    """
    Create a tiny sample dataset for testing.

    This creates a minimal dataset to verify the training pipeline works.
    """
    print("=" * 60)
    print("Creating sample dataset for testing")
    print("=" * 60)

    import json
    from PIL import Image
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Create sample images
    num_samples = 100
    samples = []

    print(f"\nCreating {num_samples} sample images...")

    for i in range(num_samples):
        # Create random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Save image
        img_path = os.path.join(output_dir, "images", f"sample_{i:05d}.jpg")
        img.save(img_path)

        # Create sample caption/instruction
        samples.append({
            "id": f"sample_{i}",
            "image": f"images/sample_{i:05d}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe this image."
                },
                {
                    "from": "gpt",
                    "value": f"This is sample image {i}. It contains random colored pixels for testing purposes."
                }
            ]
        })

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Created {num_samples} samples in {output_dir}")
    print("\nThis dataset is for testing only!")
    print("Use --laion and --llava for real training data.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download training data for AnyMAL"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--laion",
        action="store_true",
        help="Download LAION subset",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000000,
        help="Number of LAION samples to download",
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        help="Download LLaVA-Instruct-150K",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Download COCO train2017 images",
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017", "annotations"],
        help="COCO split to download",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create tiny sample dataset for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base output directory",
    )

    args = parser.parse_args()

    if not (args.all or args.laion or args.llava or args.coco or args.sample):
        parser.print_help()
        return

    success = True

    if args.sample:
        sample_dir = os.path.join(args.output_dir, "sample")
        success = create_sample_dataset(sample_dir) and success

    if args.all or args.laion:
        laion_dir = os.path.join(args.output_dir, "laion")
        success = download_laion_subset(laion_dir, args.samples) and success

    if args.all or args.llava:
        llava_dir = os.path.join(args.output_dir, "llava")
        success = download_llava_instruct(llava_dir) and success

    if args.all or args.coco:
        coco_dir = os.path.join(args.output_dir, "coco")
        success = download_coco(coco_dir, args.coco_split) and success

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
