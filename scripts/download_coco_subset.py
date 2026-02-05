#!/usr/bin/env python3
"""
Download a subset of COCO train2017 images for LLaVA training.

Usage:
    python scripts/download_coco_subset.py --num-images 5000 --output-dir /path/to/images

The script:
1. Reads the LLaVA-Instruct-150K JSON to get required image IDs
2. Downloads the first N images from COCO train2017
3. Saves them to the specified directory
"""

import os
import json
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path


COCO_BASE_URL = "https://images.cocodataset.org/train2017"


def get_llava_image_ids(json_path: str = None) -> list:
    """
    Get unique image IDs from LLaVA-Instruct-150K dataset.

    Returns:
        List of image filenames (e.g., ['000000033471.jpg', ...])
    """
    if json_path is None:
        # Download from HuggingFace if not provided
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
        )

    with open(json_path) as f:
        data = json.load(f)

    # Get unique images, preserving order
    seen = set()
    images = []
    for sample in data:
        img = sample.get("image")
        if img and img not in seen:
            seen.add(img)
            images.append(img)

    return images


def download_image(image_name: str, output_dir: str) -> tuple:
    """
    Download a single COCO image.

    Returns:
        (image_name, success, error_msg)
    """
    output_path = os.path.join(output_dir, image_name)

    # Skip if already exists
    if os.path.exists(output_path):
        return (image_name, True, "already exists")

    url = f"{COCO_BASE_URL}/{image_name}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return (image_name, True, None)
    except Exception as e:
        return (image_name, False, str(e))


def download_coco_subset(
    output_dir: str,
    num_images: int = 5000,
    json_path: str = None,
    num_workers: int = 8,
) -> dict:
    """
    Download a subset of COCO images referenced by LLaVA.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to download
        json_path: Path to LLaVA JSON (downloads if None)
        num_workers: Number of parallel download threads

    Returns:
        Dict with download statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Getting image list from LLaVA-Instruct-150K...")
    all_images = get_llava_image_ids(json_path)
    print(f"Found {len(all_images)} unique images in dataset")

    # Take first N images
    images_to_download = all_images[:num_images]
    print(f"Will download {len(images_to_download)} images")

    # Download in parallel
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_images = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_image, img, output_dir): img
            for img in images_to_download
        }

        with tqdm(total=len(images_to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                image_name, success, error = future.result()

                if success:
                    if error == "already exists":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    failed_images.append((image_name, error))

                pbar.update(1)
                pbar.set_postfix(ok=success_count, skip=skip_count, fail=fail_count)

    # Summary
    print(f"\nDownload complete:")
    print(f"  Downloaded: {success_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Failed: {fail_count}")

    if failed_images:
        print(f"\nFailed images (first 10):")
        for img, err in failed_images[:10]:
            print(f"  {img}: {err}")

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "total_requested": num_images,
        "downloaded": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "images": images_to_download,
        "failed_images": [img for img, _ in failed_images],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download COCO image subset for LLaVA")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/coco_images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=5000,
        help="Number of images to download",
    )
    parser.add_argument(
        "--json-path", "-j",
        type=str,
        default=None,
        help="Path to LLaVA JSON (downloads from HF if not provided)",
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )

    args = parser.parse_args()

    download_coco_subset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        json_path=args.json_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
