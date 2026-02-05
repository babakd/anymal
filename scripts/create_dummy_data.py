#!/usr/bin/env python
"""
Create dummy data for testing AnyMAL training pipeline.

This script generates synthetic test data without requiring any real datasets:
1. LAION-style data for Stage 1 (alignment pretraining)
2. LLaVA-style instruction data for Stage 2 (fine-tuning)

Usage:
    python scripts/create_dummy_data.py

Output structure:
    data/
    ├── laion_subset/
    │   ├── images/
    │   │   ├── 000000.jpg
    │   │   └── ... (100 images)
    │   └── metadata.json
    ├── dummy_images/
    │   ├── 000000.jpg
    │   └── ... (10 images)
    └── llava_instruct_sample.json
"""

import json
import os
import random
import numpy as np
from PIL import Image


def create_random_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a random RGB image with some structure (not just noise)."""
    # Create base color
    base_color = np.random.randint(50, 200, size=3, dtype=np.uint8)

    # Create image array
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill with base color
    img_array[:, :] = base_color

    # Add some random shapes for visual interest
    for _ in range(random.randint(3, 8)):
        shape_type = random.choice(["rectangle", "circle", "gradient"])
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        if shape_type == "rectangle":
            x1, y1 = random.randint(0, width - 20), random.randint(0, height - 20)
            x2, y2 = min(x1 + random.randint(20, 80), width), min(y1 + random.randint(20, 80), height)
            img_array[y1:y2, x1:x2] = color

        elif shape_type == "circle":
            cx, cy = random.randint(20, width - 20), random.randint(20, height - 20)
            radius = random.randint(10, 40)
            y, x = np.ogrid[:height, :width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            img_array[mask] = color

        elif shape_type == "gradient":
            x1, y1 = random.randint(0, width - 40), random.randint(0, height - 40)
            x2, y2 = min(x1 + random.randint(40, 100), width), min(y1 + random.randint(40, 100), height)
            for i, y in enumerate(range(y1, y2)):
                alpha = i / (y2 - y1)
                blend_color = (base_color * (1 - alpha) + color * alpha).astype(np.uint8)
                img_array[y, x1:x2] = blend_color

    return Image.fromarray(img_array)


def generate_caption() -> str:
    """Generate a random descriptive caption."""
    subjects = [
        "a cat", "a dog", "a person", "a car", "a house", "a tree",
        "a mountain", "a beach", "a city skyline", "a flower garden",
        "a sunset", "a forest path", "a river", "a bird", "a bicycle"
    ]

    actions = [
        "sitting in", "standing near", "walking through", "looking at",
        "next to", "in front of", "behind", "surrounded by"
    ]

    locations = [
        "a park", "the street", "a garden", "a field", "the woods",
        "a room", "the beach", "a parking lot", "nature", "the city"
    ]

    descriptors = [
        "beautiful", "colorful", "peaceful", "busy", "sunny",
        "quiet", "vibrant", "serene", "dramatic", "natural"
    ]

    # Generate various caption styles
    style = random.choice(["simple", "detailed", "artistic"])

    if style == "simple":
        return f"{random.choice(descriptors)} photo of {random.choice(subjects)}"
    elif style == "detailed":
        return f"{random.choice(subjects)} {random.choice(actions)} {random.choice(locations)}, {random.choice(descriptors)} scene"
    else:
        return f"A {random.choice(descriptors)} image showing {random.choice(subjects)} {random.choice(actions)} {random.choice(locations)}"


def generate_conversation() -> list:
    """Generate a random multi-turn conversation about an image."""
    questions = [
        "What do you see in this image?",
        "Describe this image in detail.",
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "What is happening in this scene?",
        "Can you describe the background?",
        "What mood does this image convey?",
    ]

    follow_ups = [
        "What else do you notice?",
        "Can you tell me more about the colors?",
        "Is there anything interesting in the background?",
        "What do you think the context might be?",
        "How would you describe the overall composition?",
    ]

    responses = [
        "The image shows various colorful shapes and patterns arranged in an interesting composition. There are multiple geometric forms visible against a background color.",
        "I can see a composition of different shapes and colors. The image has a mix of rectangles, circles, and gradient areas that create visual interest.",
        "This image displays an abstract arrangement of geometric shapes. The colors range from warm to cool tones, creating a balanced visual effect.",
        "The scene contains multiple overlapping shapes in various colors. There's a clear background color with shapes layered on top.",
        "Looking at this image, I notice several geometric elements arranged across the frame. The color palette creates a harmonious overall effect.",
    ]

    follow_up_responses = [
        "Additionally, I notice the way the shapes overlap creates depth in the image. The color combinations are quite pleasing to look at.",
        "The colors seem to follow a certain pattern, with complementary hues placed near each other. This creates visual harmony.",
        "The background provides a nice contrast to the foreground elements. It helps the main shapes stand out more clearly.",
        "The context appears to be an abstract or artistic composition, perhaps meant to explore color relationships and geometric forms.",
        "The composition uses the rule of thirds effectively, with key elements positioned to create balance across the frame.",
    ]

    # Build conversation
    conversation = []

    # First turn
    q1 = random.choice(questions)
    conversation.append({"from": "human", "value": f"<image>\n{q1}"})
    conversation.append({"from": "gpt", "value": random.choice(responses)})

    # Optional second turn (50% chance)
    if random.random() > 0.5:
        q2 = random.choice(follow_ups)
        conversation.append({"from": "human", "value": q2})
        conversation.append({"from": "gpt", "value": random.choice(follow_up_responses)})

    return conversation


def create_laion_data(output_dir: str, num_images: int = 100):
    """Create dummy LAION-style data for Stage 1 training."""
    print(f"Creating LAION dummy data in {output_dir}...")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []

    for i in range(num_images):
        # Create and save image
        img = create_random_image()
        img_filename = f"{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path, "JPEG", quality=85)

        # Generate caption
        caption = generate_caption()

        metadata.append({
            "image": f"images/{img_filename}",
            "caption": caption,
        })

        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{num_images} images")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {metadata_path}")
    print(f"  Total: {num_images} images")


def create_instruction_data(
    json_path: str,
    images_dir: str,
    num_samples: int = 10,
):
    """Create dummy LLaVA-style instruction data for Stage 2 training."""
    print(f"Creating instruction dummy data...")

    os.makedirs(images_dir, exist_ok=True)

    samples = []

    for i in range(num_samples):
        # Create and save image
        img = create_random_image()
        img_filename = f"{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path, "JPEG", quality=85)

        # Generate conversation
        conversation = generate_conversation()

        samples.append({
            "id": f"dummy_{i:06d}",
            "image": img_filename,
            "conversations": conversation,
        })

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {num_samples} samples to {json_path}")
    print(f"  Images saved to {images_dir}")


def main():
    """Create all dummy data."""
    # Get the project root (assuming this script is in scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    print("=" * 60)
    print("Creating dummy data for AnyMAL training pipeline")
    print("=" * 60)

    # Create LAION-style data for Stage 1
    laion_dir = os.path.join(data_dir, "laion_subset")
    create_laion_data(laion_dir, num_images=100)

    print()

    # Create instruction data for Stage 2
    instruction_json = os.path.join(data_dir, "llava_instruct_sample.json")
    dummy_images_dir = os.path.join(data_dir, "dummy_images")
    create_instruction_data(instruction_json, dummy_images_dir, num_samples=10)

    print()
    print("=" * 60)
    print("Dummy data creation complete!")
    print("=" * 60)
    print()
    print("Data locations:")
    print(f"  Stage 1 (LAION): {laion_dir}")
    print(f"  Stage 2 (Instructions): {instruction_json}")
    print(f"  Stage 2 (Images): {dummy_images_dir}")
    print()
    print("To test training:")
    print("  Stage 1: python scripts/train_pretrain.py --config configs/pretrain_image.yaml --debug")
    print("  Stage 2: python scripts/train_finetune.py --config configs/finetune.yaml --debug")


if __name__ == "__main__":
    main()
