"""
Instruction Dataset for AnyMAL Stage 2 Fine-tuning

Loads multimodal instruction tuning data (LLaVA-Instruct-150K format)
for training the model to follow visual instructions.

Educational Notes:
-----------------
Stage 2 Instruction Tuning teaches the model to:
1. Understand and follow user instructions about images
2. Generate helpful, detailed responses
3. Handle multi-turn conversations about visual content

Data Format (LLaVA-Instruct-150K):
```json
{
    "id": "unique_id",
    "image": "image_filename.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "The image shows..."},
        {"from": "human", "value": "What else do you notice?"},
        {"from": "gpt", "value": "I also notice..."}
    ]
}
```

Key differences from Stage 1:
- Multi-turn conversations (not just captions)
- Instruction-following format
- Trains LoRA adapters (not just projector)
- Smaller dataset (150K vs millions)
- More complex reasoning tasks

The <image> token is a placeholder that gets replaced with
actual image tokens from the vision encoder.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import json
import os

from .data_utils import get_image_transform, TextProcessor


class InstructionDataset(Dataset):
    """
    Instruction tuning dataset for multimodal conversations.

    Loads LLaVA-Instruct-150K style data for Stage 2 fine-tuning.

    Args:
        data_path: Path to JSON file or directory
        image_dir: Directory containing images
        tokenizer: LLaMA tokenizer
        image_size: Target image size
        max_length: Maximum text length
        split: Dataset split ("train" or "val")
        system_prompt: System prompt for conversations

    Example:
        >>> dataset = InstructionDataset(
        ...     data_path="llava_instruct_150k.json",
        ...     image_dir="./images",
        ...     tokenizer=tokenizer,
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """

    # Default system prompt for instruction following
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant that can see and understand images. "
        "Provide detailed, accurate, and helpful responses to questions about images."
    )

    # Placeholder for image tokens
    IMAGE_PLACEHOLDER = "<image>"

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str],
        tokenizer,
        image_size: int = 224,
        max_length: int = 2048,
        split: str = "train",
        system_prompt: Optional[str] = None,
        filter_to_available_images: bool = False,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Image transform
        self.transform = get_image_transform(
            image_size=image_size,
            is_train=(split == "train"),
            use_augmentation=False,  # Less augmentation for instruction tuning
        )

        # Text processor
        self.text_processor = TextProcessor(
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # Load data
        self.samples = self._load_data(data_path)

        # Filter to only samples with available images
        if filter_to_available_images and image_dir is not None:
            self.samples = self._filter_to_available_images()

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load conversation data from file."""
        if os.path.isfile(data_path):
            if data_path.endswith(".json"):
                with open(data_path, "r") as f:
                    data = json.load(f)
            elif data_path.endswith(".jsonl"):
                data = []
                with open(data_path, "r") as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise ValueError(f"Unknown file format: {data_path}")
        else:
            raise ValueError(f"Data path not found: {data_path}")

        return data

    def _filter_to_available_images(self) -> List[Dict]:
        """Filter samples to only those with available image files."""
        if self.image_dir is None:
            return self.samples

        # Get set of available images for fast lookup
        available_images = set()
        for f in os.listdir(self.image_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                available_images.add(f)

        # Filter samples
        original_count = len(self.samples)
        filtered = [
            s for s in self.samples
            if s.get("image") in available_images
        ]

        print(f"Filtered dataset: {len(filtered)}/{original_count} samples have real images")
        return filtered

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single conversation sample.

        Returns:
            Dict containing:
            - image: Processed image tensor [3, H, W]
            - input_ids: Token IDs [seq_len]
            - attention_mask: Attention mask [seq_len]
            - labels: Labels for training [seq_len]
        """
        sample = self.samples[idx]

        # Load image (real if available, otherwise reproducible dummy)
        image_tensor = self._load_image(sample)

        # Get conversations
        conversations = sample["conversations"]

        # Format conversation for training
        formatted_text, response_indices = self._format_conversation(conversations)

        # Encode text
        encoding = self._encode_with_response_masking(
            formatted_text,
            response_indices,
        )

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        }

    def _load_image(self, sample: Dict) -> torch.Tensor:
        """
        Load image from disk.

        Args:
            sample: Data sample containing 'image' key

        Returns:
            Image tensor [3, H, W]

        Raises:
            FileNotFoundError: If image_dir is set but image cannot be loaded
        """
        image_filename = sample.get("image", "")

        # If image_dir is provided, we expect real images - no fallback to dummy
        if self.image_dir is not None:
            if not image_filename:
                raise ValueError(f"Sample has no 'image' field: {sample.get('id', 'unknown')}")

            image_path = os.path.join(self.image_dir, image_filename)
            try:
                image = Image.open(image_path).convert("RGB")
                return self.transform(image)
            except (IOError, OSError) as e:
                raise FileNotFoundError(
                    f"Cannot load image {image_path}: {e}. "
                    "Use filter_to_available_images=True to filter dataset."
                )

        # No image_dir - return reproducible dummy image
        if image_filename:
            seed = hash(image_filename) % (2**32)
            generator = torch.Generator().manual_seed(seed)
            return torch.randn(3, self.image_size, self.image_size, generator=generator)
        else:
            return torch.randn(3, self.image_size, self.image_size)

    def _format_conversation(
        self,
        conversations: List[Dict[str, str]],
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Format conversations into a single string for training.

        We track where assistant responses start and end so we can
        mask the loss on user turns (only train on assistant outputs).

        Returns:
            Tuple of (formatted_text, response_indices)
            response_indices: List of (start, end) character positions
        """
        parts = []
        response_indices = []

        # Add system prompt
        parts.append(
            f"{self.text_processor.SYSTEM_HEADER}"
            f"{self.system_prompt}"
            f"{self.text_processor.END_TURN}"
        )

        # Process each turn
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            # Remove image placeholder from text
            # (image tokens are prepended separately)
            content = content.replace(self.IMAGE_PLACEHOLDER, "").strip()

            if role in ["human", "user"]:
                parts.append(
                    f"{self.text_processor.USER_HEADER}"
                    f"{content}"
                    f"{self.text_processor.END_TURN}"
                )
            elif role in ["gpt", "assistant"]:
                # Track response position for label masking
                current_pos = len("".join(parts))
                header_len = len(self.text_processor.ASSISTANT_HEADER)
                response_start = current_pos + header_len
                response_end = response_start + len(content)
                response_indices.append((response_start, response_end))

                parts.append(
                    f"{self.text_processor.ASSISTANT_HEADER}"
                    f"{content}"
                    f"{self.text_processor.END_TURN}"
                )

        formatted_text = "".join(parts)
        return formatted_text, response_indices

    def _encode_with_response_masking(
        self,
        text: str,
        response_indices: List[Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text and create labels with response masking.

        Only assistant responses are used for loss computation.
        User messages and system prompts are masked with -100.
        """
        # Tokenize the full text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Create labels - start with all masked
        labels = torch.full_like(input_ids, -100)

        # Unmask only the response tokens
        for start_char, end_char in response_indices:
            for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
                # Token is part of response if it overlaps with response span
                if token_end > start_char and token_start < end_char:
                    labels[i] = input_ids[i]

        # Mask padding tokens
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InstructionDatasetSimple(Dataset):
    """
    Simplified instruction dataset for quick experiments.

    Uses simple format without complex conversation handling.
    Good for testing and small-scale experiments.

    Args:
        data_path: Path to JSON file with entries like:
            [{"image": "path.jpg", "instruction": "...", "response": "..."}]
        image_dir: Base directory for images
        tokenizer: LLaMA tokenizer
        image_size: Target image size
        max_length: Maximum text length
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer,
        image_size: int = 224,
        max_length: int = 2048,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = get_image_transform(
            image_size=image_size,
            is_train=True,
        )

        # Load data
        with open(data_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # Format text
        instruction = sample.get("instruction", sample.get("question", ""))
        response = sample.get("response", sample.get("answer", ""))

        # Simple format: "User: {instruction}\nAssistant: {response}"
        prompt = f"User: {instruction}\nAssistant:"
        full_text = f"{prompt} {response}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels - mask the prompt portion
        labels = input_ids.clone()
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        labels[:len(prompt_tokens)] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_instruction_dataset(
    data_path: str,
    image_dir: str,
    tokenizer,
    simple: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory function to create instruction dataset.

    Args:
        data_path: Path to data file
        image_dir: Directory containing images
        tokenizer: LLaMA tokenizer
        simple: Use simplified dataset (for quick tests)
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    if simple:
        return InstructionDatasetSimple(
            data_path, image_dir, tokenizer, **kwargs
        )
    else:
        return InstructionDataset(
            data_path, image_dir, tokenizer, **kwargs
        )
