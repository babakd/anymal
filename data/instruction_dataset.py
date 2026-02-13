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
from torchvision import transforms
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import json
import os
import random

from .data_utils import get_image_transform, TextProcessor, CLIP_MEAN, CLIP_STD


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
        num_image_tokens: int = 64,
        image_token_policy: str = "fixed",
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
        split: str = "train",
        system_prompt: Optional[str] = None,
        filter_to_available_images: bool = False,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.image_token_policy = image_token_policy
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if self.image_token_policy not in {"fixed", "uniform"}:
            raise ValueError(
                f"Unsupported image_token_policy '{self.image_token_policy}'. "
                "Expected one of ['fixed', 'uniform']."
            )

        if self.image_token_policy == "uniform":
            if self.min_image_tokens is None or self.max_image_tokens is None:
                raise ValueError(
                    "image_token_policy='uniform' requires min_image_tokens and max_image_tokens."
                )
            if self.min_image_tokens <= 0 or self.max_image_tokens <= 0:
                raise ValueError("min_image_tokens and max_image_tokens must be > 0.")
            if self.min_image_tokens > self.max_image_tokens:
                raise ValueError(
                    f"min_image_tokens ({self.min_image_tokens}) must be <= "
                    f"max_image_tokens ({self.max_image_tokens})."
                )

        # Resolve the image placeholder token ID (must match AnyMAL model)
        vocab = tokenizer.get_vocab()
        if "<|reserved_special_token_0|>" in vocab:
            self.image_placeholder_token_id = vocab["<|reserved_special_token_0|>"]
        elif "<|image|>" in vocab:
            self.image_placeholder_token_id = vocab["<|image|>"]
        else:
            self.image_placeholder_token_id = None  # fallback: strip <image>

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
        formatted_text, response_indices, image_sentinel = self._format_conversation(conversations)

        num_image_tokens = self._sample_num_image_tokens()

        # Encode text
        encoding = self._encode_with_response_masking(
            formatted_text,
            response_indices,
            image_sentinel=image_sentinel,
            num_image_tokens=num_image_tokens,
        )

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
            "num_image_tokens": torch.tensor(num_image_tokens, dtype=torch.long),
        }

    def _sample_num_image_tokens(self) -> int:
        """Sample per-sample image placeholder length from policy."""
        if self.image_token_policy == "fixed":
            if self.max_image_tokens is not None:
                return int(self.max_image_tokens)
            return int(self.num_image_tokens)

        return random.randint(int(self.min_image_tokens), int(self.max_image_tokens))

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

        # No image_dir - return reproducible dummy image with CLIP normalization
        # Use rand (uniform [0,1]) + normalization instead of randn (unbounded Gaussian)
        # so dummy images match the distribution of real CLIP-normalized images.
        clip_normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        if image_filename:
            seed = hash(image_filename) % (2**32)
            generator = torch.Generator().manual_seed(seed)
            raw = torch.rand(3, self.image_size, self.image_size, generator=generator)
        else:
            raw = torch.rand(3, self.image_size, self.image_size)
        return clip_normalize(raw)

    def _format_conversation(
        self,
        conversations: List[Dict[str, str]],
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Format conversations into a single string for training.

        If image placeholder support is available, <image> is replaced with
        a sentinel string that will later be replaced with placeholder token IDs.
        Otherwise, <image> is stripped (backward compatible).

        Returns:
            Tuple of (formatted_text, response_indices)
            response_indices: List of (start, end) character positions
        """
        parts = []
        response_indices = []

        # Use a sentinel string that won't appear naturally in text.
        # After tokenization, we replace these with actual placeholder token IDs.
        IMAGE_SENTINEL = "<|image_sentinel|>"

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

            # Replace <image> with sentinel or strip it
            if self.image_placeholder_token_id is not None:
                content = content.replace(self.IMAGE_PLACEHOLDER, IMAGE_SENTINEL).strip()
            else:
                content = content.replace(self.IMAGE_PLACEHOLDER, "").strip()

            if role in ["human", "user"]:
                parts.append(
                    f"{self.text_processor.USER_HEADER}"
                    f"{content}"
                    f"{self.text_processor.END_TURN}"
                )
            elif role in ["gpt", "assistant"]:
                # Track response position for label masking
                # Include the end-of-turn token so the model learns to stop generating
                current_pos = len("".join(parts))
                header_len = len(self.text_processor.ASSISTANT_HEADER)
                response_start = current_pos + header_len
                response_end = response_start + len(content) + len(self.text_processor.END_TURN)
                response_indices.append((response_start, response_end))

                parts.append(
                    f"{self.text_processor.ASSISTANT_HEADER}"
                    f"{content}"
                    f"{self.text_processor.END_TURN}"
                )

        formatted_text = "".join(parts)
        return formatted_text, response_indices, IMAGE_SENTINEL

    def _encode_with_response_masking(
        self,
        text: str,
        response_indices: List[Tuple[int, int]],
        image_sentinel: str = "",
        num_image_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text and create labels with response masking.

        Only assistant responses are used for loss computation.
        User messages and system prompts are masked with -100.

        If image_sentinel is present in the text and we have a placeholder
        token ID, the sentinel tokens are replaced with placeholder IDs
        (N copies of the placeholder token, where N = num_image_tokens).
        """
        if num_image_tokens is None:
            num_image_tokens = self.num_image_tokens

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

        # Replace sentinel tokens with image placeholder token IDs
        if image_sentinel and self.image_placeholder_token_id is not None and image_sentinel in text:
            # Find character positions of the sentinel in the text
            sentinel_start = text.find(image_sentinel)
            sentinel_end = sentinel_start + len(image_sentinel)

            # Find which tokens correspond to the sentinel
            sentinel_token_indices = []
            for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
                if token_end > sentinel_start and token_start < sentinel_end:
                    sentinel_token_indices.append(i)

            if sentinel_token_indices:
                first_idx = sentinel_token_indices[0]
                num_sentinel_tokens = len(sentinel_token_indices)

                # We need to replace sentinel tokens with num_image_tokens placeholder tokens.
                # If sentinel tokenizes to fewer tokens, we need to expand; if more, contract.
                # Build a new input_ids tensor with the right number of placeholder tokens.
                before = input_ids[:first_idx]
                after = input_ids[first_idx + num_sentinel_tokens:]
                placeholder_block = torch.full(
                    (num_image_tokens,), self.image_placeholder_token_id,
                    dtype=input_ids.dtype
                )
                input_ids = torch.cat([before, placeholder_block, after])

                # Rebuild attention mask
                mask_before = attention_mask[:first_idx]
                mask_after = attention_mask[first_idx + num_sentinel_tokens:]
                mask_placeholder = torch.ones(num_image_tokens, dtype=attention_mask.dtype)
                attention_mask = torch.cat([mask_before, mask_placeholder, mask_after])

                # Rebuild offset_mapping for label computation
                offsets_before = offset_mapping[:first_idx]
                offsets_after = offset_mapping[first_idx + num_sentinel_tokens:]
                # Placeholder tokens get zero offsets (they'll be masked in labels)
                offsets_placeholder = torch.zeros(num_image_tokens, 2, dtype=offset_mapping.dtype)
                offset_mapping = torch.cat([offsets_before, offsets_placeholder, offsets_after])

                # Truncate or pad to max_length
                if input_ids.shape[0] > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    offset_mapping = offset_mapping[:self.max_length]
                elif input_ids.shape[0] < self.max_length:
                    pad_len = self.max_length - input_ids.shape[0]
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                    offset_mapping = torch.cat([offset_mapping, torch.zeros(pad_len, 2, dtype=offset_mapping.dtype)])

        # Create labels - start with all masked
        labels = torch.full_like(input_ids, -100)

        # Unmask only the response tokens
        for start_char, end_char in response_indices:
            for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
                # Token is part of response if it overlaps with response span
                if token_end > start_char and token_start < end_char:
                    labels[i] = input_ids[i]

        # Mask padding tokens and image placeholder tokens in labels
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        if self.image_placeholder_token_id is not None:
            labels[input_ids == self.image_placeholder_token_id] = -100

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
