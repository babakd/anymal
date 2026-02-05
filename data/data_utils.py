"""
Data Utilities for AnyMAL

Common utilities for data processing including image transforms,
text processing, and data loading helpers.

Educational Notes:
-----------------
Image Preprocessing for CLIP:
1. Resize to 224x224 (or 336x336 for high-res)
2. Center crop
3. Convert to tensor [0, 1]
4. Normalize with CLIP stats: mean=[0.48145466, 0.4578275, 0.40821073]
                             std=[0.26862954, 0.26130258, 0.27577711]

These normalization values come from the original CLIP training.
Using different values will hurt performance since the model
expects inputs in this distribution.

Text Processing for LLaMA:
1. Tokenize with LLaMA tokenizer (128K vocab, BPE)
2. Add special tokens (BOS, EOS)
3. Pad/truncate to max length
4. Create attention mask

Conversation Format for Instruction Tuning:
Uses ChatML-style format with role markers:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
{user message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant response}<|eot_id|>
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torchvision import transforms
from typing import Optional, Dict, Any, List, Tuple, Callable
from PIL import Image
import io


# CLIP normalization statistics
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_image_transform(
    image_size: int = 224,
    is_train: bool = True,
    use_augmentation: bool = True,
) -> transforms.Compose:
    """
    Get image transformation pipeline for CLIP.

    Args:
        image_size: Target image size (224 or 336)
        is_train: Whether this is for training
        use_augmentation: Whether to use data augmentation

    Returns:
        Composed transform pipeline

    Training augmentations (when enabled):
    - Random resized crop (scale 0.9-1.0)
    - Random horizontal flip

    Validation/Inference:
    - Resize to image_size
    - Center crop
    """
    if is_train and use_augmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    return transform


def process_image(
    image: Image.Image,
    transform: Optional[transforms.Compose] = None,
) -> torch.Tensor:
    """
    Process a single image for model input.

    Args:
        image: PIL Image
        transform: Optional transform pipeline

    Returns:
        Processed image tensor [3, H, W]
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    if transform is None:
        transform = get_image_transform(is_train=False)

    return transform(image)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image from bytes (for webdataset/streaming)."""
    return Image.open(io.BytesIO(image_bytes))


class TextProcessor:
    """
    Text processor for LLaMA tokenization.

    Handles:
    - Tokenization with proper special tokens
    - Padding and truncation
    - Label creation with masking

    Args:
        tokenizer: LLaMA tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy ("max_length" or "longest")
        truncation: Whether to truncate long sequences
    """

    # Special tokens for conversation format (LLaMA 3)
    SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
    USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
    ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    END_TURN = "<|eot_id|>"

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def encode_text(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single text string.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            Dict with input_ids, attention_mask
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def encode_for_training(
        self,
        text: str,
        response_start_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text with labels for training.

        Labels have -100 for tokens we don't want to predict
        (system prompt, user message - only predict assistant response).

        Args:
            text: Full conversation text
            response_start_idx: Character index where response starts

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Encode with offset mapping to accurately find token boundaries
        full_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)
        offset_mapping = full_encoding["offset_mapping"].squeeze(0)

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        if response_start_idx is not None:
            # Find token index where response starts using offset mapping
            response_token_idx = 0
            for i, (start, end) in enumerate(offset_mapping):
                if start >= response_start_idx:
                    response_token_idx = i
                    break

            # Mask all tokens before response
            labels[:response_token_idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def format_conversation(
        self,
        conversations: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Format a conversation for training.

        Args:
            conversations: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Tuple of (formatted_text, response_start_idx)
        """
        parts = []

        # Add system prompt
        if system_prompt:
            parts.append(f"{self.SYSTEM_HEADER}{system_prompt}{self.END_TURN}")

        # Add conversation turns
        response_start_idx = None
        for i, turn in enumerate(conversations):
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))

            if role in ["user", "human"]:
                parts.append(f"{self.USER_HEADER}{content}{self.END_TURN}")
            elif role in ["assistant", "gpt"]:
                # Mark where response starts (for label masking)
                if response_start_idx is None:
                    response_start_idx = len("".join(parts)) + len(self.ASSISTANT_HEADER)
                parts.append(f"{self.ASSISTANT_HEADER}{content}{self.END_TURN}")

        formatted_text = "".join(parts)
        return formatted_text, response_start_idx or 0


def collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function for DataLoader.

    Handles:
    - Filtering out None samples (from failed image loads)
    - Stacking images
    - Padding text sequences
    - Creating attention masks

    Args:
        batch: List of sample dicts (may contain None for failed loads)
        pad_token_id: Token ID for padding

    Returns:
        Batched dict with tensors, or None if all samples are invalid
    """
    # Filter out None samples (from corrupted/missing images)
    batch = [sample for sample in batch if sample is not None]

    # Return None if all samples were invalid
    if len(batch) == 0:
        return None

    # Separate different data types
    images = []
    input_ids = []
    attention_masks = []
    labels = []

    for sample in batch:
        if "image" in sample:
            images.append(sample["image"])
        input_ids.append(sample["input_ids"])
        attention_masks.append(sample["attention_mask"])
        if "labels" in sample:
            labels.append(sample["labels"])

    # Stack images if present
    result = {}
    if images:
        result["images"] = torch.stack(images)

    # Pad sequences to same length
    max_len = max(len(ids) for ids in input_ids)

    padded_ids = []
    padded_masks = []
    padded_labels = []

    for i in range(len(input_ids)):
        ids = input_ids[i]
        mask = attention_masks[i]
        pad_len = max_len - len(ids)

        # Pad input_ids (pad on right)
        padded_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
        )

        # Pad attention_mask with zeros
        padded_masks.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        )

        # Pad labels with -100 (ignore index)
        if labels:
            lab = labels[i]
            padded_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])
            )

    result["input_ids"] = torch.stack(padded_ids)
    result["attention_mask"] = torch.stack(padded_masks)
    if padded_labels:
        result["labels"] = torch.stack(padded_labels)

    return result


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Build DataLoader with optional distributed sampling.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        distributed: Whether to use distributed sampling
        collate_fn: Custom collate function

    Returns:
        Configured DataLoader
    """
    sampler = None
    is_iterable = isinstance(dataset, IterableDataset)

    # IterableDatasets (e.g. webdataset streaming) cannot use a sampler or `shuffle=True` in DataLoader.
    if is_iterable:
        shuffle = False

    if distributed and not is_iterable:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop incomplete batches for training
    )


class ImageTextCollator:
    """
    Collator class for image-text pairs.

    This is used as the collate_fn for DataLoaders.

    Args:
        tokenizer: LLaMA tokenizer
        max_length: Maximum text length
        pad_token_id: Token ID for padding
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        pad_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return collate_fn(batch, pad_token_id=self.pad_token_id)
