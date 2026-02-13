"""
LAION Dataset for AnyMAL Alignment Pretraining

Loads LAION-2B image-text pairs for training the projector to align
visual features with the LLM's text embedding space.

Educational Notes:
-----------------
LAION-2B is a large-scale dataset of 2 billion image-text pairs
scraped from the internet. For AnyMAL alignment pretraining:

1. Purpose: Teach the Perceiver Resampler to project CLIP features
   into the LLM's embedding space

2. Training Objective: Next-token prediction on captions
   - Input: [image_tokens, "Caption: "]
   - Target: [caption_tokens]
   - Loss: Cross-entropy on caption tokens only

3. Data Quality: The paper uses CAT filtering for quality control
   - Remove low-quality image-text pairs
   - Filter inappropriate content
   - We use relaion2B-en-research-safe which is pre-filtered

4. Data Format: We use webdataset format for efficient streaming
   - Images stored as .jpg/.png files
   - Captions stored as .txt files
   - Metadata in .json files

Recommended subset sizes for educational purposes:
- Quick test: 1M samples
- Small run: 10M samples
- Medium run: 50M samples
- Full replication: 200M samples
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, Dict, Any, Iterator, Callable, List
from PIL import Image
import json
import os
import glob
import re

from .data_utils import get_image_transform, TextProcessor


def _expand_brace_pattern(pattern: str) -> Optional[List[str]]:
    """
    Expand a single `{00000..00123}` brace pattern into an explicit list.

    This supports common webdataset shard patterns like:
      ./data/cc3m/{00000..00331}.tar
    """
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not match:
        return None

    start_str, end_str = match.groups()
    width = len(start_str)
    start = int(start_str)
    end = int(end_str)
    if end < start:
        return None

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    return [f"{prefix}{i:0{width}d}{suffix}" for i in range(start, end + 1)]


def _expand_webdataset_urls(data_path: str) -> List[str]:
    """
    Expand common shard patterns into a list of shard paths/URLs.

    Order:
    1) Brace expansion `{00000..00099}`
    2) Glob expansion `*.tar`
    3) Fallback to the original string
    """
    brace_expanded = _expand_brace_pattern(data_path)
    if brace_expanded:
        return brace_expanded

    globbed = sorted(glob.glob(data_path))
    if globbed:
        return globbed

    return [data_path]


class LaionDataset(Dataset):
    """
    LAION dataset for alignment pretraining.

    This is a map-style dataset for smaller subsets.
    For large-scale training, use LaionStreamingDataset.

    Args:
        data_path: Path to data directory or parquet file
        tokenizer: LLaMA tokenizer
        image_size: Target image size (224 or 336)
        max_length: Maximum text length
        split: Dataset split ("train" or "val")
        max_samples: Maximum number of samples to load
        caption_prompt: Prompt template for captions

    Example:
        >>> dataset = LaionDataset(
        ...     data_path="./data/laion_subset",
        ...     tokenizer=tokenizer,
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['image', 'input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_size: int = 224,
        max_length: int = 256,
        split: str = "train",
        max_samples: Optional[int] = None,
        caption_prompt: str = "A photo of",
        insert_image_placeholders: bool = False,
        num_image_tokens: int = 64,
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompt = caption_prompt
        self.insert_image_placeholders = insert_image_placeholders
        self.num_image_tokens = num_image_tokens

        # Set up image transform
        self.transform = get_image_transform(
            image_size=image_size,
            is_train=(split == "train"),
        )

        # Set up text processor
        self.text_processor = TextProcessor(
            tokenizer=tokenizer,
            max_length=max_length,
        )
        self.image_placeholder_token_id = self._resolve_placeholder_token_id(tokenizer)

        # Load data index
        self.samples = self._load_samples(data_path, max_samples)

    def _resolve_placeholder_token_id(self, tokenizer):
        if not self.insert_image_placeholders:
            return None
        vocab = tokenizer.get_vocab()
        if "<|reserved_special_token_0|>" in vocab:
            return vocab["<|reserved_special_token_0|>"]
        if "<|image|>" in vocab:
            return vocab["<|image|>"]
        raise ValueError(
            "insert_image_placeholders=True but tokenizer has no image placeholder token. "
            "Initialize model placeholder token before creating the dataset."
        )

    def _prepend_image_placeholders(
        self,
        encoding: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Prepend fixed-size image placeholder block for strict v2 splice."""
        if not self.insert_image_placeholders:
            return encoding

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        labels = encoding["labels"]

        placeholder_block = torch.full(
            (self.num_image_tokens,),
            self.image_placeholder_token_id,
            dtype=input_ids.dtype,
        )
        placeholder_mask = torch.ones(self.num_image_tokens, dtype=attention_mask.dtype)
        placeholder_labels = torch.full((self.num_image_tokens,), -100, dtype=labels.dtype)

        input_ids = torch.cat([placeholder_block, input_ids], dim=0)
        attention_mask = torch.cat([placeholder_mask, attention_mask], dim=0)
        labels = torch.cat([placeholder_labels, labels], dim=0)

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        elif input_ids.shape[0] < self.max_length:
            pad_len = self.max_length - input_ids.shape[0]
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)],
                dim=0,
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=labels.dtype)],
                dim=0,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _load_samples(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ) -> list:
        """
        Load sample metadata from data path.

        Supports multiple formats:
        - Directory with image files + metadata.json
        - Parquet file with URLs and captions
        - JSON lines file
        """
        samples = []

        # Check for different data formats
        if os.path.isfile(data_path):
            if data_path.endswith(".parquet"):
                samples = self._load_from_parquet(data_path)
            elif data_path.endswith(".jsonl"):
                samples = self._load_from_jsonl(data_path)
        elif os.path.isdir(data_path):
            samples = self._load_from_directory(data_path)
        else:
            raise ValueError(f"Unknown data path format: {data_path}")

        # Limit samples if specified
        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    def _load_from_directory(self, data_path: str) -> list:
        """Load from directory with images and metadata."""
        samples = []

        # Look for metadata file
        metadata_path = os.path.join(data_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            for item in metadata:
                samples.append({
                    "image_path": os.path.join(data_path, item["image"]),
                    "caption": item["caption"],
                })
        else:
            # Fall back to looking for .txt files alongside images
            for filename in os.listdir(data_path):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(data_path, filename)
                    caption_path = image_path.rsplit(".", 1)[0] + ".txt"
                    if os.path.exists(caption_path):
                        with open(caption_path, "r") as f:
                            caption = f.read().strip()
                        samples.append({
                            "image_path": image_path,
                            "caption": caption,
                        })

        return samples

    def _load_from_parquet(self, data_path: str) -> list:
        """Load from parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            samples = df.to_dict("records")
        except ImportError:
            raise ImportError("pandas required for parquet files")
        return samples

    def _load_from_jsonl(self, data_path: str) -> list:
        """Load from JSON lines file."""
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Dict containing:
            - image: Processed image tensor [3, H, W]
            - input_ids: Token IDs [seq_len]
            - attention_mask: Attention mask [seq_len]
            - labels: Labels for training [seq_len]

            Returns None if the image fails to load (corrupted/missing).
        """
        sample = self.samples[idx]

        # Load and process image
        image_path = sample.get("image_path", sample.get("image"))
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except (IOError, OSError) as e:
            # Handle corrupted or missing images
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None

        # Get caption
        caption = sample.get("caption", sample.get("text", ""))

        # Format text for training
        # The model sees: "A photo of [caption]"
        # We only compute loss on the caption part
        prompt = self.caption_prompt
        full_text = f"{prompt} {caption}"

        # Encode text
        encoding = self.text_processor.encode_for_training(
            full_text,
            response_start_idx=len(prompt) + 1,  # Start after prompt
        )
        encoding = self._prepend_image_placeholders(encoding)

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        }


class LaionStreamingDataset(IterableDataset):
    """
    Streaming LAION dataset using webdataset.

    For large-scale training where data doesn't fit in memory.
    Uses webdataset format for efficient streaming from disk or cloud.

    Args:
        data_path: Path to webdataset shards (can use glob patterns)
        tokenizer: LLaMA tokenizer
        image_size: Target image size
        max_length: Maximum text length
        buffer_size: Shuffle buffer size
        caption_prompt: Prompt template for captions

    Example:
        >>> dataset = LaionStreamingDataset(
        ...     data_path="./data/laion-{00000..00099}.tar",
        ...     tokenizer=tokenizer,
        ... )
        >>> for batch in DataLoader(dataset, batch_size=32):
        ...     print(batch["image"].shape)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_size: int = 224,
        max_length: int = 256,
        buffer_size: int = 10000,
        caption_prompt: str = "A photo of",
        insert_image_placeholders: bool = False,
        num_image_tokens: int = 64,
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.caption_prompt = caption_prompt
        self.insert_image_placeholders = insert_image_placeholders
        self.num_image_tokens = num_image_tokens

        self.transform = get_image_transform(
            image_size=image_size,
            is_train=True,
        )

        self.text_processor = TextProcessor(
            tokenizer=tokenizer,
            max_length=max_length,
        )
        self.image_placeholder_token_id = self._resolve_placeholder_token_id(tokenizer)

    def _resolve_placeholder_token_id(self, tokenizer):
        if not self.insert_image_placeholders:
            return None
        vocab = tokenizer.get_vocab()
        if "<|reserved_special_token_0|>" in vocab:
            return vocab["<|reserved_special_token_0|>"]
        if "<|image|>" in vocab:
            return vocab["<|image|>"]
        raise ValueError(
            "insert_image_placeholders=True but tokenizer has no image placeholder token."
        )

    def _prepend_image_placeholders(
        self,
        encoding: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.insert_image_placeholders:
            return encoding

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        labels = encoding["labels"]

        placeholder_block = torch.full(
            (self.num_image_tokens,),
            self.image_placeholder_token_id,
            dtype=input_ids.dtype,
        )
        placeholder_mask = torch.ones(self.num_image_tokens, dtype=attention_mask.dtype)
        placeholder_labels = torch.full((self.num_image_tokens,), -100, dtype=labels.dtype)

        input_ids = torch.cat([placeholder_block, input_ids], dim=0)
        attention_mask = torch.cat([placeholder_mask, attention_mask], dim=0)
        labels = torch.cat([placeholder_labels, labels], dim=0)

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        elif input_ids.shape[0] < self.max_length:
            pad_len = self.max_length - input_ids.shape[0]
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)],
                dim=0,
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=labels.dtype)],
                dim=0,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _create_pipeline(self):
        """Create webdataset pipeline."""
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError("webdataset required for streaming dataset")

        urls = _expand_webdataset_urls(self.data_path)

        # Split shards across distributed ranks to avoid duplicate samples.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            urls = urls[rank::world_size] or urls

        # Create pipeline
        pipeline = wds.WebDataset(urls)

        # Shuffle if buffer_size > 0
        if self.buffer_size > 0:
            pipeline = pipeline.shuffle(self.buffer_size)

        # Decode images and text
        pipeline = pipeline.decode("pil")

        # Map to our format
        pipeline = pipeline.map(self._process_sample)

        return pipeline

    def _process_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Process a single webdataset sample."""
        # Get image (try different keys)
        image = None
        for key in ["jpg", "png", "jpeg", "webp"]:
            if key in sample:
                image = sample[key]
                break

        if image is None:
            raise ValueError(f"No image found in sample: {sample.keys()}")

        # Convert to RGB and transform
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.transform(image)

        # Get caption (try different keys)
        caption = ""
        for key in ["txt", "caption", "text", "json"]:
            if key in sample:
                if key == "json":
                    caption = sample[key].get("caption", "")
                else:
                    caption = sample[key]
                break

        # Format and encode text
        full_text = f"{self.caption_prompt} {caption}"
        encoding = self.text_processor.encode_for_training(
            full_text,
            response_start_idx=len(self.caption_prompt) + 1,
        )
        encoding = self._prepend_image_placeholders(encoding)

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        pipeline = self._create_pipeline()
        return iter(pipeline)


def create_laion_dataset(
    data_path: str,
    tokenizer,
    streaming: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory function to create LAION dataset.

    Args:
        data_path: Path to data
        tokenizer: LLaMA tokenizer
        streaming: Whether to use streaming (webdataset)
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    if streaming:
        return LaionStreamingDataset(data_path, tokenizer, **kwargs)
    else:
        return LaionDataset(data_path, tokenizer, **kwargs)
