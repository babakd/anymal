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
from typing import Optional, Dict, Any, Iterator, Callable, List, Tuple
from PIL import Image
import io
import json
import os
import glob
import re
import zipfile

from .data_utils import get_vision_transform, TextProcessor


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
        vision_encoder_type: str = "clip",
        vision_model_name: Optional[str] = None,
        image_dir: Optional[str] = None,
        image_zip_path: Optional[str] = None,
        filter_to_available_images: bool = False,
        min_caption_chars: int = 1,
        deduplicate_captions: bool = False,
    ):
        super().__init__()

        self.data_path = data_path
        self.image_dir = image_dir
        self.image_zip_path = image_zip_path
        self._image_zip = None
        self._zip_member_index = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_prompt = caption_prompt
        self.insert_image_placeholders = insert_image_placeholders
        self.num_image_tokens = num_image_tokens
        self.vision_encoder_type = vision_encoder_type
        self.vision_model_name = vision_model_name
        self.filter_to_available_images = filter_to_available_images
        self.min_caption_chars = min_caption_chars
        self.deduplicate_captions = deduplicate_captions

        # Set up image transform
        self.transform = get_vision_transform(
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
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

    def _caption_text_and_start(self, caption: str) -> Tuple[str, int]:
        """Build the caption training string and response start char index."""
        if self.caption_prompt:
            return f"{self.caption_prompt} {caption}", len(self.caption_prompt) + 1
        return caption, 0

    def _extract_caption(self, item: Dict[str, Any]) -> str:
        """Extract caption text from common caption/pretrain JSON records."""
        caption = item.get("caption", item.get("text", ""))
        if caption:
            return str(caption).replace("<image>", "").strip()

        for turn in item.get("conversations", []):
            if turn.get("from", turn.get("role")) in {"gpt", "assistant"}:
                return str(turn.get("value", turn.get("content", ""))).replace("<image>", "").strip()

        return ""

    def _resolve_image_path(self, image_ref: str, data_path: str) -> str:
        """Resolve image references relative to image_dir first, then data_path."""
        if os.path.isabs(image_ref):
            return image_ref

        if self.image_dir:
            return os.path.join(self.image_dir, image_ref)

        return os.path.join(os.path.dirname(data_path), image_ref)

    def _normalize_zip_ref(self, image_ref: str) -> str:
        """Normalize JSON image refs and zip members to comparable POSIX paths."""
        return str(image_ref).replace("\\", "/").lstrip("./").lstrip("/")

    def _get_image_zip(self):
        if not self.image_zip_path:
            return None
        if self._image_zip is None:
            self._image_zip = zipfile.ZipFile(self.image_zip_path, "r")
        return self._image_zip

    def _get_zip_member_index(self) -> Dict[str, Optional[str]]:
        """
        Build a lookup for exact refs, refs under an `images/` root, and basenames.

        Basename collisions are marked ambiguous instead of guessed.
        """
        if self._zip_member_index is not None:
            return self._zip_member_index

        if not self.image_zip_path:
            self._zip_member_index = {}
            return self._zip_member_index

        member_index: Dict[str, Optional[str]] = {}
        with zipfile.ZipFile(self.image_zip_path, "r") as zf:
            members = [
                name
                for name in zf.namelist()
                if not name.endswith("/")
                and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]

        def add_key(key: str, member: str) -> None:
            key = self._normalize_zip_ref(key)
            existing = member_index.get(key)
            if existing is None and key in member_index:
                return
            if existing is not None and existing != member:
                member_index[key] = None
                return
            member_index[key] = member

        for member in members:
            norm_member = self._normalize_zip_ref(member)
            add_key(norm_member, member)
            if norm_member.startswith("images/"):
                add_key(norm_member[len("images/"):], member)
            add_key(os.path.basename(norm_member), member)

        self._zip_member_index = member_index
        return self._zip_member_index

    def _resolve_zip_member(self, image_ref: str) -> Optional[str]:
        if not self.image_zip_path:
            return None
        ref = self._normalize_zip_ref(image_ref)
        return self._get_zip_member_index().get(ref)

    def _normalize_json_item(self, item: Dict[str, Any], data_path: str) -> Optional[Dict[str, str]]:
        image_ref = item.get("image_path", item.get("image", item.get("file_name", "")))
        caption = self._extract_caption(item)
        if not image_ref or not caption:
            return None

        return {
            "image": image_ref,
            "image_path": self._resolve_image_path(str(image_ref), data_path),
            "image_zip_member": self._resolve_zip_member(str(image_ref)),
            "caption": caption,
        }

    def _filter_samples(self, samples: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Apply explicit real-image, caption-quality, and duplicate filters."""
        original_count = len(samples)

        if self.min_caption_chars > 1:
            before = len(samples)
            samples = [
                s for s in samples
                if len(str(s.get("caption", "")).strip()) >= self.min_caption_chars
            ]
            print(
                f"{source}: caption length filter kept {len(samples)}/{before} "
                f"samples (min_caption_chars={self.min_caption_chars})"
            )

        if self.filter_to_available_images:
            before = len(samples)
            if self.image_zip_path:
                samples = [
                    s for s in samples
                    if s.get("image_zip_member")
                ]
                image_root = self.image_zip_path
            else:
                samples = [
                    s for s in samples
                    if s.get("image_path") and os.path.exists(s["image_path"])
                ]
                image_root = self.image_dir or os.path.dirname(self.data_path)
            print(
                f"{source}: real-image filter kept {len(samples)}/{before} "
                f"samples from {image_root}"
            )
        else:
            print(
                f"{source}: loaded {len(samples)}/{original_count} samples; "
                "real-image filtering disabled"
            )

        if self.deduplicate_captions:
            before = len(samples)
            seen = set()
            unique = []
            for sample in samples:
                key = str(sample.get("caption", "")).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(sample)
            samples = unique
            print(f"{source}: caption de-dup kept {len(samples)}/{before} samples")

        return samples

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
            elif data_path.endswith(".json"):
                samples = self._load_from_json(data_path)
            elif data_path.endswith(".jsonl"):
                samples = self._load_from_jsonl(data_path)
        elif os.path.isdir(data_path):
            samples = self._load_from_directory(data_path)
        else:
            raise ValueError(f"Unknown data path format: {data_path}")

        samples = self._filter_samples(samples, source=self.__class__.__name__)

        # Limit samples if specified after filtering so max_samples means usable samples.
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
                item = json.loads(line)
                if isinstance(item, dict):
                    normalized = self._normalize_json_item(item, data_path)
                    samples.append(normalized or item)
        return samples

    def _load_from_json(self, data_path: str) -> list:
        """Load caption data from JSON, including LLaVA-Pretrain style files."""
        with open(data_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("data", data.get("annotations", data.get("samples", [])))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list-like JSON caption file: {data_path}")

        samples = []
        skipped = 0
        for item in data:
            if not isinstance(item, dict):
                skipped += 1
                continue
            normalized = self._normalize_json_item(item, data_path)
            if normalized is None:
                skipped += 1
                continue
            samples.append(normalized)

        if skipped:
            print(f"LaionDataset: skipped {skipped} JSON records without image/caption")
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
            if sample.get("image_zip_member"):
                with self._get_image_zip().open(sample["image_zip_member"]) as f:
                    image = Image.open(io.BytesIO(f.read())).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except (IOError, OSError, KeyError, zipfile.BadZipFile) as e:
            # Handle corrupted or missing images
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None

        # Get caption
        caption = sample.get("caption", sample.get("text", ""))

        # Format text for training. We only compute loss on the caption part.
        full_text, response_start_idx = self._caption_text_and_start(caption)

        # Encode text
        encoding = self.text_processor.encode_for_training(
            full_text,
            response_start_idx=response_start_idx,
        )
        encoding = self._prepend_image_placeholders(encoding)

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_zip"] = None
        return state


class LlavaPretrainCaptionDataset(LaionDataset):
    """
    LLaVA-Pretrain caption dataset for Stage 1 alignment.

    Supports `blip_laion_cc_sbu_558k.json` style records with either
    `caption`/`text` fields or LLaVA conversation records whose assistant turn
    contains a BLIP caption. Real image filtering is enabled by default.
    """

    def __init__(
        self,
        annotation_path: str,
        image_dir: Optional[str],
        tokenizer,
        image_zip_path: Optional[str] = None,
        caption_prompt: str = "",
        filter_to_available_images: bool = True,
        min_caption_chars: int = 3,
        deduplicate_captions: bool = True,
        **kwargs,
    ):
        super().__init__(
            data_path=annotation_path,
            image_dir=image_dir,
            image_zip_path=image_zip_path,
            tokenizer=tokenizer,
            caption_prompt=caption_prompt,
            filter_to_available_images=filter_to_available_images,
            min_caption_chars=min_caption_chars,
            deduplicate_captions=deduplicate_captions,
            **kwargs,
        )


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
        vision_encoder_type: str = "clip",
        vision_model_name: Optional[str] = None,
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.caption_prompt = caption_prompt
        self.insert_image_placeholders = insert_image_placeholders
        self.num_image_tokens = num_image_tokens
        self.vision_encoder_type = vision_encoder_type
        self.vision_model_name = vision_model_name

        self.transform = get_vision_transform(
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
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
        if self.caption_prompt:
            full_text = f"{self.caption_prompt} {caption}"
            response_start_idx = len(self.caption_prompt) + 1
        else:
            full_text = caption
            response_start_idx = 0
        encoding = self.text_processor.encode_for_training(
            full_text,
            response_start_idx=response_start_idx,
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
    dataset_type: str = "laion",
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
    if dataset_type in {"llava_pretrain", "llava_pretrain_caption"}:
        image_dir = kwargs.pop("image_dir", None)
        image_zip_path = kwargs.pop("image_zip_path", None)
        if not image_dir and not image_zip_path:
            raise ValueError("LlavaPretrainCaptionDataset requires image_dir or image_zip_path")
        if streaming:
            raise ValueError("LLaVA-Pretrain JSON caption data is map-style, not streaming")
        return LlavaPretrainCaptionDataset(
            annotation_path=data_path,
            image_dir=image_dir,
            image_zip_path=image_zip_path,
            tokenizer=tokenizer,
            **kwargs,
        )

    if streaming:
        return LaionStreamingDataset(data_path, tokenizer, **kwargs)
    else:
        return LaionDataset(data_path, tokenizer, **kwargs)
