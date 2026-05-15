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

from .chat_templates import LLAMA3_TEMPLATE, resolve_chat_template_spec


# CLIP normalization statistics
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP/SigLIP2 checkpoints in HuggingFace use processor-provided image
# statistics. These defaults match the common SigLIP processor and are used
# only if the processor config cannot be loaded.
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def _infer_hf_image_size(processor, fallback: int) -> int:
    """Infer a square image size from a HuggingFace image processor."""
    for attr in ("crop_size", "size"):
        value = getattr(processor, attr, None)
        if isinstance(value, dict):
            for key in ("height", "shortest_edge", "width"):
                if key in value:
                    return int(value[key])
        elif isinstance(value, int):
            return int(value)
    return int(fallback)


def get_siglip_image_transform(
    model_name: str = "google/siglip2-so400m-patch14-384",
    image_size: Optional[int] = None,
    is_train: bool = True,
    use_augmentation: bool = True,
    image_augmentation_mode: str = "none",
) -> transforms.Compose:
    """Get preprocessing for SigLIP/SigLIP2 vision towers."""
    mean = SIGLIP_MEAN
    std = SIGLIP_STD
    resolved_size = image_size
    image_augmentation_mode = str(image_augmentation_mode or "none").lower()
    if image_augmentation_mode not in {"none", "standard", "vqa_light"}:
        raise ValueError(
            "image_augmentation_mode must be one of: none, standard, vqa_light"
        )

    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = tuple(getattr(processor, "image_mean", mean))
        std = tuple(getattr(processor, "image_std", std))
        if resolved_size is None:
            resolved_size = _infer_hf_image_size(processor, 384)
    except Exception as e:
        print(f"Warning: could not load image processor for {model_name}: {e}. "
              "Using SigLIP fallback preprocessing.")

    if resolved_size is None:
        resolved_size = 384

    if is_train and use_augmentation and image_augmentation_mode == "vqa_light":
        return transforms.Compose([
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.15,
            ),
            transforms.RandomResizedCrop(
                resolved_size,
                scale=(0.95, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if is_train and use_augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                resolved_size,
                scale=(0.9, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize(
            resolved_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolved_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class AnyResImageTransform:
    """Deterministic global-plus-local crop pack for AnyRes-lite experiments."""

    def __init__(
        self,
        base_transform: Callable[[Image.Image], torch.Tensor],
        num_views: int = 3,
        local_crop_scale: float = 0.75,
        crop_mode: str = "global_two_crops",
    ):
        self.base_transform = base_transform
        self.num_views = int(num_views)
        self.local_crop_scale = float(local_crop_scale)
        self.crop_mode = str(crop_mode or "global_two_crops")
        if self.num_views <= 0:
            raise ValueError(f"num_views must be > 0, got {num_views}")
        if not 0.1 <= self.local_crop_scale <= 1.0:
            raise ValueError(
                "local_crop_scale must be in [0.1, 1.0], "
                f"got {local_crop_scale}"
            )

    @staticmethod
    def _clamp_crop_box(
        center_x: float,
        center_y: float,
        side: int,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        side = max(1, min(int(side), int(width), int(height)))
        left = int(round(center_x - side / 2.0))
        top = int(round(center_y - side / 2.0))
        left = max(0, min(left, int(width) - side))
        top = max(0, min(top, int(height) - side))
        return left, top, left + side, top + side

    def _local_crop_centers(self, width: int, height: int) -> List[Tuple[float, float]]:
        if width >= height * 1.05:
            return [(0.33 * width, 0.50 * height), (0.67 * width, 0.50 * height)]
        if height >= width * 1.05:
            return [(0.50 * width, 0.33 * height), (0.50 * width, 0.67 * height)]
        return [(0.35 * width, 0.35 * height), (0.65 * width, 0.65 * height)]

    def _make_local_crops(self, image: Image.Image) -> List[Image.Image]:
        width, height = image.size
        side = int(round(min(width, height) * self.local_crop_scale))
        centers = self._local_crop_centers(width, height)
        crops = []
        for center_x, center_y in centers:
            crops.append(
                image.crop(
                    self._clamp_crop_box(center_x, center_y, side, width, height)
                )
            )
        while len(crops) < max(0, self.num_views - 1):
            crops.append(crops[-1] if crops else image)
        return crops[: max(0, self.num_views - 1)]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        views = [self.base_transform(image)]
        for crop in self._make_local_crops(image):
            views.append(self.base_transform(crop))
        if len(views) > self.num_views:
            views = views[: self.num_views]
        while len(views) < self.num_views:
            views.append(views[-1].clone())
        return torch.stack(views, dim=0)


def get_vision_transform(
    vision_encoder_type: str = "clip",
    vision_model_name: Optional[str] = None,
    image_size: Optional[int] = 224,
    is_train: bool = True,
    use_augmentation: bool = True,
    image_augmentation_mode: str = "none",
    image_view_mode: str = "single",
) -> transforms.Compose:
    """Route image preprocessing by vision tower family."""
    encoder = str(vision_encoder_type or "clip").lower()
    view_mode = str(image_view_mode or "single").strip().lower().replace("-", "_")
    if encoder in {"siglip", "siglip2"}:
        if view_mode in {"anyres", "anyres_global_two_crops", "global_two_crops"}:
            base_transform = get_siglip_image_transform(
                model_name=vision_model_name or "google/siglip2-so400m-patch14-384",
                image_size=image_size,
                is_train=False,
                use_augmentation=False,
                image_augmentation_mode="none",
            )
            return AnyResImageTransform(
                base_transform=base_transform,
                num_views=3,
                local_crop_scale=0.75,
                crop_mode="global_two_crops",
            )
        if view_mode not in {"single", "none", ""}:
            raise ValueError(
                "image_view_mode for SigLIP must be one of: single, "
                "anyres_global_two_crops"
            )
        return get_siglip_image_transform(
            model_name=vision_model_name or "google/siglip2-so400m-patch14-384",
            image_size=image_size,
            is_train=is_train,
            use_augmentation=use_augmentation,
            image_augmentation_mode=image_augmentation_mode,
        )
    if view_mode not in {"single", "none", ""}:
        raise ValueError("AnyRes image_view_mode is only supported for SigLIP/SigLIP2")
    return get_image_transform(
        image_size=image_size or 224,
        is_train=is_train,
        use_augmentation=use_augmentation,
    )


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

    # Defaults preserve the historical LLaMA 3 conversation format.
    SYSTEM_HEADER = LLAMA3_TEMPLATE.system_header
    USER_HEADER = LLAMA3_TEMPLATE.user_header
    ASSISTANT_HEADER = LLAMA3_TEMPLATE.assistant_header
    END_TURN = LLAMA3_TEMPLATE.end_turn
    ASSISTANT_PREFILL = LLAMA3_TEMPLATE.assistant_prefill

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
        chat_template_family: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.chat_template_spec = resolve_chat_template_spec(
            tokenizer,
            model_name=getattr(tokenizer, "name_or_path", None),
            family=chat_template_family,
        )
        self.chat_template_family = self.chat_template_spec.family
        self.SYSTEM_HEADER = self.chat_template_spec.system_header
        self.USER_HEADER = self.chat_template_spec.user_header
        self.ASSISTANT_HEADER = self.chat_template_spec.assistant_header
        self.END_TURN = self.chat_template_spec.end_turn
        self.ASSISTANT_PREFILL = self.chat_template_spec.assistant_prefill

        # Ensure padding token is set
        # Use a dedicated pad token, not eos_token (to avoid masking
        # legitimate EOS tokens in labels during training).
        if tokenizer.pad_token is None:
            if "<|filetune_right_pad_id|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|filetune_right_pad_id|>"
            elif "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
            else:
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
        response_start_idx=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text with labels for training.

        Labels have -100 for tokens we don't want to predict
        (system prompt, user message - only predict assistant response).

        Args:
            text: Full conversation text
            response_start_idx: Character index where first response starts (int),
                or list of (start, end) char ranges for multi-turn support.

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

        if isinstance(response_start_idx, list):
            # Multi-turn: list of (start_char, end_char) ranges
            # Start with all masked, then unmask response ranges
            labels = torch.full_like(input_ids, -100)
            for start_char, end_char in response_start_idx:
                for i, (token_start, token_end) in enumerate(offset_mapping.tolist()):
                    if token_end > start_char and token_start < end_char:
                        labels[i] = input_ids[i]
            # Re-mask padding tokens
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        else:
            # Single response_start_idx (backward compatible)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            if response_start_idx is not None:
                response_token_idx = 0
                for i, (start, end) in enumerate(offset_mapping):
                    if start >= response_start_idx:
                        response_token_idx = i
                        break
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
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Format a conversation for training.

        Args:
            conversations: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Tuple of (formatted_text, response_ranges) where response_ranges
            is a list of (start_char, end_char) for all assistant responses
            (including the end-of-turn token).
        """
        parts = []
        response_ranges = []

        # Add system prompt
        if system_prompt:
            parts.append(f"{self.SYSTEM_HEADER}{system_prompt}{self.END_TURN}")

        # Add conversation turns
        for turn in conversations:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))

            if role in ["user", "human"]:
                parts.append(f"{self.USER_HEADER}{content}{self.END_TURN}")
            elif role in ["assistant", "gpt"]:
                current_pos = len("".join(parts))
                header_len = len(self.ASSISTANT_HEADER)
                response_start = current_pos + header_len
                response_end = response_start + len(content) + len(self.END_TURN)
                response_ranges.append((response_start, response_end))
                parts.append(f"{self.ASSISTANT_HEADER}{content}{self.END_TURN}")

        formatted_text = "".join(parts)
        return formatted_text, response_ranges


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
    negative_images = []
    input_ids = []
    attention_masks = []
    labels = []

    for sample in batch:
        if "image" in sample:
            images.append(sample["image"])
        if "negative_images" in sample:
            negative_images.append(sample["negative_images"])
        input_ids.append(sample["input_ids"])
        attention_masks.append(sample["attention_mask"])
        if "labels" in sample:
            labels.append(sample["labels"])

    # Stack images if present
    result = {}
    if images:
        result["images"] = torch.stack(images)
    if negative_images:
        if len(negative_images) != len(batch):
            raise ValueError(
                "Batches cannot mix contrastive samples with samples that lack negative_images"
            )
        result["negative_images"] = torch.stack(negative_images)

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
    for meta_key in (
        "sample_id",
        "image_ref",
        "question_text",
        "answer_text",
        "mixture_source",
        "source_local_index",
        "teacher_kl_weight",
        "loss_family",
        "dataset_family",
        "sample_weight",
    ):
        if any(meta_key in sample for sample in batch):
            result[meta_key] = [sample.get(meta_key) for sample in batch]

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
