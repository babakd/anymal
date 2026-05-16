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

from .data_utils import get_image_transform, get_vision_transform, TextProcessor, CLIP_MEAN, CLIP_STD
from .chat_templates import TRAINING_SYSTEM_PROMPT


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
    DEFAULT_SYSTEM_PROMPT = TRAINING_SYSTEM_PROMPT

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
        vision_encoder_type: str = "clip",
        vision_model_name: Optional[str] = None,
        split: str = "train",
        system_prompt: Optional[str] = None,
        filter_to_available_images: bool = False,
        use_augmentation: bool = False,
        image_augmentation_mode: str = "none",
        image_view_mode: str = "single",
        chat_template_family: Optional[str] = None,
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
        self.vision_encoder_type = vision_encoder_type
        self.vision_model_name = vision_model_name
        self.image_view_mode = str(image_view_mode or "single")
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.chat_template_family = chat_template_family

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
        elif "<image>" in vocab:
            self.image_placeholder_token_id = vocab["<image>"]
        else:
            self.image_placeholder_token_id = None  # fallback: strip <image>

        # Image transform
        self.transform = get_vision_transform(
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            image_size=image_size,
            is_train=(split == "train"),
            use_augmentation=use_augmentation,
            image_augmentation_mode=image_augmentation_mode,
            image_view_mode=self.image_view_mode,
        )

        # Text processor
        self.text_processor = TextProcessor(
            tokenizer=tokenizer,
            max_length=max_length,
            chat_template_family=chat_template_family,
        )
        self.chat_template_family = self.text_processor.chat_template_family

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

        normalized = []
        for sample in data:
            if "conversations" in sample:
                normalized.append(sample)
                continue
            if "image" in sample and "caption" in sample:
                normalized.append(
                    {
                        "image": sample["image"],
                        "conversations": [
                            {"from": "human", "value": "<image>\nDescribe the image."},
                            {"from": "gpt", "value": sample["caption"]},
                        ],
                    }
                )
                continue
            normalized.append(sample)

        return normalized

    def _filter_to_available_images(self) -> List[Dict]:
        """Filter samples to only those with available image files."""
        if self.image_dir is None:
            return self.samples

        # Get set of available images for fast lookup
        available_images = set()
        for f in os.listdir(self.image_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                available_images.add(f)

        # Filter samples
        original_count = len(self.samples)
        def _all_image_refs_available(sample):
            image_refs = [sample.get("image")]
            image_refs.extend(sample.get("negative_images") or [])
            return all(ref in available_images for ref in image_refs if ref)

        filtered = [
            s for s in self.samples
            if _all_image_refs_available(s)
        ]

        print(
            f"InstructionDataset: real-image filter kept {len(filtered)}/{original_count} "
            f"samples from {self.image_dir}; missing={original_count - len(filtered)}"
        )
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single conversation sample.

        Returns:
            Dict containing:
            - image: Processed image tensor [3, H, W] or [V, 3, H, W]
            - input_ids: Token IDs [seq_len]
            - attention_mask: Attention mask [seq_len]
            - labels: Labels for training [seq_len]
        """
        sample = self.samples[idx]

        # Load image (real if available, otherwise reproducible dummy)
        image_tensor = self._load_image(sample)

        # Get conversations
        conversations = sample["conversations"]
        question_text = ""
        answer_text = ""
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = str(turn.get("value", turn.get("content", "")))
            if not question_text and role in ["human", "user"]:
                question_text = content
            elif not answer_text and role in ["gpt", "assistant"]:
                answer_text = content

        # Format conversation for training. Segment-wise encoding is used so
        # role masks do not depend on tokenizer offset mappings around the
        # expanded image-placeholder block.
        segments, image_sentinel = self._format_conversation_segments(conversations)

        num_image_tokens = self._sample_num_image_tokens()

        # Encode text
        encoding = self._encode_segments_with_response_masking(
            segments,
            image_sentinel=image_sentinel,
            num_image_tokens=num_image_tokens,
        )

        result = {
            "image": image_tensor,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
            "num_image_tokens": torch.tensor(num_image_tokens, dtype=torch.long),
            "sample_id": str(sample.get("id", idx)),
            "image_ref": str(sample.get("image", "")),
            "question_text": question_text,
            "answer_text": answer_text,
        }
        for meta_key in (
            "teacher_kl_weight",
            "loss_family",
            "dataset_family",
            "sample_weight",
        ):
            if meta_key in sample:
                result[meta_key] = sample[meta_key]
        negative_images = sample.get("negative_images") or []
        if negative_images:
            result["negative_images"] = torch.stack(
                [
                    self._load_image({**sample, "image": negative_image})
                    for negative_image in negative_images
                ]
            )
        return result

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
        image_tensor = clip_normalize(raw)
        num_views = int(getattr(self.transform, "num_views", 1) or 1)
        if num_views > 1:
            return torch.stack([image_tensor.clone() for _ in range(num_views)])
        return image_tensor

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

    def _format_conversation_segments(
        self,
        conversations: List[Dict[str, str]],
    ) -> Tuple[List[Tuple[str, bool]], str]:
        """Format a conversation as tokenization segments with explicit masks."""
        segments = []
        IMAGE_SENTINEL = "<|image_sentinel|>"

        segments.append(
            (
                f"{self.text_processor.SYSTEM_HEADER}"
                f"{self.system_prompt}"
                f"{self.text_processor.END_TURN}",
                False,
            )
        )

        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if self.image_placeholder_token_id is not None:
                content = content.replace(self.IMAGE_PLACEHOLDER, IMAGE_SENTINEL).strip()
            else:
                content = content.replace(self.IMAGE_PLACEHOLDER, "").strip()

            if role in ["human", "user"]:
                segments.append(
                    (
                        f"{self.text_processor.USER_HEADER}"
                        f"{content}"
                        f"{self.text_processor.END_TURN}",
                        False,
                    )
                )
            elif role in ["gpt", "assistant"]:
                segments.append(
                    (
                        f"{self.text_processor.ASSISTANT_HEADER}"
                        f"{self.text_processor.ASSISTANT_PREFILL}",
                        False,
                    )
                )
                segments.append((f"{content}{self.text_processor.END_TURN}", True))

        return segments, IMAGE_SENTINEL

    def _tokenize_segment(self, text: str) -> List[int]:
        if not text:
            return []
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _append_segment_tokens(
        self,
        input_ids: List[int],
        labels: List[int],
        text: str,
        supervise: bool,
        image_sentinel: str,
        num_image_tokens: int,
    ) -> None:
        pieces = text.split(image_sentinel) if image_sentinel else [text]
        for piece_idx, piece in enumerate(pieces):
            piece_ids = self._tokenize_segment(piece)
            input_ids.extend(piece_ids)
            labels.extend(piece_ids if supervise else [-100] * len(piece_ids))

            if piece_idx < len(pieces) - 1 and self.image_placeholder_token_id is not None:
                input_ids.extend([self.image_placeholder_token_id] * num_image_tokens)
                labels.extend([-100] * num_image_tokens)

    def _encode_segments_with_response_masking(
        self,
        segments: List[Tuple[str, bool]],
        image_sentinel: str = "",
        num_image_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode pre-segmented conversation text with explicit role masks."""
        if num_image_tokens is None:
            num_image_tokens = self.num_image_tokens

        input_ids = []
        labels = []

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            input_ids.append(int(bos_token_id))
            labels.append(-100)

        for text, supervise in segments:
            self._append_segment_tokens(
                input_ids=input_ids,
                labels=labels,
                text=text,
                supervise=supervise,
                image_sentinel=image_sentinel,
                num_image_tokens=num_image_tokens,
            )

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        labels[input_ids == self.tokenizer.pad_token_id] = -100
        if self.image_placeholder_token_id is not None:
            labels[input_ids == self.image_placeholder_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

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
        vision_encoder_type: str = "clip",
        vision_model_name: Optional[str] = None,
        image_view_mode: str = "single",
    ):
        super().__init__()

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = get_vision_transform(
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            image_size=image_size,
            is_train=True,
            image_view_mode=image_view_mode,
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


def _summarize_license_metadata(
    source_names: List[str],
    source_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sources = []
    license_counts: Dict[str, int] = {}
    commercial_values = []
    for name, metadata in zip(source_names, source_metadata):
        license_name = metadata.get("license")
        commercial = metadata.get("commercial_use_allowed")
        if license_name is None and commercial is None:
            continue
        license_text = str(license_name or "unknown")
        license_counts[license_text] = license_counts.get(license_text, 0) + 1
        if commercial is not None:
            commercial_values.append(bool(commercial))
        sources.append(
            {
                "name": str(name),
                "license": license_text,
                "license_source": metadata.get("license_source"),
                "commercial_use_allowed": commercial,
                "dataset_family": metadata.get("dataset_family"),
                "loss_family": metadata.get("loss_family"),
            }
        )
    if not commercial_values:
        aggregate = None
    else:
        aggregate = all(commercial_values)
    return {
        "sources": sources,
        "license_counts": license_counts,
        "aggregate_commercial_use_allowed": aggregate,
    }


class InstructionMixtureDataset(Dataset):
    """
    Small wrapper for Stage 2 instruction mixtures.

    `concat` preserves natural dataset sizes. `balanced` round-robins sources
    and oversamples smaller datasets to the largest source length. `weighted`
    uses integerized per-source weights for an explicit sampling mix.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        strategy: str = "balanced",
        source_names: Optional[List[str]] = None,
        source_weights: Optional[List[float]] = None,
        source_metadata: Optional[List[Dict[str, Any]]] = None,
        length: Optional[int] = None,
        weighted_index_mode: str = "sequential",
    ):
        if not datasets:
            raise ValueError("InstructionMixtureDataset requires at least one dataset")
        if strategy not in {"balanced", "concat", "weighted"}:
            raise ValueError("strategy must be one of ['balanced', 'concat', 'weighted']")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("InstructionMixtureDataset cannot include empty datasets")

        self.datasets = datasets
        self.strategy = strategy
        self.source_names = source_names or [f"source_{i}" for i in range(len(datasets))]
        self.source_metadata = source_metadata or [{} for _ in range(len(datasets))]
        self._weighted_index_mode = str(weighted_index_mode or "sequential").strip().lower()
        if self._weighted_index_mode not in {"sequential", "hash"}:
            raise ValueError("weighted_index_mode must be one of ['sequential', 'hash']")

        if len(self.source_names) != len(self.datasets):
            raise ValueError("source_names length must match datasets length")
        if len(self.source_metadata) != len(self.datasets):
            raise ValueError("source_metadata length must match datasets length")

        self._weighted_cycle = None
        if self.strategy == "weighted":
            if source_weights is None:
                source_weights = [1.0] * len(datasets)
            if len(source_weights) != len(datasets):
                raise ValueError("source_weights length must match datasets length")
            if any(float(weight) <= 0 for weight in source_weights):
                raise ValueError("source_weights must all be > 0")

            total_weight = sum(float(weight) for weight in source_weights)
            slots = [
                max(1, round(float(weight) / total_weight * 100))
                for weight in source_weights
            ]
            self._weighted_cycle = [
                source_idx
                for source_idx, slot_count in enumerate(slots)
                for _ in range(slot_count)
            ]

        self._cumulative_lengths = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self._cumulative_lengths.append(total)

        if self.strategy == "balanced":
            self._length = max(len(dataset) for dataset in datasets) * len(datasets)
        elif self.strategy == "weighted":
            self._length = max(len(dataset) for dataset in datasets) * len(self._weighted_cycle)
        else:
            self._length = total
        if length is not None:
            self._length = int(length)
            if self._length <= 0:
                raise ValueError(f"InstructionMixtureDataset length must be > 0, got {length}")

        sizes = ", ".join(
            f"{name}={len(dataset)}" for name, dataset in zip(self.source_names, self.datasets)
        )
        print(
            f"InstructionMixtureDataset: strategy={self.strategy}, "
            f"length={self._length}, sources: {sizes}"
        )
        self.license_summary = _summarize_license_metadata(
            self.source_names,
            self.source_metadata,
        )
        if self.license_summary["sources"]:
            posture = self.license_summary["aggregate_commercial_use_allowed"]
            print(
                "InstructionMixtureDataset license summary: "
                f"aggregate_commercial_use_allowed={posture}; "
                f"licenses={self.license_summary['license_counts']}"
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        def _with_source_metadata(source_idx: int, local_idx: int) -> Dict[str, Any]:
            item = dict(self.datasets[source_idx][local_idx])
            source_name = str(self.source_names[source_idx])
            base_id = item.get("sample_id", local_idx)
            item["sample_id"] = f"{source_name}:{int(local_idx)}:{base_id}"
            item["mixture_source"] = source_name
            item["source_local_index"] = int(local_idx)
            return item

        if self.strategy == "balanced":
            source_idx = idx % len(self.datasets)
            local_idx = (idx // len(self.datasets)) % len(self.datasets[source_idx])
            return _with_source_metadata(source_idx, local_idx)

        if self.strategy == "weighted":
            source_idx = self._weighted_cycle[idx % len(self._weighted_cycle)]
            stride = idx // len(self._weighted_cycle)
            if self._weighted_index_mode == "hash":
                local_idx = (
                    stride * 1_000_003
                    + (idx + 1) * 97_531
                    + (source_idx + 1) * 31_337
                ) % len(self.datasets[source_idx])
            else:
                local_idx = stride % len(self.datasets[source_idx])
            return _with_source_metadata(source_idx, local_idx)

        for source_idx, end in enumerate(self._cumulative_lengths):
            start = 0 if source_idx == 0 else self._cumulative_lengths[source_idx - 1]
            if idx < end:
                return _with_source_metadata(source_idx, idx - start)

        raise IndexError(idx)


def build_instruction_mixture_dataset(
    mixture_config: Dict[str, Any],
    tokenizer,
    default_image_dir: Optional[str] = None,
    simple: bool = False,
    **kwargs,
) -> InstructionMixtureDataset:
    """Build a balanced/concat/weighted instruction mixture from config dictionaries."""
    entries = mixture_config.get("datasets", mixture_config.get("sources", []))
    strategy = mixture_config.get("strategy", "balanced")
    if not entries:
        raise ValueError("Instruction mixture config requires datasets")

    datasets = []
    names = []
    weights = []
    source_metadata_entries = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"data_path": entry}

        entry_kwargs = dict(kwargs)
        for key in (
            "image_size",
            "max_length",
            "num_image_tokens",
            "image_token_policy",
            "min_image_tokens",
            "max_image_tokens",
            "vision_encoder_type",
            "vision_model_name",
            "system_prompt",
            "filter_to_available_images",
            "use_augmentation",
            "image_augmentation_mode",
            "image_view_mode",
            "chat_template_family",
        ):
            if key in entry:
                entry_kwargs[key] = entry[key]

        dataset = create_instruction_dataset(
            data_path=entry["data_path"],
            image_dir=entry.get("image_dir", default_image_dir),
            tokenizer=tokenizer,
            simple=entry.get("simple", simple),
            **entry_kwargs,
        )

        max_samples = entry.get("max_samples")
        if max_samples is not None:
            if not hasattr(dataset, "samples"):
                raise ValueError(
                    f"Cannot apply max_samples to mixture source {entry.get('name', i)} "
                    "because the dataset does not expose a samples list."
                )
            samples = list(dataset.samples)
            sample_seed = entry.get("sample_seed")
            if sample_seed is not None:
                rng = random.Random(int(sample_seed))
                rng.shuffle(samples)
            dataset.samples = samples[: int(max_samples)]

        source_metadata = {}
        for key in (
            "teacher_kl_weight",
            "loss_family",
            "dataset_family",
            "sample_weight",
            "license",
            "license_source",
            "commercial_use_allowed",
        ):
            if key in entry:
                source_metadata[key] = entry[key]
        extra_metadata = entry.get("sample_metadata", entry.get("metadata", {}))
        if extra_metadata:
            if not isinstance(extra_metadata, dict):
                raise ValueError(
                    f"Mixture metadata for source {entry.get('name', i)} must be a dict"
                )
            source_metadata.update(extra_metadata)
        if source_metadata:
            if not hasattr(dataset, "samples"):
                raise ValueError(
                    f"Cannot attach source metadata to mixture source {entry.get('name', i)} "
                    "because the dataset does not expose a samples list."
                )
            dataset.samples = [
                {**sample, **source_metadata}
                for sample in dataset.samples
            ]

        datasets.append(dataset)
        names.append(entry.get("name", f"source_{i}"))
        weights.append(float(entry.get("weight", 1.0)))
        source_metadata_entries.append(
            {
                key: entry.get(key)
                for key in (
                    "license",
                    "license_source",
                    "commercial_use_allowed",
                    "dataset_family",
                    "loss_family",
                    "data_path",
                    "image_dir",
                    "weight",
                )
                if key in entry
            }
        )

    mixture_length = mixture_config.get("length", mixture_config.get("epoch_length"))
    weighted_index_mode = mixture_config.get("weighted_index_mode", "sequential")
    return InstructionMixtureDataset(
        datasets=datasets,
        strategy=strategy,
        source_names=names,
        source_weights=weights,
        source_metadata=source_metadata_entries,
        length=mixture_length,
        weighted_index_mode=weighted_index_mode,
    )


def create_instruction_dataset(
    data_path: Optional[str],
    image_dir: Optional[str],
    tokenizer,
    simple: bool = False,
    mixture_config: Optional[Dict[str, Any]] = None,
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
    if mixture_config is not None:
        return build_instruction_mixture_dataset(
            mixture_config=mixture_config,
            tokenizer=tokenizer,
            default_image_dir=image_dir,
            simple=simple,
            **kwargs,
        )

    if data_path is None:
        raise ValueError("data_path is required when mixture_config is not provided")

    if simple:
        return InstructionDatasetSimple(
            data_path, image_dir, tokenizer, **kwargs
        )
    else:
        return InstructionDataset(
            data_path, image_dir, tokenizer, **kwargs
        )
