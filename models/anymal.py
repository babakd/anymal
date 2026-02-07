"""
AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model

Main model class that combines vision encoder, projector, and LLM.

Educational Notes:
-----------------
AnyMAL Architecture Overview:
1. Modality Encoder (frozen): Converts raw input to features
   - Image: CLIP ViT-L -> [B, 257, 1024]

2. Projector (trainable): Maps encoder features to LLM space
   - Perceiver Resampler: [B, 257, 1024] -> [B, 64, 4096]

3. LLM (frozen base + trainable LoRA): Generates text
   - LLaMA 3 8B: [B, seq_len, 4096] -> logits

Forward Pass:
1. Encode image with frozen CLIP
2. Project to LLM space with Perceiver Resampler
3. Embed text tokens with LLM embeddings
4. Concatenate: [image_tokens, text_tokens]
5. Forward through LLM
6. Compute loss on text portion only

Training Stages:
- Stage 1 (Alignment): Train projector only, LLM frozen
- Stage 2 (Instruction Tuning): Train projector + LoRA adapters

Key Design Decisions:
- Frozen encoders: Leverage pretrained representations
- Learned projector: Bridge modality gap
- QLoRA: Memory-efficient LLM adaptation
- Perceiver Resampler: Compress visual tokens efficiently
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass

from .encoders import ImageEncoder
from .projectors import PerceiverResampler, LinearProjector
from .llm import LlamaWrapper


@dataclass
class AnyMALOutput:
    """Output container for AnyMAL forward pass."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class AnyMAL(nn.Module):
    """
    AnyMAL: Multimodal Language Model with modular encoder support.

    This is the main model class that orchestrates:
    - Vision encoding (frozen CLIP)
    - Modality projection (trainable Perceiver Resampler)
    - Language generation (frozen LLaMA + trainable LoRA)

    Args:
        llm_model_name: HuggingFace model name for LLM
        vision_model_name: CLIP model variant
        vision_pretrained: CLIP pretrained weights source
        projector_type: "perceiver" or "linear"
        num_image_tokens: Number of image tokens after projection
        projector_layers: Number of layers in projector
        use_qlora: Whether to use QLoRA for LLM
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        freeze_vision: Whether to freeze vision encoder
        freeze_llm: Whether to freeze LLM (except LoRA)
        use_flash_attention: Whether to use Flash Attention 2

    Example:
        >>> model = AnyMAL(
        ...     llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     vision_model_name="ViT-L-14",
        ... )
        >>> outputs = model(
        ...     images=images,
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     labels=labels,
        ... )
        >>> loss = outputs.loss
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        vision_model_name: str = "ViT-L-14",
        vision_pretrained: str = "openai",
        projector_type: str = "perceiver",
        num_image_tokens: int = 64,
        projector_layers: int = 6,
        projector_heads: int = 16,
        projector_ff_mult: int = 4,
        use_qlora: bool = True,
        use_lora: Optional[bool] = None,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = True,
        llm_device_map: Union[str, Dict[str, Union[int, str]], None] = "auto",
        llm_torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.num_image_tokens = num_image_tokens
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm

        # Initialize vision encoder (frozen by default)
        self.image_encoder = ImageEncoder(
            model_name=vision_model_name,
            pretrained=vision_pretrained,
            cache_dir=cache_dir,
            freeze=freeze_vision,
        )

        # Initialize LLM
        self.llm = LlamaWrapper(
            model_name=llm_model_name,
            use_qlora=use_qlora,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=llm_device_map,
            torch_dtype=llm_torch_dtype,
            use_flash_attention=use_flash_attention,
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
        )

        # Get dimensions for projector
        vision_dim = self.image_encoder.get_output_dim()
        llm_dim = self.llm.hidden_size

        # Initialize projector (always trainable)
        if projector_type == "perceiver":
            self.projector = PerceiverResampler(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_latents=num_image_tokens,
                num_layers=projector_layers,
                num_heads=projector_heads,
                ff_mult=projector_ff_mult,
            )
        elif projector_type == "linear":
            self.projector = LinearProjector(
                input_dim=vision_dim,
                output_dim=llm_dim,
                num_layers=2,
                pool_type="learned" if num_image_tokens < 257 else None,
                num_output_tokens=num_image_tokens,
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        # Freeze LLM base model if specified
        if freeze_llm:
            self.llm.freeze_base_model()

        # Store tokenizer reference
        self.tokenizer = self.llm.tokenizer

        # Register a special image placeholder token for positional insertion.
        # We use a reserved special token from LLaMA 3 if available; otherwise
        # add one. The dataset puts N copies of this token where <image> appears,
        # and the forward pass replaces their embeddings with projected image tokens.
        self._setup_image_placeholder_token()

    def _setup_image_placeholder_token(self):
        """Set up a dedicated token ID used as the image placeholder in input_ids."""
        # Try to use a LLaMA 3 reserved token that is not used elsewhere
        vocab = self.tokenizer.get_vocab()
        # <|reserved_special_token_0|> is token 128002 in LLaMA 3
        for candidate in ["<|reserved_special_token_0|>", "<|image|>"]:
            if candidate in vocab:
                self.image_placeholder_token_id = vocab[candidate]
                return

        # Fallback: add a new special token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
        self.llm.model.resize_token_embeddings(len(self.tokenizer))
        self.image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to LLM token space.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Image tokens [B, num_image_tokens, llm_dim]

        Process:
        1. CLIP encodes images to patch features
        2. Perceiver Resampler compresses and projects to LLM space
        """
        # Get vision features from frozen encoder
        # CLIP expects float32 input, so cast if needed
        original_dtype = images.dtype
        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        with torch.no_grad():
            vision_features = self.image_encoder(images)

        # Project to LLM space (this is trainable)
        # Cast to projector dtype if different
        projector_dtype = next(self.projector.parameters()).dtype
        if vision_features.dtype != projector_dtype:
            vision_features = vision_features.to(projector_dtype)

        image_tokens = self.projector(vision_features)

        return image_tokens

    def _splice_image_tokens(
        self,
        input_ids: torch.LongTensor,
        image_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Replace image placeholder tokens in the embedding sequence with actual image tokens.

        If no placeholder tokens are found, falls back to prepending image tokens
        (backward compatible with datasets that strip <image>).

        Returns:
            (inputs_embeds, full_attention_mask, full_labels)
        """
        batch_size = input_ids.shape[0]
        placeholder_id = self.image_placeholder_token_id

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Ensure image_tokens are on the same device and dtype
        image_tokens = image_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)

        # Check if any sample has placeholder tokens
        has_placeholders = (input_ids == placeholder_id).any()

        # Log on first call which path we take
        if not hasattr(self, '_splice_logged'):
            self._splice_logged = True
            n_placeholders = (input_ids == placeholder_id).sum().item()
            path = "SPLICE (positional insertion)" if has_placeholders else "PREPEND (fallback)"
            print(f"[AnyMAL] _splice_image_tokens: {path}, "
                  f"placeholder_id={placeholder_id}, found={n_placeholders}, "
                  f"input_ids={list(input_ids.shape)}, image_tokens={list(image_tokens.shape)}")

        if has_placeholders:
            # Positional insertion: replace placeholder token embeddings with image tokens
            new_embeds_list = []
            new_mask_list = []
            new_labels_list = []

            for b in range(batch_size):
                ids = input_ids[b]
                placeholder_mask = (ids == placeholder_id)

                if not placeholder_mask.any():
                    # No placeholders in this sample — prepend (fallback)
                    new_embeds_list.append(torch.cat([image_tokens[b], text_embeds[b]], dim=0))
                    if attention_mask is not None:
                        img_mask = torch.ones(self.num_image_tokens, device=ids.device, dtype=attention_mask.dtype)
                        new_mask_list.append(torch.cat([img_mask, attention_mask[b]], dim=0))
                    if labels is not None:
                        img_labels = torch.full((self.num_image_tokens,), -100, device=ids.device, dtype=labels.dtype)
                        new_labels_list.append(torch.cat([img_labels, labels[b]], dim=0))
                    continue

                # Find positions of placeholder tokens
                placeholder_indices = placeholder_mask.nonzero(as_tuple=True)[0]
                first_placeholder = placeholder_indices[0].item()

                # Split text embeddings around the placeholder block
                before = text_embeds[b, :first_placeholder]
                after = text_embeds[b, first_placeholder + len(placeholder_indices):]

                new_embeds_list.append(torch.cat([before, image_tokens[b], after], dim=0))

                if attention_mask is not None:
                    mask_before = attention_mask[b, :first_placeholder]
                    mask_after = attention_mask[b, first_placeholder + len(placeholder_indices):]
                    img_mask = torch.ones(self.num_image_tokens, device=ids.device, dtype=attention_mask.dtype)
                    new_mask_list.append(torch.cat([mask_before, img_mask, mask_after], dim=0))

                if labels is not None:
                    labels_before = labels[b, :first_placeholder]
                    labels_after = labels[b, first_placeholder + len(placeholder_indices):]
                    img_labels = torch.full((self.num_image_tokens,), -100, device=ids.device, dtype=labels.dtype)
                    new_labels_list.append(torch.cat([labels_before, img_labels, labels_after], dim=0))

            # Pad to same length and stack
            inputs_embeds = self._pad_and_stack_embeds(new_embeds_list, text_embeds.device)
            full_attention_mask = self._pad_and_stack_1d(new_mask_list, 0) if new_mask_list else None
            full_labels = self._pad_and_stack_1d(new_labels_list, -100) if new_labels_list else None
        else:
            # No placeholders found — fall back to prepending image tokens
            inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

            if attention_mask is not None:
                image_attention = torch.ones(
                    batch_size, self.num_image_tokens,
                    device=text_embeds.device, dtype=attention_mask.dtype
                )
                full_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
            else:
                full_attention_mask = None

            if labels is not None:
                image_labels = torch.full(
                    (batch_size, self.num_image_tokens),
                    fill_value=-100,
                    device=text_embeds.device,
                    dtype=labels.dtype,
                )
                full_labels = torch.cat([image_labels, labels], dim=1)
            else:
                full_labels = None

        return inputs_embeds, full_attention_mask, full_labels

    @staticmethod
    def _pad_and_stack_embeds(embed_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Pad embedding sequences to same length and stack into batch."""
        max_len = max(e.shape[0] for e in embed_list)
        dim = embed_list[0].shape[1]
        padded = []
        for e in embed_list:
            pad_len = max_len - e.shape[0]
            if pad_len > 0:
                e = torch.cat([e, torch.zeros(pad_len, dim, device=device, dtype=e.dtype)], dim=0)
            padded.append(e)
        return torch.stack(padded)

    @staticmethod
    def _pad_and_stack_1d(tensor_list: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Pad 1D tensors to same length and stack into batch."""
        max_len = max(t.shape[0] for t in tensor_list)
        padded = []
        for t in tensor_list:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), pad_value, device=t.device, dtype=t.dtype)])
            padded.append(t)
        return torch.stack(padded)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> AnyMALOutput:
        """
        Forward pass for multimodal input.

        If input_ids contain image placeholder tokens, image embeddings are
        inserted at those positions. Otherwise, image tokens are prepended
        (backward compatible).

        Args:
            images: Input images [B, 3, H, W]
            input_ids: Text token IDs [B, text_seq_len]
            attention_mask: Attention mask for text [B, text_seq_len]
            labels: Labels for language modeling [B, text_seq_len]
            image_tokens: Pre-computed image tokens (optional)
            return_dict: Whether to return AnyMALOutput

        Returns:
            AnyMALOutput with loss and logits
        """
        batch_size = input_ids.shape[0] if input_ids is not None else images.shape[0]

        # Encode images if not pre-computed
        if image_tokens is None and images is not None:
            image_tokens = self.encode_images(images)

        # Handle text-only case
        if image_tokens is None:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return AnyMALOutput(
                loss=outputs.loss,
                logits=outputs.logits,
            )

        # Splice image tokens into the sequence (at placeholder or prepended)
        inputs_embeds, full_attention_mask, full_labels = self._splice_image_tokens(
            input_ids, image_tokens, attention_mask, labels
        )

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )

        return AnyMALOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text given image and optional text prompt.

        Args:
            images: Input images [B, 3, H, W]
            input_ids: Text prompt tokens [B, text_seq_len]
            attention_mask: Attention mask for text
            image_tokens: Pre-computed image tokens (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs [B, generated_seq_len]
        """
        batch_size = input_ids.shape[0] if input_ids is not None else images.shape[0]

        # Encode images if needed
        if image_tokens is None and images is not None:
            image_tokens = self.encode_images(images)

        # Text-only generation
        if image_tokens is None:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )

        # Build inputs_embeds with image tokens spliced in
        if input_ids is not None:
            inputs_embeds, attention_mask, _ = self._splice_image_tokens(
                input_ids, image_tokens, attention_mask, labels=None
            )
        else:
            device = images.device if images is not None else image_tokens.device
            text_embeds_dtype = self.llm.get_input_embeddings().weight.dtype
            inputs_embeds = image_tokens.to(dtype=text_embeds_dtype)
            attention_mask = torch.ones(
                batch_size, self.num_image_tokens,
                device=device, dtype=torch.bool
            )

        # Generate
        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )

        # HF `generate()` returns only newly generated tokens when `inputs_embeds` is provided.
        # For consistency with text-only generation (and to simplify downstream decoding),
        # prepend the original text prompt tokens when available.
        if input_ids is not None and isinstance(generated, torch.Tensor):
            # Strip placeholder tokens from the prepended input_ids
            mask = input_ids != self.image_placeholder_token_id
            # For simplicity, just prepend original input_ids (placeholders won't affect decoding)
            return torch.cat([input_ids, generated], dim=1)

        return generated

    def set_training_stage(self, stage: int) -> None:
        """
        Configure model for specific training stage.

        Stage 1 (Alignment Pretraining):
        - Only projector is trainable
        - Vision encoder frozen
        - LLM completely frozen (no LoRA)

        Stage 2 (Instruction Tuning):
        - Projector trainable
        - Vision encoder frozen
        - LLM LoRA adapters trainable

        Args:
            stage: Training stage (1 or 2)
        """
        if stage == 1:
            # Stage 1: Only train projector
            self.image_encoder.freeze()

            # Freeze all LLM parameters including LoRA
            for param in self.llm.parameters():
                param.requires_grad = False

            # Unfreeze projector
            for param in self.projector.parameters():
                param.requires_grad = True

            print("Stage 1: Training projector only")

        elif stage == 2:
            # Stage 2: Train projector + LoRA
            self.image_encoder.freeze()

            # Freeze LLM base, unfreeze LoRA
            self.llm.freeze_base_model()

            # Unfreeze projector
            for param in self.projector.parameters():
                param.requires_grad = True

            print("Stage 2: Training projector + LoRA")

        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.print_trainable_parameters()

    def print_trainable_parameters(self) -> None:
        """Print trainable parameter counts for each component."""
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        # Count for each component
        vision_total, vision_train = count_params(self.image_encoder)
        proj_total, proj_train = count_params(self.projector)
        llm_total, llm_train = count_params(self.llm)

        print(f"\nTrainable Parameters:")
        print(f"  Vision Encoder: {vision_train:,} / {vision_total:,}")
        print(f"  Projector:      {proj_train:,} / {proj_total:,}")
        print(f"  LLM:            {llm_train:,} / {llm_total:,}")
        print(f"  Total:          {vision_train + proj_train + llm_train:,} / "
              f"{vision_total + proj_total + llm_total:,}")

    def save_pretrained(self, save_path: str) -> None:
        """
        Save model components.

        Saves:
        - Projector weights
        - LLM LoRA adapters
        - Tokenizer

        Args:
            save_path: Directory to save model
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save projector
        torch.save(
            self.projector.state_dict(),
            os.path.join(save_path, "projector.pt")
        )

        # Save LLM (LoRA weights)
        self.llm.save_pretrained(os.path.join(save_path, "llm"))

        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs,
    ) -> "AnyMAL":
        """
        Load pretrained AnyMAL model.

        Args:
            save_path: Directory containing saved model
            llm_model_name: Base LLM model name
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded AnyMAL model
        """
        import os

        # Initialize model
        model = cls(llm_model_name=llm_model_name, **kwargs)

        # Load projector
        projector_path = os.path.join(save_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path))

        # Load LLM LoRA weights
        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            from peft import PeftModel
            # Check if model already has LoRA (is a PeftModel)
            base_model = model.llm.model
            if hasattr(base_model, "base_model"):
                # Model is already a PeftModel, get the underlying base model
                base_model = base_model.base_model
            model.llm.model = PeftModel.from_pretrained(
                base_model,
                llm_path,
            )

        return model


def create_anymal(
    llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    vision_model_name: str = "ViT-L-14",
    **kwargs,
) -> AnyMAL:
    """
    Factory function to create AnyMAL model.

    Args:
        llm_model_name: HuggingFace model name for LLM
        vision_model_name: CLIP model variant
        **kwargs: Additional arguments

    Returns:
        Configured AnyMAL instance
    """
    return AnyMAL(
        llm_model_name=llm_model_name,
        vision_model_name=vision_model_name,
        **kwargs,
    )
