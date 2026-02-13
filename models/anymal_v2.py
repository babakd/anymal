"""
AnyMALv2 core architecture.

V2 stack:
- SigLIP2 vision encoder
- Learned token compressor
- MLP bottleneck projector
- Strict placeholder/token alignment checks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoders import SigLIP2Encoder
from .llm import LlamaWrapper
from model_metadata import validate_checkpoint_architecture, write_model_metadata
from .projectors import MLPBottleneckProjector, TokenCompressor


@dataclass
class AnyMALv2Output:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class AnyMALv2(nn.Module):
    """AnyMAL v2 multimodal model."""

    architecture = "anymal_v2"

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        vision_encoder_type: str = "siglip2",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        token_compressor_type: str = "learned",
        bottleneck_dim: int = 2048,
        max_image_tokens: int = 256,
        min_image_tokens: Optional[int] = None,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        use_qlora: bool = True,
        use_lora: Optional[bool] = None,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = True,
        llm_device_map: Union[str, Dict[str, Union[int, str]], None] = "auto",
        llm_torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if max_image_tokens <= 0:
            raise ValueError(f"max_image_tokens must be > 0, got {max_image_tokens}")
        if min_image_tokens is not None and min_image_tokens <= 0:
            raise ValueError(f"min_image_tokens must be > 0, got {min_image_tokens}")
        if min_image_tokens is not None and min_image_tokens > max_image_tokens:
            raise ValueError(
                f"min_image_tokens ({min_image_tokens}) must be <= max_image_tokens ({max_image_tokens})"
            )

        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.max_image_tokens = max_image_tokens
        self.min_image_tokens = min_image_tokens or max_image_tokens
        # Keep this for compatibility with existing diagnostics/utilities.
        self.num_image_tokens = max_image_tokens

        if vision_encoder_type != "siglip2":
            raise ValueError(
                f"Unsupported vision_encoder_type '{vision_encoder_type}'. Expected 'siglip2'."
            )
        self.vision_encoder_type = vision_encoder_type
        self.token_compressor_type = token_compressor_type
        self.bottleneck_dim = bottleneck_dim

        self.image_encoder = SigLIP2Encoder(
            model_name=vision_model_name,
            cache_dir=cache_dir,
            freeze=freeze_vision,
        )

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

        vision_dim = self.image_encoder.get_output_dim()
        llm_dim = self.llm.hidden_size

        self.token_compressor = TokenCompressor(
            input_dim=vision_dim,
            max_tokens=max_image_tokens,
            compressor_type=token_compressor_type,
        )
        self.projector = MLPBottleneckProjector(
            input_dim=vision_dim,
            output_dim=llm_dim,
            bottleneck_dim=bottleneck_dim,
        )

        if freeze_llm:
            self.llm.freeze_base_model()

        self.tokenizer = self.llm.tokenizer
        self._setup_image_placeholder_token()

    def _setup_image_placeholder_token(self):
        vocab = self.tokenizer.get_vocab()
        for candidate in ["<|reserved_special_token_0|>", "<|image|>"]:
            if candidate in vocab:
                self.image_placeholder_token_id = vocab[candidate]
                return
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
        self.llm.model.resize_token_embeddings(len(self.tokenizer))
        self.image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

    def _extract_placeholder_counts(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        strict: bool = True,
    ) -> torch.LongTensor:
        """Return per-sample placeholder counts with strict contiguity checks."""
        if input_ids is None:
            raise ValueError("input_ids must be provided for multimodal v2 forward.")

        placeholder_id = self.image_placeholder_token_id
        counts: List[int] = []

        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            mask = ids == placeholder_id
            if attention_mask is not None:
                mask = mask & attention_mask[b].bool()
            indices = mask.nonzero(as_tuple=True)[0]
            count = int(indices.numel())
            counts.append(count)

            if not strict:
                continue

            if count == 0:
                raise ValueError(
                    f"AnyMALv2 strict splice requires image placeholder tokens, "
                    f"but sample {b} has none."
                )
            is_contiguous = (indices[-1] - indices[0] + 1) == count
            if not is_contiguous:
                raise ValueError(
                    f"AnyMALv2 strict splice expects one contiguous placeholder block, "
                    f"but sample {b} has non-contiguous placeholders."
                )

        return torch.tensor(counts, device=input_ids.device, dtype=torch.long)

    def encode_images(
        self,
        images: torch.Tensor,
        target_num_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Encode and compress images.

        Returns:
            image_tokens: [B, max_image_tokens, llm_dim]
            image_token_mask: [B, max_image_tokens]
            image_token_counts: [B]
        """
        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        if self.freeze_vision:
            with torch.no_grad():
                vision_features = self.image_encoder(images)
        else:
            vision_features = self.image_encoder(images)

        compressor_dtype = next(self.token_compressor.parameters()).dtype
        if vision_features.dtype != compressor_dtype:
            vision_features = vision_features.to(compressor_dtype)

        compressed_tokens, token_mask, token_counts = self.token_compressor(
            vision_features,
            target_num_tokens=target_num_tokens,
        )
        image_tokens = self.projector(compressed_tokens)
        image_tokens = image_tokens * token_mask.unsqueeze(-1).to(dtype=image_tokens.dtype)
        return image_tokens, token_mask, token_counts

    @staticmethod
    def _pad_and_stack_embeds(
        embed_list: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
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
    def _pad_and_stack_1d(
        tensor_list: List[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        max_len = max(t.shape[0] for t in tensor_list)
        padded = []
        for t in tensor_list:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), pad_value, device=t.device, dtype=t.dtype)])
            padded.append(t)
        return torch.stack(padded)

    def _splice_image_tokens_strict(
        self,
        input_ids: torch.LongTensor,
        image_tokens: torch.Tensor,
        image_token_mask: torch.BoolTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Strict per-sample placeholder replacement with count checks."""
        placeholder_id = self.image_placeholder_token_id
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_tokens = image_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        image_token_mask = image_token_mask.to(device=text_embeds.device)

        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            ph_mask = ids == placeholder_id
            if attention_mask is not None:
                ph_mask = ph_mask & attention_mask[b].bool()

            ph_indices = ph_mask.nonzero(as_tuple=True)[0]
            ph_count = int(ph_indices.numel())
            if ph_count == 0:
                raise ValueError(
                    f"AnyMALv2 strict splice requires placeholder tokens, but sample {b} has none."
                )
            if (ph_indices[-1] - ph_indices[0] + 1) != ph_count:
                raise ValueError(
                    f"AnyMALv2 strict splice expects one contiguous placeholder block, "
                    f"but sample {b} is non-contiguous."
                )

            valid_token_count = int(image_token_mask[b].sum().item())
            if valid_token_count != ph_count:
                raise ValueError(
                    f"Placeholder/token mismatch in sample {b}: "
                    f"placeholders={ph_count}, image_tokens={valid_token_count}."
                )

            first = int(ph_indices[0].item())
            before = text_embeds[b, :first]
            after = text_embeds[b, first + ph_count :]
            image_block = image_tokens[b, :valid_token_count]

            new_embeds = torch.cat([before, image_block, after], dim=0)
            new_embeds_list.append(new_embeds)

            if attention_mask is not None:
                mask_before = attention_mask[b, :first]
                mask_after = attention_mask[b, first + ph_count :]
                mask_image = torch.ones(valid_token_count, device=ids.device, dtype=attention_mask.dtype)
                new_mask_list.append(torch.cat([mask_before, mask_image, mask_after], dim=0))

            if labels is not None:
                labels_before = labels[b, :first]
                labels_after = labels[b, first + ph_count :]
                labels_image = torch.full(
                    (valid_token_count,),
                    fill_value=-100,
                    device=ids.device,
                    dtype=labels.dtype,
                )
                new_labels_list.append(torch.cat([labels_before, labels_image, labels_after], dim=0))

        inputs_embeds = self._pad_and_stack_embeds(new_embeds_list, text_embeds.device)
        full_attention_mask = self._pad_and_stack_1d(new_mask_list, 0) if new_mask_list else None
        full_labels = self._pad_and_stack_1d(new_labels_list, -100) if new_labels_list else None
        return inputs_embeds, full_attention_mask, full_labels

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ) -> AnyMALv2Output:
        # Text-only path
        if images is None and image_tokens is None:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return AnyMALv2Output(loss=outputs.loss, logits=outputs.logits)

        if input_ids is None:
            raise ValueError("AnyMALv2 multimodal forward requires input_ids for strict placeholder splice.")

        if image_tokens is None:
            requested_counts = self._extract_placeholder_counts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict=True,
            )
            image_tokens, image_token_mask, _ = self.encode_images(
                images=images,
                target_num_tokens=requested_counts,
            )
        elif image_token_mask is None:
            image_token_mask = torch.ones(
                image_tokens.shape[:2],
                dtype=torch.bool,
                device=image_tokens.device,
            )

        inputs_embeds, full_attention_mask, full_labels = self._splice_image_tokens_strict(
            input_ids=input_ids,
            image_tokens=image_tokens,
            image_token_mask=image_token_mask,
            attention_mask=attention_mask,
            labels=labels,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        return AnyMALv2Output(loss=outputs.loss, logits=outputs.logits)

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        if images is None and image_tokens is None:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("AnyMALv2 multimodal generate requires input_ids with placeholders.")

        if image_tokens is None:
            requested_counts = self._extract_placeholder_counts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict=True,
            )
            image_tokens, image_token_mask, _ = self.encode_images(
                images=images,
                target_num_tokens=requested_counts,
            )
        elif image_token_mask is None:
            image_token_mask = torch.ones(
                image_tokens.shape[:2],
                dtype=torch.bool,
                device=image_tokens.device,
            )

        inputs_embeds, full_attention_mask, _ = self._splice_image_tokens_strict(
            input_ids=input_ids,
            image_tokens=image_tokens,
            image_token_mask=image_token_mask,
            attention_mask=attention_mask,
            labels=None,
        )

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )
        if isinstance(generated, torch.Tensor):
            return torch.cat([input_ids, generated], dim=1)
        return generated

    def set_training_stage(self, stage: int) -> None:
        if stage == 1:
            self.image_encoder.freeze()
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.token_compressor.parameters():
                param.requires_grad = True
            print("Stage 1: Training token compressor + projector only")
        elif stage == 2:
            self.image_encoder.freeze()
            self.llm.freeze_base_model()
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.token_compressor.parameters():
                param.requires_grad = True
            print("Stage 2: Training token compressor + projector + LoRA")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.print_trainable_parameters()

    def print_trainable_parameters(self) -> None:
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        vision_total, vision_train = count_params(self.image_encoder)
        comp_total, comp_train = count_params(self.token_compressor)
        proj_total, proj_train = count_params(self.projector)
        llm_total, llm_train = count_params(self.llm)

        print("\nTrainable Parameters:")
        print(f"  Vision Encoder:    {vision_train:,} / {vision_total:,}")
        print(f"  Token Compressor:  {comp_train:,} / {comp_total:,}")
        print(f"  Projector:         {proj_train:,} / {proj_total:,}")
        print(f"  LLM:               {llm_train:,} / {llm_total:,}")
        print(
            f"  Total:             {vision_train + comp_train + proj_train + llm_train:,} / "
            f"{vision_total + comp_total + proj_total + llm_total:,}"
        )

    def save_pretrained(self, save_path: str) -> None:
        import os

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))
        torch.save(
            self.token_compressor.state_dict(),
            os.path.join(save_path, "token_compressor.pt"),
        )
        self.llm.save_pretrained(os.path.join(save_path, "llm"))
        write_model_metadata(
            save_path,
            architecture=self.architecture,
            extra={
                "vision_encoder_type": self.vision_encoder_type,
                "token_compressor_type": self.token_compressor_type,
                "max_image_tokens": self.max_image_tokens,
                "min_image_tokens": self.min_image_tokens,
            },
        )
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs,
    ) -> "AnyMALv2":
        import os
        from peft import PeftModel

        validate_checkpoint_architecture(save_path, expected_architecture=cls.architecture)

        model = cls(llm_model_name=llm_model_name, **kwargs)
        projector_path = os.path.join(save_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path))

        compressor_path = os.path.join(save_path, "token_compressor.pt")
        if os.path.exists(compressor_path):
            model.token_compressor.load_state_dict(torch.load(compressor_path))

        llm_path = os.path.join(save_path, "llm")
        if os.path.exists(llm_path):
            base_model = model.llm.model
            if hasattr(base_model, "base_model"):
                base_model = base_model.base_model
            model.llm.model = PeftModel.from_pretrained(base_model, llm_path)
        return model
