"""
LLaMA Wrapper for AnyMAL

Wraps LLaMA-3-8B for multimodal input with QLoRA support.

Educational Notes:
-----------------
LLaMA Architecture Overview (LLaMA 3 8B):
- Hidden size: 4096
- Intermediate size (FFN): 14336
- Layers: 32
- Attention heads: 32
- KV heads: 8 (Grouped Query Attention)
- Vocab size: 128256
- Context length: 8192

Key Multimodal Integration:
1. LLaMA normally takes token IDs -> looks up embeddings -> processes
2. We bypass token lookup by directly injecting embedding tensors
3. Image tokens are concatenated with text embeddings
4. Format: [image_tokens, text_tokens] -> LLaMA -> logits

QLoRA (Quantized Low-Rank Adaptation):
- Quantize base model to 4-bit (NF4 format) -> ~4GB for 8B model
- Add small trainable LoRA adapters to attention layers
- Only train adapters (~0.1% of parameters)
- Memory efficient: can train 8B model on single GPU

LoRA Math:
Original: y = Wx
LoRA:     y = Wx + BAx, where B (r x d), A (d x r), r << d
- r = rank (typically 8-64)
- Only B and A are trained, W is frozen
- Merge at inference: W' = W + BA

Why freeze LLM?
- LLM already has strong language understanding
- We only need to teach it to "understand" image tokens
- Prevents catastrophic forgetting of language abilities
- Much faster training (fewer gradients to compute)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


class LlamaWrapper(nn.Module):
    """
    Wrapper for LLaMA-3 that supports multimodal input and QLoRA training.

    This wrapper:
    1. Loads LLaMA with optional 4-bit quantization
    2. Adds LoRA adapters for efficient fine-tuning
    3. Provides interface for multimodal forward pass

    Args:
        model_name: HuggingFace model name or path
        use_qlora: Whether to use 4-bit quantization
        lora_r: LoRA rank (dimension of low-rank matrices)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        lora_target_modules: Which modules to apply LoRA to
        load_in_8bit: Use 8-bit quantization instead of 4-bit
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        torch_dtype: Model dtype (bfloat16 recommended for training)
        use_flash_attention: Whether to use Flash Attention 2
        gradient_checkpointing: Enable gradient checkpointing for memory

    Example:
        >>> llm = LlamaWrapper(
        ...     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     use_qlora=True,
        ...     lora_r=64,
        ... )
        >>> # Get embeddings for multimodal input
        >>> text_embeds = llm.get_input_embeddings()(input_ids)
        >>> # Concatenate with image tokens
        >>> inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        >>> # Forward through LLM
        >>> outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
    """

    # Default LoRA target modules for LLaMA
    DEFAULT_LORA_TARGETS = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # FFN (MLP)
    ]

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        use_qlora: bool = True,
        use_lora: Optional[bool] = None,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_qlora = use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.use_lora = use_lora

        # Configure quantization
        bnb_config = None
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load base model
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "cache_dir": cache_dir,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # Set padding token if not set
        # Use LLaMA 3's dedicated pad token instead of eos_token to avoid
        # masking legitimate end-of-sequence tokens during label creation.
        if self.tokenizer.pad_token is None:
            if "<|filetune_right_pad_id|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|filetune_right_pad_id|>"
            elif "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Prepare for k-bit training if using quantization
        if use_qlora or load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=gradient_checkpointing,
            )

        enable_lora = use_qlora if use_lora is None else use_lora

        # Add LoRA adapters
        if enable_lora and lora_r > 0:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or self.DEFAULT_LORA_TARGETS,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)

        # Store model dimensions
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the token embedding layer.

        Used to embed text tokens before concatenating with image tokens.

        Returns:
            Token embedding module
        """
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward pass through the LLM.

        For multimodal input, use inputs_embeds instead of input_ids.
        The inputs_embeds should be: [image_tokens, text_embeddings]

        Args:
            input_ids: Token IDs [B, seq_len] (mutually exclusive with inputs_embeds)
            attention_mask: Attention mask [B, total_seq_len]
            inputs_embeds: Direct embeddings [B, total_seq_len, hidden_size]
            labels: Labels for language modeling loss [B, seq_len]
            past_key_values: Cached key/values for efficient generation
            use_cache: Whether to return key/value cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return ModelOutput or tuple

        Returns:
            CausalLMOutput with loss, logits, and optional hidden states

        Note on labels:
            Labels should be aligned with the TEXT portion only.
            Image tokens don't have labels (we don't predict them).
            The loss is computed only on text token predictions.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text given multimodal input.

        Args:
            inputs_embeds: [image_tokens + text_embeddings] for multimodal
            input_ids: Text-only input (alternative to inputs_embeds)
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy decoding)
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs [B, seq_len]
        """
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    def freeze_base_model(self) -> None:
        """
        Freeze all base model parameters.

        Only LoRA adapters remain trainable.
        """
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """
        Unfreeze base model parameters.

        Warning: This will significantly increase memory usage.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def print_trainable_parameters(self) -> None:
        """Print number of trainable vs total parameters."""
        trainable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        print(f"Trainable parameters: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")

    def save_pretrained(self, save_path: str, save_base_model: bool = False) -> bool:
        """
        Save model weights.

        Args:
            save_path: Directory to save model
            save_base_model: If False, skip writing full base LLM weights when no LoRA adapters exist.

        Returns:
            True if any LLM weights were written, False otherwise.
        """
        # For PeftModel, save_pretrained writes adapter weights (small).
        is_peft_model = hasattr(self.model, "peft_config")

        if is_peft_model or save_base_model:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            return True

        # Base model only + save_base_model=False: skip heavy write.
        return False

    def merge_and_save(self, save_path: str) -> None:
        """
        Merge LoRA weights into base model and save.

        This creates a full model without LoRA adapters.
        Useful for deployment.

        Args:
            save_path: Directory to save merged model
        """
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model parameters."""
        return next(self.model.parameters()).dtype


def create_llama_wrapper(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    use_qlora: bool = True,
    lora_r: int = 64,
    **kwargs,
) -> LlamaWrapper:
    """
    Factory function to create LLaMA wrapper.

    Common configurations:
    - Training: use_qlora=True, lora_r=64
    - Inference: use_qlora=False, load merged model

    Args:
        model_name: HuggingFace model name or path
        use_qlora: Whether to use 4-bit quantization with LoRA
        lora_r: LoRA rank
        **kwargs: Additional arguments

    Returns:
        Configured LlamaWrapper instance
    """
    return LlamaWrapper(
        model_name=model_name,
        use_qlora=use_qlora,
        lora_r=lora_r,
        **kwargs,
    )
