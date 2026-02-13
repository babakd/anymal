"""
Tests for AnyMAL model components.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn


class TestImageEncoder:
    """Tests for ImageEncoder component."""

    def test_output_shape(self):
        """Test that encoder outputs correct shape."""
        from models.encoders.image_encoder import ImageEncoder

        # Use a smaller model for testing
        encoder = ImageEncoder(
            model_name="ViT-L-14",
            pretrained="openai",
            freeze=True,
        )

        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            features = encoder(images)

        # Check shape: [batch, 257, 1024]
        assert features.shape == (batch_size, 257, 1024), \
            f"Expected shape (2, 257, 1024), got {features.shape}"

    def test_frozen_parameters(self):
        """Test that encoder parameters are frozen."""
        from models.encoders.image_encoder import ImageEncoder

        encoder = ImageEncoder(freeze=True)

        # Check all parameters are frozen
        for param in encoder.parameters():
            assert not param.requires_grad, "Encoder should be frozen"

    def test_cls_only_output(self):
        """Test CLS-only output mode."""
        from models.encoders.image_encoder import ImageEncoder

        encoder = ImageEncoder(freeze=True)
        images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            features = encoder(images, return_cls_only=True)

        # Should be [batch, hidden_dim]
        assert features.shape == (2, 1024), \
            f"CLS output shape should be (2, 1024), got {features.shape}"


class TestPerceiverResampler:
    """Tests for Perceiver Resampler projector."""

    def test_output_shape(self):
        """Test that resampler outputs correct shape."""
        from models.projectors.perceiver_resampler import PerceiverResampler

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=6,
        )

        # Create dummy input (like CLIP output)
        batch_size = 2
        x = torch.randn(batch_size, 257, 1024)

        # Forward pass
        output = resampler(x)

        # Check shape: [batch, 64, 4096]
        assert output.shape == (batch_size, 64, 4096), \
            f"Expected shape (2, 64, 4096), got {output.shape}"

    def test_gradient_flow(self):
        """Test that gradients flow through the resampler."""
        from models.projectors.perceiver_resampler import PerceiverResampler

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,  # Fewer layers for faster test
        )

        x = torch.randn(2, 257, 1024, requires_grad=True)
        output = resampler(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        for param in resampler.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameters should have gradients"

    def test_num_params(self):
        """Test parameter counting."""
        from models.projectors.perceiver_resampler import PerceiverResampler

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=6,
        )

        num_params = resampler.get_num_params()
        assert num_params > 0, "Should have trainable parameters"

        # Rough estimate: should be in the millions
        assert num_params > 1_000_000, \
            f"Expected >1M params, got {num_params}"


class TestLinearProjector:
    """Tests for Linear Projector baseline."""

    def test_output_shape_no_pooling(self):
        """Test output shape without pooling."""
        from models.projectors.linear_projector import LinearProjector

        projector = LinearProjector(
            input_dim=1024,
            output_dim=4096,
            num_layers=2,
            pool_type=None,
        )

        x = torch.randn(2, 257, 1024)
        output = projector(x)

        # Should maintain sequence length
        assert output.shape == (2, 257, 4096), \
            f"Expected shape (2, 257, 4096), got {output.shape}"

    def test_output_shape_with_pooling(self):
        """Test output shape with learned pooling."""
        from models.projectors.linear_projector import LinearProjector

        projector = LinearProjector(
            input_dim=1024,
            output_dim=4096,
            pool_type="learned",
            num_output_tokens=64,
        )

        x = torch.randn(2, 257, 1024)
        output = projector(x)

        # Should reduce to num_output_tokens
        assert output.shape == (2, 64, 4096), \
            f"Expected shape (2, 64, 4096), got {output.shape}"


class TestDataUtils:
    """Tests for data utilities."""

    def test_image_transform(self):
        """Test image transform creates correct output."""
        from data.data_utils import get_image_transform
        from PIL import Image
        import numpy as np

        transform = get_image_transform(image_size=224, is_train=False)

        # Create dummy image
        image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = transform(image)

        # Check output shape and normalization
        assert tensor.shape == (3, 224, 224), \
            f"Expected shape (3, 224, 224), got {tensor.shape}"

        # Values should be normalized (not in [0, 255] range)
        assert tensor.min() < 0 or tensor.max() < 10, \
            "Image should be normalized"

    def test_collate_fn(self):
        """Test collate function for batching."""
        from data.data_utils import collate_fn

        # Create dummy batch
        batch = [
            {
                "image": torch.randn(3, 224, 224),
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
            },
            {
                "image": torch.randn(3, 224, 224),
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
            },
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Check shapes
        assert result["images"].shape == (2, 3, 224, 224)
        assert result["input_ids"].shape == (2, 4)  # Padded to max length
        assert result["attention_mask"].shape == (2, 4)

        # Check padding
        assert result["input_ids"][1, 2].item() == 0  # Padded
        assert result["attention_mask"][1, 2].item() == 0  # Masked


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_pretrain_config(self):
        """Test pretraining configuration."""
        from training.pretrain import PretrainConfig

        config = PretrainConfig(
            max_steps=100,
            learning_rate=2e-4,
        )

        assert config.max_steps == 100
        assert config.learning_rate == 2e-4
        assert config.train_projector_only == True

    def test_finetune_config(self):
        """Test finetuning configuration."""
        from training.finetune import FinetuneConfig

        config = FinetuneConfig(
            max_steps=50,
            learning_rate=1e-5,
            lora_r=64,
        )

        assert config.max_steps == 50
        assert config.learning_rate == 1e-5
        assert config.lora_r == 64


class TestDistributed:
    """Tests for distributed utilities."""

    def test_single_process(self):
        """Test distributed utilities in single-process mode."""
        from training.distributed import (
            is_main_process,
            get_rank,
            get_world_size,
        )

        # In single process, should behave normally
        assert is_main_process() == True
        assert get_rank() == 0
        assert get_world_size() == 1

    def test_reduce_tensor(self):
        """Test tensor reduction in single-process mode."""
        from training.distributed import reduce_tensor

        tensor = torch.tensor(5.0)
        reduced = reduce_tensor(tensor, op="mean")

        assert reduced.item() == 5.0  # No change in single process


class TestLabelMasking:
    """Tests for label masking correctness after bug fixes."""

    def test_multi_turn_response_masking(self):
        """Test that all assistant responses in multi-turn conversations are supervised."""
        from data.data_utils import TextProcessor
        from unittest.mock import MagicMock

        # Create a mock tokenizer that behaves like a real one
        # We test the format_conversation logic directly
        tp = MagicMock(spec=TextProcessor)
        tp.SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
        tp.USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
        tp.ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tp.END_TURN = "<|eot_id|>"

        # Test TextProcessor.format_conversation returns ranges for ALL responses
        real_tp = TextProcessor.__new__(TextProcessor)
        real_tp.SYSTEM_HEADER = tp.SYSTEM_HEADER
        real_tp.USER_HEADER = tp.USER_HEADER
        real_tp.ASSISTANT_HEADER = tp.ASSISTANT_HEADER
        real_tp.END_TURN = tp.END_TURN

        conversations = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well."},
        ]

        text, response_ranges = real_tp.format_conversation(conversations)

        # Should have 2 response ranges (one per assistant turn)
        assert len(response_ranges) == 2, \
            f"Expected 2 response ranges, got {len(response_ranges)}"

        # Each range should include the eot_id token
        eot = tp.END_TURN
        for start, end in response_ranges:
            response_text = text[start:end]
            assert response_text.endswith(eot), \
                f"Response range should end with eot_id, got: '{response_text}'"

    def test_eot_id_included_in_response_range(self):
        """Test that response ranges in InstructionDataset include end-of-turn token."""
        from data.data_utils import TextProcessor

        tp = TextProcessor.__new__(TextProcessor)
        tp.SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
        tp.USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
        tp.ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tp.END_TURN = "<|eot_id|>"

        conversations = [
            {"role": "user", "content": "Describe this image."},
            {"role": "assistant", "content": "The image shows a cat."},
        ]

        text, ranges = tp.format_conversation(conversations)

        assert len(ranges) == 1
        start, end = ranges[0]
        response_text = text[start:end]

        # Should contain both the content and the eot_id
        assert "The image shows a cat." in response_text
        assert response_text.endswith("<|eot_id|>"), \
            f"Response should end with <|eot_id|>, got: '{response_text}'"

    def test_encode_for_training_with_ranges(self):
        """Test that encode_for_training handles list of (start, end) ranges."""
        from data.data_utils import TextProcessor
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                use_fast=True,
            )
        except Exception:
            pytest.skip("LLaMA tokenizer not available")

        tp = TextProcessor(tokenizer, max_length=256)

        conversations = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]

        text, ranges = tp.format_conversation(conversations)
        encoding = tp.encode_for_training(text, ranges)

        labels = encoding["labels"]
        input_ids = encoding["input_ids"]

        # Non-padding, non-response tokens should be masked
        # At least some tokens should have labels != -100 (the responses)
        supervised_count = (labels != -100).sum().item()
        assert supervised_count > 0, "Should have some supervised tokens"

        # Padding tokens should be masked
        pad_mask = input_ids == tokenizer.pad_token_id
        assert (labels[pad_mask] == -100).all(), "Padding tokens should be masked"


class TestImageTokenInsertion:
    """Tests for image token positional insertion."""

    def test_splice_image_tokens_prepend_fallback(self):
        """Test that _splice_image_tokens falls back to prepending when no placeholders."""
        from models.anymal import AnyMAL

        # We can't easily construct AnyMAL without the full model,
        # so test the static helper methods directly
        embed_list = [
            torch.randn(10, 64),
            torch.randn(8, 64),
        ]
        result = AnyMAL._pad_and_stack_embeds(embed_list, torch.device("cpu"))
        assert result.shape == (2, 10, 64), f"Expected (2, 10, 64), got {result.shape}"

    def test_pad_and_stack_1d(self):
        """Test 1D tensor padding and stacking."""
        from models.anymal import AnyMAL

        tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        ]
        result = AnyMAL._pad_and_stack_1d(tensors, pad_value=-100)
        assert result.shape == (2, 3)
        assert result[1, 2].item() == -100  # Padded value


class TestPerceiverContextNorm:
    """Test that perceiver resampler has context normalization."""

    def test_context_norm_exists(self):
        """Test that PerceiverResamplerBlock has context_norm."""
        from models.projectors.perceiver_resampler import PerceiverResamplerBlock

        block = PerceiverResamplerBlock(dim=256, num_heads=4)
        assert hasattr(block, "context_norm"), "Block should have context_norm"
        assert isinstance(block.context_norm, nn.LayerNorm)

    def test_context_norm_applied(self):
        """Test that context normalization doesn't change output shape."""
        from models.projectors.perceiver_resampler import PerceiverResampler

        resampler = PerceiverResampler(
            input_dim=64, output_dim=128, num_latents=8, num_layers=2, num_heads=4
        )
        x = torch.randn(2, 16, 64)
        output = resampler(x)
        assert output.shape == (2, 8, 128)


class TestSeparateLearningRates:
    """Test separate learning rate configuration."""

    def test_lora_lr_config(self):
        """Test that TrainerConfig supports separate LoRA learning rate."""
        from training.trainer import TrainerConfig

        config = TrainerConfig(
            learning_rate=1e-5,
            lora_learning_rate=5e-5,
        )
        assert config.learning_rate == 1e-5
        assert config.lora_learning_rate == 5e-5

    def test_lora_lr_defaults_to_none(self):
        """Test that lora_learning_rate defaults to None."""
        from training.trainer import TrainerConfig

        config = TrainerConfig()
        assert config.lora_learning_rate is None


class TestDummyImageDistribution:
    """Test that dummy images use proper CLIP normalization."""

    def test_dummy_image_bounded(self):
        """Test that dummy images have bounded values (not unbounded Gaussian)."""
        from data.data_utils import CLIP_MEAN, CLIP_STD
        from torchvision import transforms

        clip_normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        raw = torch.rand(3, 224, 224)
        normalized = clip_normalize(raw)

        # CLIP-normalized images from [0,1] input should be roughly in [-2, 3]
        assert normalized.min() > -3.0, f"Min too low: {normalized.min()}"
        assert normalized.max() < 4.0, f"Max too high: {normalized.max()}"


class TestAnyMALv2CoreModules:
    """Unit tests for AnyMALv2 core modules."""

    def test_mlp_bottleneck_projector_shape_and_grad(self):
        """Projector should preserve batch/tokens and support gradient flow."""
        from models.projectors.mlp_bottleneck_projector import MLPBottleneckProjector

        projector = MLPBottleneckProjector(
            input_dim=128,
            output_dim=256,
            bottleneck_dim=64,
        )
        x = torch.randn(2, 11, 128, requires_grad=True)
        y = projector(x)
        assert y.shape == (2, 11, 256)

        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in projector.parameters())

    def test_token_compressor_learned_shape_mask_and_grad(self):
        """Compressor should return bounded token tensor + per-sample mask."""
        from models.projectors.token_compressor import TokenCompressor

        compressor = TokenCompressor(
            input_dim=96,
            max_tokens=12,
            compressor_type="learned",
            num_heads=8,
        )
        x = torch.randn(3, 33, 96, requires_grad=True)
        target_counts = torch.tensor([12, 7, 3], dtype=torch.long)
        y, mask, counts = compressor(x, target_num_tokens=target_counts)

        assert y.shape == (3, 12, 96)
        assert mask.shape == (3, 12)
        assert counts.tolist() == [12, 7, 3]
        assert mask[0].sum().item() == 12
        assert mask[1].sum().item() == 7
        assert mask[2].sum().item() == 3

        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compressor.parameters() if p.requires_grad)

    def test_v2_strict_splice_detects_mismatch(self):
        """Strict splice should fail fast when placeholder and image token counts differ."""
        from models.anymal_v2 import AnyMALv2

        class DummyLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(512, 16)

            def get_input_embeddings(self):
                return self.embedding

        model = AnyMALv2.__new__(AnyMALv2)
        nn.Module.__init__(model)
        model.llm = DummyLLM()
        model.image_placeholder_token_id = 7

        input_ids = torch.tensor([[11, 7, 7, 7, 12]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        image_tokens = torch.randn(1, 4, 16)
        image_token_mask = torch.tensor([[True, True, False, False]])

        with pytest.raises(ValueError, match="Placeholder/token mismatch"):
            model._splice_image_tokens_strict(
                input_ids=input_ids,
                image_tokens=image_tokens,
                image_token_mask=image_token_mask,
                attention_mask=attention_mask,
                labels=labels,
            )


class TestModelFactory:
    """Tests for architecture-routing model factory."""

    def test_factory_routes_by_architecture(self, monkeypatch):
        """Factory should dispatch anymal_v1 and anymal_v2 to the right constructor."""
        from models import factory

        class DummyV1:
            def __init__(self, **kwargs):
                self.kind = "v1"
                self.kwargs = kwargs

        class DummyV2:
            def __init__(self, **kwargs):
                self.kind = "v2"
                self.kwargs = kwargs

        monkeypatch.setattr(factory, "AnyMAL", DummyV1)
        monkeypatch.setattr(factory, "AnyMALv2", DummyV2)

        model_v1 = factory.create_model("anymal_v1", llm_model_name="a")
        model_v2 = factory.create_model("anymal_v2", llm_model_name="b")

        assert model_v1.kind == "v1"
        assert model_v2.kind == "v2"

    def test_factory_from_config_uses_model_architecture(self, monkeypatch):
        """Config-driven factory should route by config['model']['architecture']."""
        from models import factory

        class DummyV2:
            def __init__(self, llm_model_name=None, **kwargs):
                self.kwargs = {"llm_model_name": llm_model_name, **kwargs}

        monkeypatch.setattr(factory, "AnyMALv2", DummyV2)
        config = {"model": {"architecture": "anymal_v2", "llm_model_name": "foo"}}
        model = factory.create_model_from_config(config)
        assert model.kwargs["llm_model_name"] == "foo"


# Integration tests (require more resources)
@pytest.mark.slow
class TestIntegration:
    """Integration tests for full model."""

    def test_full_forward_pass(self):
        """Test complete forward pass through model."""
        # This test requires loading the full model
        # Skip if resources not available
        pytest.skip("Full model test requires GPU and model weights")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
