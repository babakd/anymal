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
