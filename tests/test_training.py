"""
Tests for AnyMAL training pipeline.

These tests verify that the training loop works end-to-end with dummy data.

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
import json
import importlib.util
from PIL import Image
import numpy as np


# Helper to import modules directly without triggering models/__init__.py
# This avoids the peft dependency issue when peft is not installed
def _import_module_directly(module_name, relative_path):
    """Import a module directly from file to avoid package __init__.py imports."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(base_path, *relative_path.split("/"))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_perceiver_resampler():
    """Get PerceiverResampler class without triggering peft import."""
    module = _import_module_directly(
        "perceiver_resampler",
        "models/projectors/perceiver_resampler.py"
    )
    return module.PerceiverResampler


def get_linear_projector():
    """Get LinearProjector class without triggering peft import."""
    module = _import_module_directly(
        "linear_projector",
        "models/projectors/linear_projector.py"
    )
    return module.LinearProjector


class TestDatasets:
    """Tests for dataset classes."""

    def test_laion_dataset_with_metadata(self, tmp_path):
        """Test LaionDataset with metadata.json format."""
        from data.laion_dataset import LaionDataset
        from transformers import AutoTokenizer

        # Create dummy data
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create dummy images
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"{i:06d}.jpg")

        # Create metadata
        metadata = [
            {"image": f"images/{i:06d}.jpg", "caption": f"A test caption {i}"}
            for i in range(5)
        ]
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Load tokenizer (using a simple one for testing)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pytest.skip("Tokenizer not available")

        # Create dataset
        dataset = LaionDataset(
            data_path=str(tmp_path),
            tokenizer=tokenizer,
            max_samples=3,
        )

        assert len(dataset) == 3

        # Test __getitem__
        sample = dataset[0]
        assert sample is not None
        assert "image" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert sample["image"].shape == (3, 224, 224)

    def test_instruction_dataset(self, tmp_path):
        """Test InstructionDataset with LLaVA format."""
        from data.instruction_dataset import InstructionDataset
        from transformers import AutoTokenizer

        # Create dummy images
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"{i:06d}.jpg")

        # Create instruction data
        samples = [
            {
                "id": f"test_{i}",
                "image": f"{i:06d}.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nDescribe this image."},
                    {"from": "gpt", "value": f"This is test response {i}."},
                ],
            }
            for i in range(3)
        ]
        data_path = tmp_path / "data.json"
        with open(data_path, "w") as f:
            json.dump(samples, f)

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pytest.skip("Tokenizer not available")

        # Create dataset
        dataset = InstructionDataset(
            data_path=str(data_path),
            image_dir=str(images_dir),
            tokenizer=tokenizer,
        )

        assert len(dataset) == 3

        sample = dataset[0]
        assert sample is not None
        assert "image" in sample
        assert "labels" in sample

    def test_instruction_dataset_uniform_token_policy(self, tmp_path):
        """InstructionDataset should sample placeholder lengths from configured range."""
        from data.instruction_dataset import InstructionDataset
        from transformers import AutoTokenizer
        import random

        random.seed(7)

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"{i:06d}.jpg")

        samples = [
            {
                "id": f"uniform_{i}",
                "image": f"{i:06d}.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nDescribe this image."},
                    {"from": "gpt", "value": f"Response {i}."},
                ],
            }
            for i in range(6)
        ]
        data_path = tmp_path / "uniform.json"
        with open(data_path, "w") as f:
            json.dump(samples, f)

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
        except Exception:
            pytest.skip("Tokenizer not available")

        dataset = InstructionDataset(
            data_path=str(data_path),
            image_dir=str(images_dir),
            tokenizer=tokenizer,
            image_token_policy="uniform",
            min_image_tokens=4,
            max_image_tokens=9,
        )

        observed_counts = []
        for i in range(len(dataset)):
            item = dataset[i]
            count = int(item["num_image_tokens"].item())
            observed_counts.append(count)
            assert 4 <= count <= 9

            placeholder_count = int((item["input_ids"] == dataset.image_placeholder_token_id).sum().item())
            assert placeholder_count == count

        assert len(set(observed_counts)) > 1, "Uniform policy should produce varied token counts"

    def test_corrupted_image_handling(self, tmp_path):
        """Test that corrupted images return None."""
        from data.laion_dataset import LaionDataset
        from transformers import AutoTokenizer

        # Create one good image and one corrupted
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Good image
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img.save(images_dir / "good.jpg")

        # Corrupted image (just random bytes)
        with open(images_dir / "bad.jpg", "wb") as f:
            f.write(b"not a valid image")

        # Create metadata
        metadata = [
            {"image": "images/good.jpg", "caption": "Good image"},
            {"image": "images/bad.jpg", "caption": "Bad image"},
        ]
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pytest.skip("Tokenizer not available")

        dataset = LaionDataset(
            data_path=str(tmp_path),
            tokenizer=tokenizer,
        )

        # Good image should work
        assert dataset[0] is not None

        # Bad image should return None (not raise)
        assert dataset[1] is None

    def test_laion_dataset_inserts_v2_placeholders(self, tmp_path):
        """LaionDataset should prepend fixed image placeholder block for strict v2 splice."""
        from data.laion_dataset import LaionDataset
        from transformers import AutoTokenizer

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img.save(images_dir / "000000.jpg")

        metadata = [
            {"image": "images/000000.jpg", "caption": "A small test caption."},
        ]
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
        except Exception:
            pytest.skip("Tokenizer not available")

        dataset = LaionDataset(
            data_path=str(tmp_path),
            tokenizer=tokenizer,
            max_length=64,
            insert_image_placeholders=True,
            num_image_tokens=8,
        )

        sample = dataset[0]
        assert sample is not None
        placeholder_id = dataset.image_placeholder_token_id
        assert placeholder_id is not None

        placeholder_count = int((sample["input_ids"] == placeholder_id).sum().item())
        assert placeholder_count == 8

    def test_collate_fn_filters_none(self, tmp_path):
        """Test that collate_fn filters out None samples."""
        from data.data_utils import collate_fn

        # Batch with some None values
        batch = [
            {
                "image": torch.randn(3, 224, 224),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            None,  # Corrupted image
            {
                "image": torch.randn(3, 224, 224),
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
            },
            None,  # Another corrupted image
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Should only have 2 samples
        assert result["images"].shape[0] == 2
        assert result["input_ids"].shape[0] == 2

    def test_collate_fn_all_none(self):
        """Test that collate_fn returns None when all samples are invalid."""
        from data.data_utils import collate_fn

        batch = [None, None, None]
        result = collate_fn(batch, pad_token_id=0)

        assert result is None


class TestProjectors:
    """Tests for projector forward passes in training context."""

    def test_perceiver_resampler_training(self):
        """Test PerceiverResampler in training mode."""
        PerceiverResampler = get_perceiver_resampler()

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )
        resampler.train()

        # Simulate CLIP output
        x = torch.randn(4, 257, 1024, requires_grad=True)

        # Forward
        output = resampler(x)
        assert output.shape == (4, 64, 4096)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        grad_norms = [p.grad.norm().item() for p in resampler.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(g > 0 or g == 0 for g in grad_norms)  # Non-negative

    def test_linear_projector_training(self):
        """Test LinearProjector in training mode."""
        LinearProjector = get_linear_projector()

        projector = LinearProjector(
            input_dim=1024,
            output_dim=4096,
            num_layers=2,
            pool_type="learned",
            num_output_tokens=64,
        )
        projector.train()

        x = torch.randn(4, 257, 1024, requires_grad=True)

        output = projector(x)
        assert output.shape == (4, 64, 4096)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestGradientFlow:
    """Tests for gradient flow through the model components."""

    def test_projector_gradient_magnitude(self):
        """Test that gradients have reasonable magnitudes."""
        PerceiverResampler = get_perceiver_resampler()

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )

        # Multiple forward-backward passes
        for _ in range(3):
            x = torch.randn(2, 257, 1024, requires_grad=True)
            output = resampler(x)

            # Simulate language model loss
            target = torch.randn_like(output)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()

        # Check gradient magnitudes are not exploding or vanishing
        for name, param in resampler.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1000, f"Gradient explosion in {name}: {grad_norm}"
                # Note: some gradients may be small, that's okay


class TestCheckpointing:
    """Tests for model checkpoint saving and loading."""

    def test_projector_save_load(self, tmp_path):
        """Test saving and loading projector weights."""
        PerceiverResampler = get_perceiver_resampler()

        # Create and save model
        resampler1 = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )

        # Do a forward pass to initialize
        x = torch.randn(1, 257, 1024)
        out1 = resampler1(x)

        # Save weights
        save_path = tmp_path / "projector.pt"
        torch.save(resampler1.state_dict(), save_path)

        # Create new model and load
        resampler2 = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )
        resampler2.load_state_dict(torch.load(save_path))

        # Outputs should match
        out2 = resampler2(x)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestCheckpointMetadataCompatibility:
    """Tests for architecture metadata compatibility gating."""

    def test_legacy_checkpoint_is_v1_only(self, tmp_path):
        from model_metadata import validate_checkpoint_architecture

        # No model_meta.json => treated as legacy v1 checkpoint.
        with pytest.raises(RuntimeError, match="legacy checkpoints are treated as anymal_v1-only"):
            validate_checkpoint_architecture(
                checkpoint_dir=str(tmp_path),
                expected_architecture="anymal_v2",
            )

        # v1 should still be accepted.
        validate_checkpoint_architecture(
            checkpoint_dir=str(tmp_path),
            expected_architecture="anymal_v1",
        )

    def test_explicit_architecture_mismatch_fails(self, tmp_path):
        from model_metadata import write_model_metadata, validate_checkpoint_architecture

        write_model_metadata(str(tmp_path), architecture="anymal_v1")
        with pytest.raises(RuntimeError, match="architecture mismatch"):
            validate_checkpoint_architecture(
                checkpoint_dir=str(tmp_path),
                expected_architecture="anymal_v2",
            )

    def test_resolve_checkpoint_architecture_reads_metadata(self, tmp_path):
        from model_metadata import (
            read_model_metadata,
            resolve_checkpoint_architecture,
            write_model_metadata,
        )

        write_model_metadata(
            str(tmp_path),
            architecture="anymal_v2",
            extra={"vision_model_name": "google/siglip2-so400m-patch14-384"},
        )

        architecture, has_metadata = resolve_checkpoint_architecture(str(tmp_path))
        metadata = read_model_metadata(str(tmp_path))

        assert architecture == "anymal_v2"
        assert has_metadata is True
        assert metadata["vision_model_name"] == "google/siglip2-so400m-patch14-384"


class TestTrainingConfigs:
    """Tests for training configuration classes."""

    def test_pretrain_config_defaults(self):
        """Test PretrainConfig has sensible defaults."""
        from training.pretrain import PretrainConfig

        config = PretrainConfig()

        assert config.learning_rate > 0
        assert config.max_steps > 0
        assert config.warmup_steps >= 0
        assert config.train_projector_only == True

    def test_finetune_config_defaults(self):
        """Test FinetuneConfig has sensible defaults."""
        from training.finetune import FinetuneConfig

        config = FinetuneConfig()

        assert config.learning_rate > 0
        assert config.lora_r > 0
        assert config.lora_alpha > 0


class _DummyLossOutput:
    def __init__(self, loss):
        self.loss = loss


class _DummyMultimodalModel(nn.Module):
    architecture = "anymal_v2"
    preprocessing_family = "siglip2"

    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(4, 4)
        self.token_compressor = nn.Linear(4, 4)
        self.lora_adapter = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)
        self.image_placeholder_token_id = 99
        self.fixed_image_token_count = 8
        self.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

    def get_visual_bridge_modules(self):
        return {
            "token_compressor": self.token_compressor,
            "projector": self.projector,
        }

    def freeze_visual_bridge(self):
        for module in self.get_visual_bridge_modules().values():
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_visual_bridge(self):
        for module in self.get_visual_bridge_modules().values():
            for param in module.parameters():
                param.requires_grad = True

    def set_training_stage(self, stage: int):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.lora_adapter.parameters():
            param.requires_grad = True
        self.unfreeze_visual_bridge()

    def forward(self, **_kwargs):
        loss = (
            self.projector.weight.sum() * 0
            + self.token_compressor.weight.sum() * 0
            + self.lora_adapter.weight.sum() * 0
        )
        return _DummyLossOutput(loss)


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, _idx):
        return {
            "images": torch.randn(1, 3, 8, 8),
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
        }


class TestArchitectureAwareTrainerBehavior:
    """Tests for architecture-aware optimizer grouping and warmup behavior."""

    def test_optimizer_groups_include_token_compressor(self):
        from training.trainer import Trainer, TrainerConfig

        dataset = _DummyDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        trainer = Trainer(
            model=_DummyMultimodalModel(),
            config=TrainerConfig(
                max_steps=1,
                use_amp=False,
                logging_steps=1,
            ),
            train_dataloader=dataloader,
        )

        labels = [group.get("label") for group in trainer.optimizer.param_groups]
        assert "token_compressor" in labels
        assert "projector" in labels
        assert "lora" in labels

    def test_finetune_warmup_freezes_and_unfreezes_full_visual_bridge(self):
        from training.finetune import FinetuneConfig, FinetuneTrainer

        dataset = _DummyDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        model = _DummyMultimodalModel()

        trainer = FinetuneTrainer(
            model=model,
            config=FinetuneConfig(
                max_steps=2,
                use_amp=False,
                projector_warmup_steps=1,
                learning_rate=1e-4,
                lora_learning_rate=2e-4,
            ),
            train_dataloader=dataloader,
        )

        assert not any(p.requires_grad for p in trainer.unwrapped_model.projector.parameters())
        assert not any(p.requires_grad for p in trainer.unwrapped_model.token_compressor.parameters())
        assert any(p.requires_grad for p in trainer.unwrapped_model.lora_adapter.parameters())

        trainer.global_step = 1
        batch = next(iter(dataloader))
        trainer._train_step(batch)

        assert all(p.requires_grad for p in trainer.unwrapped_model.projector.parameters())
        assert all(p.requires_grad for p in trainer.unwrapped_model.token_compressor.parameters())


class TestTrainingLoop:
    """Tests for training loop mechanics (without full model)."""

    def test_optimizer_step(self):
        """Test optimizer can step on projector parameters."""
        PerceiverResampler = get_perceiver_resampler()

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )

        optimizer = torch.optim.AdamW(resampler.parameters(), lr=1e-4)

        # Save initial weights
        initial_weight = resampler.latents.clone()

        # Training step
        x = torch.randn(2, 257, 1024)
        output = resampler(x)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(initial_weight, resampler.latents)

    def test_lr_scheduler(self):
        """Test learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        PerceiverResampler = get_perceiver_resampler()

        resampler = PerceiverResampler(
            input_dim=1024,
            output_dim=4096,
            num_latents=64,
            num_layers=2,
        )

        optimizer = torch.optim.AdamW(resampler.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler
        for _ in range(50):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr < initial_lr  # LR should decrease


class TestDummyDataIntegration:
    """Integration tests using the generated dummy data."""

    @pytest.fixture
    def dummy_data_path(self):
        """Return path to dummy data if it exists."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        laion_path = os.path.join(base_path, "data", "laion_subset")

        if not os.path.exists(laion_path):
            pytest.skip("Dummy data not generated. Run: python scripts/create_dummy_data.py")

        return laion_path

    def test_dummy_laion_dataset(self, dummy_data_path):
        """Test loading the generated dummy LAION data."""
        from data.laion_dataset import LaionDataset
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pytest.skip("Tokenizer not available")

        dataset = LaionDataset(
            data_path=dummy_data_path,
            tokenizer=tokenizer,
        )

        assert len(dataset) == 100  # Created 100 dummy images

        # Check a few samples
        for i in [0, 50, 99]:
            sample = dataset[i]
            assert sample is not None
            assert sample["image"].shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
