"""
SigLIP2 vision encoder wrapper for AnyMALv2.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

try:
    from transformers import Siglip2VisionModel
except Exception:  # pragma: no cover - fallback for older transformers
    Siglip2VisionModel = None


class SigLIP2Encoder(nn.Module):
    """
    HuggingFace SigLIP2 vision wrapper.

    Defaults to a So400m-family checkpoint.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        cache_dir: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        model_type = getattr(config, "model_type", "")

        should_try_siglip2_vision = (
            Siglip2VisionModel is not None and str(model_type).startswith("siglip2")
        )

        if should_try_siglip2_vision:
            try:
                loaded_model = Siglip2VisionModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                )
            except Exception:
                # Some checkpoints expose a mixed or legacy model_type and fail with
                # Siglip2VisionModel strict loading; AutoModel is more permissive.
                loaded_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            loaded_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # Some checkpoints load a full multimodal wrapper (with text + vision).
        # Keep only the vision tower for image feature extraction.
        self.model = getattr(loaded_model, "vision_model", loaded_model)

        config = getattr(self.model, "config", None)
        self.hidden_dim = getattr(config, "hidden_size", None)
        if self.hidden_dim is None:
            self.hidden_dim = getattr(config, "vision_hidden_size", None)
        if self.hidden_dim is None:
            vision_cfg = getattr(config, "vision_config", None)
            self.hidden_dim = getattr(vision_cfg, "hidden_size", None)
        if self.hidden_dim is None:
            self.hidden_dim = getattr(config, "projection_dim", None)
        if self.hidden_dim is None and hasattr(self.model, "embeddings"):
            patch_embedding = getattr(self.model.embeddings, "patch_embedding", None)
            weight = getattr(patch_embedding, "weight", None)
            if weight is not None and weight.ndim > 0:
                self.hidden_dim = int(weight.shape[0])
        if self.hidden_dim is None:
            raise ValueError(f"Could not infer hidden dim from model config: {model_name}")

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Some SigLIP checkpoints are trained at 384 and error on smaller
        # resolutions unless positional embeddings are interpolated.
        try:
            outputs = self.model(
                pixel_values=images,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
        except TypeError:
            # Older/newer model wrappers may not expose interpolate_pos_encoding.
            outputs = self.model(pixel_values=images, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.unsqueeze(1)
        raise RuntimeError(
            f"SigLIP2 encoder output for {self.model_name} does not expose token features."
        )

    def get_output_dim(self) -> int:
        return int(self.hidden_dim)
