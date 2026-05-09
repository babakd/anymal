"""
SigLIP2 vision encoder wrapper for AnyMALv2.
"""

from typing import Optional, Sequence

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

    def forward(
        self,
        images: torch.Tensor,
        output_hidden_states: bool = False,
        hidden_state_indices: Optional[Sequence[int]] = None,
    ):
        # Some SigLIP checkpoints are trained at 384 and error on smaller
        # resolutions unless positional embeddings are interpolated.
        kwargs = {
            "pixel_values": images,
            "return_dict": True,
        }
        config = getattr(self.model, "config", None)
        previous_output_hidden_states = None
        if output_hidden_states:
            kwargs["output_hidden_states"] = True
            if config is not None and hasattr(config, "output_hidden_states"):
                previous_output_hidden_states = config.output_hidden_states
                config.output_hidden_states = True
        try:
            try:
                outputs = self.model(**kwargs, interpolate_pos_encoding=True)
            except TypeError:
                # Older/newer model wrappers may not expose interpolate_pos_encoding.
                outputs = self.model(**kwargs)
        finally:
            if (
                output_hidden_states
                and config is not None
                and previous_output_hidden_states is not None
            ):
                config.output_hidden_states = previous_output_hidden_states
        if output_hidden_states:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                vision_output = getattr(outputs, "vision_model_output", None)
                hidden_states = getattr(vision_output, "hidden_states", None)
            if not hidden_states:
                indices = tuple(hidden_state_indices or (-3, -2, -1))
                hooked = self._forward_with_hidden_state_hooks(
                    images=images,
                    hidden_state_indices=indices,
                )
                if hooked is not None:
                    return hooked
                if (
                    len(indices) == 1
                    and hasattr(outputs, "last_hidden_state")
                    and outputs.last_hidden_state is not None
                ):
                    return (outputs.last_hidden_state,)
                raise RuntimeError(
                    f"SigLIP2 encoder output for {self.model_name} does not expose hidden states."
                )
            indices = tuple(hidden_state_indices or (-3, -2, -1))
            return tuple(hidden_states[idx] for idx in indices)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.unsqueeze(1)
        raise RuntimeError(
            f"SigLIP2 encoder output for {self.model_name} does not expose token features."
        )

    def _forward_with_hidden_state_hooks(
        self,
        images: torch.Tensor,
        hidden_state_indices: Sequence[int],
    ):
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(encoder, "layer", None)
        if layers is None:
            return None

        num_layers = len(layers)
        if num_layers <= 0:
            return None

        layer_indices = []
        for idx in hidden_state_indices:
            idx = int(idx)
            if idx == 0:
                return None
            layer_idx = idx - 1 if idx > 0 else num_layers + idx
            if layer_idx < 0 or layer_idx >= num_layers:
                raise IndexError(
                    f"Hidden state index {idx} maps to encoder layer {layer_idx}, "
                    f"but {self.model_name} has {num_layers} layers."
                )
            layer_indices.append(layer_idx)

        captured = {}
        handles = []

        def _capture(layer_idx):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, (list, tuple)) else output
                captured[layer_idx] = value

            return hook

        for layer_idx in sorted(set(layer_indices)):
            handles.append(layers[layer_idx].register_forward_hook(_capture(layer_idx)))

        kwargs = {
            "pixel_values": images,
            "return_dict": True,
        }
        try:
            try:
                self.model(**kwargs, interpolate_pos_encoding=True)
            except TypeError:
                self.model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()

        if any(layer_idx not in captured for layer_idx in layer_indices):
            missing = [layer_idx for layer_idx in layer_indices if layer_idx not in captured]
            raise RuntimeError(
                f"Could not capture SigLIP hidden state layers {missing} for {self.model_name}."
            )
        return tuple(captured[layer_idx] for layer_idx in layer_indices)

    def get_output_dim(self) -> int:
        return int(self.hidden_dim)
