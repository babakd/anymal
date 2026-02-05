"""
Image Encoder for AnyMAL

Wraps CLIP ViT to extract image features for multimodal learning.

Educational Notes:
-----------------
CLIP (Contrastive Language-Image Pre-training) was trained on 400M image-text
pairs using a contrastive objective. The vision encoder learns rich visual
representations that are semantically aligned with text.

Key insight: We use CLIP FROZEN during training. It already has excellent
visual representations - we just need to learn how to project them into
the LLM's embedding space.

Architecture Details (ViT-L/14):
- Patch size: 14x14 pixels
- Image size: 224x224 (standard) or 336x336 (high-res)
- Number of patches: (224/14)^2 = 256 spatial + 1 CLS = 257 total
- Hidden dimension: 1024
- Layers: 24 transformer blocks
- Attention heads: 16

Output shape: [batch_size, 257, 1024]
- 257 = 256 spatial patch tokens + 1 CLS token
- 1024 = hidden dimension

For ViT-G/14 (larger model):
- Hidden dimension: 1664
- Layers: 48
- Output shape: [batch_size, 257, 1664]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import open_clip


class ImageEncoder(nn.Module):
    """
    CLIP Vision Transformer wrapper for extracting image features.

    This encoder is kept FROZEN during training. The pretrained CLIP
    features are projected to the LLM space via the Perceiver Resampler.

    Args:
        model_name: CLIP model variant (e.g., "ViT-L-14", "ViT-G-14")
        pretrained: Pretrained weights source ("openai", "laion2b_s34b_b88k", etc.)
        cache_dir: Directory to cache downloaded weights
        freeze: Whether to freeze encoder weights (default: True)

    Example:
        >>> encoder = ImageEncoder("ViT-L-14", pretrained="openai")
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = encoder(images)
        >>> print(features.shape)  # [4, 257, 1024]
    """

    # Mapping of model names to their hidden dimensions
    HIDDEN_DIMS = {
        "ViT-L-14": 1024,
        "ViT-L-14-336": 1024,
        "ViT-H-14": 1280,
        "ViT-G-14": 1664,
        "ViT-bigG-14": 1664,
    }

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        cache_dir: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained

        # Load CLIP model using open_clip
        # open_clip provides access to both OpenAI and LAION trained models
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )

        # Extract just the vision encoder
        self.visual = model.visual

        # Store hidden dimension for downstream use
        self.hidden_dim = self.HIDDEN_DIMS.get(model_name, 1024)

        # Determine number of patches based on model
        # Standard ViT-L-14 with 224x224 images: (224/14)^2 + 1 = 257
        if "336" in model_name:
            self.num_patches = (336 // 14) ** 2 + 1  # 577 patches
        else:
            self.num_patches = (224 // 14) ** 2 + 1  # 257 patches

        # Freeze encoder by default
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters (for fine-tuning experiments)."""
        for param in self.visual.parameters():
            param.requires_grad = True
        self.visual.train()

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model parameters."""
        return next(self.visual.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.visual.parameters()).device

    def forward(
        self,
        images: torch.Tensor,
        return_cls_only: bool = False,
    ) -> torch.Tensor:
        """
        Extract image features from input images.

        Args:
            images: Input images [B, 3, H, W] normalized for CLIP
            return_cls_only: If True, return only CLS token [B, hidden_dim]
                           If False, return all patch tokens [B, num_patches, hidden_dim]

        Returns:
            Image features tensor
            - If return_cls_only: [B, hidden_dim]
            - Otherwise: [B, num_patches, hidden_dim]

        Educational Note:
        ----------------
        The forward pass of CLIP ViT:
        1. Patch embedding: Split image into 14x14 patches, project to hidden_dim
        2. Add positional embeddings (learned)
        3. Prepend learnable CLS token
        4. Pass through transformer layers
        5. Output: all tokens including CLS

        For multimodal LLMs, we typically use ALL patch tokens (not just CLS)
        because we want to preserve spatial information for tasks like:
        - "What's in the top-left corner?"
        - "Count the objects"
        - "Read the text in the image"
        """
        # CLIP ViT forward pass
        # We need to extract intermediate features, not just the final output
        features = self._forward_features(images)

        if return_cls_only:
            # Return only the CLS token (first token)
            return features[:, 0, :]

        return features

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract all patch features from the vision transformer.

        This method replicates the CLIP ViT forward pass but returns
        all tokens instead of just the projected CLS token.
        """
        visual = self.visual

        # Patch embedding
        x = visual.conv1(x)  # [B, hidden_dim, H/patch, W/patch]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, hidden_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, hidden_dim]

        # Prepend CLS token
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches + 1, hidden_dim]

        # Add positional embeddings
        x = x + visual.positional_embedding.to(x.dtype)

        # Pre-transformer layer norm (some models have this)
        if hasattr(visual, 'ln_pre'):
            x = visual.ln_pre(x)

        # Transformer blocks
        # Note: open_clip uses different attribute names depending on version
        if hasattr(visual, 'transformer'):
            x = visual.transformer(x)
        elif hasattr(visual, 'blocks'):
            for block in visual.blocks:
                x = block(x)
        else:
            raise AttributeError("Cannot find transformer blocks in visual encoder")

        # Post-transformer layer norm
        if hasattr(visual, 'ln_post'):
            x = visual.ln_post(x)

        return x

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.hidden_dim

    def get_num_patches(self) -> int:
        """Return the number of output tokens (patches + CLS)."""
        return self.num_patches


def create_image_encoder(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    **kwargs,
) -> ImageEncoder:
    """
    Factory function to create image encoder.

    Common configurations:
    - ViT-L-14 + openai: Standard CLIP, good balance of speed/quality
    - ViT-L-14 + laion2b_s34b_b88k: LAION trained, slightly better on some tasks
    - ViT-G-14 + laion2b_s34b_b88k: Larger model, best quality but slower

    Args:
        model_name: CLIP model variant
        pretrained: Pretrained weights source
        **kwargs: Additional arguments passed to ImageEncoder

    Returns:
        Configured ImageEncoder instance
    """
    return ImageEncoder(model_name=model_name, pretrained=pretrained, **kwargs)
