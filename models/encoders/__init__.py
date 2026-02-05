"""
Modality Encoders

Currently implemented:
- ImageEncoder: CLIP ViT-L/14 or ViT-G wrapper

Future extensions:
- VideoEncoder: InternVideo wrapper
- AudioEncoder: CLAP wrapper
- IMUEncoder: IMU2CLIP wrapper
"""

from .image_encoder import ImageEncoder

__all__ = ["ImageEncoder"]
