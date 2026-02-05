"""
AnyMAL Training Module

Provides training loops for:
- Stage 1: Alignment pretraining (train projector only)
- Stage 2: Instruction tuning (train projector + LoRA)
"""

from .trainer import Trainer
from .pretrain import PretrainTrainer
from .finetune import FinetuneTrainer
from .distributed import setup_distributed, cleanup_distributed

__all__ = [
    "Trainer",
    "PretrainTrainer",
    "FinetuneTrainer",
    "setup_distributed",
    "cleanup_distributed",
]
