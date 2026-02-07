"""
Deterministic Train/Validation Split for AnyMAL

Provides a reproducible train/val split using a fixed random seed.
"""

import random
from torch.utils.data import Subset


def deterministic_train_val_split(dataset, val_fraction=0.05, seed=42):
    """
    Split a dataset into train and validation subsets deterministically.

    Uses a fixed random seed to ensure the same split every time,
    regardless of other random state in the program.

    Args:
        dataset: PyTorch Dataset to split
        val_fraction: Fraction of data for validation (default: 5%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset) as torch.utils.data.Subset objects
    """
    n = len(dataset)
    indices = list(range(n))

    # Use isolated RNG to avoid affecting global state
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(n * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
