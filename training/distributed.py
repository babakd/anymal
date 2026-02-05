"""
Distributed Training Utilities for AnyMAL

Provides utilities for multi-GPU training with PyTorch DDP.

Educational Notes:
-----------------
Distributed Data Parallel (DDP) Training:
1. Each GPU runs a copy of the model
2. Data is split across GPUs (different batches)
3. Gradients are synchronized after backward pass
4. All GPUs update weights with same gradients

Key concepts:
- World size: Total number of processes (usually = num GPUs)
- Rank: Unique ID for each process (0 to world_size-1)
- Local rank: GPU ID on current node (0 to num_gpus_per_node-1)

Environment variables (set by torchrun):
- WORLD_SIZE: Total processes
- RANK: Global process rank
- LOCAL_RANK: Local GPU ID
- MASTER_ADDR: Master node address
- MASTER_PORT: Master node port

Memory optimization tips:
1. Gradient checkpointing: Trade compute for memory
2. Mixed precision (bf16): 2x memory reduction
3. Gradient accumulation: Simulate larger batches
4. ZeRO optimization: Shard optimizer states (via FSDP or DeepSpeed)
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from contextlib import contextmanager


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int]:
    """
    Initialize distributed training environment.

    Should be called at the start of each process.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        init_method: Initialization method ("env://" uses environment variables)

    Returns:
        Tuple of (rank, local_rank, world_size)

    Example:
        >>> rank, local_rank, world_size = setup_distributed()
        >>> print(f"Process {rank}/{world_size} on GPU {local_rank}")
    """
    # Check if distributed environment is available
    if not dist.is_available():
        print("Distributed training not available, running single-GPU")
        return 0, 0, 1

    # Check for environment variables (set by torchrun)
    if "RANK" not in os.environ:
        print("Distributed environment not set up, running single-GPU")
        return 0, 0, 1

    # Get distributed info from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Synchronize all processes
    dist.barrier()

    if rank == 0:
        print(f"Initialized distributed training: {world_size} GPUs")

    return rank, local_rank, world_size


def cleanup_distributed():
    """
    Clean up distributed training.

    Should be called at the end of training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(
    tensor: torch.Tensor,
    op: str = "mean",
) -> torch.Tensor:
    """
    Reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("mean", "sum", "max", "min")

    Returns:
        Reduced tensor (on all ranks)
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    tensor = tensor.clone()

    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown reduction op: {op}")

    return tensor


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes to rank 0.

    Args:
        tensor: Tensor to gather

    Returns:
        Concatenated tensor on rank 0, original tensor on other ranks
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    world_size = get_world_size()

    # Create list to hold gathered tensors
    if is_main_process():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered = None

    dist.gather(tensor, gathered, dst=0)

    if is_main_process():
        return torch.cat(gathered, dim=0)
    return tensor


@contextmanager
def main_process_first():
    """
    Context manager to ensure main process runs first.

    Useful for operations like downloading files.

    Example:
        >>> with main_process_first():
        ...     download_dataset()  # Only rank 0 downloads
    """
    if not is_main_process():
        synchronize()

    try:
        yield
    finally:
        if is_main_process():
            synchronize()


def print_rank_0(message: str):
    """Print message only on rank 0."""
    if is_main_process():
        print(message)


class DistributedContext:
    """
    Context manager for distributed training.

    Handles setup and cleanup automatically.

    Example:
        >>> with DistributedContext() as ctx:
        ...     print(f"Rank {ctx.rank}/{ctx.world_size}")
        ...     # Training code here
    """

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

    def __enter__(self):
        self.rank, self.local_rank, self.world_size = setup_distributed(
            backend=self.backend
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
        return False


def compute_effective_batch_size(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: Optional[int] = None,
) -> int:
    """
    Compute the effective global batch size.

    Effective batch = per_device × accumulation × world_size

    Args:
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of accumulation steps
        world_size: Number of GPUs (auto-detected if None)

    Returns:
        Effective global batch size
    """
    if world_size is None:
        world_size = get_world_size()

    return per_device_batch_size * gradient_accumulation_steps * world_size
