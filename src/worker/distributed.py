"""Distributed Data Parallel (DDP) setup and utilities."""

from __future__ import annotations

import os
import time

import structlog
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

logger = structlog.get_logger()


def setup_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
) -> None:
    """Initialize the distributed process group for DDP training.

    Uses gloo backend (CPU). Production would use nccl for GPU.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    logger.info(
        "distributed_initialized",
        rank=rank,
        world_size=world_size,
        backend="gloo",
    )


def get_distributed_sampler(
    dataset: Dataset,
    rank: int,
    world_size: int,
) -> DistributedSampler:
    """Create a DistributedSampler that shards data across workers."""
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )


def is_main_process(rank: int) -> bool:
    """Check if this is the main (rank 0) process."""
    return rank == 0


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("distributed_cleanup_complete")


def log_gradient_sync_time(model: torch.nn.Module) -> float:
    """Measure time spent in AllReduce gradient synchronization.

    This is the communication bottleneck in distributed training.
    Returns sync time in milliseconds.
    """
    if not dist.is_initialized():
        return 0.0

    # Measure a dummy all-reduce to estimate sync overhead
    dummy = torch.zeros(1)
    start = time.perf_counter()
    dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
    sync_time_ms = (time.perf_counter() - start) * 1000

    logger.info("gradient_sync_time_ms", sync_time=sync_time_ms)
    return sync_time_ms
