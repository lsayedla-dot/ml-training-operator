"""Tests for DDP distributed training setup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import TensorDataset

from src.worker.distributed import (
    cleanup_distributed,
    get_distributed_sampler,
    is_main_process,
)


def test_is_main_process_rank_zero():
    """is_main_process returns True only for rank 0."""
    assert is_main_process(0) is True
    assert is_main_process(1) is False
    assert is_main_process(5) is False


def test_distributed_sampler_shards_data():
    """DistributedSampler shards dataset correctly across workers."""
    data = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 4, (100,))
    dataset = TensorDataset(data, labels)

    sampler_0 = get_distributed_sampler(dataset, rank=0, world_size=4)
    sampler_1 = get_distributed_sampler(dataset, rank=1, world_size=4)

    # Each worker should get approximately N/world_size samples
    indices_0 = list(iter(sampler_0))
    indices_1 = list(iter(sampler_1))

    assert len(indices_0) == 25  # 100 / 4
    assert len(indices_1) == 25
    # Indices should be different for different ranks
    assert indices_0 != indices_1


def test_rank_aware_logging():
    """Only rank 0 should log (verified by is_main_process gate)."""
    # This is a pattern test — the actual logging gate is in train.py
    for rank in range(4):
        should_log = is_main_process(rank)
        if rank == 0:
            assert should_log is True
        else:
            assert should_log is False


def test_checkpoint_gated_to_rank_zero():
    """Checkpoint saving should only happen on rank 0."""
    # Pattern: only rank 0 saves checkpoints
    for rank in range(4):
        should_save = is_main_process(rank)
        if rank == 0:
            assert should_save is True
        else:
            assert should_save is False


@patch("src.worker.distributed.dist")
def test_setup_initializes_process_group(mock_dist):
    """DDP setup initializes process group correctly."""
    from src.worker.distributed import setup_distributed

    mock_dist.init_process_group = MagicMock()
    setup_distributed(rank=0, world_size=2, master_addr="localhost", master_port=29500)
    mock_dist.init_process_group.assert_called_once_with(backend="gloo", rank=0, world_size=2)


@patch("src.worker.distributed.dist")
def test_cleanup_destroys_process_group(mock_dist):
    """Cleanup destroys the process group."""
    mock_dist.is_initialized.return_value = True
    cleanup_distributed()
    mock_dist.destroy_process_group.assert_called_once()
