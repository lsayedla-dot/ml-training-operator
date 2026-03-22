"""Tests for checkpoint save/load/resume."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from src.worker.checkpoint import list_checkpoints, load_latest_checkpoint, save_checkpoint


def _make_model():
    """Create a simple test model."""
    return nn.Linear(10, 4)


def test_save_checkpoint(tmp_path):
    """Save checkpoint creates a file."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(
        model, optimizer, epoch=3, metrics={"loss": 0.5}, checkpoint_dir=str(tmp_path)
    )
    assert os.path.exists(path)
    assert "epoch_0003" in path


def test_load_latest_checkpoint(tmp_path):
    """Load latest checkpoint restores model state."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())

    # Save two checkpoints
    save_checkpoint(model, optimizer, epoch=1, metrics={"loss": 0.8}, checkpoint_dir=str(tmp_path))
    save_checkpoint(model, optimizer, epoch=3, metrics={"loss": 0.3}, checkpoint_dir=str(tmp_path))

    # Load latest
    new_model = _make_model()
    new_optimizer = torch.optim.Adam(new_model.parameters())
    result = load_latest_checkpoint(str(tmp_path), new_model, new_optimizer)

    assert result is not None
    epoch, metrics = result
    assert epoch == 3
    assert metrics["loss"] == 0.3


def test_load_empty_dir(tmp_path):
    """Loading from empty dir returns None."""
    model = _make_model()
    result = load_latest_checkpoint(str(tmp_path), model)
    assert result is None


def test_list_checkpoints(tmp_path):
    """List checkpoints returns sorted list."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())

    save_checkpoint(model, optimizer, epoch=5, metrics={}, checkpoint_dir=str(tmp_path))
    save_checkpoint(model, optimizer, epoch=2, metrics={}, checkpoint_dir=str(tmp_path))
    save_checkpoint(model, optimizer, epoch=8, metrics={}, checkpoint_dir=str(tmp_path))

    checkpoints = list_checkpoints(str(tmp_path))
    assert len(checkpoints) == 3
    assert checkpoints[0].epoch == 2
    assert checkpoints[-1].epoch == 8


def test_resume_training(tmp_path):
    """Save + load preserves model weights."""
    model = _make_model()
    original_weight = model.weight.data.clone()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, epoch=1, metrics={}, checkpoint_dir=str(tmp_path))

    new_model = _make_model()
    load_latest_checkpoint(str(tmp_path), new_model)
    assert torch.allclose(new_model.weight.data, original_weight)
