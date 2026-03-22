"""Checkpoint management for training jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import structlog
import torch

logger = structlog.get_logger()


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""

    epoch: int
    path: str
    loss: float | None = None
    accuracy: float | None = None


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
) -> str:
    """Save a training checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info("checkpoint_saved", epoch=epoch, path=path)
    return path


def load_latest_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict] | None:
    """Load the latest checkpoint. Returns (epoch, metrics) or None."""
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None

    latest = checkpoints[-1]
    checkpoint = torch.load(latest.path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    logger.info("checkpoint_loaded", epoch=epoch, path=latest.path)
    return epoch, metrics


def list_checkpoints(checkpoint_dir: str) -> list[CheckpointInfo]:
    """List all checkpoints in the directory, sorted by epoch."""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.startswith("checkpoint_epoch_") and fname.endswith(".pt"):
            path = os.path.join(checkpoint_dir, fname)
            try:
                # Extract epoch from filename
                epoch = int(fname.replace("checkpoint_epoch_", "").replace(".pt", ""))
                checkpoints.append(CheckpointInfo(epoch=epoch, path=path))
            except ValueError:
                continue

    return sorted(checkpoints, key=lambda c: c.epoch)
