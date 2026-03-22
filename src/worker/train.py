"""Main training script — runs inside K8s Job pods."""

from __future__ import annotations

import json
import os
import signal

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.worker.checkpoint import load_latest_checkpoint, save_checkpoint
from src.worker.dataset import NuScenesClassificationDataset
from src.worker.distributed import (
    cleanup_distributed,
    get_distributed_sampler,
    is_main_process,
    setup_distributed,
)
from src.worker.model import AVObjectClassifier

logger = structlog.get_logger()


def _get_config() -> dict:
    """Parse training configuration from environment variables."""
    return {
        "model_type": os.environ.get("MODEL_TYPE", "resnet18"),
        "dataset": os.environ.get("DATASET", "nuscenes-mini"),
        "epochs": int(os.environ.get("EPOCHS", "10")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "0.001")),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", "checkpoints"),
        "data_dir": os.environ.get("DATA_DIR", "data/nuscenes-mini"),
        "checkpoint_interval": int(os.environ.get("CHECKPOINT_INTERVAL", "2")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": int(os.environ.get("MASTER_PORT", "29500")),
        "synthetic": os.environ.get("SYNTHETIC", "false").lower() == "true",
        "enable_optimization": os.environ.get("ENABLE_OPTIMIZATION", "false").lower() == "true",
    }


def train() -> None:
    """Main training entry point."""
    config = _get_config()
    rank = config["rank"]
    world_size = config["world_size"]
    is_distributed = world_size > 1
    is_main = is_main_process(rank)

    if is_main:
        logger.info("training_starting", config=config)

    # Setup distributed training
    if is_distributed:
        setup_distributed(rank, world_size, config["master_addr"], config["master_port"])

    # Create dataset
    dataset = NuScenesClassificationDataset(
        data_dir=config["data_dir"],
        split="train",
        synthetic=config["synthetic"],
    )
    val_dataset = NuScenesClassificationDataset(
        data_dir=config["data_dir"],
        split="val",
        synthetic=config["synthetic"],
        num_synthetic_samples=50,
    )

    # Create data loaders
    if is_distributed:
        train_sampler = get_distributed_sampler(dataset, rank, world_size)
        train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Create model
    model = AVObjectClassifier(model_type=config["model_type"], pretrained=False)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_epoch = 0
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    base_model = model.module if is_distributed else model
    result = load_latest_checkpoint(checkpoint_dir, base_model, optimizer)
    if result is not None:
        start_epoch, _ = result
        start_epoch += 1
        if is_main:
            logger.info("resuming_from_checkpoint", start_epoch=start_epoch)

    # Graceful shutdown handler
    should_stop = False

    def signal_handler(signum, frame):
        nonlocal should_stop
        should_stop = True
        if is_main:
            logger.warning("sigterm_received", msg="saving checkpoint before exit")

    signal.signal(signal.SIGTERM, signal_handler)

    # Training loop
    best_accuracy = 0.0
    results = {"epochs": [], "final_accuracy": 0.0}

    for epoch in range(start_epoch, config["epochs"]):
        if should_stop:
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / max(len(train_loader), 1)
        train_accuracy = correct / max(total, 1)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = val_correct / max(val_total, 1)

        if is_main:
            logger.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                train_accuracy=round(train_accuracy, 4),
                val_accuracy=round(val_accuracy, 4),
            )

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
            results["epochs"].append(epoch_result)

            # Checkpoint
            if (epoch + 1) % config["checkpoint_interval"] == 0 or epoch == config["epochs"] - 1:
                metrics = {"loss": train_loss, "accuracy": train_accuracy}
                save_checkpoint(base_model, optimizer, epoch, metrics, checkpoint_dir)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

    results["final_accuracy"] = best_accuracy

    # Post-training optimization (rank 0 only)
    if is_main and config["enable_optimization"]:
        logger.info("starting_optimization_pipeline")
        try:
            from src.worker.optimize import run_optimization_pipeline

            sample_input = torch.randn(1, 3, 224, 224)
            artifacts_dir = os.path.join(checkpoint_dir, "artifacts")
            benchmarks = run_optimization_pipeline(base_model, sample_input, artifacts_dir)
            results["optimization"] = benchmarks
            logger.info("optimization_complete")
        except Exception as e:
            logger.error("optimization_failed", error=str(e))
            results["optimization_error"] = str(e)

    # Save final results (rank 0 only)
    if is_main:
        results_path = os.path.join(checkpoint_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("training_complete", results_path=results_path, best_accuracy=best_accuracy)

    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    train()
