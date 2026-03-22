"""Scaling benchmark: measures training throughput across worker counts."""

from __future__ import annotations

import json
import os
import time

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.worker.dataset import NuScenesClassificationDataset
from src.worker.model import AVObjectClassifier

logger = structlog.get_logger()


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, int]:
    """Train one epoch and return (duration_seconds, samples_processed)."""
    model.train()
    total_samples = 0
    start = time.perf_counter()

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_samples += images.size(0)

    duration = time.perf_counter() - start
    return duration, total_samples


def _run_single_worker_benchmark(
    num_samples: int = 200,
    batch_size: int = 32,
    epochs: int = 2,
) -> dict:
    """Run a single-worker training benchmark with synthetic data."""
    dataset = NuScenesClassificationDataset(synthetic=True, num_synthetic_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AVObjectClassifier(model_type="resnet18", pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_duration = 0.0
    total_samples = 0
    for _ in range(epochs):
        duration, samples = _train_epoch(model, dataloader, optimizer, criterion)
        total_duration += duration
        total_samples += samples

    return {
        "duration_seconds": total_duration,
        "samples_processed": total_samples,
        "samples_per_second": total_samples / max(total_duration, 1e-6),
    }


def _run_multi_worker_benchmark(
    num_workers: int,
    num_samples: int = 200,
    batch_size: int = 32,
    epochs: int = 2,
) -> dict:
    """Simulate multi-worker training using torch.multiprocessing.spawn."""
    import torch.multiprocessing as mp
    from torch.utils.data.distributed import DistributedSampler

    results_dict: dict = mp.Manager().dict()

    def worker_fn(rank: int, world_size: int, results: dict) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29599"
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

        dataset = NuScenesClassificationDataset(synthetic=True, num_synthetic_samples=num_samples)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        model = AVObjectClassifier(model_type="resnet18", pretrained=False)
        model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        total_duration = 0.0
        total_samples = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            duration, samples = _train_epoch(model, dataloader, optimizer, criterion)
            total_duration += duration
            total_samples += samples

        results[rank] = {
            "duration_seconds": total_duration,
            "samples_processed": total_samples,
        }
        torch.distributed.destroy_process_group()

    mp.spawn(worker_fn, args=(num_workers, results_dict), nprocs=num_workers, join=True)

    # Aggregate results
    total_samples = sum(r["samples_processed"] for r in results_dict.values())
    max_duration = max(r["duration_seconds"] for r in results_dict.values())

    return {
        "duration_seconds": max_duration,
        "samples_processed": total_samples,
        "samples_per_second": total_samples / max(max_duration, 1e-6),
    }


def run_scaling_benchmark(
    max_workers: int = 4,
    output_dir: str = "benchmarks",
) -> dict:
    """Run training with 1, 2, and 4 workers, measuring throughput scaling.

    This analysis helps decide cluster sizing — the kind of work
    Motional's ML Infra team does to optimize training costs.
    """
    os.makedirs(output_dir, exist_ok=True)
    worker_counts = [w for w in [1, 2, 4] if w <= max_workers]

    report = {"worker_results": {}, "scaling_efficiency": {}}

    # Single worker baseline
    logger.info("benchmark_starting", workers=1)
    baseline = _run_single_worker_benchmark()
    report["worker_results"]["1"] = baseline
    baseline_throughput = baseline["samples_per_second"]

    for num_workers in worker_counts:
        if num_workers == 1:
            continue
        logger.info("benchmark_starting", workers=num_workers)
        try:
            result = _run_multi_worker_benchmark(num_workers)
            report["worker_results"][str(num_workers)] = result

            # Scaling efficiency: ideal is 100% (linear scaling)
            efficiency = result["samples_per_second"] / (num_workers * baseline_throughput) * 100
            report["scaling_efficiency"][str(num_workers)] = round(efficiency, 1)
        except Exception as e:
            logger.error("benchmark_failed", workers=num_workers, error=str(e))
            report["worker_results"][str(num_workers)] = {"error": str(e)}

    # Write report
    report_path = os.path.join(output_dir, "scaling_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("scaling_benchmark_complete", report_path=report_path)
    return report
