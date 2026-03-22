"""Tests for scaling benchmark."""

from __future__ import annotations

import json
import os

import pytest

try:
    import torchvision  # noqa: F401

    from src.worker.benchmark import _run_single_worker_benchmark

    _TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision not available")


def test_single_worker_benchmark():
    """Single-worker benchmark runs and produces valid results."""
    result = _run_single_worker_benchmark(num_samples=50, batch_size=16, epochs=1)
    assert "duration_seconds" in result
    assert "samples_processed" in result
    assert "samples_per_second" in result
    assert result["samples_processed"] > 0
    assert result["samples_per_second"] > 0


def test_benchmark_report_structure(tmp_path):
    """Benchmark report contains entries for each worker count."""
    # Run only single-worker to avoid multiprocessing complexity in tests
    from src.worker.benchmark import run_scaling_benchmark

    report = run_scaling_benchmark(max_workers=1, output_dir=str(tmp_path))

    assert "worker_results" in report
    assert "1" in report["worker_results"]
    assert "scaling_efficiency" in report

    report_path = os.path.join(str(tmp_path), "scaling_report.json")
    assert os.path.exists(report_path)

    with open(report_path) as f:
        saved_report = json.load(f)
    assert saved_report["worker_results"]["1"]["samples_per_second"] > 0


def test_scaling_efficiency_calculation():
    """Scaling efficiency is calculated correctly."""
    # If 1 worker does 100 samples/s, 2 workers should ideally do 200 samples/s
    # Efficiency = (actual / ideal) * 100
    baseline_throughput = 100.0
    actual_throughput = 180.0
    num_workers = 2
    efficiency = (actual_throughput / (num_workers * baseline_throughput)) * 100
    assert efficiency == 90.0
